"""
KVCIS PoC - Step 3: Train Importance Probe

Trains a linear probe to predict token importance from activations:
1. Load activation → importance training data
2. Train Ridge regression probe
3. Evaluate on held-out test set
4. Save trained probe

Output:
  - probe/regression/weights.npy: Probe weights
  - probe/regression/bias.npy: Probe bias
  - probe/regression/metrics.json: Training metrics
"""

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from pathlib import Path
import json
import argparse
import joblib


def train_probe(
    activations: np.ndarray,
    importance: np.ndarray,
    alpha: float = 1.0,
    test_size: float = 0.2,
) -> dict:
    """Train a linear probe to predict importance from activations."""
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        activations, importance, test_size=test_size, random_state=42
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    print(f"Feature dimension: {X_train.shape[1]}")
    
    # Train Ridge regression
    print(f"\nTraining Ridge probe (alpha={alpha})...")
    probe = Ridge(alpha=alpha)
    probe.fit(X_train, y_train)
    
    # Evaluate
    y_train_pred = probe.predict(X_train)
    y_test_pred = probe.predict(X_test)
    
    metrics = {
        "train_r2": float(r2_score(y_train, y_train_pred)),
        "test_r2": float(r2_score(y_test, y_test_pred)),
        "train_mse": float(mean_squared_error(y_train, y_train_pred)),
        "test_mse": float(mean_squared_error(y_test, y_test_pred)),
        "train_corr": float(np.corrcoef(y_train, y_train_pred)[0, 1]),
        "test_corr": float(np.corrcoef(y_test, y_test_pred)[0, 1]),
    }
    
    return probe, metrics, (X_test, y_test, y_test_pred)


def analyze_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metadata: list = None,
) -> dict:
    """Analyze probe predictions."""
    
    analysis = {}
    
    # Overall stats
    analysis["correlation"] = float(np.corrcoef(y_true, y_pred)[0, 1])
    analysis["r2"] = float(r2_score(y_true, y_pred))
    
    # Prediction ranges
    analysis["pred_min"] = float(y_pred.min())
    analysis["pred_max"] = float(y_pred.max())
    analysis["pred_mean"] = float(y_pred.mean())
    analysis["pred_std"] = float(y_pred.std())
    
    # High importance token detection
    high_threshold = 0.8
    true_high = y_true >= high_threshold
    pred_high = y_pred >= high_threshold
    
    if true_high.sum() > 0:
        # How many true high-importance tokens did we catch?
        recall = (true_high & pred_high).sum() / true_high.sum()
        analysis["high_importance_recall"] = float(recall)
    
    if pred_high.sum() > 0:
        # How many predicted high-importance are actually high?
        precision = (true_high & pred_high).sum() / pred_high.sum()
        analysis["high_importance_precision"] = float(precision)
    
    return analysis


def main():
    parser = argparse.ArgumentParser(description="KVCIS Step 3: Train Probe")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./probe")
    parser.add_argument("--alpha", type=float, default=1.0, help="Ridge regularization")
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading data from {data_dir}...")
    activations = np.load(data_dir / "activations.npy")
    importance = np.load(data_dir / "importance.npy")
    
    print(f"Loaded {len(importance)} samples")
    print(f"Activation shape: {activations.shape}")
    print(f"Importance range: [{importance.min():.4f}, {importance.max():.4f}]")

    # Load metadata if available
    metadata = None
    metadata_path = data_dir / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)

    # Train probe
    probe, metrics, (X_test, y_test, y_pred) = train_probe(
        activations, importance,
        alpha=args.alpha,
        test_size=args.test_size,
    )

    print(f"\n--- Training Results ---")
    print(f"Train R²: {metrics['train_r2']:.4f}")
    print(f"Test R²:  {metrics['test_r2']:.4f}")
    print(f"Train Correlation: {metrics['train_corr']:.4f}")
    print(f"Test Correlation:  {metrics['test_corr']:.4f}")

    # Analyze predictions
    analysis = analyze_predictions(y_test, y_pred)
    print(f"\n--- Prediction Analysis ---")
    print(f"Prediction range: [{analysis['pred_min']:.4f}, {analysis['pred_max']:.4f}]")
    print(f"Prediction mean: {analysis['pred_mean']:.4f}")
    if "high_importance_recall" in analysis:
        print(f"High importance recall: {analysis['high_importance_recall']:.4f}")
    if "high_importance_precision" in analysis:
        print(f"High importance precision: {analysis['high_importance_precision']:.4f}")

    # Save probe
    probe_dir = output_dir / "regression"
    probe_dir.mkdir(parents=True, exist_ok=True)
    
    np.save(probe_dir / "weights.npy", probe.coef_)
    np.save(probe_dir / "bias.npy", np.array([probe.intercept_]))
    
    # Save full sklearn model too
    joblib.dump(probe, probe_dir / "probe.joblib")
    
    # Save metrics
    all_metrics = {**metrics, **analysis, "alpha": args.alpha}
    with open(probe_dir / "metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    print(f"\nProbe saved to {probe_dir}")
    print(f"  weights.npy: {probe.coef_.shape}")
    print(f"  bias.npy: {probe.intercept_}")
    print(f"  probe.joblib: sklearn model")

    # Test probe loading
    print("\n--- Verification ---")
    loaded_weights = np.load(probe_dir / "weights.npy")
    loaded_bias = np.load(probe_dir / "bias.npy")
    
    # Manual prediction
    manual_pred = X_test @ loaded_weights + loaded_bias[0]
    diff = np.abs(manual_pred - y_pred).max()
    print(f"Max difference between sklearn and manual prediction: {diff:.10f}")
    
    if diff < 1e-6:
        print("✓ Probe weights verified")
    else:
        print("⚠ Warning: Prediction mismatch")

    print("\n✓ Step 3 complete - probe trained and saved")


if __name__ == "__main__":
    main()
