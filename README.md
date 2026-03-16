# KVCIS: Activation-Based Token Importance Prediction for Intelligent KV-Cache Compression

This repository contains the code, data, and experimental results for the paper:

> **KVCIS: Activation-Based Token Importance Prediction for Intelligent KV-Cache Compression**  
> Glen Messenger  

## Abstract

We introduce KVCIS (KV-Cache Importance Scoring), a novel approach to KV-cache compression that predicts token importance from intermediate-layer activations before attention is computed. Unlike existing methods that make compression decisions based on attention scores computed during generation, KVCIS enables proactive compression at cache insertion time. A simple linear probe achieves R² = 0.998 overall and R² = 0.68–0.79 for discriminating among content tokens. Extensive validation demonstrates **50% memory reduction with zero degradation on NarrativeQA**, while uniform quantization degrades by 7.8% at the same compression ratio.

## Key Results

| Method | Memory Reduction | NarrativeQA F1 | F1 Δ |
|--------|------------------|----------------|------|
| Baseline | 0% | 0.0642 | — |
| **KVCIS** | **50%** | **0.0642** | **0.0%** |
| Uniform-INT8 | 50% | 0.0592 | -7.8% |
| H2O | 40% | 0.0642 | 0.0% |

## Repository Structure

```
kvcis/
├── code/                    # Source code
│   ├── step1_single_prompt.py    # Activation extraction test
│   ├── step2_collect_data.py     # Training data collection
│   ├── step3_train_probe.py      # Probe training
│   ├── step4_compression_eval.py # Compression evaluation
│   ├── longctx_eval.py           # Long-context experiments
│   └── requirements.txt          # Dependencies
├── results/                 # Experimental results
│   ├── baseline_comparison.json  # Table 1 data
│   ├── downstream_tasks.json     # Table 2 data
│   └── longctx_scaling.json      # Table 3 data
├── scripts/                 # Reproduction scripts
│   ├── run_all.sh               # Full reproduction
│   └── run_quick.sh             # Quick validation
└── README.md
```

## Quick Start

### Requirements

- Python 3.10+
- PyTorch 2.0+
- CUDA-capable GPU (24GB+ VRAM recommended)
- Access to Llama-3.1-8B-Instruct via Hugging Face

### Installation

```bash
git clone https://github.com/your-username/kvcis.git
cd kvcis
pip install -r code/requirements.txt
```

### Reproduce Main Results

```bash
# Full reproduction (requires ~2 hours on A100)
./scripts/run_all.sh

# Quick validation (requires ~15 minutes)
./scripts/run_quick.sh
```

### Step-by-Step Execution

```bash
cd code

# Step 1: Verify activation extraction
python step1_single_prompt.py --model meta-llama/Llama-3.1-8B-Instruct

# Step 2: Collect training data (500 prompts)
python step2_collect_data.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --n-prompts 500 \
    --output-dir ../data

# Step 3: Train importance probe
python step3_train_probe.py \
    --data-dir ../data \
    --output-dir ../data/probe

# Step 4: Evaluate compression
python step4_compression_eval.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --probe-path ../data/probe/regression \
    --n-texts 50

# Step 5: Long-context evaluation
python longctx_eval.py \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --probe-path ../data/probe/regression \
    --context-lengths 512 1024 2048 \
    --n-texts 20
```

## Method Overview

KVCIS operates in two phases:

### Training (Offline)
1. Process diverse prompts through the model
2. Extract activations at layer 12 (~37% depth)
3. Generate tokens while recording attention patterns
4. Train linear probe: `importance = activation · weights + bias`

### Inference (Online)
1. During prefill, extract activations at layer 12
2. Predict importance via single dot product (~0.1ms)
3. Store high-importance tokens at fp16, others at int8
4. Proceed with generation using compressed cache

```
Importance > 0.9  →  fp16 (full precision)
Importance ≤ 0.9  →  int8 (quantized)
```

## Results Reproduction

### Table 1: Baseline Comparison (256 tokens)

| Method | Mem. ↓ | PPL | PPL Δ |
|--------|--------|-----|-------|
| Baseline | 0% | 27.62 | — |
| Uniform-INT8 | 50% | 28.84 | +4.40% |
| H2O-20 | 41% | 27.63 | +0.04% |
| **KVCIS** | **45%** | **27.78** | **+0.57%** |

### Table 2: Downstream Tasks

| Method | Mem. ↓ | NarrativeQA F1 | Passkey Acc. |
|--------|--------|----------------|--------------|
| Baseline | 0% | 0.0642 | 100% |
| Uniform-INT8 | 50% | 0.0592 | 100% |
| **KVCIS** | **50%** | **0.0642** | **100%** |

### Table 3: Long-Context Scaling

| Context | KVCIS PPL Δ | Uniform-INT8 PPL Δ | Advantage |
|---------|-------------|--------------------| ----------|
| 256 | +0.57% | +4.40% | 7.7× |
| 1024 | -0.07% | +0.47% | 6.7× |
| 2048 | +0.14% | +0.73% | 5.2× |

## Citation

```bibtex
@article{messenger2026kvcis,
  title={KVCIS: Activation-Based Token Importance Prediction for Intelligent KV-Cache Compression},
  author={Messenger, Glen},
  year={2026}
}
```
## License

Apache 2.0
