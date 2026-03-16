"""
KVCIS PoC - Step 4: Compression Simulation

Simulates KV-cache compression and measures quality impact:
1. Load trained probe
2. For each test text:
   - Run prefill, get KV cache
   - Score tokens with probe
   - Simulate quantization based on importance
   - Measure perplexity on held-out tokens
3. Compare against baselines

Baselines:
- No compression (baseline)
- Uniform INT8 (quantize everything)
- KVCIS (importance-based)
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from pathlib import Path
import json
import argparse
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import math
from datasets import load_dataset


def quantize_tensor(tensor: torch.Tensor, bits: int) -> torch.Tensor:
    """Simple min-max uniform quantization."""
    original_dtype = tensor.dtype
    tensor = tensor.float()
    
    if bits >= 16:
        return tensor.to(original_dtype)
    if bits == 0:
        return torch.zeros_like(tensor).to(original_dtype)
    
    t_min = tensor.min()
    t_max = tensor.max()
    t_range = t_max - t_min + 1e-8
    
    n_levels = 2 ** bits
    scaled = ((tensor - t_min) / t_range * (n_levels - 1)).round()
    result = scaled / (n_levels - 1) * t_range + t_min
    return result.to(original_dtype)


def cache_to_list(cache) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Convert DynamicCache to list of (K, V) tuples."""
    if cache is None:
        return []
    
    if isinstance(cache, (list, tuple)):
        result = []
        for item in cache:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                result.append((item[0], item[1]))
            else:
                raise ValueError(f"Unexpected cache item format: {type(item)}")
        return result
    
    if hasattr(cache, 'key_cache') and hasattr(cache, 'value_cache'):
        return list(zip(cache.key_cache, cache.value_cache))
    
    if hasattr(cache, 'to_legacy_cache'):
        legacy = cache.to_legacy_cache()
        return [(item[0], item[1]) for item in legacy]
    
    raise ValueError(f"Cannot convert cache of type {type(cache)}")


def list_to_cache(kv_list: List[Tuple[torch.Tensor, torch.Tensor]]) -> DynamicCache:
    """Convert list of (K, V) tuples back to DynamicCache."""
    cache = DynamicCache()
    for k, v in kv_list:
        cache.update(k, v, layer_idx=len(cache))
    return cache


class KVCISCompressor:
    """KVCIS compression using trained probe."""
    
    def __init__(
        self,
        model: AutoModelForCausalLM,
        probe_path: str,
        extraction_layer: int = 12,
        high_threshold: float = 0.9,
        high_bits: int = 16,
        low_bits: int = 8,
    ):
        self.model = model
        self.extraction_layer = extraction_layer
        self.high_threshold = high_threshold
        self.high_bits = high_bits
        self.low_bits = low_bits
        self.device = next(model.parameters()).device
        
        # Load probe
        probe_dir = Path(probe_path)
        self.weights = np.load(probe_dir / "weights.npy")
        self.bias = np.load(probe_dir / "bias.npy")[0]
        
        # Convert to torch
        self.weights_t = torch.from_numpy(self.weights).float().to(self.device)
        self.bias_t = torch.tensor(self.bias).float().to(self.device)
        
        # Hook storage
        self._activations = None
        self._hook_handle = None
    
    def _hook_fn(self, module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        self._activations = hidden.detach()
    
    def setup_hook(self):
        layer = self.model.model.layers[self.extraction_layer]
        self._hook_handle = layer.register_forward_hook(self._hook_fn)
    
    def remove_hook(self):
        if self._hook_handle:
            self._hook_handle.remove()
            self._hook_handle = None
    
    def predict_importance(self, activations: torch.Tensor) -> torch.Tensor:
        """Predict importance scores from activations."""
        # activations: [batch, seq_len, hidden_dim]
        act_float = activations.float()
        # Linear probe: importance = act @ weights + bias
        scores = torch.matmul(act_float, self.weights_t) + self.bias_t
        # Clamp to [0, 1]
        return scores.clamp(0, 1)
    
    def compress(self, past_key_values, importance_scores: torch.Tensor) -> Tuple:
        """Compress KV cache based on importance scores."""
        kv_list = cache_to_list(past_key_values)
        
        # importance_scores: [batch, seq_len] or [seq_len]
        if importance_scores.dim() == 2:
            importance_scores = importance_scores[0]  # Take first batch
        
        seq_len = importance_scores.shape[0]
        
        compressed = []
        n_high = 0
        n_low = 0
        
        for k, v in kv_list:
            # k, v shape: [batch, heads, seq_len, head_dim]
            k_new = k.clone()
            v_new = v.clone()
            
            for pos in range(min(seq_len, k.shape[2])):
                if importance_scores[pos] >= self.high_threshold:
                    # High importance - keep fp16
                    bits = self.high_bits
                    n_high += 1
                else:
                    # Low importance - quantize
                    bits = self.low_bits
                    n_low += 1
                
                if bits < 16:
                    k_new[:, :, pos, :] = quantize_tensor(k[:, :, pos, :], bits)
                    v_new[:, :, pos, :] = quantize_tensor(v[:, :, pos, :], bits)
            
            compressed.append((k_new, v_new))
        
        # Calculate memory ratio
        n_layers = len(kv_list)
        total_positions = n_high + n_low
        if total_positions > 0:
            # Each position counted n_layers times
            high_ratio = n_high / total_positions
            memory_ratio = high_ratio * 1.0 + (1 - high_ratio) * (self.low_bits / self.high_bits)
        else:
            memory_ratio = 1.0
        
        stats = {
            "method": "KVCIS",
            "n_high": n_high // n_layers,
            "n_low": n_low // n_layers,
            "memory_ratio": memory_ratio,
            "threshold": self.high_threshold,
        }
        
        return list_to_cache(compressed), stats


class UniformQuantizer:
    """Uniform quantization baseline."""
    
    def __init__(self, bits: int = 8):
        self.bits = bits
    
    def compress(self, past_key_values) -> Tuple:
        kv_list = cache_to_list(past_key_values)
        
        compressed = []
        for k, v in kv_list:
            k_q = quantize_tensor(k, self.bits)
            v_q = quantize_tensor(v, self.bits)
            compressed.append((k_q, v_q))
        
        memory_ratio = self.bits / 16
        stats = {
            "method": f"Uniform-INT{self.bits}",
            "memory_ratio": memory_ratio,
            "bits": self.bits,
        }
        
        return list_to_cache(compressed), stats


def load_eval_texts(n_texts: int = 100, max_length: int = 512) -> List[str]:
    """Load evaluation texts from WikiText."""
    try:
        dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
    except:
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    
    texts = []
    current = ""
    
    for item in dataset:
        text = item.get("text", "")
        if not text.strip():
            continue
        current += " " + text
        if len(current) > max_length * 4:  # Rough char estimate
            texts.append(current.strip())
            current = ""
            if len(texts) >= n_texts:
                break
    
    return texts[:n_texts]


def evaluate_compression(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    texts: List[str],
    compressor: Optional[object] = None,
    kvcis: Optional[KVCISCompressor] = None,
    max_length: int = 512,
    eval_ratio: float = 0.3,
) -> Dict:
    """Evaluate compression method on texts."""
    
    device = next(model.parameters()).device
    total_loss = 0.0
    total_tokens = 0
    memory_ratios = []
    
    is_kvcis = kvcis is not None
    method_name = "KVCIS" if is_kvcis else (
        compressor.compress.__self__.__class__.__name__ if compressor else "Baseline"
    )
    if hasattr(compressor, 'bits'):
        method_name = f"Uniform-INT{compressor.bits}"
    
    for text in tqdm(texts, desc=f"Evaluating {method_name}"):
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        ).to(device)
        
        seq_len = inputs.input_ids.shape[1]
        if seq_len < 20:
            continue
        
        # Split into prompt and eval
        prompt_len = int(seq_len * (1 - eval_ratio))
        prompt_ids = inputs.input_ids[:, :prompt_len]
        target_ids = inputs.input_ids[:, prompt_len:]
        
        if target_ids.shape[1] < 5:
            continue
        
        with torch.no_grad():
            # Prefill
            if is_kvcis:
                kvcis.setup_hook()
            
            outputs = model(
                prompt_ids,
                use_cache=True,
                return_dict=True,
            )
            
            if is_kvcis:
                kvcis.remove_hook()
                activations = kvcis._activations
                importance = kvcis.predict_importance(activations)
                compressed_kv, stats = kvcis.compress(outputs.past_key_values, importance)
            elif compressor:
                compressed_kv, stats = compressor.compress(outputs.past_key_values)
            else:
                compressed_kv = outputs.past_key_values
                stats = {"method": "Baseline", "memory_ratio": 1.0}
            
            memory_ratios.append(stats.get("memory_ratio", 1.0))
            
            # Evaluate on targets
            eval_outputs = model(
                target_ids,
                past_key_values=compressed_kv,
                return_dict=True,
            )
            
            # Compute loss
            shift_logits = eval_outputs.logits[:, :-1, :].contiguous()
            shift_labels = target_ids[:, 1:].contiguous()
            
            loss_fn = torch.nn.CrossEntropyLoss(reduction='sum')
            loss = loss_fn(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
            
            total_loss += loss.item()
            total_tokens += shift_labels.numel()
    
    if total_tokens == 0:
        return {"error": "No valid texts"}
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    
    return {
        "method": method_name,
        "perplexity": perplexity,
        "avg_loss": avg_loss,
        "total_tokens": total_tokens,
        "memory_ratio": np.mean(memory_ratios),
        "memory_reduction_pct": (1 - np.mean(memory_ratios)) * 100,
    }


def main():
    parser = argparse.ArgumentParser(description="KVCIS Step 4: Compression Evaluation")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--probe-path", type=str, required=True)
    parser.add_argument("--extraction-layer", type=int, default=12)
    parser.add_argument("--n-texts", type=int, default=50)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--high-threshold", type=float, default=0.9)
    parser.add_argument("--output-dir", type=str, default="./results")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="eager",
    )
    model.eval()

    # Load eval texts
    print(f"\nLoading {args.n_texts} evaluation texts...")
    texts = load_eval_texts(args.n_texts, args.max_length)
    print(f"Loaded {len(texts)} texts")

    results = []

    # Baseline (no compression)
    print("\n--- Baseline (no compression) ---")
    baseline = evaluate_compression(model, tokenizer, texts, None, None, args.max_length)
    results.append(baseline)
    print(f"Perplexity: {baseline['perplexity']:.4f}")

    # Uniform INT8
    print("\n--- Uniform INT8 ---")
    uniform8 = UniformQuantizer(bits=8)
    u8_result = evaluate_compression(model, tokenizer, texts, uniform8, None, args.max_length)
    results.append(u8_result)
    ppl_change = (u8_result['perplexity'] / baseline['perplexity'] - 1) * 100
    print(f"Perplexity: {u8_result['perplexity']:.4f} ({ppl_change:+.2f}%)")
    print(f"Memory reduction: {u8_result['memory_reduction_pct']:.1f}%")

    # KVCIS
    print("\n--- KVCIS ---")
    kvcis = KVCISCompressor(
        model=model,
        probe_path=args.probe_path,
        extraction_layer=args.extraction_layer,
        high_threshold=args.high_threshold,
    )
    kvcis_result = evaluate_compression(model, tokenizer, texts, None, kvcis, args.max_length)
    results.append(kvcis_result)
    ppl_change = (kvcis_result['perplexity'] / baseline['perplexity'] - 1) * 100
    print(f"Perplexity: {kvcis_result['perplexity']:.4f} ({ppl_change:+.2f}%)")
    print(f"Memory reduction: {kvcis_result['memory_reduction_pct']:.1f}%")

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Method':<20} {'Memory Red.':>12} {'PPL':>10} {'PPL Δ':>10}")
    print("-"*55)
    for r in results:
        ppl_delta = (r['perplexity'] / baseline['perplexity'] - 1) * 100
        print(f"{r['method']:<20} {r['memory_reduction_pct']:>11.1f}% {r['perplexity']:>10.4f} {ppl_delta:>+9.2f}%")

    # KVCIS advantage
    if u8_result['perplexity'] > baseline['perplexity']:
        u8_degradation = (u8_result['perplexity'] / baseline['perplexity'] - 1) * 100
        kvcis_degradation = (kvcis_result['perplexity'] / baseline['perplexity'] - 1) * 100
        if kvcis_degradation > 0 and u8_degradation > 0:
            advantage = u8_degradation / kvcis_degradation
            print(f"\nKVCIS advantage: {advantage:.1f}x better quality than Uniform-INT8")

    # Save results
    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_dir}/results.json")
    print("\n✓ Step 4 complete - compression evaluation done")


if __name__ == "__main__":
    main()
