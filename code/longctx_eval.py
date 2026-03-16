"""
KVCIS Long-Context Evaluation

Tests compression methods at various context lengths (512 to 16K+).
Compares:
- Baseline (no compression)
- Uniform INT8
- KVCIS (ours)
- H2O (attention-based) - optional, OOMs at long context
- StreamingLLM (sink + window) - optional

Usage:
    python longctx_eval.py --probe-path ./probe/regression --context-lengths 512 1024 2048 4096
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


# Import from step4
from step4_compression_eval import (
    quantize_tensor,
    cache_to_list,
    list_to_cache,
    KVCISCompressor,
    UniformQuantizer,
)


class H2OCompressor:
    """H2O: Heavy-Hitter Oracle - evict based on attention scores."""
    
    def __init__(self, keep_ratio: float = 0.2):
        self.keep_ratio = keep_ratio
        self._attention_scores = None
    
    def set_attention(self, attention_scores: torch.Tensor):
        """Set attention scores from prefill."""
        # attention_scores: [batch, heads, seq, seq] from last layer
        # Sum attention received by each token
        self._attention_scores = attention_scores[0].mean(dim=0).sum(dim=0).float().cpu()
    
    def compress(self, past_key_values) -> Tuple:
        kv_list = cache_to_list(past_key_values)
        
        if self._attention_scores is None:
            return list_to_cache(kv_list), {"method": "H2O", "memory_ratio": 1.0}
        
        seq_len = self._attention_scores.shape[0]
        n_keep = max(1, int(seq_len * self.keep_ratio))
        
        # Keep top-k by attention
        _, keep_indices = torch.topk(self._attention_scores, n_keep)
        keep_mask = torch.zeros(seq_len, dtype=torch.bool)
        keep_mask[keep_indices] = True
        
        compressed = []
        for k, v in kv_list:
            # k, v: [batch, heads, seq, head_dim]
            k_new = k.clone()
            v_new = v.clone()
            
            for pos in range(min(seq_len, k.shape[2])):
                if not keep_mask[pos]:
                    # Evict by zeroing (or could quantize heavily)
                    k_new[:, :, pos, :] = quantize_tensor(k[:, :, pos, :], 8)
                    v_new[:, :, pos, :] = quantize_tensor(v[:, :, pos, :], 8)
            
            compressed.append((k_new, v_new))
        
        memory_ratio = self.keep_ratio + (1 - self.keep_ratio) * 0.5  # kept + quantized
        
        stats = {
            "method": f"H2O-{int(self.keep_ratio*100)}",
            "keep_ratio": self.keep_ratio,
            "n_keep": n_keep,
            "memory_ratio": memory_ratio,
        }
        
        return list_to_cache(compressed), stats


class StreamingLLMCompressor:
    """StreamingLLM: Keep sink tokens + recent window."""
    
    def __init__(self, n_sink: int = 4, recent_window: int = 256):
        self.n_sink = n_sink
        self.recent_window = recent_window
    
    def compress(self, past_key_values) -> Tuple:
        kv_list = cache_to_list(past_key_values)
        
        if not kv_list:
            return list_to_cache(kv_list), {"method": "StreamingLLM", "memory_ratio": 1.0}
        
        seq_len = kv_list[0][0].shape[2]
        
        # Keep first n_sink + last recent_window
        keep_mask = torch.zeros(seq_len, dtype=torch.bool)
        keep_mask[:self.n_sink] = True  # Sink tokens
        keep_mask[-self.recent_window:] = True  # Recent window
        
        n_keep = keep_mask.sum().item()
        
        compressed = []
        for k, v in kv_list:
            k_new = k.clone()
            v_new = v.clone()
            
            for pos in range(seq_len):
                if not keep_mask[pos]:
                    k_new[:, :, pos, :] = quantize_tensor(k[:, :, pos, :], 8)
                    v_new[:, :, pos, :] = quantize_tensor(v[:, :, pos, :], 8)
            
            compressed.append((k_new, v_new))
        
        memory_ratio = n_keep / seq_len + (1 - n_keep / seq_len) * 0.5
        
        stats = {
            "method": "StreamingLLM",
            "n_sink": self.n_sink,
            "recent_window": self.recent_window,
            "n_keep": n_keep,
            "memory_ratio": memory_ratio,
        }
        
        return list_to_cache(compressed), stats


def load_long_texts(context_length: int, n_texts: int = 20) -> List[str]:
    """Load texts of sufficient length."""
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
        
        if len(current) > context_length * 5:  # Rough char estimate
            texts.append(current.strip())
            current = ""
            if len(texts) >= n_texts:
                break
    
    return texts[:n_texts]


def evaluate_at_context_length(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    texts: List[str],
    context_length: int,
    compressor: Optional[object] = None,
    kvcis: Optional[KVCISCompressor] = None,
    use_h2o: bool = False,
    h2o_compressor: Optional[H2OCompressor] = None,
    eval_ratio: float = 0.3,
) -> Dict:
    """Evaluate at specific context length."""
    
    device = next(model.parameters()).device
    total_loss = 0.0
    total_tokens = 0
    memory_ratios = []
    
    # Determine method name
    if kvcis:
        method_name = "KVCIS"
    elif h2o_compressor:
        method_name = f"H2O-{int(h2o_compressor.keep_ratio*100)}"
    elif compressor:
        method_name = getattr(compressor, 'method_name', 'Compressor')
        if hasattr(compressor, 'bits'):
            method_name = f"Uniform-INT{compressor.bits}"
        elif hasattr(compressor, 'n_sink'):
            method_name = "StreamingLLM"
    else:
        method_name = "Baseline"
    
    for text in tqdm(texts, desc=f"{method_name} @ {context_length}"):
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=context_length,
        ).to(device)
        
        seq_len = inputs.input_ids.shape[1]
        if seq_len < context_length * 0.5:
            continue
        
        prompt_len = int(seq_len * (1 - eval_ratio))
        prompt_ids = inputs.input_ids[:, :prompt_len]
        target_ids = inputs.input_ids[:, prompt_len:]
        
        if target_ids.shape[1] < 5:
            continue
        
        with torch.no_grad():
            # Prefill
            if kvcis:
                kvcis.setup_hook()
            
            outputs = model(
                prompt_ids,
                use_cache=True,
                return_dict=True,
                output_attentions=use_h2o,
            )
            
            # Apply compression
            if kvcis:
                kvcis.remove_hook()
                importance = kvcis.predict_importance(kvcis._activations)
                compressed_kv, stats = kvcis.compress(outputs.past_key_values, importance)
            elif h2o_compressor and use_h2o:
                attn = outputs.attentions[-1]  # Last layer attention
                h2o_compressor.set_attention(attn)
                compressed_kv, stats = h2o_compressor.compress(outputs.past_key_values)
            elif compressor:
                compressed_kv, stats = compressor.compress(outputs.past_key_values)
            else:
                compressed_kv = outputs.past_key_values
                stats = {"method": "Baseline", "memory_ratio": 1.0}
            
            memory_ratios.append(stats.get("memory_ratio", 1.0))
            
            # Evaluate
            eval_outputs = model(
                target_ids,
                past_key_values=compressed_kv,
                return_dict=True,
            )
            
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
        return {"method": method_name, "error": "No valid texts"}
    
    perplexity = math.exp(total_loss / total_tokens)
    
    return {
        "context_length": context_length,
        "method": method_name,
        "perplexity": perplexity,
        "memory_ratio": np.mean(memory_ratios),
        "memory_reduction_pct": (1 - np.mean(memory_ratios)) * 100,
        "total_tokens": total_tokens,
    }


def main():
    parser = argparse.ArgumentParser(description="KVCIS Long-Context Evaluation")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--probe-path", type=str, required=True)
    parser.add_argument("--extraction-layer", type=int, default=12)
    parser.add_argument("--context-lengths", type=int, nargs="+", default=[512, 1024, 2048])
    parser.add_argument("--n-texts", type=int, default=20)
    parser.add_argument("--skip-h2o", action="store_true", help="Skip H2O (OOMs at long context)")
    parser.add_argument("--output-dir", type=str, default="./longctx_results")
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

    all_results = []

    for ctx_len in args.context_lengths:
        print(f"\n{'='*60}")
        print(f"CONTEXT LENGTH: {ctx_len}")
        print(f"{'='*60}")
        
        texts = load_long_texts(ctx_len, args.n_texts)
        print(f"Loaded {len(texts)} texts")
        
        # Baseline
        baseline = evaluate_at_context_length(
            model, tokenizer, texts, ctx_len
        )
        all_results.append(baseline)
        print(f"Baseline PPL: {baseline['perplexity']:.4f}")
        
        # Uniform INT8
        uniform8 = UniformQuantizer(bits=8)
        u8_result = evaluate_at_context_length(
            model, tokenizer, texts, ctx_len, compressor=uniform8
        )
        all_results.append(u8_result)
        ppl_delta = (u8_result['perplexity'] / baseline['perplexity'] - 1) * 100
        print(f"Uniform-INT8: PPL={u8_result['perplexity']:.4f} ({ppl_delta:+.2f}%)")
        
        # KVCIS
        kvcis = KVCISCompressor(
            model=model,
            probe_path=args.probe_path,
            extraction_layer=args.extraction_layer,
        )
        kvcis_result = evaluate_at_context_length(
            model, tokenizer, texts, ctx_len, kvcis=kvcis
        )
        all_results.append(kvcis_result)
        ppl_delta = (kvcis_result['perplexity'] / baseline['perplexity'] - 1) * 100
        print(f"KVCIS: PPL={kvcis_result['perplexity']:.4f} ({ppl_delta:+.2f}%), Mem={kvcis_result['memory_reduction_pct']:.1f}%")
        
        # StreamingLLM
        window = max(64, ctx_len // 4)
        streaming = StreamingLLMCompressor(n_sink=4, recent_window=window)
        s_result = evaluate_at_context_length(
            model, tokenizer, texts, ctx_len, compressor=streaming
        )
        all_results.append(s_result)
        ppl_delta = (s_result['perplexity'] / baseline['perplexity'] - 1) * 100
        print(f"StreamingLLM: PPL={s_result['perplexity']:.4f} ({ppl_delta:+.2f}%)")
        
        # H2O (optional, OOMs at long context)
        if not args.skip_h2o and ctx_len <= 2048:
            try:
                h2o = H2OCompressor(keep_ratio=0.2)
                h2o_result = evaluate_at_context_length(
                    model, tokenizer, texts, ctx_len,
                    use_h2o=True, h2o_compressor=h2o
                )
                all_results.append(h2o_result)
                ppl_delta = (h2o_result['perplexity'] / baseline['perplexity'] - 1) * 100
                print(f"H2O-20: PPL={h2o_result['perplexity']:.4f} ({ppl_delta:+.2f}%)")
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"H2O: Skipped (OOM)")
                else:
                    raise

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    for ctx_len in args.context_lengths:
        ctx_results = [r for r in all_results if r.get("context_length") == ctx_len]
        baseline_ppl = next((r['perplexity'] for r in ctx_results if r['method'] == 'Baseline'), None)
        
        print(f"\n--- {ctx_len} tokens ---")
        print(f"{'Method':<18} {'Mem Red.':>10} {'PPL':>10} {'PPL Δ':>10}")
        print("-"*50)
        
        for r in sorted(ctx_results, key=lambda x: x['perplexity']):
            ppl_delta = (r['perplexity'] / baseline_ppl - 1) * 100 if baseline_ppl else 0
            print(f"{r['method']:<18} {r['memory_reduction_pct']:>9.1f}% {r['perplexity']:>10.4f} {ppl_delta:>+9.2f}%")

    # Save results
    with open(output_dir / "longctx_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to {output_dir}/longctx_results.json")


if __name__ == "__main__":
    main()
