# KVCIS: Activation-Based Token Importance Prediction for KV-Cache Compression

Prompt dataset and results for the KVCIS paper.

## Dataset

`prompts.txt` contains 500 curated prompts across four categories used for probe training:

| Category | Count | Description |
|----------|-------|-------------|
| General | 125 | OpenWebText-style general content |
| Instruction | 125 | Alpaca-style instruction-following |
| Code | 125 | Programming and technical content |
| Question | 125 | Factual and analytical questions |

All prompts are used to train a linear probe that predicts token importance from intermediate-layer activations.

## Results Summary

### Perplexity Evaluation (Llama-3.1-8B-Instruct)

| Method | Memory ↓ | PPL Δ |
|--------|----------|-------|
| Baseline | 0% | — |
| Uniform-INT8 | 50% | +4.40% |
| **KVCIS** | **45%** | **+0.57%** |

### Downstream Tasks (n=50)

| Method | Memory ↓ | NarrativeQA F1 | Passkey Acc. |
|--------|----------|----------------|--------------|
| Baseline | 0% | 0.0642 | 100% |
| Uniform-INT8 | 50% | 0.0592 | 100% |
| **KVCIS** | **50%** | **0.0642** | **100%** |

KVCIS achieves 50% memory reduction with zero degradation on NarrativeQA, while uniform quantization degrades by 7.8%.

## Code

Experimental code available from the author upon request.

## Citation

```bibtex
@article{messenger2026kvcis,
  title={KVCIS: Activation-Based Token Importance Prediction for Intelligent KV-Cache Compression},
  author={Messenger, Glen},
  year={2026},
  note={Preprint}
}
```

## License

Apache 2.0
