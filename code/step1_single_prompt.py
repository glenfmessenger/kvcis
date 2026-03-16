"""
KVCIS PoC - Step 1: Single Prompt Test

Validates that we can:
1. Run a prompt through the model
2. Extract activations from a specific layer
3. Collect attention patterns during generation

This is a sanity check before full data collection.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse


def main():
    parser = argparse.ArgumentParser(description="KVCIS Step 1: Single Prompt Test")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--extraction-layer", type=int, default=12)
    parser.add_argument("--prompt", type=str, default="The capital of France is")
    args = parser.parse_args()

    print(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation="eager",  # Need eager for attention extraction
    )
    model.eval()

    # Storage for activations
    activations = {}

    def hook_fn(module, input, output):
        # output is tuple: (hidden_states, ...)
        hidden = output[0] if isinstance(output, tuple) else output
        activations["layer"] = hidden.detach().float().cpu()  # Convert bf16 to float32

    # Register hook at extraction layer
    layer = model.model.layers[args.extraction_layer]
    handle = layer.register_forward_hook(hook_fn)

    # Tokenize prompt
    inputs = tokenizer(args.prompt, return_tensors="pt").to("cuda")
    prompt_length = inputs.input_ids.shape[1]

    print(f"\nPrompt: {args.prompt}")
    print(f"Prompt tokens: {prompt_length}")
    print(f"Extraction layer: {args.extraction_layer}")

    # Forward pass with attention output
    with torch.no_grad():
        outputs = model(
            inputs.input_ids,
            output_attentions=True,
            return_dict=True,
        )

    handle.remove()

    # Check activations
    layer_activations = activations["layer"]
    print(f"\nActivations shape: {layer_activations.shape}")
    print(f"  Expected: [1, {prompt_length}, {model.config.hidden_size}]")

    # Check attention
    attn = outputs.attentions[args.extraction_layer]
    print(f"\nAttention shape: {attn.shape}")
    print(f"  Expected: [1, num_heads, {prompt_length}, {prompt_length}]")

    # Generate a few tokens and track attention
    print("\n--- Generation Test ---")
    current_ids = inputs.input_ids.clone()
    
    for step in range(5):
        with torch.no_grad():
            out = model(
                current_ids,
                output_attentions=True,
                return_dict=True,
            )
        
        # Get attention from last token to all previous tokens
        attn = out.attentions[args.extraction_layer]
        # attn shape: [batch, heads, seq_len, seq_len]
        # We want: attention FROM last token TO prompt tokens
        last_token_attn = attn[0, :, -1, :prompt_length].mean(dim=0).float().cpu()
        
        # Sample next token
        next_token = out.logits[0, -1, :].argmax()
        current_ids = torch.cat([current_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
        
        decoded = tokenizer.decode(next_token)
        print(f"  Step {step+1}: '{decoded}' | Attn to prompt: min={last_token_attn.min():.4f}, max={last_token_attn.max():.4f}")

    # Show final generation
    full_output = tokenizer.decode(current_ids[0], skip_special_tokens=True)
    print(f"\nFull output: {full_output}")

    # Show attention distribution to prompt tokens
    print("\n--- Attention Distribution to Prompt Tokens ---")
    tokens = [tokenizer.decode([t]) for t in inputs.input_ids[0]]
    for i, (tok, attn_val) in enumerate(zip(tokens, last_token_attn.tolist())):
        bar = "█" * int(attn_val * 50)
        print(f"  [{i}] {tok:15s} {attn_val:.4f} {bar}")

    print("\n✓ Step 1 complete - activation extraction and attention tracking working")


if __name__ == "__main__":
    main()
