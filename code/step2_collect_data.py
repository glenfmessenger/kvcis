"""
KVCIS PoC - Step 2: Collect Training Data

Collects activation → importance pairs for probe training:
1. Run diverse prompts through the model
2. For each token, extract activation at layer N
3. Generate tokens and measure attention received (importance)
4. Save (activation, importance) pairs for training

Output:
  - activations.npy: [n_tokens, hidden_dim]
  - importance.npy: [n_tokens]
  - metadata.json: collection statistics
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from pathlib import Path
import json
import argparse
from typing import List, Dict, Tuple
from tqdm import tqdm
from dataclasses import dataclass
from datasets import load_dataset


@dataclass
class TokenData:
    activation: np.ndarray
    importance: float
    token_id: int
    token_str: str
    position: int
    prompt_idx: int


def load_diverse_prompts(n_prompts: int = 500) -> List[str]:
    """Load diverse prompts from multiple sources."""
    prompts = []
    
    # OpenWebText for general text
    try:
        print("Loading OpenWebText...")
        dataset = load_dataset("openwebtext", split="train", streaming=True)
        count = 0
        for item in dataset:
            text = item.get("text", "")
            if len(text) > 100:
                # Take first 500 chars as prompt
                prompts.append(text[:500])
                count += 1
                if count >= n_prompts // 2:
                    break
    except Exception as e:
        print(f"Could not load OpenWebText: {e}")
    
    # Alpaca for instructions
    try:
        print("Loading Alpaca...")
        dataset = load_dataset("tatsu-lab/alpaca", split="train")
        for item in list(dataset)[:n_prompts // 4]:
            instruction = item.get("instruction", "")
            input_text = item.get("input", "")
            if instruction:
                prompt = f"Instruction: {instruction}"
                if input_text:
                    prompt += f"\nInput: {input_text}"
                prompts.append(prompt)
    except Exception as e:
        print(f"Could not load Alpaca: {e}")
    
    # Code prompts
    code_prompts = [
        "def fibonacci(n):",
        "class DatabaseConnection:",
        "import pandas as pd\n\ndef load_data(path):",
        "async def fetch_url(url):",
        "# Binary search implementation\ndef binary_search(arr, target):",
    ]
    prompts.extend(code_prompts * (n_prompts // 20))
    
    # Ensure we have enough
    if len(prompts) < n_prompts:
        # Pad with simple prompts
        simple = [
            "The weather today is",
            "In the year 2024,",
            "The most important thing about",
            "Scientists have discovered that",
            "The history of artificial intelligence",
        ]
        while len(prompts) < n_prompts:
            prompts.extend(simple)
    
    return prompts[:n_prompts]


class DataCollector:
    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        extraction_layer: int = 12,
        generation_steps: int = 30,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.extraction_layer = extraction_layer
        self.generation_steps = generation_steps
        self.device = next(model.parameters()).device
        
        self._activations = None
        self._hook_handle = None
    
    def _hook_fn(self, module, input, output):
        hidden = output[0] if isinstance(output, tuple) else output
        self._activations = hidden[0].detach().float().cpu().numpy()
    
    def setup_hook(self):
        layer = self.model.model.layers[self.extraction_layer]
        self._hook_handle = layer.register_forward_hook(self._hook_fn)
    
    def remove_hook(self):
        if self._hook_handle:
            self._hook_handle.remove()
            self._hook_handle = None
    
    def collect_single_prompt(
        self,
        prompt: str,
        prompt_idx: int,
        max_prompt_tokens: int = 128,
    ) -> List[TokenData]:
        """Collect data for a single prompt."""
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_prompt_tokens,
        ).to(self.device)
        
        input_ids = inputs.input_ids
        prompt_length = input_ids.shape[1]
        
        if prompt_length < 5:
            return []
        
        # Forward pass with hook to get activations
        self.setup_hook()
        with torch.no_grad():
            self.model(input_ids, use_cache=False)
        self.remove_hook()
        
        activations = self._activations  # [prompt_length, hidden_dim]
        
        # Generate tokens and accumulate attention
        attention_received = np.zeros(prompt_length)
        current_ids = input_ids.clone()
        
        for step in range(self.generation_steps):
            with torch.no_grad():
                outputs = self.model(
                    current_ids,
                    output_attentions=True,
                    return_dict=True,
                )
            
            # Attention from last token to prompt tokens
            attn = outputs.attentions[self.extraction_layer]
            last_attn = attn[0, :, -1, :prompt_length].mean(dim=0).float().cpu().numpy()
            attention_received += last_attn
            
            # Sample next token
            next_token = outputs.logits[0, -1, :].argmax()
            current_ids = torch.cat([current_ids, next_token.unsqueeze(0).unsqueeze(0)], dim=1)
            
            if next_token.item() == self.tokenizer.eos_token_id:
                break
        
        # Normalize importance to 0-1
        if attention_received.max() > 0:
            importance = attention_received / attention_received.max()
        else:
            importance = attention_received
        
        # Create TokenData for each prompt token
        token_data = []
        token_ids = input_ids[0].tolist()
        
        for pos in range(prompt_length):
            token_data.append(TokenData(
                activation=activations[pos],
                importance=float(importance[pos]),
                token_id=token_ids[pos],
                token_str=self.tokenizer.decode([token_ids[pos]]),
                position=pos,
                prompt_idx=prompt_idx,
            ))
        
        return token_data
    
    def collect_dataset(
        self,
        prompts: List[str],
        max_prompt_tokens: int = 128,
    ) -> Tuple[np.ndarray, np.ndarray, List[Dict]]:
        """Collect data from multiple prompts."""
        
        all_data = []
        
        for i, prompt in enumerate(tqdm(prompts, desc="Processing prompts")):
            try:
                token_data = self.collect_single_prompt(prompt, i, max_prompt_tokens)
                all_data.extend(token_data)
            except Exception as e:
                print(f"\nError on prompt {i}: {e}")
                continue
        
        if not all_data:
            return np.array([]), np.array([]), []
        
        # Convert to arrays
        activations = np.array([d.activation for d in all_data])
        importance = np.array([d.importance for d in all_data])
        
        metadata = [
            {
                "token_id": d.token_id,
                "token_str": d.token_str,
                "position": d.position,
                "prompt_idx": d.prompt_idx,
                "importance": d.importance,
            }
            for d in all_data
        ]
        
        return activations, importance, metadata


def main():
    parser = argparse.ArgumentParser(description="KVCIS Step 2: Collect Training Data")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--extraction-layer", type=int, default=12)
    parser.add_argument("--n-prompts", type=int, default=500)
    parser.add_argument("--max-prompt-tokens", type=int, default=128)
    parser.add_argument("--generation-steps", type=int, default=30)
    parser.add_argument("--output-dir", type=str, default="./data")
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

    # Load prompts
    print(f"\nLoading {args.n_prompts} prompts...")
    prompts = load_diverse_prompts(args.n_prompts)
    print(f"Loaded {len(prompts)} prompts")

    # Collect data
    collector = DataCollector(
        model=model,
        tokenizer=tokenizer,
        extraction_layer=args.extraction_layer,
        generation_steps=args.generation_steps,
    )

    activations, importance, metadata = collector.collect_dataset(
        prompts,
        max_prompt_tokens=args.max_prompt_tokens,
    )

    print(f"\nCollected {len(importance)} token samples")

    # Save data
    np.save(output_dir / "activations.npy", activations)
    np.save(output_dir / "importance.npy", importance)
    
    with open(output_dir / "metadata.json", "w") as f:
        # Convert numpy types for JSON serialization
        json.dump(metadata, f)

    # Save collection config
    config = {
        "model": args.model,
        "extraction_layer": args.extraction_layer,
        "n_prompts": args.n_prompts,
        "max_prompt_tokens": args.max_prompt_tokens,
        "generation_steps": args.generation_steps,
        "total_tokens": len(importance),
    }
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)

    print(f"\nData saved to {output_dir}")
    print(f"  activations.npy: {activations.shape}")
    print(f"  importance.npy: {importance.shape}")

    # Show importance distribution
    print(f"\nImportance distribution:")
    print(f"  Min: {importance.min():.4f}")
    print(f"  Max: {importance.max():.4f}")
    print(f"  Mean: {importance.mean():.4f}")
    print(f"  Std: {importance.std():.4f}")

    # Show BOS token importance
    bos_importance = [m["importance"] for m in metadata if m["position"] == 0]
    if bos_importance:
        print(f"\nBOS token importance (position 0):")
        print(f"  Mean: {np.mean(bos_importance):.4f}")
        print(f"  This should be close to 1.0 (attention sink)")

    print("\n✓ Step 2 complete - training data collected")


if __name__ == "__main__":
    main()
