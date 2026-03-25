#!/usr/bin/env python3
"""Proper perplexity evaluation with LoRA checkpoint."""
import torch
import math
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import sys
sys.path.insert(0, "/home/jetson/lisa_proj")
from federated.client import apply_lora_to_model

def load_model_with_lora(checkpoint_path):
    """Load model, apply LoRA, then load checkpoint."""
    print("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B",
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    print(f"  Base model keys: {len(model.state_dict())}")
    
    print("Applying LoRA to model...")
    lora_count = apply_lora_to_model(model, rank=4, alpha=8.0, dropout=0.05)
    print(f"  LoRA applied to {lora_count} layers")
    print(f"  Model keys after LoRA: {len(model.state_dict())}")
    
    print(f"Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    print(f"  Checkpoint keys: {len(ckpt)}")
    
    print("Loading checkpoint into LoRA model...")
    model.load_state_dict(ckpt, strict=False)
    print(f"  Model keys after load: {len(model.state_dict())}")
    
    lora_keys = sum(1 for k in model.state_dict().keys() if "lora_" in k)
    print(f"  LoRA keys in model: {lora_keys}")
    
    return model

def compute_perplexity(model, tokenizer):
    """Compute perplexity on wikitext."""
    print("\nLoading wikitext...")
    ds = load_dataset("wikitext", "wikitext-2-v1", split="test")
    texts = [t for t in ds["text"] if t.strip() and len(t.strip()) > 20][:30]
    print(f"  Using {len(texts)} texts")
    
    print("Tokenizing...")
    enc = tokenizer(texts, max_length=64, padding="max_length", truncation=True, return_tensors="pt")
    
    print("Computing perplexity...")
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for i in range(len(enc["input_ids"])):
            ids = enc["input_ids"][i:i+1]
            out = model(ids)
            shift_logits = out.logits[..., :-1, :]
            shift_labels = ids[..., 1:]
            loss = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                reduction="sum"
            )
            num_tokens = (shift_labels != tokenizer.pad_token_id).sum().item()
            total_loss += loss.item()
            total_tokens += num_tokens
    
    ppl = math.exp(total_loss / max(total_tokens, 1))
    return ppl

def main():
    import os
    
    checkpoint_dir = "/home/jetson/lisa_proj/checkpoints"
    
    print("=" * 60)
    print("PERPLEXITY EVALUATION WITH PROPER LoRA LOADING")
    print("=" * 60)
    
    # Get checkpoints
    checkpoints = []
    for f in os.listdir(checkpoint_dir):
        if f.startswith("model_round_") and f.endswith(".pt"):
            path = os.path.join(checkpoint_dir, f)
            checkpoints.append((os.path.getmtime(path), path, f))
    
    checkpoints.sort(key=lambda x: x[0], reverse=True)
    
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
    
    results = {}
    
    # Baseline (no checkpoint)
    print("\n" + "=" * 40)
    print("BASELINE MODEL (no training)")
    print("=" * 40)
    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B",
        torch_dtype=torch.float32,
        device_map="cpu"
    )
    base_ppl = compute_perplexity(base_model, tokenizer)
    results["baseline"] = base_ppl
    print(f"\n*** Baseline Perplexity: {base_ppl:.4f} ***\n")
    del base_model
    
    # Check latest checkpoints
    for _, path, name in checkpoints[:4]:
        print("\n" + "=" * 40)
        print(f"CHECKPOINT: {name}")
        print("=" * 40)
        
        model = load_model_with_lora(path)
        ppl = compute_perplexity(model, tokenizer)
        results[name] = ppl
        print(f"\n*** Perplexity: {ppl:.4f} ***\n")
        
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Baseline:    {results.get('baseline', 'N/A'):.4f}")
    
    if 'baseline' in results:
        for name in sorted(results.keys()):
            if name != 'baseline':
                ppl = results[name]
                improvement = ((results['baseline'] - ppl) / results['baseline']) * 100
                print(f"{name}: {ppl:.4f} ({improvement:+.2f}%)")
    
    return results

if __name__ == "__main__":
    main()
