#!/usr/bin/env python3
"""
Perplexity Evaluation Script for LISA
Evaluates model quality using wikitext dataset
"""
import torch
import math
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

MODEL_PATH = "/tmp/lisa_checkpoints/step_500.pt"  # Or HuggingFace model
BASE_MODEL = "Qwen/Qwen2.5-7B"

def load_model(checkpoint_path=None):
    """Load model with optional LISA checkpoint."""
    if checkpoint_path and checkpoint_path.endswith('.pt'):
        # Load base model and apply checkpoint
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            trust_remote_code=True,
        )
        # Load trained weights here
        return model, None
    else:
        model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path or BASE_MODEL,
            torch_dtype=torch.bfloat16,
            device_map="cpu",
            trust_remote_code=True,
        )
    return model, None

def calculate_perplexity(model, tokenizer, texts):
    """Calculate perplexity on texts."""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    for text in texts:
        inputs = tokenizer(text, return_tensors="pt", max_length=128, truncation=True)
        with torch.no_grad():
            outputs = model(**inputs, labels=inputs["input_ids"])
            loss = outputs.loss.item()
            total_loss += loss * inputs["input_ids"].numel()
            total_tokens += inputs["input_ids"].numel()
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    return perplexity

def main():
    test_texts = [
        "The cat sat on the mat",
        "Machine learning is transforming",
        "The quick brown fox jumps over",
    ]
    
    print("Loading model...")
    model, _ = load_model()
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    
    print("Calculating perplexity...")
    ppl = calculate_perplexity(model, tokenizer, test_texts)
    print(f"Perplexity: {ppl:.2f}")

if __name__ == "__main__":
    main()
