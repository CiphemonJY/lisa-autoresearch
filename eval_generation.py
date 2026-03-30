#!/usr/bin/env python3
"""
Generation Quality Test for LISA
Tests model outputs with various prompts
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE_MODEL = "Qwen/Qwen2.5-7B"

PROMPTS = [
    "The future of AI is",
    "Once upon a time",
    "In a galaxy far far away",
    "The best way to learn programming is",
]

def test_generation(model, tokenizer, prompt, max_length=50):
    """Generate text and return."""
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_length=max_length,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    
    print("\nGeneration Test Results:")
    print("=" * 50)
    for prompt in PROMPTS:
        result = test_generation(model, tokenizer, prompt)
        print(f"Prompt: {prompt}")
        print(f"Output: {result}")
        print("-" * 50)

if __name__ == "__main__":
    main()
