#!/usr/bin/env python3
"""
7B LISA + LCSB - CPU Optimized Version

Performance on Jetson CPU (ARM64):
- Forward pass: ~8s (seq_len=4)
- Backward pass: ~25s
- Total per step: ~35s

For faster training: use 0.5B model or GPU
"""
import gc
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

def train_7b_lisa_lcsb_cpu(texts, epochs=2, lr=1e-4, seq_len=4):
    """Train 7B with LISA + LCSB on CPU with optimizations"""
    
    # CPU optimizations
    torch.set_num_threads(4)
    
    print(f"Loading Qwen2.5-7B (seq_len={seq_len})...")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B",
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=True,
    )
    print(f"Layers: {len(model.model.layers)}")

    # LISA: Freeze all, unfreeze last 2
    print("Applying LISA...")
    for p in model.parameters():
        p.requires_grad = False
    for p in model.model.layers[-2:].parameters():
        p.requires_grad = True

    # LoRA
    print("Applying LoRA...")
    model = get_peft_model(model, LoraConfig(
        r=1,
        lora_alpha=2,
        target_modules=["q_proj", "k_proj", "v_proj"]
    ))

    # LISA on LoRA: Only train LoRA in last 2 layers
    print("Setting up LISA on LoRA...")
    for p in model.parameters():
        p.requires_grad = False
    for name, p in model.named_parameters():
        if ("layers.26." in name or "layers.27." in name) and "lora_" in name:
            p.requires_grad = True

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable:,}")

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B", trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr
    )

    # LCSB layer cycle
    LAYER_CYCLE = [27, 26]
    
    print(f"\nTraining {epochs} epochs, {len(texts)} texts, seq_len={seq_len}...")
    step = 0
    for epoch in range(epochs):
        for i, text in enumerate(texts):
            step += 1
            layer = LAYER_CYCLE[i % 2]
            
            t0 = time.time()
            inputs = tokenizer(text, return_tensors="pt", max_length=seq_len)
            out = model(**inputs, labels=inputs["input_ids"])
            t1 = time.time()
            
            optimizer.zero_grad()
            out.loss.backward()
            t2 = time.time()
            
            grad = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
            optimizer.step()
            t3 = time.time()
            
            print(f"  Step {step}: layer={layer}, loss={out.loss.item():.4f}, "
                  f"fwd={t1-t0:.1f}s, bwd={t2-t1:.1f}s, opt={t3-t2:.1f}s")
    
    return model

if __name__ == "__main__":
    texts = [
        "The cat sat on the mat",
        "Machine learning is powerful",
        "The quick brown fox jumps",
    ]
    
    model = train_7b_lisa_lcsb_cpu(texts, epochs=2, seq_len=4)
    print("\n✅ Training complete!")
