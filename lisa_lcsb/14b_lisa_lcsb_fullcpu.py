#!/usr/bin/env python3
"""
14B LISA+LCSB - Full CPU, no offload during training
Keep entire model in RAM, just train LoRA on last 2 layers
"""
import gc
import torch
import time
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

LOG_FILE = "/tmp/lisa_14b_fullcpu_log.txt"
CHECKPOINT_DIR = "/tmp/lisa_14b_fullcpu_checkpoints"

def log(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a") as f:
        f.write(line + "\n")

log("="*60)
log("14B LISA+LCSB - FULL CPU MODE")
log("="*60)

MODEL_NAME = "Qwen/Qwen2.5-14B"
log(f"Model: {MODEL_NAME}")

log("\n[1] Loading 14B model on CPU (no offload)...")
log("    This uses ~28GB RAM, will swap heavily")

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="cpu",
    trust_remote_code=True,
    low_cpu_mem_usage=True,
)

log(f"Model loaded: {len(model.model.layers)} layers")

# LISA - freeze all, unfreeze last 2
log("\n[2] Applying LISA...")
num_layers = len(model.model.layers)
for p in model.parameters():
    p.requires_grad = False
for p in model.model.layers[-2:].parameters():
    p.requires_grad = True

# LoRA
log("[3] Applying LoRA...")
model = get_peft_model(model, LoraConfig(
    r=1, lora_alpha=2,
    target_modules=["q_proj", "k_proj", "v_proj"]
))

# LISA on LoRA
log("[4] Setting up LISA on LoRA...")
target_layers = [num_layers - 2, num_layers - 1]
for p in model.parameters():
    p.requires_grad = False
for name, p in model.named_parameters():
    if any(f"layers.{l}." in name for l in target_layers) and "lora_" in name:
        p.requires_grad = True

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
log(f"Trainable params: {trainable:,}")

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=1e-4
)

TEXTS = [
    "The cat sat on the mat",
    "Machine learning works",
    "AI is powerful",
]

LAYER_CYCLE = target_layers[::-1]
TOTAL_STEPS = 500

log(f"\n[5] Training {TOTAL_STEPS} steps...")
log("="*60)

step = 0
start_time = time.time()

try:
    while step < TOTAL_STEPS:
        step += 1
        text_idx = (step - 1) % len(TEXTS)
        layer = LAYER_CYCLE[(step - 1) % len(LAYER_CYCLE)]
        
        gc.collect()
        
        t0 = time.time()
        inputs = tokenizer(TEXTS[text_idx], return_tensors="pt", max_length=8)
        
        out = model(**inputs, labels=inputs["input_ids"])
        t1 = time.time()
        
        optimizer.zero_grad()
        out.loss.backward()
        t2 = time.time()
        
        grad = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
        optimizer.step()
        t3 = time.time()
        
        loss = out.loss.item()
        
        log(f"Step {step}/{TOTAL_STEPS}: layer={layer}, loss={loss:.4f}, "
            f"fwd={t1-t0:.1f}s, bwd={t2-t1:.1f}s")
        
        if step % 50 == 0:
            checkpoint = {
                "step": step,
                "loss": loss,
                "model_state": {k: v.clone() for k, v in model.named_parameters() if v.requires_grad}
            }
            torch.save(checkpoint, f"{CHECKPOINT_DIR}/step_{step}.pt")
            log(f"  -> Checkpoint saved")
        
        if step % 10 == 0:
            gc.collect()

except KeyboardInterrupt:
    log("Interrupted!")
except Exception as e:
    log(f"Error at step {step}: {e}")
    import traceback
    traceback.print_exc()

log("="*60)
log(f"DONE! Steps: {step}, Final loss: {loss}")
log(f"Total time: {(time.time() - start_time)/3600:.1f} hours")
log("="*60)
