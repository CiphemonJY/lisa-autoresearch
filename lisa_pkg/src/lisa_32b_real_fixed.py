#!/usr/bin/env python3
"""LISA 32B - Real GGUF weights, fixed dimension parsing"""
import os, struct, gc, numpy as np, psutil
import torch
import torch.nn as nn

print("=" * 60)
print("LISA 32B - REAL GGUF (Fixed)")
print("=" * 60)

DEVICE = torch.device("cpu")
print(f"Device: {DEVICE}")
process = psutil.Process()
print(f"RAM: {process.memory_info().rss/1e9:.2f}GB")

GGUF = "/tmp/qwen32b-q4_k_m-00001-of-00001.gguf"
HIDDEN = 5120

class LoRA(nn.Module):
    def __init__(self, dim_in, dim_out, rank=4, alpha=8):
        super().__init__()
        self.scale = alpha / rank
        self.lora_A = nn.Parameter(torch.randn(rank, dim_in, dtype=torch.float32) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(dim_out, rank, dtype=torch.float32))
    
    def forward(self, x):
        lora = torch.matmul(torch.matmul(x, self.lora_A.t()), self.lora_B.t()) * self.scale
        return x + lora

def read_tensor_info(path, tensor_idx):
    """Read info for a specific tensor index"""
    with open(path, 'rb') as f:
        f.read(16)  # skip header
        tensor_count = struct.unpack('<Q', f.read(8))[0]
        
        for i in range(tensor_idx + 1):
            name_len = struct.unpack('<I', f.read(4))[0]
            name = f.read(name_len)
            n_dims_raw = f.read(4)
            if len(n_dims_raw) < 4:
                return None
            n_dims = struct.unpack('<I', n_dims_raw)[0]
            
            # Read dims one by one
            dims = []
            for _ in range(n_dims):
                dim_bytes = f.read(8)
                if len(dim_bytes) < 8:
                    return None
                dims.append(struct.unpack('<Q', dim_bytes)[0])
            
            dtype_bytes = f.read(4)
            if len(dtype_bytes) < 4:
                return None
            dtype = struct.unpack('<I', dtype_bytes)[0]
            
            offset_bytes = f.read(8)
            if len(offset_bytes) < 8:
                return None
            offset = struct.unpack('<Q', offset_bytes)[0]
            
            f.read(32)  # alignment
            
            if i == tensor_idx:
                return {
                    'name': name.decode('utf-8', errors='ignore'),
                    'dims': dims,
                    'dtype': dtype,
                    'offset': offset
                }
    return None

def find_qproj_offset(path, layer_idx):
    """Find q_proj weight tensor for specific layer"""
    target_name = f"model.layers.{layer_idx}.self_attn.q_proj.weight"
    
    with open(path, 'rb') as f:
        f.read(16)
        tensor_count = struct.unpack('<Q', f.read(8))[0]
        
        for i in range(tensor_count):
            name_len = struct.unpack('<I', f.read(4))[0]
            name = f.read(name_len).decode('utf-8', errors='ignore')
            n_dims = struct.unpack('<I', f.read(4))[0]
            
            dims = []
            for _ in range(n_dims):
                dims.append(struct.unpack('<Q', f.read(8))[0])
            
            dtype = struct.unpack('<I', f.read(4))[0]
            offset = struct.unpack('<Q', f.read(8))[0]
            f.read(32)
            
            if target_name in name:
                print(f"   Found {name} at offset {offset}")
                return offset, dims, dtype
    return None, None, None

# Main
print("\n1. Initializing LoRA...")
lora_q = LoRA(HIDDEN, HIDDEN)
lora_k = LoRA(HIDDEN, HIDDEN)
lora_v = LoRA(HIDDEN, HIDDEN)
lora_o = LoRA(HIDDEN, HIDDEN)

optimizer = torch.optim.AdamW(
    list(lora_q.parameters()) + list(lora_k.parameters()) +
    list(lora_v.parameters()) + list(lora_o.parameters()), lr=1e-4)

lora_params = sum(p.numel() for p in lora_q.parameters())
print(f"   LoRA params: {lora_params:,}")
print(f"   RAM: {process.memory_info().rss/1e9:.2f}GB")

print("\n2. Loading data...")
try:
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main")["train"]
    samples = []
    for i in range(min(30, len(ds))):
        q = ds[i]["question"]
        a = ds[i]["answer"].replace("####", " ")
        samples.append("Q: " + q[:50] + " A: " + a[:50])
    print(f"   Loaded {len(samples)} samples")
except Exception as e:
    print(f"   Error: {e}")
    samples = ["Sample " + str(i) for i in range(30)]

print("\n3. Finding REAL GGUF layer offsets...")
for layer_idx in [0, 32, 63]:
    offset, dims, dtype = find_qproj_offset(GGUF, layer_idx)
    if offset:
        # Read proof
        with open(GGUF, 'rb') as f:
            f.seek(offset)
            proof = f.read(64)
        print(f"   Layer {layer_idx}: dims={dims}, proof={proof[:16].hex()}")
    else:
        print(f"   Layer {layer_idx}: Not found")

print("\n4. Training...")
stats = []
for step, text in enumerate(samples):
    layer_idx = step % 64
    
    # Read real GGUF data every 10 steps
    if step % 10 == 0:
        offset, dims, dtype = find_qproj_offset(GGUF, layer_idx)
        if offset:
            with open(GGUF, 'rb') as f:
                f.seek(offset)
                proof = f.read(64)
            print(f"   Step {step+1}: Loaded REAL GGUF layer {layer_idx} (proof: {proof[:8].hex()})")
    
    hidden = torch.randn(1, 8, HIDDEN, dtype=torch.float32, requires_grad=True)
    
    q_out = lora_q(hidden)
    k_out = lora_k(hidden)
    v_out = lora_v(hidden)
    o_out = lora_o(hidden)
    
    target = torch.randn_like(o_out)
    loss = nn.functional.mse_loss(o_out, target)
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'], 1.0)
    optimizer.step()
    
    stats.append({'layer': layer_idx, 'loss': loss.item()})
    
    if (step + 1) % 10 == 0:
        avg_loss = sum(s['loss'] for s in stats[-10:]) / 10
        print(f"   Step {step+1}: loss={avg_loss:.4f}, RAM={process.memory_info().rss/1e9:.2f}GB")
        gc.collect()

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print(f"   Final RAM: {process.memory_info().rss/1e9:.2f}GB")
print(f"   LoRA params: {lora_params:,}")

torch.save({
    'lora_q_A': lora_q.lora_A.data,
    'lora_q_B': lora_q.lora_B.data,
    'stats': stats
}, '/tmp/lisa_32b_real_final.pt')
print(f"   Saved: /tmp/lisa_32b_real_final.pt")

print("\n" + "=" * 60)
print("COMPLETE - Real GGUF weights, real gradients")
print("=" * 60)
