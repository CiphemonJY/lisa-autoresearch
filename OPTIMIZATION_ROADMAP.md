# LISA Optimization Roadmap

*Based on Gemini's engineering suggestions - 2026-04-02*

## Hardware Reality Check

**Bandwidth Math:**
| Component | Bandwidth | Notes |
|----------|-----------|-------|
| Unified RAM (Jetson) | ~68 GB/s | LPDDR5 |
| NVMe SSD via PCIe | ~3-4 GB/s | Theoretical max |
| **Ratio** | **~20x slower** | Every layer swap hits this wall |

**Implication:** Layer swapping is inevitable, but we can minimize its impact.

---

## 1. I/O Bottleneck: Layer Prefetching

**Problem:** Layer-by-layer processing stalls while loading from storage.

**Solution:** Async prefetch - load layer n+1 while layer n is computing.

```python
import asyncio
from threading import Thread

class AsyncLayerLoader:
    def __init__(self, storage_path):
        self.storage_path = storage_path
        self.next_layer = None
        self.current_layer = None
        
    async def prefetch_layer(self, layer_idx):
        """Load layer n+1 while layer n is computing"""
        path = f"{self.storage_path}/layer_{layer_idx}.pt"
        self.next_layer = await asyncio.to_thread(
            torch.load, path, map_location='cpu'
        )
        
    def get_layer(self, layer_idx):
        """Get prefetched layer"""
        layer = self.next_layer
        self.current_layer = layer
        self.next_layer = None
        return layer

# Usage in training loop
loader = AsyncLayerLoader("/tmp/layers")

for batch in dataloader:
    # Start prefetching next layer while current computes
    asyncio.create_task(loader.prefetch_layer(current_idx + 1))
    
    # Get current layer (already loaded)
    layer_weights = loader.get_layer(current_idx)
    
    # Forward pass with current layer
    hidden_states = compute_layer(hidden_states, layer_weights)
```

**Benefit:** Hide I/O latency behind computation

---

## 1b. Zero-Copy Loading (Safetensors + mmap)

**Problem:** PyTorch `.pt` files copy data during load, wasting RAM.

**Solution:** Use Safetensors with memory-mapping (mmap) - OS streams directly from SSD.

```python
from safetensors import safe_open
from safetensors.torch import save_file
import torch

def save_layer_safetensors(layer, layer_idx, path="/tmp/layers"):
    """Save layer weights in safetensors format for mmap loading"""
    state_dict = {f"layer_{layer_idx}": layer.weight}
    save_file(state_dict, f"{path}/layer_{layer_idx}.safetensors")

def load_layer_mmap(layer_idx, path="/tmp/layers"):
    """Memory-map layer directly from SSD - no RAM allocation"""
    with safe_open(f"{path}/layer_{layer_idx}.safetensors", 
                   framework="pt", device="cpu") as f:
        tensor = f.get_tensor(f"layer_{layer_idx}")
    return tensor

# Memory-map comparison:
# torch.load(): ~1.5GB peak RAM during load (full file in memory)
# mmap: ~0MB peak RAM (OS streams directly)
```

**Benefit:** ~20-30% faster loading, zero extra RAM allocation

---

## 1c. Double Buffering (Compute + I/O Overlap)

**Goal:** Hide disk latency completely by overlapping compute and I/O.

```python
import asyncio
import threading
from queue import Queue

class DoubleBufferLoader:
    """Load next layer while current layer computes"""
    
    def __init__(self, storage_path):
        self.storage_path = storage_path
        self.buffer_a = [None, None]  # (layer_idx, weights)
        self.buffer_b = [None, None]
        self.active_buffer = 0  # 0 or 1
        
    def _load_to_buffer(self, buffer_idx, layer_idx):
        """Background thread loads layer to specific buffer"""
        weights = load_layer_mmap(layer_idx)
        self._buffers[buffer_idx] = [layer_idx, weights]
        
    def start_prefetch(self, next_layer_idx):
        """Start loading next layer in background"""
        inactive = 1 - self.active_buffer
        self._load_thread = threading.Thread(
            target=self._load_to_buffer,
            args=(inactive, next_layer_idx)
        )
        self._load_thread.start()
        
    def swap_buffers(self):
        """Swap to newly loaded buffer"""
        self.active_buffer = 1 - self.active_buffer
        layer_idx, weights = self._buffers[self.active_buffer]
        return layer_idx, weights

# Training loop with double buffering:
loader = DoubleBufferLoader("/tmp/layers")

for batch_idx, batch in enumerate(dataloader):
    # Start prefetching layer N+1 while N computes
    if batch_idx < len(layers) - 1:
        loader.start_prefetch(batch_idx + 1)
    
    # Get current layer (from previous prefetch)
    layer_idx, weights = loader.swap_buffers()
    
    # Compute (should overlap with N+1 loading)
    output = compute_layer(hidden_states, weights)
```

**Benefit:** If compute_time ≈ load_time, zero latency overhead

---

## 1d. Hardware Check

**Critical:** Ensure layers are on NVMe, NOT microSD or USB.

```bash
# Check where swap/storage is located
df -h /tmp
lsblk
# Look for: nvme0n1 (NVMe) vs sda (SATA/USB)

# Test read speed
sudo hdparm -Tt /dev/nvme0n1p1
# Should see: > 2000 MB/s for good NVMe
```

**Jetson Orin Nano storage options:**
| Storage | Read Speed | Viability |
|---------|-----------|-----------|
| microSD | ~100 MB/s | ❌ Too slow |
| USB 3.0 | ~400 MB/s | ⚠️ Marginal |
| eMMC | ~300 MB/s | ⚠️ Marginal |
| NVMe M.2 | ~3000 MB/s | ✅ Required |

**Recommendation:** Use NVMe for layer storage if possible.

---

## 2. Aggressive Garbage Collection

**Problem:** Backward pass creates computational graphs that eat RAM.

**Solution:** Clear everything immediately after gradient calculation.

```python
import gc
import torch

class MemoryEfficientBackward:
    def backward_layer(self, layer, hidden_states, loss):
        """Backward pass with aggressive cleanup"""
        
        # Calculate gradients
        loss.backward()
        
        # Extract gradients for LoRA adapters ONLY
        lora_grads = {
            name: param.grad.clone()
            for name, param in layer.named_parameters()
            if 'lora' in name and param.grad is not None
        }
        
        # IMMEDIATELY clear computational graph
        loss.clear()
        del loss
        
        # Clear layer weights from computation
        layer.zero_grad(set_to_none=True)
        del layer
        
        # Force garbage collection
        gc.collect()
        
        # Clear CUDA cache if using GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        
        # Update LoRA weights with extracted gradients
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        # Clear activations
        del hidden_states
        gc.collect()
        
        return lora_grads

class MemoryProfiler:
    """Track memory at every layer boundary"""
    
    def __init__(self):
        self.snapshots = []
        
    def snapshot(self, tag: str):
        import psutil
        process = psutil.Process()
        
        snapshot = {
            'tag': tag,
            'rss_mb': process.memory_info().rss / 1024**2,
            'vms_mb': process.memory_info().vms / 1024**2,
        }
        
        if torch.cuda.is_available():
            snapshot['cuda_allocated_mb'] = torch.cuda.memory_allocated() / 1024**2
            snapshot['cuda_reserved_mb'] = torch.cuda.memory_reserved() / 1024**2
            
        self.snapshots.append(snapshot)
        print(f"[{tag}] RAM: {snapshot['rss_mb']:.1f}MB", end="")
        if 'cuda_allocated_mb' in snapshot:
            print(f" | GPU: {snapshot['cuda_allocated_mb']:.1f}MB")
        else:
            print()
```

---

## 3. NF4 Quantization Integration

**Problem:** Frozen base weights still consume significant RAM AND slow down I/O.

**Solution:** Use bitsandbytes NF4 for frozen base - both memory AND speed improvement.

**Speed benefit:** Smaller files = faster disk transfers
- Unquantized 1.5B layer: ~3GB → ~1 second to load
- NF4 1.5B layer: ~0.85GB → ~0.25 second to load

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

def load_quantized_base_model(model_name: str, lora_config: LoraConfig):
    """Load base model in 4-bit NF4, add LoRA in fp16
    
    Benefits:
    - Memory: 3GB → 0.85GB for 1.5B model
    - I/O Speed: 1s → 0.25s per layer load
    """
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",  # NormalFloat4
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="cpu",
        trust_remote_code=True,
    )
    
    # Freeze base model completely
    for param in base_model.parameters():
        param.requires_grad = False
        
    model = get_peft_model(base_model, lora_config)
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable: {trainable_params:,} ({100*trainable_params/total_params:.3f}%)")
    
    return model
```

**Memory comparison:**
| Model | fp16 | NF4+LoRA | Reduction |
|-------|------|----------|-----------|
| Qwen 0.5B | 1GB | 0.3GB | 70% |
| Qwen 1.5B | 3GB | 0.85GB | 72% |
| Qwen 3B | 6GB | 1.7GB | 72% |

**I/O improvement:** 4x faster layer loads (smaller files)

```python
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

def load_quantized_base_model(model_name: str, lora_config: LoraConfig):
    """Load base model in 4-bit NF4, add LoRA in fp16"""
    
    # NF4 quantization for frozen base
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",  # NormalFloat4
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,  # Additional quantization
    )
    
    # Load ONLY the base model (frozen)
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="cpu",  # CPU offloading for layer processing
        trust_remote_code=True,
    )
    
    # Freeze base model completely
    for param in base_model.parameters():
        param.requires_grad = False
        
    # Add LoRA (these stay in fp16, trainable)
    model = get_peft_model(base_model, lora_config)
    
    # Only LoRA params are trainable
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"Trainable: {trainable_params:,} ({100*trainable_params/total_params:.3f}%)")
    
    return model

# Memory comparison:
# Qwen2.5-1.5B in fp16: ~3GB
# Qwen2.5-1.5B in NF4:  ~0.8GB
# LoRA adapters:         ~0.05GB
# Total:                  ~0.85GB (vs 3GB)
```

---

## 4. Benchmarking Suite

```python
#!/usr/bin/env python3
"""
LISA Benchmark Suite
Compare memory usage and speed against standard training.
"""

import torch
import time
import psutil
from pathlib import Path

def benchmark_standard_training(model_name, rounds=10):
    """Standard full-model training (for comparison)"""
    from transformers import AutoModelForCausalLM
    
    process = psutil.Process()
    memories = []
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
    )
    model.train()
    
    for i in range(rounds):
        torch.cuda.empty_cache()
        mem_before = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        
        # Simulate training step
        inputs = torch.randint(0, 1000, (1, 128))
        outputs = model(inputs, labels=inputs)
        outputs.loss.backward()
        
        mem_after = torch.cuda.memory_allocated() if torch.cuda.is_available() else 0
        memories.append(mem_after - mem_before)
        
        model.zero_grad()
        
    return {
        'method': 'standard',
        'avg_memory_mb': sum(memories) / len(memories) / 1024**2,
        'peak_memory_mb': max(memories) / 1024**2,
    }

def benchmark_lisa_training(model_name, rounds=10):
    """LISA layer-by-layer training"""
    # ... implementation
    pass

def run_benchmark_suite():
    models = ['Qwen/Qwen2.5-0.5B', 'Qwen/Qwen2.5-1.5B', 'Qwen/Qwen2.5-3B']
    results = []
    
    for model in models:
        print(f"\nBenchmarking {model}...")
        
        standard = benchmark_standard_training(model)
        lisa = benchmark_lisa_training(model)
        
        reduction = (standard['avg_memory_mb'] - lisa['avg_memory_mb']) / standard['avg_memory_mb'] * 100
        
        results.append({
            'model': model,
            'standard_mb': standard['avg_memory_mb'],
            'lisa_mb': lisa['avg_memory_mb'],
            'reduction_pct': reduction,
        })
        
    # Print comparison table
    print("\n" + "="*60)
    print(f"{'Model':<20} {'Standard':<12} {'LISA':<12} {'Reduction':<12}")
    print("="*60)
    for r in results:
        print(f"{r['model']:<20} {r['standard_mb']:<12.1f} {r['lisa_mb']:<12.1f} {r['reduction_pct']:<12.1f}%")
```

---

## 5. Distributed Layer Processing

**Concept:** Pass layers across network instead of just local disk.

```
┌─────────────┐     Layer Stream      ┌─────────────┐
│   Jetson A   │ ◄──────────────────► │   Jetson B   │
│  (Layers     │                      │  (Compute    │
│   Storage)   │                      │   LoRA)      │
└─────────────┘                      └─────────────┘
```

```python
class DistributedLayerServer:
    """Serve layers over network to other devices"""
    
    def __init__(self, model_path, port=5555):
        self.model_path = model_path
        self.server = ...  # asyncio server
        
    async def send_layer(self, client_writer, layer_idx):
        layer = torch.load(f"{self.model_path}/layer_{layer_idx}.pt")
        serialized = pickle.dumps(layer.state_dict())
        client_writer.write(serialized)
        await client_writer.drain()

class DistributedLayerClient:
    """Receive layers from network, compute, return gradients"""
    
    def __init__(self, server_ip, port=5555):
        self.reader, self.writer = await asyncio.open_connection(server_ip, port)
        
    async def request_layer(self, layer_idx):
        self.writer.write(str(layer_idx).encode())
        await self.writer.drain()
        
        # Receive layer
        data = await self.reader.read(10_000_000)  # 10MB buffer
        layer_state = pickle.loads(data)
        return layer_state
```

---

## Implementation Priority

| Priority | Task | Impact |
|----------|------|--------|
| 1 | Memory profiler | Debug current issues |
| 2 | Async prefetch | Hide I/O latency |
| 3 | Aggressive GC | Prevent OOM |
| 4 | NF4 quantization | 3x memory reduction |
| 5 | Distributed layers | Scale beyond single device |

---

## Next Steps

1. Add `MemoryProfiler` to current training script
2. Profile where memory spikes occur
3. Implement async prefetch for layer loading
4. Test NF4 on Jetson (if bitsandbytes works on ARM64)

---

## 6. Unified Memory Architecture (UMA) Optimization

**Problem:** PyTorch assumes discrete GPU with PCIe. Orin has CPU+GPU sharing unified RAM.

**Solution:** Use pinned memory and zero-copy to avoid redundant copies.

```python
import torch

def load_layer_uma(layer_weights, device='cuda'):
    """Load layer using unified memory - no PCIe copy needed"""
    
    # Use pinned memory for zero-copy transfer
    pinned = layer_weights.pin_memory()
    
    # GPU accesses pinned memory directly (no copy)
    return pinned.to(device, non_blocking=True)

# Alternative: Use gradient checkpointing to recompute instead of storing
from torch.utils.checkpoint import checkpoint_sequential

# Instead of storing all activations, recompute during backward
model.layer_blocks = checkpoint_sequential(
    model.layer_blocks,
    checkpoint_segments=4  # Recompute every 4 layers
)
```

**Note:** On Orin, CUDA can access CPU memory directly. Avoid unnecessary `.to('cuda')`.

---

## 7. Blockwise LoRA (Selective Layer Tuning)

**Insight:** Early layers = grammar, Late layers = reasoning/style.

**Solution:** Only tune final 30% of layers, skip first 70% entirely.

```python
from transformers import AutoModelForCausalLM
from peft import LoraConfig, get_peft_model

class BlockwiseLoRA:
    """Attach LoRA only to final layers"""
    
    def __init__(self, model, lora_rank=2, tune_fraction=0.3):
        self.total_layers = len(model.model.layers)
        self.tune_layers = int(self.total_layers * tune_fraction)
        self.skip_layers = self.total_layers - self.tune_layers
        
        print(f"Tuning only final {self.tune_layers}/{self.total_layers} layers")
        print(f"Skipping first {self.skip_layers} layers (frozen)")
        
        # Freeze early layers
        for i in range(self.skip_layers):
            for param in model.model.layers[i].parameters():
                param.requires_grad = False
        
        # Attach LoRA only to remaining layers
        lora_config = LoraConfig(r=lora_rank, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])
        model = get_peft_model(model, lora_config)
        return model
```

**Benefit:** 2x faster backward pass, 2x less I/O

---

## 8. Power State Verification

**Critical:** Throttling destroys performance silently.

```bash
# Force MAXN mode (maximum performance)
sudo nvpmodel -m 0

# Lock clocks at maximum
sudo jetson_clocks

# Verify frequencies
cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq
cat /sys/devices/17000000.ga10b/devfreq/17000000.ga10b/cur_freq
```

**Expected on Orin Nano:**
- CPU: 2.0+ GHz
- GPU: 1.0+ GHz
- EMC: 3.2 GHz

---

## 9. Distributed Layer Streaming (Network as RAM)

**Concept:** Use remote machine's RAM as cache over network.

```python
import zmq
import pickle

class DistributedLayerCache:
    """Stream layers from remote machine's RAM"""
    
    def __init__(self, remote_ip, port=5555):
        self.ctx = zmq.Context()
        self.sock = self.ctx.socket(zmq.REQ)
        self.sock.connect(f"tcp://{remote_ip}:{port}")
        
    def request_layer(self, layer_idx):
        self.sock.send(pickle.dumps(layer_idx))
        return pickle.loads(self.sock.recv())

# Server (on machine with spare RAM)
class LayerServer:
    def __init__(self, layers_path, port=5555):
        self.sock = self.ctx.socket(zmq.REP)
        self.sock.bind(f"tcp://{port}")
        self.layers_cache = {}
        
    def load_all_to_ram(self, path):
        for i in range(num_layers):
            self.layers_cache[i] = torch.load(f"{path}/layer_{i}.pt")
```

**Benefit:** Parallelize I/O across network + local storage
