# LISA Optimization Roadmap

*Based on Gemini's engineering suggestions - 2026-04-02*

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

**Problem:** Frozen base weights still consume significant RAM.

**Solution:** Use bitsandbytes NF4 for frozen base, keep LoRA in fp16.

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
