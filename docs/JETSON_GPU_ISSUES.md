# Jetson Orin GPU Investigation

## Issue
CUDA allocator fails when loading 7B+ models directly to GPU on Jetson Orin.

## Error Observed
```
NvMapMemAllocInternalTagged: 1075072515 error 12
NvMapMemHandleAlloc: error 0
RuntimeError: NVML_SUCCESS == r INTERNAL ASSERT FAILED at "/opt/pytorch/c10/cuda/CUDACachingAllocator.cpp":1131
```

## System Info
- Hardware: Jetson Orin (8GB GPU)
- OS: Ubuntu 22.04.5 LTS
- Driver: 540.4.0
- CUDA: 12.6
- PyTorch: 2.8.0

## When It Occurs
- Loading 7B model with `device_map="cuda"`
- Loading 14B model with `device_map="sequential"`  
- Any attempt to allocate >1GB on GPU

## Possible Causes
1. **Memory fragmentation** - Previous processes may have fragmented CUDA memory
2. **Driver bug** - NVIDIA driver 540.4.0 may have allocator issues
3. **PyTorch incompatibility** - PyTorch 2.8.0 CUDA allocator not optimized for Orin
4. **Power management** - Orin power states may affect GPU memory allocation

## Solutions to Investigate

### 1. Clear CUDA Cache
```python
torch.cuda.empty_cache()
torch.cuda.synchronize()
gc.collect()
```

### 2. Environment Variables
```bash
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
```

### 3. Different Loading Strategy
```python
# Instead of device_map="cuda"
model = model.cuda()  # Explicit placement
```

### 4. Reboot
Clean reboot may clear GPU state completely.

### 5. Update Driver/PyTorch
May require JetPack update.

## Status
**Unresolved** - GPU training still broken as of 2026-03-29.

## Next Steps
- [ ] Try PYTORCH_CUDA_ALLOC_CONF
- [ ] Test after clean reboot
- [ ] Investigate JetPack updates
- [ ] Consider using CPU-only training permanently
