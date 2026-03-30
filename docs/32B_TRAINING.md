# 32B Model Training on Jetson Orin

## Memory Analysis

### Hardware Available
- Jetson Orin: 8GB GPU + 7GB RAM + 23GB Swap = 38GB Total

### Model Requirements

| Configuration | Memory | Fit? |
|--------------|--------|------|
| 32B FP16 (full) | ~64GB | ❌ No |
| 32B FP16 + gradients | ~72GB | ❌ No |
| 32B QLoRA 4-bit | ~16GB | ⚠️ Tight |
| 32B QLoRA + LoRA | ~17GB | ⚠️ Tight |
| 32B QLoRA + partial offload | ~12GB | ✅ Yes |

### Training Requirements
- 32B QLoRA: ~16GB for weights
- LoRA adapters: ~100MB
- Activations (batch 1, seq 8): ~500MB
- Gradients: ~500MB
- **Total needed: ~17GB**
- **Available: 38GB**

## Recommended Strategy

### Option A: QLoRA + Full CPU Offload
Load quantized weights, offload to CPU except active layer.

### Option B: QLoRA + Sequential Loading
Load layer-by-layer, only keep one layer in memory.

## Scripts
- `train_32b_full_stack.py` - Full stack training
- `lisa_full_stack.py` - Unified script for all models
- `test_qlora_jetson.py` - Test QLoRA compatibility
