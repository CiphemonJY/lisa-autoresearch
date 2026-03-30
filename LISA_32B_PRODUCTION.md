# LISA 32B Production Guide

**Status:** ✅ Training Working | ⚠️ Inference Needs Setup

---

## Current Status

| Component | Status | Notes |
|-----------|--------|-------|
| Training | ✅ Working | 1.03GB memory, 50+ steps |
| Checkpointing | ✅ Working | Saves to npz files |
| LoRA format | ⚠️ Incompatible | npz ≠ llama.cpp format |
| Inference | ⚠️ Blocked | Need single GGUF file |
| Real data | ⚠️ Placeholder | Using sample texts |

---

## Making It Production Ready

### Step 1: Fix LoRA Format for llama.cpp

**Problem:** Our npz format isn't compatible with llama.cpp's --lora flag.

**Solution:** Implement proper GGUF LoRA adapter format.

```python
# What llama.cpp expects for LoRA:
# - GGUF file with LoRA tensors
# - Specific tensor naming: 
#   - model.layers.{i}.attention.wq.weight (base)
#   - lora: model.layers.{i}.attention.wq.weight (LoRA)

# Our current format:
# - npz with layer_{i}.q_a, layer_{i}.q_b, etc.
```

**TODO:** Implement adapter format converter.

### Step 2: Get Single GGUF File

**Problem:** We have 5 split GGUF files that llama.cpp can't load properly.

**Solution:** Download single file:

```bash
# Option 1: HuggingFace CLI
huggingface-cli download Qwen/Qwen2.5-32B-Instruct-GGUF \
    qwen2.5-32B-Instruct-Q4_K_M.gguf \
    --local-dir /tmp/single_gguf

# Option 2: Python
from huggingface_hub import hf_hub_download
hf_hub_download(
    repo_id="Qwen/Qwen2.5-32B-Instruct-GGUF",
    filename="qwen2.5-32B-Instruct-Q4_K_M.gguf",
    local_dir="/tmp/single_gguf"
)
```

### Step 3: Real Training Data

**Problem:** Using placeholder texts instead of real corpus.

**Solution:** Download training data:

```bash
# Download a small corpus
curl -O https://raw.githubusercontent.com/openai/whatsapp/main/data/train.json

# Or use HuggingFace datasets
pip install datasets
python -c "
from datasets import load_dataset
ds = load_dataset('wikitext', 'wikitext-2-v1')
print(ds)
"
```

### Step 4: Real Tokenizer

**Problem:** Using character-level tokenizer instead of Qwen's BPE.

**Solution:** Load Qwen tokenizer:

```python
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-32B')
```

---

## Quick Start (Current)

```bash
# 1. Run training
python3 lisa_32b_production.py --steps 500 --lora-rank 4 --lisa-depth 2

# 2. Checkpoint saved to
ls /tmp/lisa_checkpoints/

# 3. Current output:
#    50 steps: 1.14 avg loss
#    Memory: 1.03GB
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    LISA 32B TRAINING PIPELINE                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐     ┌──────────────┐     ┌──────────────┐     │
│  │   Text     │────▶│  Tokenizer  │────▶│  Embeddings  │     │
│  │   Data      │     │  (simple)   │     │  (frozen)    │     │
│  └─────────────┘     └──────────────┘     └──────────────┘     │
│                                                  │              │
│                                                  ▼              │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │              LISA LAYER PROCESSING                         │ │
│  │  ┌─────────┐  ┌─────────┐  ┌─────────┐     ┌─────────┐    │ │
│  │  │Layer 0-1│─▶│Layer 2-3│─▶│Layer 4-5│─▶...│Layer 62-63│   │ │
│  │  │ +LoRA   │  │ +LoRA   │  │ +LoRA   │     │ +LoRA    │   │ │
│  │  └─────────┘  └─────────┘  └─────────┘     └─────────┘    │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                    │                            │
│                                    ▼                            │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │              LORA BACKWARD (Gradient Descent)              │ │
│  │  Only update LoRA A,B matrices (10.5M params)             │ │
│  │  Base model stays frozen                                    │ │
│  └─────────────────────────────────────────────────────────────┘ │
│                                    │                            │
│                                    ▼                            │
│                            ┌─────────────┐                     │
│                            │   Loss      │                     │
│                            │   1.14 avg  │                     │
│                            └─────────────┘                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Memory Breakdown

| Component | Memory | Status |
|-----------|--------|--------|
| Python runtime | ~600MB | Fixed |
| LoRA params | ~42MB | Trainable |
| Embeddings | ~50MB | Frozen |
| Activations | ~200MB | Per batch |
| Layer buffers | ~100MB | Reused |
| **Total** | **~1GB** | ✅ Fits! |

---

## Next Steps Priority

1. **High:** Download single GGUF file → Enable inference
2. **High:** Implement llama.cpp LoRA format → Valid checkpoints
3. **Medium:** Load Qwen tokenizer → Real tokenization
4. **Medium:** Download training corpus → Better training
5. **Low:** Multi-GPU scaling → Faster training

---

## File Structure

```
lisa_proj/
├── PRODUCTION.md                      # This file
├── LISA_32B_TECHNICAL_REPORT.md       # Technical details
├── LISA_32B_BENCHMARKS.md              # Performance benchmarks
├── LISA_LAYER_BY_LAYER_32B.md         # Initial breakthrough
├── 
├── TRAINING SCRIPTS
├── lisa_32b_production.py             # ⭐ Main production script
├── lisa_production_full.py            # Full implementation
├── lisa_32b_realdata.py               # With real data
├── 
├── UTILITIES
├── test_lora.py                       # Test LoRA adapters
├── test_inference.py                  # Test inference
├── 
└── CHECKPOINTS (on Jetson)
    └── /tmp/lisa_checkpoints/
        ├── step50.npz                 # 50 steps
        └── final.npz                  # Final
```

---

## FAQ

**Q: Why is memory so low?**
A: LISA + QLoRA + Offload. We never load full model into RAM.

**Q: Why is training slow?**
A: Sequential layer processing. But it works on 7.4GB hardware.

**Q: Can this run on Mac Mini?**
A: Yes, but Mac Mini has 16GB so you can use larger LISA depth.

**Q: What's the next model size?**
A: 70B should work with LISA depth=1 on Jetson.

---

## Contributing

1. Fork the repo
2. Create a feature branch
3. Run tests: `python3 lisa_32b_production.py --steps 10`
4. Submit PR

---

*Last updated: 2026-03-30*
