# LISA Usage Guide

Train 32B-120B language models on limited RAM (8GB or less).

## Supported Model Formats

| Training | Inference |
|----------|-----------|
| HuggingFace (.bin, .safetensors) | GGUF (.gguf) |

## Quick Start

### 1. Installation

```bash
git clone https://github.com/CiphemonJY/LISA_FTM.git
cd LISA_FTM
pip install transformers torch accelerate peft
```

### 2. Training a Model

```bash
# 70B model training
python lisa_pkg/src/lisa_70b_v2.py

# 120B model training
python lisa_pkg/src/lisa_120b_training.py
```

The scripts will download a base model and apply LISA training with LoRA adapters.

### 3. Running Inference

```bash
# Point to your GGUF model files
python lisa_pkg/src/lisa_inference_prod.py --gguf-dir /path/to/your/model/
```

## Finding Models

### HuggingFace (for training)
- [Qwen 2.5 7B](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct)
- [Qwen 2.5 32B](https://huggingface.co/Qwen/Qwen2.5-32B-Instruct)
- [Llama 3 70B](https://huggingface.co/meta-llama/Meta-Llama-3-70B-Instruct) (requires approval)
- [Mistral 7B](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)

### GGUF Models (for inference)
- [TheBloke's GGUF Collection](https://huggingface.co/TheBloke)
- [LLaMA 3 GGUF](https://huggingface.co/QuantFactory/Llama-3-8B-Instruct-GGUF)
- [Qwen 2.5 GGUF](https://huggingface.co/Qwen)

## Hardware Requirements

| Model Size | Minimum RAM | Recommended |
|------------|-------------|-------------|
| 7B | 6GB | 8GB |
| 32B | 6GB | 12GB |
| 70B | 6GB | 16GB |
| 120B | 7.4GB | 16GB |

## How LISA Works

1. **Layer-by-Layer Processing**: Load one transformer layer at a time
2. **LoRA Adapters**: Only train ~0.1% of parameters
3. **Memory Offload**: Keep most weights on disk, not in RAM

This enables training 120B models on hardware that traditionally needs 240GB+.

## Example: Custom Training

```python
from lisa_pkg.src.lisa_70b_v2 import LISATrainer

# Your dataset
texts = ["Hello world", "How are you?"]

# Initialize trainer
trainer = LISATrainer()

# Train
for text in texts:
    result = trainer.train_step(text)
    print(f"Loss: {result['loss']:.4f}")

# Save adapter
trainer.save_adapter("/output/path")
```

## Troubleshooting

**Out of memory?**
- Reduce batch size
- Use 4-bit quantization (QLoRA)
- Close other applications

**GGUF not loading?**
- Ensure GGUF files are in a single directory
- Check file permissions
- Verify GGUF magic bytes (should be `GGUF`)

**Training slow?**
- Training is intentionally slower than full model training
- This is the trade-off for memory efficiency
- 120B training takes ~hours on limited hardware vs impossible normally

## Questions?

Open an issue on GitHub or check the main README.
