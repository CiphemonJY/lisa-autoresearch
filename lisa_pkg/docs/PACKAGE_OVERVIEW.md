# LISA - Layer-Indexed Sequential Adapters

**Train 32B, 70B, and 120B language models on limited RAM hardware.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## The Problem

Training large language models requires massive memory:
- 32B model: 64GB+ RAM needed
- 70B model: 140GB+ RAM needed  
- 120B model: 240GB+ RAM needed

Most researchers and hobbyists have 8-16GB of RAM. Traditional methods simply don't work.

## The Solution: LISA

LISA enables large model training on limited hardware through:

1. **Layer-by-Layer Processing**: Instead of loading all weights, process one layer at a time
2. **QLoRA Integration**: 4-bit quantized base model stays frozen
3. **LoRA Adapters**: Only train small adapter weights (MB instead of GB)
4. **Memory Offload**: Layers loaded from disk on-demand

## Hardware Results

| Model | Traditional RAM | LISA RAM | Savings |
|-------|----------------|----------|---------|
| 32B | 64GB | **4GB** | 94% |
| 70B | 140GB | **6GB** | 96% |
| 120B | 240GB | **8GB** | 97% |

Tested on: **Jetson Orin (7.4GB RAM)**

## Quick Start

```bash
# Clone the repo
git clone https://github.com/CiphemonJY/LISA_FTM.git
cd LISA_FTM

# Train a 70B model
python lisa_pkg/src/lisa_70b_v2.py

# Run inference
python lisa_pkg/src/lisa_inference_prod.py
```

## Python API

```python
from lisa_pkg.src.lisa_70b_v2 import LISATrainer, CONFIG

# Initialize for 70B model
trainer = LISATrainer(CONFIG)

# Train
for text in dataset:
    result = trainer.train_step(text)
    
# Save adapter
trainer.lora.save("my_adapter.npz")
```

## Project Structure

```
lisa_pkg/
├── src/                    # Core LISA implementations
│   ├── lisa_70b_v2.py     # 70B training
│   ├── lisa_120b_training.py  # 120B training
│   └── lisa_inference_prod.py # LISA inference
├── examples/               # Example scripts
│   ├── train_70b.py
│   ├── train_120b.py
│   └── inference.py
├── docs/                  # Documentation
└── README.md

docs/
├── LISA_70B_RESULTS.md    # 70B benchmark results
├── LISA_120B_RESULTS.md   # 120B benchmark results
└── LISA_INFERENCE.md       # Inference documentation
```

## Key Papers/Concepts

- **QLoRA**: Efficient fine-tuning of quantized LLMs
- **LoRA**: Low-Rank Adaptation
- **Layer-wise training**: Progressive layer training

## Documentation

- [70B Results](docs/LISA_70B_RESULTS.md)
- [120B Results](docs/LISA_120B_RESULTS.md)  
- [Inference Guide](docs/LISA_INFERENCE.md)
- [Technical Details](docs/LISA_TECHNICAL_PAPER.md)

## License

MIT License - see [LICENSE](LICENSE)

## Citation

If you use LISA in your research, please cite:

```bibtex
@software{lisa2026,
  title={LISA: Layer-Indexed Sequential Adapters},
  author={LISA Team},
  year={2026},
  url={https://github.com/CiphemonJY/LISA_FTM}
}
```
