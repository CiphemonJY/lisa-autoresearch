# LISA Package - Easy to Use

Train 32B, 70B, and 120B language models on limited RAM hardware.

## What is LISA?

LISA (Layer-Indexed Sequential Adapters) enables training massive language models on consumer hardware:

| Model | Traditional RAM | LISA RAM |
|-------|---------------|----------|
| 32B | 64GB+ | **4GB** |
| 70B | 140GB+ | **6GB** |
| 120B | 240GB+ | **8GB** |

## Quick Start

```bash
# Train 70B model
python lisa_pkg/src/lisa_70b_v2.py

# Train 120B model  
python lisa_pkg/src/lisa_120b_training.py

# Run LISA inference
python lisa_pkg/src/lisa_inference_prod.py
```

## Python API

```python
from lisa_pkg.src.lisa_70b_v2 import LISATrainer, CONFIG

trainer = LISATrainer(CONFIG)
for text in dataset:
    result = trainer.train_step(text)
```

## Package Structure

```
lisa_pkg/
├── src/
│   ├── lisa_70b_v2.py          # 70B training
│   ├── lisa_120b_training.py  # 120B training
│   ├── lisa_inference_prod.py  # LISA inference
│   └── lisa_fullstack_realtraining.py # Full stack
├── examples/
│   ├── train_70b.py
│   ├── train_120b.py
│   └── inference.py
├── scripts/
│   └── train.sh
├── docs/
│   └── PACKAGE_OVERVIEW.md
└── README.md
```

## Key Features

- **Memory Efficient**: Train 120B on 8GB RAM
- **Simple API**: Just call `train_step()` 
- **LoRA Adapters**: Only train MB of weights
- **Layer-by-Layer**: Process one layer at a time

## Documentation

See `docs/PACKAGE_OVERVIEW.md` for full details.

## License

MIT
