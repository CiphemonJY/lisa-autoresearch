# Federated LISA

**Train AI together — any device, any hardware, any location.**

This project implements federated fine-tuning of large language models. Every device — a laptop, a server, a Mac Mini — trains locally on its own data and shares learned updates with a central server. No data leaves the device. Everyone benefits from everyone's contributions.

Built on **LISA** (Layer-wise Importance Sampling): only train the layers that matter most, making fine-tuning fast and cheap on any hardware.

---

## What It Does

```
┌─────────────┐      gradients      ┌─────────────┐
│  PC (you)   │  ───────────────>  │   Server    │
│  TinyLlama  │                    │  aggregates │
│  trains     │  <───────────────  │  + distributes│
│  locally    │    model update    │  averaged   │
└─────────────┘                    └─────────────┘
       ▲                                ▲
       │                                │
┌─────────────┐                  ┌─────────────┐
│  Mac Mini   │                  │  GPU Server │
│  pythia-70m │                  │  7B model   │
└─────────────┘                  └─────────────┘
```

- Each device trains locally on its own data
- Only gradient updates (not data) are shared
- Server averages updates and distributes improvements to all clients
- **Any model works** — from 70M to 70B params

---

## Quick Start

### 1. Install dependencies
```bash
pip install torch transformers datasets huggingface_hub numpy \
  cryptography requests psutil pyyaml fastapi uvicorn pytest
```

### 2. Start the server
```bash
python -m federated.server --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --rounds 3 --port 8080
```

### 3. Run a client (same machine or another device)
```bash
python fed_client.py --host 127.0.0.1 --port 8080
```

### Or run locally without a server
```bash
python main.py --mode train --model EleutherAI/pythia-70m --steps 100
```

---

## Project Structure

```
lisa-autoresearch/
├── federated/
│   ├── client.py       # FederatedClient: train + exchange gradients
│   ├── server.py       # FederatedServer: aggregate + distribute
│   └── ...
├── lisa/
│   ├── train_torch.py  # LISA layer-wise training (PyTorch, CPU/GPU)
│   └── offload_torch.py # Disk-offloaded training for 7B+ on 16GB RAM
├── inference/
│   └── engine.py       # Checkpoint → inference pipeline
├── distributed/
│   ├── p2p.py          # P2P gradient exchange
│   └── discovery.py    # Client discovery for federation
├── api/
│   └── server.py       # FastAPI server + HTTP client runners
├── tests/
│   ├── test_federated.py          # 25 federated unit tests
│   └── test_http_integration.py   # HTTP server integration
├── main.py             # Unified CLI: simulate, train, hardware, server, client
├── real_training.py    # Standalone training with real dataset
└── fed_client.py       # Federated client entry point
```

---

## Training Modes

| Mode | Description | Hardware |
|------|-------------|----------|
| `simulate` | Federated simulation with 3 local clients | CPU |
| `train` | Single-device LISA training (LoRA + layer selection) | CPU/GPU |
| `hardware` | Detect hardware and recommend settings | Any |
| `server` | Run federated coordination server | Any |
| `client` | Connect to server as federated participant | Any |

```bash
# Simulate a federated round
python main.py --mode simulate --clients 3 --rounds 3

# LISA training on CPU
python main.py --mode train --model EleutherAI/pythia-70m --iters 200

# Real training with wikitext dataset
python real_training.py --model EleutherAI/pythia-70m --steps 200
```

---

## Hardware Constraints

The magic of federated learning: **every device contributes, regardless of hardware.**

| Device | RAM | Can Train |
|--------|-----|-----------|
| This PC | 8 GB | ✅ TinyLlama-1.1B, pythia-70m |
| Mac Mini M2 | 16 GB | ✅ 7B models |
| Cloud GPU | 80 GB | ✅ 32B+ models |

No single machine needs to train the entire model. Each device trains what it can, shares what it learns, and receives aggregated improvements from the network.

---

## LISA: Layer-wise Importance Sampling

LISA reduces compute by selectively training only the most important layers per round:

- **Bottom layers** (always trained): Capture foundational language patterns
- **Top layers** (always trained): Handle task-specific outputs
- **Middle layers** (randomly sampled): Randomly selected each round

```
Round 0: Train layers [0, 1, 8, 17, 20, 21]
Round 1: Train layers [0, 1, 6, 15, 20, 21]
Round 2: Train layers [0, 1, 10, 14, 20, 21]
```

Result: ~70% compute reduction with minimal quality loss.

Combined with **LoRA** (Low-Rank Adaptation), only 0.2-0.3% of model parameters are trained per step — enabling large models on small hardware.

---

## Federated Privacy

Data never leaves the local device:

1. Model trains on local data
2. Only gradient tensors (not data) are sent to server
3. Server aggregates via FedAvg — individual contributions are indistinguishable
4. Optional: differential privacy noise can be added to gradients client-side

This makes federated learning suitable for:
- Healthcare data (HIPAA compliance)
- Enterprise data (no data leaves the firewall)
- Personal devices (your data stays yours)

---

## Results

| Model | Dataset | Steps | Time | Final Loss |
|-------|---------|-------|------|------------|
| Pythia-70m (real training) | Wikitext-2 | 200 | ~10 min | 0.01 |
| TinyLlama-1.1B (federated) | Synthetic | 5 rounds | ~2 min/round | 1.87 |
| distilgpt2 | Synthetic | 5 | ~30s | 7.71 |

---

## GitHub

**https://github.com/CiphemonJY/lisa-autoresearch**

Clone and run:
```bash
git clone https://github.com/CiphemonJY/lisa-autoresearch
cd lisa-autoresearch
pip install torch transformers datasets huggingface_hub
python main.py --mode simulate
```
