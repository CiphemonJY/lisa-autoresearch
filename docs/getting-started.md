# Getting Started with LISA Federated Fine-Tuning

**Train AI together — any device, any hardware, any location.**

LISA (Layer-wise Importance Sampling) enables federated fine-tuning of LLMs. Every device trains locally on its own data and shares learned updates with a central server. No data leaves your device.

---

## 1. Quick Start

Three steps to run your first federated round with TinyLlama:

### Step 1 — Install dependencies

```bash
pip install torch transformers datasets huggingface_hub numpy cryptography requests psutil pyyaml fastapi uvicorn pytest
```

### Step 2 — Start the server (Terminal 1)

```bash
cd lisa-autoresearch
python -m federated.server --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --rounds 3 --port 8080
```

The server will:
- Download the model
- Wait for clients to connect
- Aggregate gradients and distribute model updates

### Step 3 — Run a client (Terminal 2)

```bash
cd lisa-autoresearch
python fed_client.py --host 127.0.0.1 --port 8080
```

The client will connect, train LoRA layers locally, send gradients to the server, and receive the aggregated model update.

> **Tip:** You can run multiple clients on different machines pointing to the same server — each contributes to the federation.

---

## 2. Installation

### pip (recommended)

```bash
pip install torch transformers datasets huggingface_hub
```

### From source

```bash
git clone https://github.com/CiphemonJY/lisa-autoresearch
cd lisa-autoresearch
pip install -e .
```

Or install all dependencies:

```bash
pip install -r requirements.txt
```

### Verify your install

```bash
python main.py --mode hardware
```

This prints your CPU/GPU info and recommends settings.

---

## 3. Basic Concepts

### What is Federated Learning?

Federated learning lets multiple devices train a shared model without sharing raw data:

1. A **central server** holds the global model
2. Each **client** downloads the model, trains it locally on their own data
3. Clients send only **gradient updates** (not data) back to the server
4. Server **averages** all updates and improves the global model
5. All clients download the improved model and repeat

```
┌──────────────┐    gradients    ┌──────────────┐
│  Your Laptop │ ──────────────► │    Server     │
│  trains      │                 │  aggregates  │
│  locally     │ ◄────────────── │  + distributes│
└──────────────┘  model update  └──────────────┘
       ▲                                ▲
       │                                │
┌──────────────┐                 ┌──────────────┐
│  Mac Mini    │                 │  GPU Server   │
│  (different  │                 │  (7B model)   │
│   data)      │                 │               │
└──────────────┘                 └──────────────┘
```

### What does LISA do?

LISA = **Layer-wise Importance Sampling**

Instead of training all layers every round, LISA selectively trains only the most important layers:

- **Bottom layers** (always trained): Capture foundational language patterns
- **Top layers** (always trained): Handle task-specific outputs
- **Middle layers** (randomly sampled each round): Random selection

```
Round 0: Train layers [0, 1, 8, 17, 20, 21]
Round 1: Train layers [0, 1, 6, 15, 20, 21]
Round 2: Train layers [0, 1, 10, 14, 20, 21]
```

This gives ~70% compute reduction with minimal quality loss.

### Why LoRA?

**LoRA** (Low-Rank Adaptation) adds small trainable matrices alongside frozen model weights:

- Only **0.2–0.3%** of model parameters are trained per step
- TinyLlama-1.1B fine-tunes comfortably on **8GB RAM** (CPU)
- 7B models work on a **Mac Mini M2** (16GB)

No data leaves your device — just small gradient updates.

---

## 4. Federated Training Walkthrough

### Start the server

```bash
python -m federated.server --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --rounds 3 --port 8080
```

Key flags:
- `--model` — Which model to aggregate (any causal LLM from HuggingFace)
- `--rounds` — Number of federated rounds (default: 3)
- `--port` — TCP port to listen on (default: 8080)
- `--min-clients` — Minimum clients needed before aggregation (default: 2)

Expected output:
```
FederatedServer initialized (model=TinyLlama/TinyLlama-1.1B-Chat-v1.0)
  Rounds: 3
  Min clients/round: 2
  Socket server listening on port 8080
```

### Connect a client

On another machine (or a second terminal on the same machine):

```bash
python fed_client.py --host 127.0.0.1 --port 8080
```

Or specify a different model:

```bash
python fed_client.py --host 192.168.1.100 --port 8080 --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
```

> **Important:** Clients must use the **same model architecture** as the server. Different model sizes of the same architecture (e.g., TinyLlama-1.1B vs TinyLlama-500M) are not directly compatible.

### What happens each round?

```
Round N begins
│
├── 1. Client downloads current model from server
├── 2. Client applies LoRA to trainable layers
├── 3. Client trains locally on local data (no data sent to server)
├── 4. Client extracts gradients from LoRA layers only
├── 5. Client sends gradients to server (NOT raw data)
│
├── Server receives gradients from all connected clients
├── Server validates gradients (checks norm, sample count)
├── Server aggregates via FedAvg (weighted average by sample count)
├── Server applies aggregated gradient to global model
├── Server saves checkpoint
│
└── Server sends updated model back to all clients
    └── Clients apply update and prepare for next round
```

### Run multiple clients

Start the server once, then run clients on as many machines as you want:

```bash
# Terminal 2
python fed_client.py --host 127.0.0.1 --port 8080

# Terminal 3 (or another machine)
python fed_client.py --host 127.0.0.1 --port 8080

# Terminal 4 (or a friend's machine)
python fed_client.py --host 127.0.0.1 --port 8080
```

Each client trains independently and contributes its gradient. The server waits for at least `--min-clients` before aggregating.

---

## 5. Configuration

### config/default.yaml

The project ships with a default config at `config/default.yaml`. It is loaded automatically if it exists.

```yaml
model:
  name: "EleutherAI/pythia-70m"
  lora_rank: 4
  lora_alpha: 8

training:
  lr: 3e-4
  epochs: 3
  batch_size: 4
  seq_length: 128
  steps: 200

federated:
  server_host: "127.0.0.1"
  server_port: 8080
  rounds: 3
  local_steps: 5

offload:
  enabled: false
  layer_groups: 4
```

All values can be overridden by CLI arguments.

### Per-client overrides for federated

When a client connects, it inherits defaults from `config/default.yaml`. To override specific settings per client, use CLI flags:

```bash
# Client with custom training steps
python fed_client.py --host 127.0.0.1 --port 8080 --model TinyLlama/TinyLlama-1.1B-Chat-v1.0

# Server with custom rounds
python -m federated.server --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --rounds 10 --min-clients 3
```

### Load a custom config file

```bash
python main.py --config path/to/custom_config.yaml --mode train
```

---

## 6. Gradient Compression

Gradient compression reduces network bandwidth when sending updates. Use the `--compression` flag on the **server**:

```bash
python -m federated.server --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --rounds 3 --compression both
```

| Flag value | Description |
|---|---|
| `none` | Send gradients uncompressed (default) |
| `sparsify` | Keep only top K% largest gradients by magnitude |
| `quantize` | Quantize gradients to `N` bits (default: 8-bit) |
| `both` | Sparsify first, then quantize the sparse result |

### Tuning compression

```bash
python -m federated.server \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --compression both \
  --compression-k 0.1 \    # Keep top 10% of gradient values
  --compression-bits 8      # Quantize to 8-bit
```

### How compression helps

- `sparsify --compression-k 0.05`: Only top 5% of gradient values are transmitted — 20x bandwidth reduction
- `quantize --compression-bits 8`: 32-bit floats → 8-bit integers — 4x reduction
- `both`: Combine for maximum compression

### Client-side behavior

Clients receive compressed updates from the server and automatically decompress them. No extra flags needed on the client side.

---

## 7. Authentication

The server can require a shared secret token from clients to prevent unauthorized access.

### Generate a token

```bash
python -m federated.server --gen-token
```

This prints a random token (e.g., `Kx9mT2...`). Copy it — you'll need it for clients.

### Start server with auth

```bash
python -m federated.server \
  --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
  --rounds 3 \
  --auth-token "your-token-here"
```

### Connect client with token

```bash
python fed_client.py \
  --host 127.0.0.1 \
  --port 8080 \
  --auth-token "your-token-here"
```

> **Note:** If `--auth-token` is set on the server but a client connects without one (or with a wrong token), the connection is rejected.

---

## 8. Local Training (No Federation)

You can train a model locally without any server — useful for experimentation or when you don't have multiple machines.

### Basic local training

```bash
python main.py --mode train --model EleutherAI/pythia-70m --steps 100
```

This uses LISA (layer selection + LoRA) to train on your local machine.

### Run with real dataset (Wikitext)

```bash
python real_training.py --model EleutherAI/pythia-70m --steps 200
```

### Available modes in main.py

| Mode | Description |
|---|---|
| `train` | Single-device LISA training with LoRA |
| `simulate` | Federated simulation with 3 in-process clients |
| `hardware` | Detect hardware and recommend settings |
| `server` | Run federated coordination server |
| `client` | Connect to server as federated participant |

### Hardware detection

```bash
python main.py --mode hardware
```

Outputs CPU/GPU info and recommendations for model size and batch size.

---

## 9. Inference

After training, generate text from your fine-tuned checkpoint:

```bash
python inference_demo.py --dir output/real_training
```

This loads the last checkpoint from the specified directory and runs inference.

### How it works

The `inference_demo.py` script:
1. Loads the trained model checkpoint
2. Tokenizes your prompt
3. Generates text using the fine-tuned weights
4. Prints the result

### With a custom checkpoint

```bash
python inference_demo.py --dir checkpoints
```

### In Python

```python
from inference.engine import InferenceEngine

engine = InferenceEngine("output/real_training")
result = engine.generate("The fundamentals of machine learning are", max_new_tokens=50)
print(result)
```

---

## 10. Hardware Requirements

### What works on CPU?

| Model | RAM | Notes |
|---|---|---|
| EleutherAI/pythia-70m | 4 GB | Fast, good for testing |
| distilgpt2 | 4 GB | Very lightweight |
| TinyLlama-1.1B | 8 GB | Minimum for chat-style models |

### GPU recommendations

| GPU VRAM | Recommended models |
|---|---|
| 6 GB | TinyLlama-1.1B, pythia-160m |
| 8 GB | TinyLlama-1.1B, pythia-410m |
| 16 GB | 7B models (qwen2.5-7B, Llama-3.1-8B) |
| 24 GB | 13B models |
| 40+ GB | 70B+ models |

### Disk-offloaded training (large models on small RAM)

For 7B+ models on machines with limited RAM:

```bash
python main.py --mode offload --model Qwen/Qwen2.5-7B-Instruct --iters 50
```

This offloads layers to disk as needed, enabling huge models on modest hardware.

### Federated learning and hardware

The key advantage of federated learning: **no single machine needs to train the whole model**. Each device trains what it can (LoRA layers only), shares the small gradient update, and benefits from the aggregated result.

---

## 10. Running on Raspberry Pi / Jetson Nano

The `client_minimal.py` script is a single-file federated client designed for memory-constrained edge devices. It implements the same gradient-exchange protocol as `fed_client.py` but uses only the core dependencies.

### Hardware profiles

| Device | RAM | Recommended settings |
|--------|-----|----------------------|
| **RPi Zero 2 W** | 512 MB | `--device cpu --batch-size 1 --local-steps 3` |
| **Jetson Nano** | 4 GB | `--device cuda --batch-size 4 --local-steps 3` |
| **ARM64 board** | 1–2 GB | `--device cpu --batch-size 1 --lora-rank 4` |

### Requirements (minimal)

```bash
# RPi Zero 2 W / ARM64
pip install torch transformers datasets

# Jetson Nano (with CUDA)
pip install torch transformers datasets
# Ensure JetPack CUDA libraries are installed system-wide
```

For RAM monitoring (optional), install `psutil`:

```bash
pip install psutil
```

### Quick start

**1. Connect to a federated server:**

```bash
python client_minimal.py \
  --server 127.0.0.1:8080 \
  --model EleutherAI/pythia-70m \
  --device auto \
  --local-steps 3
```

**2. With GPU (Jetson Nano):**

```bash
python client_minimal.py \
  --server 127.0.0.1:8080 \
  --model EleutherAI/pythia-70m \
  --device cuda \
  --local-steps 5 \
  --batch-size 4
```

**3. Standalone (no server — saves gradients locally):**

```bash
python client_minimal.py \
  --model EleutherAI/pythia-70m \
  --local-steps 3 \
  --rounds 3
```

### CLI flags

| Flag | Default | Description |
|------|---------|-------------|
| `--server` | `http://127.0.0.1:8080` | Federated server address |
| `--model` | `EleutherAI/pythia-70m` | HuggingFace model ID |
| `--auth-token` | (env: `LISA_AUTH_TOKEN`) | Auth token for server |
| `--device` | `auto` | `auto`, `cpu`, or `cuda` |
| `--local-steps` | `3` | Training steps per round |
| `--lora-rank` | `4` | LoRA rank (memory vs quality) |
| `--batch-size` | `4` | Training batch size |
| `--rounds` | `3` | Number of federated rounds |
| `--timeout` | `300` | Socket timeout (seconds) |

### Memory usage

`client_minimal.py` prints RAM usage at startup and every 10 training steps:

```
RAM step 0 (after model load): 420.3MB
CUDA:      NVIDIA Jetson Nano
VRAM:      4.0GB
```

On the RPi Zero 2 W, only LoRA adapter gradients are held in RAM — the base model weights stay on disk/CPU. The full gradient dict is never materialised alongside the model.

### Jetson Nano GPU notes

- `torch.cuda.is_available()` auto-detects CUDA — `--device auto` selects GPU by default
- VRAM is logged at startup: `torch.cuda.get_device_properties(0).total_memory`
- Gradient compression (sparsify / quantize) works identically on GPU — tensors are moved to CPU before sending

### RPi Zero 2 W notes

- Load only the tokenizer + config first; model params live in CPU RAM
- LoRA rank 4 is the default (minimal trainable footprint)
- Use `--batch-size 1` to keep peak RAM below 512 MB
- Gradients are sent as base64-encoded pickle (same protocol as `fed_client.py`)

## Troubleshooting

### "Connection refused" when client connects

- Make sure the server is running first
- Check that `--port` matches on both server and client
- Check firewall settings if connecting across machines

### Out of memory

- Use a smaller model (pythia-70m instead of TinyLlama-1.1B)
- Reduce batch size in `config/default.yaml`
- Use GPU if available (model auto-detects CUDA)

### Model download is slow

Models are downloaded from HuggingFace on first run. Set `HF_HOME` to a directory with more space:

```bash
export HF_HOME=/path/to/large/disk
```

### Client disconnected mid-round

The server handles disconnects gracefully:
- Timeout is 30 seconds per socket operation
- Disconnected clients' gradients are excluded from the current round
- Round continues with remaining active clients
- Clients can reconnect for the next round

### Server shows "fewer clients than min_clients"

Lower `--min-clients` or wait for more clients to connect:

```bash
python -m federated.server --min-clients 1
```
