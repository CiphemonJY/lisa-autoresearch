# Config Directory

This directory holds configuration files for LISA_FTM federated learning.

## Default Config

`default.yaml` is loaded automatically if it exists at `config/default.yaml`. You can
override it with the `--config` CLI flag.

## Per-Client Configuration (Federated Learning)

In federated learning, each client may run on different hardware (GPU, CPU, memory-constrained).
You can create client-specific config files to override defaults per-machine:

### Approach 1: Client-specific config files

Create a config for each client machine:

```
config/
  default.yaml          # Server defaults
  client-hospital-1.yaml # Override for hospital-1's hardware
  client-lab-2.yaml     # Override for lab-2's hardware
```

Run each client with:
```bash
python main.py --mode client --client-id hospital-1 --config config/client-hospital-1.yaml
```

### Approach 2: Environment variables

Set hardware-specific values via environment variables (these take precedence over YAML):

```bash
export LISA_MODEL_NAME="EleutherAI/pythia-70m"
export LISA_LR="0.0003"
export LISA_EPOCHS="5"
python main.py --mode client --client-id hospital-1
```

### Approach 3: Hardware auto-detection

Run hardware detection first to see what the machine supports:
```bash
python main.py --mode hardware
```

Then create a client config based on the recommended settings.

## Config Precedence (highest to lowest)

1. **CLI arguments** (--lr, --epochs, --model, etc.)
2. **Custom config file** (--config path/to/config.yaml)
3. **Environment variables** (LISA_* prefix)
4. **default.yaml** (config/default.yaml)
5. **Hardcoded defaults in main.py**

## Config Keys Reference

```yaml
model:
  name: str              # HuggingFace model ID
  lora_rank: int         # LoRA rank (4 = small, 16 = medium)
  lora_alpha: int        # LoRA alpha scaling

training:
  lr: float              # Learning rate
  epochs: int            # Number of local epochs per round
  batch_size: int        # Training batch size
  seq_length: int        # Max sequence length
  steps: int             # Max training steps (client)

federated:
  server_host: str       # Federated server hostname
  server_port: int       # Federated server port
  rounds: int            # Total federated rounds
  local_steps: int       # Local gradient steps per round

offload:
  enabled: bool         # Enable disk offloading for large models
  layer_groups: int       # Number of layer groups for offloading
```

## Example: Client with Limited RAM

For a machine with only 8GB RAM that can't fit a full model:

```yaml
# config/low-memory-client.yaml
model:
  name: "EleutherAI/pythia-70m"  # Small model
  lora_rank: 4
  lora_alpha: 8

training:
  lr: 3e-4
  epochs: 2
  batch_size: 2
  seq_length: 128

federated:
  rounds: 5
  local_steps: 3

offload:
  enabled: true
  layer_groups: 8  # More groups = lower memory usage
```

## Example: GPU Client with More Power

```yaml
# config/gpu-client.yaml
model:
  name: "microsoft/phi-2"  # Larger model
  lora_rank: 16
  lora_alpha: 32

training:
  lr: 5e-4
  epochs: 5
  batch_size: 8
  seq_length: 512

offload:
  enabled: false
```
