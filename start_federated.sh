#!/bin/bash
#===============================================================================
# start_federated.sh — Unified LISA Federated Training Launcher
#
# Usage:
#   # On Mac Mini (MLX):
#   bash start_federated.sh --role client --server SERVER_IP:8080 \
#     --model Qwen/Qwen2.5-7B --device mlx --layers 0,1,2,3,4,5,6,7,8 \
#     --api-key YOUR_KEY --rounds 10
#
#   # On Jetson (CUDA):
#   bash start_federated.sh --role client --server SERVER_IP:8080 \
#     --model Qwen/Qwen2.5-3B --device cuda --layers 20,21,22,23,24,25,26,27 \
#     --api-key YOUR_KEY --rounds 10
#
#   # As pool coordinator (server-side, on Jetson or head node):
#   bash start_federated.sh --role pool --server http://SERVER_IP:8080 \
#     --api-key YOUR_KEY --rounds 10
#
# Requirements:
#   Mac:  mlx, transformers, psutil
#   Linux/Jetson: torch, transformers, psutil, requests
#===============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---- Defaults ----------------------------------------------------------------
ROLE="client"
SERVER=""
MODEL="microsoft/phi-2"
DEVICE="auto"
LAYERS=""
TRAIN_STEPS=10
ROUNDS=10
API_KEY=""
COMPRESSION="topk"
COMPRESSION_RATIO="0.1"
ASSIGN_STRATEGY="bottom_top"
MIN_CLIENTS=1
POOL_TIMEOUT=300
POOL_DELAY=5.0

# ---- CLI Parsing --------------------------------------------------------------
while [[ $# -gt 0 ]]; do
    case $1 in
        --role)
            ROLE="$2"; shift 2 ;;
        --server)
            SERVER="$2"; shift 2 ;;
        --model)
            MODEL="$2"; shift 2 ;;
        --device)
            DEVICE="$2"; shift 2 ;;
        --layers)
            LAYERS="$2"; shift 2 ;;
        --train-steps)
            TRAIN_STEPS="$2"; shift 2 ;;
        --rounds)
            ROUNDS="$2"; shift 2 ;;
        --api-key)
            API_KEY="$2"; shift 2 ;;
        --compression)
            COMPRESSION="$2"; shift 2 ;;
        --compression-ratio)
            COMPRESSION_RATIO="$2"; shift 2 ;;
        --assign-strategy)
            ASSIGN_STRATEGY="$2"; shift 2 ;;
        --min-clients)
            MIN_CLIENTS="$2"; shift 2 ;;
        --timeout)
            POOL_TIMEOUT="$2"; shift 2 ;;
        --delay-between-rounds)
            POOL_DELAY="$2"; shift 2 ;;
        --help|-h)
            echo "Usage: $0 [--role client|pool] [options]"
            echo ""
            echo "Client options:"
            echo "  --server HOST:PORT    Federated server address (required)"
            echo "  --model MODEL_ID      HuggingFace model (default: microsoft/phi-2)"
            echo "  --device auto|mlx|cpu|cuda  Device selection (default: auto)"
            echo "  --layers N,N,N        Comma-separated layer indices this device trains"
            echo "  --train-steps N       Training steps per round (default: 10)"
            echo "  --rounds N            Number of federated rounds (default: 10)"
            echo "  --api-key KEY         Server auth key (default: \"\")"
            echo "  --compression none|topk|quantize  Gradient compression (default: topk)"
            echo "  --compression-ratio R  Compression ratio for topk (default: 0.1)"
            echo ""
            echo "Pool options:"
            echo "  --server http://HOST:PORT  Federated server URL (required)"
            echo "  --api-key KEY        Server auth key (required)"
            echo "  --model MODEL_ID      Model for layer detection (default: microsoft/phi-2)"
            echo "  --rounds N            Number of rounds (default: 10)"
            echo "  --assign-strategy bottom_top|interleaved|balanced"
            echo "  --min-clients N      Min clients per round (default: 1)"
            echo "  --timeout N          Round timeout seconds (default: 300)"
            echo "  --delay-between-rounds N  Seconds between rounds (default: 5.0)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"; exit 1 ;;
    esac
done

# ---- Platform Detection -------------------------------------------------------
detect_platform() {
    local os_type platform
    os_type=$(uname -s)
    if [[ "$os_type" == "Darwin" ]]; then
        platform="macos"
    elif [[ "$os_type" == "Linux" ]]; then
        platform="linux"
    else
        platform="unknown"
    fi
    echo "$platform"
}

detect_device() {
    local requested="$1"
    local platform="$2"

    if [[ "$requested" != "auto" ]]; then
        echo "$requested"
        return
    fi

    if [[ "$platform" == "macos" ]]; then
        # Check for MLX
        if python3 -c "import mlx" 2>/dev/null; then
            echo "mlx"
        else
            echo "cpu"
        fi
    elif [[ "$platform" == "linux" ]]; then
        # Check for CUDA
        if python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q True; then
            echo "cuda"
        else
            echo "cpu"
        fi
    else
        echo "cpu"
    fi
}

# ---- Validation ---------------------------------------------------------------
validate_client() {
    if [[ -z "$SERVER" ]]; then
        echo "ERROR: --server is required for client role"
        exit 1
    fi
    if [[ -z "$LAYERS" ]]; then
        echo "ERROR: --layers is required for client role"
        exit 1
    fi
    echo "[Launcher] Starting LISA federated CLIENT"
    echo "  Server:        $SERVER"
    echo "  Model:         $MODEL"
    echo "  Device:        $DEVICE (auto-detected)"
    echo "  LISA Layers:   $LAYERS"
    echo "  Train Steps:   $TRAIN_STEPS"
    echo "  Rounds:        $ROUNDS"
    echo "  Compression:   $COMPRESSION (ratio=$COMPRESSION_RATIO)"
}

validate_pool() {
    if [[ -z "$SERVER" ]]; then
        echo "ERROR: --server is required for pool role"
        exit 1
    fi
    echo "[Launcher] Starting LISA federated POOL coordinator"
    echo "  Server:        $SERVER"
    echo "  Model:         $MODEL (for layer detection)"
    echo "  Rounds:        $ROUNDS"
    echo "  Assign:        $ASSIGN_STRATEGY"
    echo "  Min Clients:   $MIN_CLIENTS"
}

# ---- Install missing deps ----------------------------------------------------
check_deps() {
    local missing=()
    python3 -c "import mlx" 2>/dev/null || true
    python3 -c "import torch" 2>/dev/null || missing+=("torch")
    python3 -c "import transformers" 2>/dev/null || missing+=("transformers")
    python3 -c "import requests" 2>/dev/null || missing+=("requests")
    python3 -c "import psutil" 2>/dev/null || missing+=("psutil")
    python3 -c "import numpy" 2>/dev/null || missing+=("numpy")

    if [[ ${#missing[@]} -gt 0 ]]; then
        echo "[Launcher] Missing packages: ${missing[*]}"
        echo "[Launcher] Install with: pip install ${missing[*]}"
        # Don't exit — might be in venv
    fi
}

# ---- Signal handling for clean exit -----------------------------------------
cleanup() {
    echo ""
    echo "[Launcher] Caught shutdown signal — cleaning up..."
    echo "[Launcher] Round $ROUNDS completed or interrupted."
    echo "[Launcher] Exiting cleanly."
    exit 0
}
trap cleanup SIGINT SIGTERM

# ---- Main --------------------------------------------------------------------
PLATFORM=$(detect_platform)
echo "[Launcher] Detected platform: $PLATFORM"

check_deps

if [[ "$ROLE" == "client" ]]; then
    validate_client
    DETECTED_DEVICE=$(detect_device "$DEVICE" "$PLATFORM")
    echo "[Launcher] Running on: $PLATFORM | Device: $DETECTED_DEVICE"

    CMD="python3 -c \"
import sys; sys.path.insert(0, '$SCRIPT_DIR')
from federated_lisa import create_client, main
client, device = create_client(
    model='$MODEL',
    layers=[$LAYERS],
    server='$SERVER',
    api_key='$API_KEY',
    device='$DEVICE',
    train_steps=$TRAIN_STEPS,
    compression='$COMPRESSION',
    compression_ratio=float('$COMPRESSION_RATIO'),
)
client.load_model()
client.run($ROUNDS)
\""
    eval "$CMD"

elif [[ "$ROLE" == "pool" ]]; then
    validate_pool
    python3 -c "
import sys; sys.path.insert(0, '$SCRIPT_DIR')
from federated_pool import FederatedPool
pool = FederatedPool(
    server_url='$SERVER',
    pool_api_key='$API_KEY',
    model_name='$MODEL',
    round_timeout=$POOL_TIMEOUT,
)
print('[Pool] Discovering devices...')
pool.discover_devices()
print('[Pool] Assigning layers...')
pool.assign_layers(strategy='$ASSIGN_STRATEGY')
import json
print(json.dumps(pool.status(), indent=2))
print(f'[Pool] Running $ROUNDS rounds...')
pool.run(
    num_rounds=$ROUNDS,
    min_clients_per_round=$MIN_CLIENTS,
    delay_between_rounds=float('$POOL_DELAY'),
)
print('[Pool] Final status:')
print(json.dumps(pool.status(), indent=2))
"

else
    echo "ERROR: Unknown role: $ROLE"
    echo "Valid roles: client, pool"
    exit 1
fi

echo "[Launcher] Done."
