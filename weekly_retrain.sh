#!/bin/bash
# Weekly Model Retraining Script
# Retrains LLM personality model with latest conversation data
# Scheduled: Sunday 4 AM

set -e

WORKSPACE="$HOME/~/.lisa"
TRAINING="$WORKSPACE/training-data"
LOGS="$WORKSPACE/logs/training"
DATE=$(date +%Y-%m-%d)
VERSION="$(date +%Y%m%d)"

# Create log directory
mkdir -p "$LOGS"

echo "=== Weekly Retraining Started: $(date) ===" >> "$LOGS/weekly_$DATE.log"

cd "$TRAINING"

# Step 1: Gather latest conversation data
echo "Gathering conversation data..." >> "$LOGS/weekly_$DATE.log"
if [ -f "$WORKSPACE/scripts/gather_training_data.py" ]; then
    python3 "$WORKSPACE/scripts/gather_training_data.py" >> "$LOGS/weekly_$DATE.log" 2>&1 || echo "Warning: gather_training_data.py failed" >> "$LOGS/weekly_$DATE.log"
else
    echo "Warning: gather_training_data.py not found, using existing data" >> "$LOGS/weekly_$DATE.log"
fi

# Step 2: Prepare MLX format data for Qwen
echo "Preparing MLX training data for Qwen..." >> "$LOGS/weekly_$DATE.log"
if [ -f "convert_to_qwen_format.py" ]; then
    python3 convert_to_qwen_format.py >> "$LOGS/weekly_$DATE.log" 2>&1 || echo "Warning: Qwen conversion failed" >> "$LOGS/weekly_$DATE.log"
else
    # Create Qwen format data on the fly
    python3 -c "
import json
from pathlib import Path

# Read training data
data = Path('training_data.jsonl').read_text().strip().split('\n')
mlx_dir = Path('mlx_data_qwen')
mlx_dir.mkdir(exist_ok=True)

converted = []
for line in data:
    if not line.strip():
        continue
    item = json.loads(line)
    text = item.get('text', '')
    if 'USER:' in text and 'ASSISTANT:' in text:
        parts = text.split('ASSISTANT:')
        user = parts[0].replace('USER:', '').strip()
        assistant = parts[1].strip() if len(parts) > 1 else ''
        converted.append({'text': f'<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n{assistant}<|im_end|>'})
    else:
        converted.append({'text': text})

with open(mlx_dir / 'train.jsonl', 'w') as f:
    for item in converted:
        f.write(json.dumps(item) + '\n')
with open(mlx_dir / 'valid.jsonl', 'w') as f:
    for item in converted[:3]:
        f.write(json.dumps(item) + '\n')
print(f'Prepared {len(converted)} samples for Qwen')
" >> "$LOGS/weekly_$DATE.log" 2>&1 || echo "Warning: Qwen conversion failed" >> "$LOGS/weekly_$DATE.log"
fi

# Step 3: Check training data exists
if [ ! -f "training_data.jsonl" ] || [ ! -s "training_data.jsonl" ]; then
    echo "ERROR: No training data found. Skipping training." >> "$LOGS/weekly_$DATE.log"
    exit 0
fi

TRAIN_SAMPLES=$(wc -l < training_data.jsonl)
echo "Training samples: $TRAIN_SAMPLES" >> "$LOGS/weekly_$DATE.log"

if [ "$TRAIN_SAMPLES" -lt 3 ]; then
    echo "WARNING: Too few training samples ($TRAIN_SAMPLES). Minimum 3 required." >> "$LOGS/weekly_$DATE.log"
    exit 0
fi

# Step 4: Run training with TinyLlama (HuggingFace repo ID)
echo "Starting training..." >> "$LOGS/weekly_$DATE.log"

# Step 4: Check for autoresearch best config
echo "Checking autoresearch results..." >> "$LOGS/weekly_$DATE.log"
BEST_CONFIG_FILE="$TRAINING/../training-data/autoresearch_best.tsv"
if [ -f "$BEST_CONFIG_FILE" ]; then
    # Read best config (skip header)
    BEST_LINE=$(tail -1 "$BEST_CONFIG_FILE")
    BEST_NAME=$(echo "$BEST_LINE" | cut -f2)
    BEST_LR=$(echo "$BEST_LINE" | cut -f3)
    BEST_ITERS=$(echo "$BEST_LINE" | cut -f4)
    BEST_BATCH=$(echo "$BEST_LINE" | cut -f5)
    BEST_LOSS=$(echo "$BEST_LINE" | cut -f6)
    
    echo "Autoresearch best config found:" >> "$LOGS/weekly_$DATE.log"
    echo "  Name: $BEST_NAME" >> "$LOGS/weekly_$DATE.log"
    echo "  LR: $BEST_LR" >> "$LOGS/weekly_$DATE.log"
    echo "  Iters: $BEST_ITERS" >> "$LOGS/weekly_$DATE.log"
    echo "  Batch: $BEST_BATCH" >> "$LOGS/weekly_$DATE.log"
    echo "  Val loss: $BEST_LOSS" >> "$LOGS/weekly_$DATE.log"
    
    # Use autoresearch params if they exist
    LEARNING_RATE="$BEST_LR"
    ITERS="$BEST_ITERS"
    BATCH_SIZE="$BEST_BATCH"
else
    echo "No autoresearch config found, using defaults" >> "$LOGS/weekly_$DATE.log"
    # Default config for Qwen 7B 4-bit
    LEARNING_RATE="1e-5"
    ITERS="500"
    BATCH_SIZE="1"
fi

# Use 4-bit quantized Qwen 7B (fits in memory with better quality)
# Alternative: TinyLlama/TinyLlama-1.1B-Chat-v1.0 (smaller, faster)
MODEL_ID="mlx-community/Qwen2.5-7B-Instruct-4bit"
ADAPTER_PATH="adapters/model_7b_$VERSION"

echo "  Model: $MODEL_ID" >> "$LOGS/weekly_$DATE.log"
echo "  Adapter: $ADAPTER_PATH" >> "$LOGS/weekly_$DATE.log"
echo "  Learning rate: $LEARNING_RATE" >> "$LOGS/weekly_$DATE.log"
echo "  Iterations: $ITERS" >> "$LOGS/weekly_$DATE.log"
echo "  Batch size: $BATCH_SIZE" >> "$LOGS/weekly_$DATE.log"

# Data directory for Qwen format
MLX_DATA="mlx_data_qwen"

python3 -m mlx_lm.lora \
    --model "$MODEL_ID" \
    --data "$MLX_DATA" \
    --train \
    --batch-size $BATCH_SIZE \
    --learning-rate $LEARNING_RATE \
    --iters $ITERS \
    --adapter-path "$ADAPTER_PATH" \
    --grad-checkpoint \
    --seed 42 \
    >> "$LOGS/weekly_$DATE.log" 2>&1

# Step 5: Validate the model
echo "Validating trained model..." >> "$LOGS/weekly_$DATE.log"

if [ -d "$ADAPTER_PATH" ]; then
    echo "✅ Model trained successfully: $ADAPTER_PATH" >> "$LOGS/weekly_$DATE.log"
    
    # Record training success
    echo "{
  \"date\": \"$DATE\",
  \"adapter_path\": \"$ADAPTER_PATH\",
  \"version\": \"$VERSION\",
  \"model\": \"$MODEL_ID\",
  \"iterations\": 500,
  \"samples\": $TRAIN_SAMPLES
}" > checkpoints/training_results_$VERSION.json
    
    # Update latest pointer
    ln -sf "$ADAPTER_PATH" adapters/latest 2>/dev/null || true
else
    echo "❌ Training failed - adapter not created" >> "$LOGS/weekly_$DATE.log"
fi

# Step 6: Cleanup old adapters (keep last 5)
echo "Cleaning up old adapters..." >> "$LOGS/weekly_$DATE.log"
cd adapters
ls -t | grep -v "latest" | tail -n +6 | xargs rm -rf 2>/dev/null || true
cd ..

echo "" >> "$LOGS/weekly_$DATE.log"
echo "=== Weekly Retraining Complete: $(date) ===" >> "$LOGS/weekly_$DATE.log"
echo "Model saved to: $ADAPTER_PATH" >> "$LOGS/weekly_$DATE.log"