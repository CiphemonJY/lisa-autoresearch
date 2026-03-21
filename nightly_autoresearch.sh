#!/bin/bash
# Nightly Auto-Research Script
# Runs autonomous training experiments on Qwen 7B 4-bit to discover optimal configs
# Scheduled: 2 AM daily
# Integrates with weekly retrain via results.tsv

set -e

WORKSPACE="$HOME/~/.lisa"
AUTORESEARCH="$WORKSPACE/autoresearch-mlx"
LOGS="$WORKSPACE/logs/autoresearch"
DATE=$(date +%Y-%m-%d)

# Create log directory
mkdir -p "$LOGS"

# Log start
echo "=== Auto-Research Started: $(date) ===" >> "$LOGS/$DATE.log"

cd "$AUTORESEARCH" 2>/dev/null || {
    echo "ERROR: autoresearch-mlx directory not found" >> "$LOGS/$DATE.log"
    echo "Skipping auto-research" >> "$LOGS/$DATE.log"
    exit 0
}

# Create log for today
touch "$LOGS/$DATE.log"

# Qwen 7B 4-bit configuration
MODEL_ID="mlx-community/Qwen2.5-7B-Instruct-4bit"
DATA_DIR="$WORKSPACE/training-data/mlx_data_qwen"

# Check if training data exists
if [ ! -d "$DATA_DIR" ]; then
    # Create Qwen format data if needed
    echo "Preparing Qwen training data..." >> "$LOGS/$DATE.log"
    python3 -c "
import json
from pathlib import Path

data = Path('$WORKSPACE/training-data/training_data.jsonl').read_text().strip().split('\n')
mlx_dir = Path('$DATA_DIR')
mlx_dir.mkdir(exist_ok=True, parents=True)

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
print(f'Prepared {len(converted)} samples')
" >> "$LOGS/$DATE.log" 2>&1
fi

# Experiment configurations to test
# Format: "name|learning_rate|iters|batch_size"
EXPERIMENTS=(
    "baseline|1e-5|100|1"
    "high_lr|5e-5|100|1"
    "low_lr|5e-6|100|1"
    "more_iters|1e-5|200|1"
    "batch_2|1e-5|100|2"
)

# Run experiments
EXPERIMENT_COUNT=0
BEST_VAL_LOSS=999
BEST_CONFIG=""

echo "" >> "$LOGS/$DATE.log"
echo "Running Qwen 7B 4-bit experiments..." >> "$LOGS/$DATE.log"
echo "Model: $MODEL_ID" >> "$LOGS/$DATE.log"
echo "Data: $DATA_DIR" >> "$LOGS/$DATE.log"
echo "" >> "$LOGS/$DATE.log"

for exp in "${EXPERIMENTS[@]}"; do
    IFS='|' read -r NAME LR ITERS BATCH <<< "$exp"
    
    echo "=== Experiment: $NAME ===" >> "$LOGS/$DATE.log"
    echo "Config: lr=$LR, iters=$ITERS, batch=$BATCH" >> "$LOGS/$DATE.log"
    
    ADAPTER_PATH="$WORKSPACE/training-data/adapters/autoresearch_$DATE"
    
    # Run training
    python3 -m mlx_lm.lora \
        --model "$MODEL_ID" \
        --data "$DATA_DIR" \
        --train \
        --batch-size "$BATCH" \
        --learning-rate "$LR" \
        --iters "$ITERS" \
        --adapter-path "$ADAPTER_PATH" \
        --grad-checkpoint \
        --seed 42 \
        >> "$LOGS/$DATE.log" 2>&1
    
    # Extract validation loss (last Val loss line)
    VAL_LOSS=$(grep "Val loss" "$LOGS/$DATE.log" | tail -1 | sed 's/.*Val loss \([0-9.]*\).*/\1/' || echo "999")
    
    # Record result
    echo "Result: val_loss=$VAL_LOSS" >> "$LOGS/$DATE.log"
    
    # Track best
    if (( $(echo "$VAL_LOSS < $BEST_VAL_LOSS" | bc -l) )); then
        BEST_VAL_LOSS=$VAL_LOSS
        BEST_CONFIG="$NAME|$LR|$ITERS|$BATCH"
        echo "New best config!" >> "$LOGS/$DATE.log"
    fi
    
    EXPERIMENT_COUNT=$((EXPERIMENT_COUNT + 1))
    
    # Clean up adapter (we just want results)
    rm -rf "$ADAPTER_PATH" 2>/dev/null || true
    
    # Small break between experiments
    sleep 5
done

# Save best config to results file for weekly retrain
RESULTS_FILE="$WORKSPACE/training-data/autoresearch_best.tsv"
echo -e "date\tconfig\tlearning_rate\titers\tbatch_size\tval_loss" > "$RESULTS_FILE"
echo -e "$DATE\t$BEST_CONFIG\t$BEST_VAL_LOSS" >> "$RESULTS_FILE"

# Also save to research memory
mkdir -p "$WORKSPACE/memory/research" 2>/dev/null
echo "$(date): Best config = $BEST_CONFIG, val_loss = $BEST_VAL_LOSS" >> "$WORKSPACE/memory/research/autoresearch_history.txt"

# Summary
echo "" >> "$LOGS/$DATE.log"
echo "=== Auto-Research Complete ===" >> "$LOGS/$DATE.log"
echo "Experiments run: $EXPERIMENT_COUNT" >> "$LOGS/$DATE.log"
echo "Best config: $BEST_CONFIG" >> "$LOGS/$DATE.log"
echo "Best val_loss: $BEST_VAL_LOSS" >> "$LOGS/$DATE.log"
echo "Saved to: $RESULTS_FILE" >> "$LOGS/$DATE.log"
echo "Finished at: $(date)" >> "$LOGS/$DATE.log"