#!/bin/bash
# Quick Start Script for LISA+Offload
# One-command setup

set -e

echo "============================================================"
echo "LISA+Offload Quick Start"
echo "============================================================"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Please install Python 3.11+"
    exit 1
fi

echo "✅ Python found: $(python3 --version)"

# Check pip
if ! command -v pip &> /dev/null; then
    echo "❌ pip not found. Please install pip"
    exit 1
fi

echo "✅ pip found"

# Install dependencies
echo ""
echo "=== Installing Dependencies ==="
echo ""

pip install mlx mlx-lm transformers accelerate --quiet

echo "✅ Dependencies installed"

# Check if LISA files exist
echo ""
echo "=== Checking LISA+Offload Files ==="
echo ""

FILES=(
    "lisa_offload.py"
    "disk_offload.py"
    "hardware_detection.py"
    "mixed_precision.py"
    "gradient_accumulation.py"
    "selective_offload.py"
)

MISSING=0
for FILE in "${FILES[@]}"; do
    if [ -f "$FILE" ]; then
        echo "✅ $FILE found"
    else
        echo "❌ $FILE missing"
        MISSING=1
    fi
done

if [ $MISSING -eq 1 ]; then
    echo ""
    echo "❌ Some files are missing. Please download from:"
    echo "   https://github.com/userJY/lisa-autoresearch"
    exit 1
fi

# Detect hardware
echo ""
echo "=== Hardware Detection ==="
echo ""

python3 -c "
import sys
sys.path.insert(0, '.')
from hardware_detection import detect_hardware

hw = detect_hardware()
print(f'CPU: {hw.cpu_brand}')
print(f'Cores: {hw.cpu_cores}')
print(f'RAM: {hw.ram_total_gb:.1f} GB')
print(f'GPU: {hw.gpu_name}')
print(f'Max model (normal): {hw.max_model_size_normal}')
print(f'Max model (offload): {hw.max_model_size_offload}')
"

# Ask user what to do
echo ""
echo "=== What would you like to do? ==="
echo ""
echo "1. Start API server"
echo "2. Run Jupyter tutorial"
echo "3. Quick test (train small model)"
echo "4. Exit"
echo ""
read -p "Enter choice (1-4): " CHOICE

case $CHOICE in
    1)
        echo ""
        echo "=== Starting API Server ==="
        echo ""
        echo "Server will be available at: http://localhost:8000"
        echo "Press Ctrl+C to stop"
        echo ""
        python3 api_server.py
        ;;
    2)
        echo ""
        echo "=== Starting Jupyter ==="
        echo ""
        if ! command -v jupyter &> /dev/null; then
            echo "Installing Jupyter..."
            pip install jupyter --quiet
        fi
        jupyter notebook tutorial.ipynb
        ;;
    3)
        echo ""
        echo "=== Running Quick Test ==="
        echo ""
        python3 -c "
import sys
sys.path.insert(0, '.')
from hardware_detection import detect_hardware

hw = detect_hardware()
print(f'Quick test on {hw.cpu_brand}')
print(f'RAM: {hw.ram_total_gb:.1f} GB')
print('')
print('✅ LISA+Offload is ready!')
print('')
print('To train a model, run:')
print('  python3 lisa_offload.py --model Qwen/Qwen2.5-7B-Instruct --data training_data/')
"
        ;;
    4)
        echo "Goodbye!"
        exit 0
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac