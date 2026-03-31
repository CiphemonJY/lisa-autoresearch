#!/bin/bash
# LISA Training Launcher
# Usage: ./train.sh [32b|70b|120b]

MODEL=${1:-70b}

echo "============================================"
echo "LISA Training - $MODEL Model"
echo "============================================"

case $MODEL in
    32b)
        python3 -m lisa_pkg.src.lisa_32b_training ;;
    70b)
        python3 -m lisa_pkg.src.lisa_70b_v2 ;;
    120b)
        python3 -m lisa_pkg.src.lisa_120b_training ;;
    *)
        echo "Usage: $0 [32b|70b|120b]"
        exit 1 ;;
esac
