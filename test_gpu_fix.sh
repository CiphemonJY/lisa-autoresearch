#!/bin/bash
# Test GPU fix strategies on Jetson
# Run this after rebooting

echo "Testing GPU fixes..."

# Fix 1: Environment variable
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:512"
echo "Set PYTORCH_CUDA_ALLOC_CONF"

# Test 1: Basic CUDA
python3 -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'Device: {torch.cuda.get_device_name(0)}')
torch.cuda.empty_cache()
print('Cache cleared')
"

# Test 2: Load small model on GPU
python3 -c "
import torch
from transformers import AutoModelForCausalLM

torch.cuda.empty_cache()
print('Loading tiny model...')
try:
    model = AutoModelForCausalLM.from_pretrained(
        'microsoft/TinyLlama-1.1B-Chat-v1.0',
        torch_dtype=torch.float16,
        device_map='cuda',
    )
    print('SUCCESS: Tiny model loaded on GPU')
except Exception as e:
    print(f'FAILED: {e}')
"

# Test 3: Try Qwen 7B with explicit cuda
python3 -c "
import torch
from transformers import AutoModelForCausalLM

torch.cuda.empty_cache()
print('Loading Qwen 7B...')
try:
    model = AutoModelForCausalLM.from_pretrained(
        'Qwen/Qwen2.5-7B',
        torch_dtype=torch.bfloat16,
    )
    model = model.cuda()
    print('SUCCESS: Qwen 7B on GPU')
except Exception as e:
    print(f'FAILED: {e}')
"
