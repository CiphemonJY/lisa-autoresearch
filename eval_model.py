import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load tokenizer
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B', trust_remote_code=True)

# Load base model
print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen2.5-0.5B', 
    trust_remote_code=True, 
    torch_dtype=torch.float16,
    device_map="cpu"
)

# Load checkpoint state dict
print("Loading round 10 checkpoint...")
checkpoint_path = '/home/jetson/lisa_proj/checkpoints/round_10_v1/model.pt'
state_dict = torch.load(checkpoint_path, map_location='cpu')

# Load into model
base_model.load_state_dict(state_dict, strict=False)
print("Model loaded successfully!\n")

# Test prompts
prompts = [
    'Hello, how are you?',
    'The capital of France is',
    'Once upon a time',
    'Write a haiku:'
]

for prompt in prompts:
    print(f'=== Prompt: {prompt!r} ===')
    inputs = tokenizer(prompt, return_tensors='pt')
    outputs = base_model.generate(**inputs, max_new_tokens=50, do_sample=True, temperature=0.7)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    print()
