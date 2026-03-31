#!/usr/bin/env python3
"""
Example: Run LISA Inference

This example shows how to run inference using LISA's
layer-by-layer approach for memory efficiency.
"""
import sys
sys.path.insert(0, '..')

from lisa_pkg.src.lisa_inference_prod import LISAInference

def main():
    print("=" * 60)
    print("LISA Inference Example")
    print("=" * 60)
    
    # Initialize inference engine
    # Assumes GGUF model files and LoRA adapter exist
    gguf_dir = "/tmp/qwen32b_q4_parts"
    lora_path = "/tmp/lisa_32b_final.npz"
    
    print("\n🚀 Initializing LISA Inference...")
    engine = LISAInference(gguf_dir, lora_path)
    
    # Generate text
    prompt = "Artificial intelligence is"
    next_token = engine.generate(prompt, max_tokens=10)
    
    print(f"\n✅ Generated token ID: {next_token}")
    print("\nNote: For real text output, integrate with a tokenizer.")

if __name__ == "__main__":
    main()
