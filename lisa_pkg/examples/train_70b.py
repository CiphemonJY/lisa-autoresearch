#!/usr/bin/env python3
"""
Example: Train 70B Model with LISA

This example shows how to use LISA to train a 70B model
on hardware with only 8GB RAM.
"""
import sys
sys.path.insert(0, '..')

from lisa_pkg.src.lisa_70b_v2 import LISATrainer, CONFIG, TRAINING_DATA

def main():
    print("=" * 60)
    print("LISA 70B Training Example")
    print("=" * 60)
    
    # Initialize trainer
    trainer = LISATrainer(CONFIG)
    
    # Train for a few steps
    print("\n🔄 Training...")
    for i, text in enumerate(TRAINING_DATA[:10]):
        result = trainer.train_step(text)
        if (i + 1) % 5 == 0:
            print(f"  Step {i+1}: loss={result['loss']:.4f}, mem={result['mem_gb']:.2f}GB")
    
    # Save checkpoint
    trainer.lora.save("/tmp/lisa_70b_example.npz")
    print("\n✅ Trained adapter saved to /tmp/lisa_70b_example.npz")

if __name__ == "__main__":
    main()
