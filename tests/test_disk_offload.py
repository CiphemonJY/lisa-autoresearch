#!/usr/bin/env python3
"""Tests for disk-offload training."""

import sys
from pathlib import Path
import pytest

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from lisa.offload import DiskOffloadedTrainer


class TestDiskOffloadedTrainer:
    """Tests for DiskOffloadedTrainer class."""

    def test_initialization(self):
        """Test trainer initialization."""
        trainer = DiskOffloadedTrainer(
            model_id="test-model",
            layer_groups=6,
            max_memory_gb=5.0,
        )
        
        assert trainer.model_id == "test-model"
        assert trainer.layer_groups == 6
        assert trainer.max_memory_gb == 5.0

    def test_memory_estimation_7b(self):
        """Test memory estimation for 7B model."""
        trainer = DiskOffloadedTrainer(
            model_id="Qwen2.5-7B-Instruct-4bit",
            layer_groups=6,
        )
        
        size = trainer.estimate_model_size()
        
        assert size["params_billion"] == 7
        assert size["model_size_gb"] == 3.5  # 7B * 0.5 (4-bit)
        assert size["peak_memory_gb"] < 5.0

    def test_memory_estimation_32b(self):
        """Test memory estimation for 32B model."""
        trainer = DiskOffloadedTrainer(
            model_id="Qwen2.5-32B-Instruct-4bit",
            layer_groups=6,
        )
        
        size = trainer.estimate_model_size()
        
        assert size["params_billion"] == 32
        assert size["model_size_gb"] == 16.0  # 32B * 0.5 (4-bit)
        assert size["peak_memory_gb"] < 10.0

    def test_memory_check(self):
        """Test memory check functionality."""
        # Should pass with high enough limit
        trainer = DiskOffloadedTrainer(
            model_id="Qwen2.5-7B-Instruct-4bit",
            max_memory_gb=10.0,
        )
        
        assert trainer.check_memory() is True

    def test_memory_check_fails(self):
        """Test memory check fails with low limit."""
        trainer = DiskOffloadedTrainer(
            model_id="Qwen2.5-32B-Instruct-4bit",
            layer_groups=6,
            max_memory_gb=1.0,  # Too low
        )
        
        assert trainer.check_memory() is False

    def test_layer_group_estimation(self):
        """Test layer group size estimation."""
        trainer = DiskOffloadedTrainer(
            model_id="Qwen2.5-32B-Instruct-4bit",
            layer_groups=6,
        )
        
        size = trainer.estimate_model_size()
        
        # Each group should be ~1/6 of model
        assert size["group_size_gb"] < size["model_size_gb"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
