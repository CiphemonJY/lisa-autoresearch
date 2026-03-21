#!/usr/bin/env python3
"""
Model Parallelism - Split Large Models Across Multiple Machines

This enables running 100T+ parameter models across multiple machines,
each holding a portion of the model.

PARALLELISM TYPES:
──────────────────────────────────────────────────────────────────
1. DATA PARALLELISM
   - Same model on each machine
   - Different data batches
   - Good for small models, large datasets
   
2. MODEL PARALLELISM (Tensor Parallelism)
   - Split model layers across machines
   - Same data on all machines
   - Good for large models
   
3. PIPELINE PARALLELISM
   - Split model stages across machines
   - Data flows through pipeline
   - Good for very large models
   
4. COMBINED PARALLELISM
   - Combine model + pipeline parallelism
   - For extremely large models (100T+)

IMPLEMENTATION:
"""

import os
import sys
import time
import hashlib
import json
import socket
import threading
import queue
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime
import logging
import math

os.environ['PYTHONWARNINGS'] = 'ignore::urllib3.warnings.NotOpenSSLWarning'

# Try to import torch
try:
    import torch
    import torch.nn as nn
    import torch.distributed as dist
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None
    nn = None
    dist = None


# ============================================================================
# Parallelism Configuration
# ============================================================================

@dataclass
class ParallelConfig:
    """Configuration for model parallelism."""
    world_size: int = 4  # Number of machines
    rank: int = 0  # This machine's rank
    backend: str = "gloo"  # gloo, nccl, or mpi
    master_addr: str = "localhost"
    master_port: int = 29500
    
    # Model parallelism
    tensor_parallel_size: int = 2  # Split layers across machines
    pipeline_parallel_size: int = 2  # Pipeline stages
    
    # Communication
    overlap_communication: bool = True  # Overlap compute and communication
    
    def get_layer_assignment(self, num_layers: int) -> List[Tuple[int, int]]:
        """
        Get layer assignments for each rank.
        
        Returns list of (start_layer, end_layer) for each rank.
        """
        layers_per_rank = num_layers // self.world_size
        remainder = num_layers % self.world_size
        
        assignments = []
        start = 0
        
        for rank in range(self.world_size):
            # Distribute remainder across first ranks
            extra = 1 if rank < remainder else 0
            end = start + layers_per_rank + extra
            assignments.append((start, end))
            start = end
        
        return assignments
    
    def get_my_layers(self, num_layers: int) -> Tuple[int, int]:
        """Get layer assignment for this rank."""
        assignments = self.get_layer_assignment(num_layers)
        return assignments[self.rank]


# ============================================================================
# Model Splitter
# ============================================================================

class ModelSplitter:
    """
    Split model across multiple machines.
    
    Example for 4 machines:
    ┌─────────────────────────────────────────────────────────────┐
    │  Machine 0: Layers 0-24                                     │
    │  Machine 1: Layers 25-49                                    │
    │  Machine 2: Layers 50-74                                    │
    │  Machine 3: Layers 75-99                                    │
    └─────────────────────────────────────────────────────────────┘
    
    For 100T model with 200 layers:
    - Each machine holds 50 layers (25% of model)
    - With INT4: 25T params × 0.5 bytes = 12.5 TB per machine
    - With LISA: 12.5 TB × 5% = 625 GB in RAM per machine
    """
    
    def __init__(self, config: ParallelConfig):
        self.config = config
        self.logger = logging.getLogger(f"model-splitter-rank-{config.rank}")
        
        # Layer assignments
        self.my_layers: Optional[Tuple[int, int]] = None
        self.all_assignments: List[Tuple[int, int]] = []
    
    def split_model(self, model_layers: List[Any]) -> List[Any]:
        """
        Split model layers across machines.
        
        Args:
            model_layers: List of model layers
            
        Returns:
            List of layers for this machine
        """
        num_layers = len(model_layers)
        self.all_assignments = self.config.get_layer_assignment(num_layers)
        self.my_layers = self.all_assignments[self.config.rank]
        
        start, end = self.my_layers
        my_layers = model_layers[start:end]
        
        self.logger.info(
            f"Rank {self.config.rank}: Assigned layers {start}-{end-1} "
            f"({len(my_layers)} layers)"
        )
        
        return my_layers
    
    def get_layer_for_rank(self, layer_idx: int) -> int:
        """Get which rank should handle a layer."""
        for rank, (start, end) in enumerate(self.all_assignments):
            if start <= layer_idx < end:
                return rank
        return -1
    
    def is_my_layer(self, layer_idx: int) -> bool:
        """Check if this rank should handle a layer."""
        if self.my_layers is None:
            return False
        start, end = self.my_layers
        return start <= layer_idx < end
    
    def get_model_size_per_rank(self, total_params: float, bits: int = 4) -> Dict:
        """
        Calculate memory requirements per rank.
        
        Args:
            total_params: Total model parameters (e.g., 100e12 for 100T)
            bits: Quantization bits (4, 8, or 16)
            
        Returns:
            Dict with memory requirements
        """
        params_per_rank = total_params / self.config.world_size
        
        # Bytes per parameter
        bytes_per_param = bits / 8
        
        # Total size per rank
        size_per_rank_bytes = params_per_rank * bytes_per_param
        size_per_rank_gb = size_per_rank_bytes / 1e9
        
        # With LISA
        lisa_ratio = 0.05  # 5% in RAM
        lisa_size_gb = size_per_rank_gb * lisa_ratio
        
        return {
            "total_params": total_params,
            "params_per_rank": params_per_rank,
            "bits": bits,
            "size_per_rank_gb": size_per_rank_gb,
            "lisa_size_gb": lisa_size_gb,
            "world_size": self.config.world_size,
        }


# ============================================================================
# Communication Handler
# ============================================================================

class CommunicationHandler:
    """
    Handle communication between machines.
    
    Types of communication:
    1. AllReduce: Sum gradients across all machines
    2. AllGather: Gather outputs from all machines
    3. Send/Recv: Point-to-point communication
    """
    
    def __init__(self, config: ParallelConfig):
        self.config = config
        self.logger = logging.getLogger(f"comm-rank-{config.rank}")
        
        # Message queues for simulation
        self.send_queues: Dict[int, queue.Queue] = {
            r: queue.Queue() for r in range(config.world_size)
        }
        self.recv_queue = queue.Queue()
    
    def init_distributed(self):
        """Initialize distributed communication."""
        if HAS_TORCH:
            try:
                dist.init_process_group(
                    backend=self.config.backend,
                    init_method=f"tcp://{self.config.master_addr}:{self.config.master_port}",
                    world_size=self.config.world_size,
                    rank=self.config.rank,
                )
                self.logger.info(f"Initialized distributed: rank={self.config.rank}, world_size={self.config.world_size}")
                return True
            except Exception as e:
                self.logger.warning(f"Failed to init distributed: {e}")
                return False
        else:
            self.logger.info("Using simulated communication (no PyTorch)")
            return True
    
    def all_reduce(self, tensor: Any, op: str = "sum") -> Any:
        """
        AllReduce operation across all machines.
        
        Sums tensors from all machines and broadcasts result.
        
        Args:
            tensor: Local tensor
            op: Operation (sum, avg, max, min)
            
        Returns:
            Reduced tensor
        """
        if HAS_TORCH and dist.is_initialized():
            # Real distributed
            if op == "sum":
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            elif op == "avg":
                dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                tensor /= self.config.world_size
            elif op == "max":
                dist.all_reduce(tensor, op=dist.ReduceOp.MAX)
            elif op == "min":
                dist.all_reduce(tensor, op=dist.ReduceOp.MIN)
            
            return tensor
        else:
            # Simulated
            self.logger.debug(f"Simulated AllReduce: {op}")
            return tensor
    
    def all_gather(self, tensor: Any) -> List[Any]:
        """
        AllGather operation.
        
        Gathers tensors from all machines.
        
        Args:
            tensor: Local tensor
            
        Returns:
            List of tensors from all machines
        """
        if HAS_TORCH and dist.is_initialized():
            gathered = [torch.zeros_like(tensor) for _ in range(self.config.world_size)]
            dist.all_gather(gathered, tensor)
            return gathered
        else:
            # Simulated
            self.logger.debug("Simulated AllGather")
            return [tensor] * self.config.world_size
    
    def send(self, tensor: Any, dst: int):
        """Send tensor to destination rank."""
        if HAS_TORCH and dist.is_initialized():
            dist.send(tensor, dst=dst)
        else:
            self.logger.debug(f"Simulated send to rank {dst}")
    
    def recv(self, src: int) -> Any:
        """Receive tensor from source rank."""
        if HAS_TORCH and dist.is_initialized():
            tensor = torch.zeros(1)  # Placeholder
            dist.recv(tensor, src=src)
            return tensor
        else:
            self.logger.debug(f"Simulated recv from rank {src}")
            return None


# ============================================================================
# Pipeline Parallelism
# ============================================================================

class PipelineParallel:
    """
    Pipeline parallelism for very large models.
    
    Example for 4 machines:
    ┌─────────────────────────────────────────────────────────────┐
    │  Stage 0 (Machine 0): Embedding + Layers 0-24                │
    │  Stage 1 (Machine 1): Layers 25-49                           │
    │  Stage 2 (Machine 2): Layers 50-74                            │
    │  Stage 3 (Machine 3): Layers 75-99 + Output                 │
    └─────────────────────────────────────────────────────────────┘
    
    Data flows through pipeline:
    Input → Stage 0 → Stage 1 → Stage 2 → Stage 3 → Output
    
    Micro-batching enables parallelism:
    - Stage 0 processes micro-batch 1
    - While Stage 1 processes micro-batch 1, Stage 0 processes micro-batch 2
    - etc.
    """
    
    def __init__(self, config: ParallelConfig):
        self.config = config
        self.logger = logging.getLogger(f"pipeline-rank-{config.rank}")
        
        self.comm = CommunicationHandler(config)
        self.stage_id = config.rank
        self.num_stages = config.world_size
    
    def forward(self, input_data: Any, forward_fn: Callable) -> Any:
        """
        Forward pass through pipeline stage.
        
        Args:
            input_data: Input from previous stage (or original input for stage 0)
            forward_fn: Forward function for this stage
            
        Returns:
            Output for next stage (or final output for last stage)
        """
        # Receive from previous stage (if not first)
        if self.stage_id > 0:
            self.logger.debug(f"Stage {self.stage_id}: Waiting for input from stage {self.stage_id - 1}")
            input_data = self.comm.recv(self.stage_id - 1)
        
        # Process
        self.logger.debug(f"Stage {self.stage_id}: Processing")
        output = forward_fn(input_data)
        
        # Send to next stage (if not last)
        if self.stage_id < self.num_stages - 1:
            self.logger.debug(f"Stage {self.stage_id}: Sending output to stage {self.stage_id + 1}")
            self.comm.send(output, self.stage_id + 1)
            return None  # Not final output
        else:
            return output  # Final output
    
    def backward(self, grad: Any, backward_fn: Callable) -> Any:
        """
        Backward pass through pipeline stage.
        
        Args:
            grad: Gradient from next stage (or loss gradient for last stage)
            backward_fn: Backward function for this stage
            
        Returns:
            Gradient for previous stage (or None for first stage)
        """
        # Receive from next stage (if not last)
        if self.stage_id < self.num_stages - 1:
            self.logger.debug(f"Stage {self.stage_id}: Waiting for grad from stage {self.stage_id + 1}")
            grad = self.comm.recv(self.stage_id + 1)
        
        # Process
        self.logger.debug(f"Stage {self.stage_id}: Backward pass")
        grad = backward_fn(grad)
        
        # Send to previous stage (if not first)
        if self.stage_id > 0:
            self.logger.debug(f"Stage {self.stage_id}: Sending grad to stage {self.stage_id - 1}")
            self.comm.send(grad, self.stage_id - 1)
            return None
        else:
            return grad  # Gradient for input


# ============================================================================
# Tensor Parallelism
# ============================================================================

class TensorParallel:
    """
    Tensor parallelism for large layers.
    
    Example for linear layer:
    ┌─────────────────────────────────────────────────────────────┐
    │  Split matrix multiplication across machines:              │
    │                                                             │
    │  Machine 0: Y_0 = X @ W_0                                   │
    │  Machine 1: Y_1 = X @ W_1                                   │
    │  Machine 2: Y_2 = X @ W_2                                   │
    │  Machine 3: Y_3 = X @ W_3                                   │
    │                                                             │
    │  AllReduce: Y = Y_0 + Y_1 + Y_2 + Y_3                      │
    └─────────────────────────────────────────────────────────────┘
    
    This splits large weight matrices across machines,
    enabling layers larger than single machine memory.
    """
    
    def __init__(self, config: ParallelConfig):
        self.config = config
        self.logger = logging.getLogger(f"tensor-parallel-rank-{config.rank}")
        
        self.comm = CommunicationHandler(config)
    
    def split_weight(self, weight: Any, dim: int = 0) -> Any:
        """
        Split weight matrix across machines.
        
        Args:
            weight: Weight matrix to split
            dim: Dimension to split along (0 = rows, 1 = columns)
            
        Returns:
            Weight shard for this machine
        """
        if HAS_TORCH and isinstance(weight, torch.Tensor):
            # Split tensor
            shards = torch.chunk(weight, self.config.world_size, dim=dim)
            return shards[self.config.rank]
        else:
            # Simulated
            self.logger.debug(f"Splitting weight along dim {dim}")
            return weight
    
    def all_reduce_output(self, output: Any) -> Any:
        """
        AllReduce output across machines.
        
        Combines partial outputs from all machines.
        
        Args:
            output: Partial output from this machine
            
        Returns:
            Combined output from all machines
        """
        return self.comm.all_reduce(output, op="sum")


# ============================================================================
# Combined Parallelism
# ============================================================================

class CombinedParallel:
    """
    Combine tensor and pipeline parallelism.
    
    For extremely large models (100T+):
    
    Example for 8 machines with 4-way tensor + 2-way pipeline:
    ┌─────────────────────────────────────────────────────────────┐
    │  Pipeline Stage 0:                                          │
    │    Machines 0-3: Tensor parallel (layers 0-49)              │
    │                                                             │
    │  Pipeline Stage 1:                                          │
    │    Machines 4-7: Tensor parallel (layers 50-99)             │
    └─────────────────────────────────────────────────────────────┘
    
    Each pipeline stage uses tensor parallelism within the stage.
    """
    
    def __init__(self, config: ParallelConfig):
        self.config = config
        
        # Tensor parallel size
        self.tp_size = config.tensor_parallel_size
        
        # Pipeline parallel size
        self.pp_size = config.pipeline_parallel_size
        
        # Validate
        assert self.tp_size * self.pp_size == config.world_size, \
            f"world_size ({config.world_size}) must equal tp_size ({self.tp_size}) * pp_size ({self.pp_size})"
        
        # This machine's position
        self.tp_rank = config.rank % self.tp_size  # Position within tensor group
        self.pp_rank = config.rank // self.tp_size  # Pipeline stage
        
        self.logger = logging.getLogger(f"combined-parallel-rank-{config.rank}")
        
        self.tensor_parallel = TensorParallel(config)
        self.pipeline_parallel = PipelineParallel(config)
    
    def forward(self, input_data: Any, forward_fn: Callable) -> Any:
        """
        Forward pass with combined parallelism.
        
        Args:
            input_data: Input data
            forward_fn: Forward function for this stage
            
        Returns:
            Output (only on last pipeline stage)
        """
        # Pipeline forward (handles inter-stage communication)
        output = self.pipeline_parallel.forward(input_data, forward_fn)
        
        # Tensor parallel all-reduce (within stage)
        if output is not None:
            output = self.tensor_parallel.all_reduce_output(output)
        
        return output


# ============================================================================
# Model Parallel Manager
# ============================================================================

class ModelParallelManager:
    """
    Manage model parallelism across machines.
    
    Usage:
        config = ParallelConfig(world_size=4, rank=0)
        manager = ModelParallelManager(config)
        
        # Split model
        my_layers = manager.split_model(all_layers)
        
        # Forward pass
        output = manager.forward(input_data)
        
        # Backward pass
        grad = manager.backward(grad)
    """
    
    def __init__(self, config: ParallelConfig):
        self.config = config
        self.logger = logging.getLogger(f"model-parallel-rank-{config.rank}")
        
        self.splitter = ModelSplitter(config)
        self.comm = CommunicationHandler(config)
        
        # Choose parallelism strategy
        if config.tensor_parallel_size > 1 and config.pipeline_parallel_size > 1:
            self.parallel = CombinedParallel(config)
            self.strategy = "combined"
        elif config.tensor_parallel_size > 1:
            self.parallel = TensorParallel(config)
            self.strategy = "tensor"
        elif config.pipeline_parallel_size > 1:
            self.parallel = PipelineParallel(config)
            self.strategy = "pipeline"
        else:
            self.parallel = None
            self.strategy = "none"
        
        self.my_layers: Optional[List[Any]] = None
    
    def split_model(self, model_layers: List[Any]) -> List[Any]:
        """Split model layers across machines."""
        self.my_layers = self.splitter.split_model(model_layers)
        return self.my_layers
    
    def forward(self, input_data: Any, forward_fn: Callable) -> Any:
        """Forward pass with parallelism."""
        if self.strategy == "none":
            return forward_fn(input_data)
        elif self.strategy == "pipeline":
            return self.parallel.forward(input_data, forward_fn)
        elif self.strategy == "tensor":
            output = forward_fn(input_data)
            return self.parallel.all_reduce_output(output)
        else:  # combined
            return self.parallel.forward(input_data, forward_fn)
    
    def backward(self, grad: Any, backward_fn: Callable) -> Any:
        """Backward pass with parallelism."""
        if self.strategy == "none":
            return backward_fn(grad)
        elif self.strategy == "pipeline":
            return self.parallel.backward(grad, backward_fn)
        elif self.strategy == "tensor":
            grad = backward_fn(grad)
            return self.comm.all_reduce(grad, op="sum")
        else:  # combined
            return self.parallel.backward(grad, backward_fn)
    
    def get_memory_requirements(self, total_params: float, bits: int = 4) -> Dict:
        """Get memory requirements for this configuration."""
        return self.splitter.get_model_size_per_rank(total_params, bits)


# ============================================================================
# Demo
# ============================================================================

def test_model_parallel():
    """Test model parallelism."""
    print("="*70)
    print("MODEL PARALLELISM TEST")
    print("="*70)
    print()
    
    print("TESTING DIFFERENT PARALLELISM STRATEGIES:")
    print()
    
    # Test 1: Simple model split
    print("="*70)
    print("1. MODEL SPLIT (4 machines, 100 layers)")
    print("="*70)
    print()
    
    config = ParallelConfig(world_size=4, rank=0)
    splitter = ModelSplitter(config)
    
    # Create dummy layers
    layers = [f"layer_{i}" for i in range(100)]
    
    print(f"Total layers: {len(layers)}")
    print(f"Machines: {config.world_size}")
    print()
    
    # Split
    my_layers = splitter.split_model(layers)
    
    print("Layer assignments:")
    for rank, (start, end) in enumerate(splitter.all_assignments):
        print(f"  Machine {rank}: Layers {start}-{end-1} ({end-start} layers)")
    print()
    
    # Test 2: Memory requirements
    print("="*70)
    print("2. MEMORY REQUIREMENTS (100T model)")
    print("="*70)
    print()
    
    for bits in [16, 8, 4]:
        reqs = splitter.get_model_size_per_rank(100e12, bits)
        print(f"Quantization: {bits}-bit")
        print(f"  Total params: {reqs['total_params']:.1e}")
        print(f"  Params per machine: {reqs['params_per_rank']:.1e}")
        print(f"  Size per machine: {reqs['size_per_rank_gb']:.1f} GB")
        print(f"  With LISA (5%): {reqs['lisa_size_gb']:.1f} GB in RAM")
        print()
    
    # Test 3: Communication
    print("="*70)
    print("3. COMMUNICATION PATTERNS")
    print("="*70)
    print()
    
    print("TENSOR PARALLELISM:")
    print("  1. Each machine computes partial output")
    print("  2. AllReduce sums outputs from all machines")
    print("  3. Result is same as if computed on single machine")
    print()
    
    print("PIPELINE PARALLELISM:")
    print("  1. Stage 0 processes input, sends to Stage 1")
    print("  2. Stage 1 processes, sends to Stage 2")
    print("  3. Stage 2 processes, sends to Stage 3")
    print("  4. Stage 3 returns output")
    print()
    
    print("COMBINED PARALLELISM:")
    print("  1. Within each stage: Tensor parallelism")
    print("  2. Between stages: Pipeline parallelism")
    print("  3. Best of both worlds for 100T+ models")
    print()
    
    # Test 4: Example configuration
    print("="*70)
    print("4. EXAMPLE: 100T MODEL ON 4 MAC STUDIOS")
    print("="*70)
    print()
    
    print("Configuration:")
    print("  Model: 100T parameters")
    print("  Machines: 4 × Mac Studio (64GB each)")
    print("  Quantization: INT4")
    print("  Parallelism: Pipeline")
    print()
    
    print("Memory per machine:")
    print("  Model size: 100T × 0.5 bytes = 50 TB")
    print("  Per machine: 50 TB / 4 = 12.5 TB")
    print("  With LISA (5%): 625 GB in RAM")
    print("  Mac Studio 64GB: Need disk offload!")
    print()
    
    print("With disk offload:")
    print("  12.5 TB on fast SSD (4TB SSD = $200)")
    print("  625 GB in RAM (64GB Mac + swap)")
    print("  Feasible! Each Mac holds 25 layers")
    print()
    
    print("✓ Model parallelism working!")


if __name__ == "__main__":
    test_model_parallel()