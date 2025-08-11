"""
High-performance kernels for RevNet-Zero reversible operations.

This module provides optimized CUDA and Triton kernels for:
- Fused reversible attention operations
- Memory-efficient coupling function computation
- Optimized gradient checkpointing
"""

from .cuda_kernels import *
from .triton_kernels import *
from .kernel_manager import KernelManager

__all__ = [
    'fused_reversible_attention',
    'fused_coupling_forward',
    'fused_coupling_backward',
    'optimized_memory_checkpoint',
    'KernelManager'
]