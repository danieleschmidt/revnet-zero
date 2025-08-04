"""
Optimization module for RevNet-Zero.

This module provides advanced optimization techniques including
kernel fusion, memory optimization, and performance tuning.
"""

from .performance import (
    KernelFusion,
    MemoryOptimizer,
    InferenceOptimizer,
    PerformanceProfiler,
    OptimizationSuite,
    optimize_model_for_inference,
)

__all__ = [
    'KernelFusion',
    'MemoryOptimizer', 
    'InferenceOptimizer',
    'PerformanceProfiler',
    'OptimizationSuite',
    'optimize_model_for_inference',
]