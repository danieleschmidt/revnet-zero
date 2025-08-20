"""
Experimental Validation Framework for RevNet-Zero

This module provides comprehensive experimental validation for breakthrough
algorithms with rigorous statistical analysis and publication-ready results.

Key Features:
- Multi-seed statistical validation
- Effect size analysis (Cohen's d)
- Multiple comparison correction
- Publication-ready metrics and visualizations
- Computational and memory efficiency profiling
"""

from .breakthrough_validation import (
    BreakthroughAlgorithmBenchmark,
    StatisticalValidator,
    ExperimentConfig,
    run_full_validation_suite
)

__all__ = [
    'BreakthroughAlgorithmBenchmark',
    'StatisticalValidator',
    'ExperimentConfig',
    'run_full_validation_suite'
]