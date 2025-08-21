"""
Information-Theoretic Optimization for RevNet-Zero

This module implements theoretical foundations for information-preserving
reversible neural networks using principles from information theory.

Key Features:
- Mutual information preservation in coupling functions
- Information bottleneck optimization
- Variational information bounds
- Fisher information geometry
- Minimum description length principles
"""

from .information_preserving_coupling import (
    InformationPreservingCoupling,
    InformationTheoreticOptimizer,
    MutualInformationEstimator,
    InformationBottleneckLayer,
    VariationalInformationMaximization,
    InformationTheoreticConfig
)

__all__ = [
    'InformationPreservingCoupling',
    'InformationTheoreticOptimizer', 
    'MutualInformationEstimator',
    'InformationBottleneckLayer',
    'VariationalInformationMaximization',
    'InformationTheoreticConfig'
]