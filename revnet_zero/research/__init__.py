"""
Advanced research capabilities for RevNet-Zero.

This module provides comprehensive tools for:
- Comparative analysis of reversible vs standard transformers
- Scaling laws derivation and analysis  
- Novel algorithm experimentation
- Statistical validation and reproducibility
"""

from .comparative_studies import *
from .scaling_analysis import *
from .experimental_architectures import *
from .reproducibility import *

__all__ = [
    'ComparativeStudySuite',
    'ScalingLawsAnalyzer', 
    'HierarchicalReversibleAttention',
    'NovelCouplingExplorer',
    'ExperimentManager',
    'ReproducibilityFramework'
]