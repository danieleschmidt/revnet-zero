"""
Quantum Computing Integration for RevNet-Zero

This module implements quantum-inspired algorithms and quantum error correction
techniques for neural networks, representing breakthrough research in the
intersection of quantum computing and machine learning.
"""

from .quantum_error_correction import (
    QuantumErrorCorrectedLayer,
    QuantumStabilizer,
    QuantumRedundancyEncoder,
    QECConfig,
    QECCode,
    apply_quantum_error_correction,
    create_qec_enhanced_model
)

__all__ = [
    'QuantumErrorCorrectedLayer',
    'QuantumStabilizer',
    'QuantumRedundancyEncoder', 
    'QECConfig',
    'QECCode',
    'apply_quantum_error_correction',
    'create_qec_enhanced_model'
]