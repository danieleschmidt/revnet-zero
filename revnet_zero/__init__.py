"""
RevNet-Zero: Memory-efficient reversible transformers for long-context training.

This library implements reversible transformer layers with on-the-fly activation
recomputation, reducing GPU memory usage by >70% during training.
"""

__version__ = "1.0.0"
__author__ = "RevNet-Zero Team"
__email__ = "team@revnet-zero.org"

from .models.reversible_transformer import ReversibleTransformer
from .layers.reversible_attention import ReversibleAttention
from .layers.reversible_ffn import ReversibleFFN
from .layers.coupling_layers import AdditiveCoupling, AffineCoupling
from .memory.scheduler import MemoryScheduler, AdaptiveScheduler
from .utils.conversion import convert_to_reversible
from .training.trainer import LongContextTrainer

__all__ = [
    "ReversibleTransformer",
    "ReversibleAttention", 
    "ReversibleFFN",
    "AdditiveCoupling",
    "AffineCoupling",
    "MemoryScheduler",
    "AdaptiveScheduler",
    "convert_to_reversible",
    "LongContextTrainer",
]