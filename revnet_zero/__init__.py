"""
revnet_zero: Memory-efficient reversible transformers with activation recomputation.

The key idea (Gomez et al., 2017 "The Reversible Residual Network"):
  Forward:      y1 = x1 + F(x2),  y2 = x2 + G(y1)
  Reconstruct:  x2 = y2 - G(y1),  x1 = y1 - F(x2)

No intermediate activations need to be stored — they are recomputed on demand
during the backward pass, giving O(1) memory w.r.t. depth.
"""

from .model import (
    RevBlock,
    RevTransformerBlock,
    RevTransformer,
    StandardTransformer,
)

__all__ = [
    "RevBlock",
    "RevTransformerBlock",
    "RevTransformer",
    "StandardTransformer",
]
