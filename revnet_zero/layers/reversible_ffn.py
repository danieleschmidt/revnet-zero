"""
Reversible feedforward network implementation.

This module implements memory-efficient feedforward networks using reversible
neural networks for transformer architectures.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union

from .coupling_layers import BaseCoupling, AdditiveCoupling


class ReversibleFFNFunction(torch.autograd.Function):
    """
    Custom autograd function for reversible feedforward computation.
    """
    
    @staticmethod
    def forward(ctx, x, ffn_layer, coupling_fn):
        """
        Forward pass with minimal memory storage.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            ffn_layer: The FFN computation layer
            coupling_fn: Coupling function for reversible transformation
            
        Returns:
            Output tensor after reversible FFN
        """
        ctx.ffn_layer = ffn_layer
        ctx.coupling_fn = coupling_fn
        
        with torch.no_grad():
            # Split input for reversible computation
            x1, x2 = coupling_fn.split_input(x)
            
            # Apply coupling transformation
            y1, y2 = coupling_fn.forward(x1, x2)
            
            # Store only minimal information for backward pass
            ctx.save_for_backward(y1, y2)
            
            # Return concatenated output
            return coupling_fn.cat_output(y1, y2)
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass with activation recomputation.
        """
        y1, y2 = ctx.saved_tensors
        ffn_layer = ctx.ffn_layer
        coupling_fn = ctx.coupling_fn
        
        # Split gradient
        grad_y1, grad_y2 = coupling_fn.split_input(grad_output)
        
        # Recompute forward pass with gradients enabled
        y1.requires_grad_(True)
        y2.requires_grad_(True)
        
        # Reconstruct input activations
        x1, x2 = coupling_fn.inverse(y1, y2)
        
        # Recompute forward transformation
        z1, z2 = coupling_fn.forward(x1, x2)
        
        # Compute gradients through recomputed activations
        torch.autograd.backward([z1, z2], [grad_y1, grad_y2])
        
        return x1.grad + x2.grad, None, None


class FeedForwardNetwork(nn.Module):
    """
    Standard feedforward network with optional gating.
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        activation: str = "relu",
        dropout: float = 0.1,
        use_gating: bool = False,
        bias: bool = True,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.d_ff = d_ff
        self.activation = activation
        self.dropout = dropout
        self.use_gating = use_gating
        
        # First linear layer
        self.w1 = nn.Linear(d_model, d_ff, bias=bias)
        
        # Gating layer for GLU variants
        if use_gating:
            self.w_gate = nn.Linear(d_model, d_ff, bias=bias)
        
        # Second linear layer
        self.w2 = nn.Linear(d_ff, d_model, bias=bias)
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
        
        # Activation function
        self.activation_fn = self._get_activation_fn(activation)
    
    def _get_activation_fn(self, activation: str):
        """Get activation function by name."""
        if activation == "relu":
            return F.relu
        elif activation == "gelu":
            return F.gelu
        elif activation == "swish" or activation == "silu":
            return F.silu
        elif activation == "tanh":
            return torch.tanh
        else:
            raise ValueError(f"Unknown activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply feedforward transformation.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            FFN output [batch_size, seq_len, d_model]
        """
        # First linear transformation
        hidden = self.w1(x)
        
        if self.use_gating:
            # Gated Linear Unit (GLU) or variants
            gate = self.w_gate(x)
            if self.activation == "gelu":
                # GeGLU: GELU(x * W1) * (x * W_gate)
                hidden = self.activation_fn(hidden) * gate
            else:
                # Standard GLU: (x * W1) * sigmoid(x * W_gate)
                hidden = hidden * torch.sigmoid(gate)
        else:
            # Standard activation
            hidden = self.activation_fn(hidden)
        
        # Dropout
        hidden = self.dropout_layer(hidden)
        
        # Second linear transformation
        output = self.w2(hidden)
        
        return output


class ReversibleFFN(nn.Module):
    """
    Reversible feedforward network for memory-efficient training.
    
    This layer implements feedforward networks using reversible neural networks,
    allowing activation recomputation during backpropagation to save memory.
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        coupling: Union[str, BaseCoupling] = "additive",
        activation: str = "relu",
        dropout: float = 0.1,
        use_gating: bool = False,
        layer_norm_eps: float = 1e-5,
        bias: bool = True,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.d_ff = d_ff
        self.activation = activation
        self.dropout = dropout
        
        # Feedforward network
        self.ffn = FeedForwardNetwork(
            d_model=d_model,
            d_ff=d_ff,
            activation=activation,
            dropout=dropout,
            use_gating=use_gating,
            bias=bias,
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        
        # Coupling function
        if isinstance(coupling, str):
            if coupling == "additive":
                self.coupling_fn = AdditiveCoupling(d_model)
            else:
                raise ValueError(f"Unknown coupling type: {coupling}")
        else:
            self.coupling_fn = coupling
        
        # Residual dropout
        self.dropout_layer = nn.Dropout(dropout)
    
    def forward(
        self, 
        hidden_states: torch.Tensor,
        use_reversible: bool = True
    ) -> torch.Tensor:
        """
        Apply reversible feedforward network.
        
        Args:
            hidden_states: Input tensor [batch_size, seq_len, d_model]
            use_reversible: Whether to use reversible computation
            
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        if use_reversible and self.training:
            # Use reversible computation during training
            return ReversibleFFNFunction.apply(
                hidden_states, self.ffn, self.coupling_fn
            )
        else:
            # Standard computation for inference or testing
            return self._forward_standard(hidden_states)
    
    def _forward_standard(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Standard (non-reversible) forward pass for inference.
        
        Args:
            hidden_states: Input tensor
            
        Returns:
            Output tensor
        """
        # Layer norm
        normed_input = self.layer_norm(hidden_states)
        
        # Feedforward network
        ffn_output = self.ffn(normed_input)
        
        # Residual connection with dropout
        output = hidden_states + self.dropout_layer(ffn_output)
        
        return output
    
    def estimate_memory_usage(self, batch_size: int, seq_len: int) -> dict:
        """
        Estimate memory usage for this layer.
        
        Args:
            batch_size: Batch size
            seq_len: Sequence length
            
        Returns:
            Dictionary with memory estimates in bytes
        """
        # Intermediate activation memory
        intermediate_memory = batch_size * seq_len * self.d_ff * 4  # float32
        
        # Input/output activation memory
        io_memory = batch_size * seq_len * self.d_model * 4  # float32
        
        # Reversible memory (only store coupling outputs)
        reversible_memory = batch_size * seq_len * self.d_model * 4  # float32
        
        return {
            "standard_memory": intermediate_memory + io_memory,
            "reversible_memory": reversible_memory,
            "memory_saved": intermediate_memory,
            "reduction_ratio": intermediate_memory / (intermediate_memory + io_memory)
        }


class GeGLUFFN(ReversibleFFN):
    """
    Reversible FFN with Gated GELU activation (GeGLU).
    
    GeGLU has shown to be more effective than standard activations
    in transformer models.
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        coupling: Union[str, BaseCoupling] = "additive",
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-5,
        bias: bool = True,
    ):
        super().__init__(
            d_model=d_model,
            d_ff=d_ff,
            coupling=coupling,
            activation="gelu",
            dropout=dropout,
            use_gating=True,
            layer_norm_eps=layer_norm_eps,
            bias=bias,
        )