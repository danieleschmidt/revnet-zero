"""
Reversible self-attention layer implementation.

This module implements memory-efficient self-attention using reversible neural networks,
enabling gradient computation without storing intermediate activations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
import math

from .coupling_layers import BaseCoupling, AdditiveCoupling


class ReversibleAttentionFunction(torch.autograd.Function):
    """
    Custom autograd function for reversible attention computation.
    
    This function implements the forward and backward pass for reversible attention,
    enabling activation recomputation during backpropagation for memory efficiency.
    """
    
    @staticmethod
    def forward(ctx, x, attention_layer, coupling_fn, attention_mask=None):
        """
        Forward pass with minimal memory storage.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            attention_layer: The attention computation layer
            coupling_fn: Coupling function for reversible transformation
            attention_mask: Optional attention mask
            
        Returns:
            Output tensor after reversible attention
        """
        ctx.attention_layer = attention_layer
        ctx.coupling_fn = coupling_fn
        ctx.attention_mask = attention_mask
        
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
        
        Args:
            grad_output: Gradient w.r.t. output
            
        Returns:
            Gradients w.r.t. inputs
        """
        y1, y2 = ctx.saved_tensors
        attention_layer = ctx.attention_layer
        coupling_fn = ctx.coupling_fn
        attention_mask = ctx.attention_mask
        
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
        
        return x1.grad + x2.grad, None, None, None


class MultiHeadAttention(nn.Module):
    """
    Standard multi-head attention implementation optimized for reversible layers.
    """
    
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1,
                 use_flash_attention: bool = False):
        super().__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.dropout = dropout
        self.use_flash_attention = use_flash_attention
        
        # Linear projections
        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.w_o = nn.Linear(d_model, d_model, bias=False)
        
        self.dropout_layer = nn.Dropout(dropout)
        
        # Scale factor for attention
        self.scale = 1.0 / math.sqrt(self.d_head)
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply multi-head attention.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            attention_mask: Optional attention mask [batch_size, seq_len, seq_len]
            
        Returns:
            Attention output [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.shape
        
        # Compute Q, K, V
        q = self.w_q(x).view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        k = self.w_k(x).view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        v = self.w_v(x).view(batch_size, seq_len, self.num_heads, self.d_head).transpose(1, 2)
        
        if self.use_flash_attention and hasattr(F, 'scaled_dot_product_attention'):
            # Use PyTorch's flash attention if available
            attn_output = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attention_mask, dropout_p=self.dropout if self.training else 0.0
            )
        else:
            # Standard attention computation
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
            
            if attention_mask is not None:
                scores = scores.masked_fill(attention_mask == 0, -1e9)
            
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout_layer(attn_weights)
            attn_output = torch.matmul(attn_weights, v)
        
        # Reshape and apply output projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, d_model
        )
        
        return self.w_o(attn_output)


class ReversibleAttention(nn.Module):
    """
    Reversible self-attention layer for memory-efficient training.
    
    This layer implements self-attention using reversible neural networks,
    allowing activation recomputation during backpropagation to save memory.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        coupling: Union[str, BaseCoupling] = "additive",
        dropout: float = 0.1,
        use_flash_attention: bool = False,
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.dropout = dropout
        
        # Attention layer
        self.attention = MultiHeadAttention(
            d_model=d_model,
            num_heads=num_heads,
            dropout=dropout,
            use_flash_attention=use_flash_attention
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
        attention_mask: Optional[torch.Tensor] = None,
        use_reversible: bool = True
    ) -> torch.Tensor:
        """
        Apply reversible attention.
        
        Args:
            hidden_states: Input tensor [batch_size, seq_len, d_model]
            attention_mask: Optional attention mask
            use_reversible: Whether to use reversible computation (for testing)
            
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        if use_reversible and self.training:
            # Use reversible computation during training
            return ReversibleAttentionFunction.apply(
                hidden_states, self.attention, self.coupling_fn, attention_mask
            )
        else:
            # Standard computation for inference or testing
            return self._forward_standard(hidden_states, attention_mask)
    
    def _forward_standard(
        self, 
        hidden_states: torch.Tensor, 
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Standard (non-reversible) forward pass for inference.
        
        Args:
            hidden_states: Input tensor
            attention_mask: Optional attention mask
            
        Returns:
            Output tensor
        """
        # Layer norm
        normed_input = self.layer_norm(hidden_states)
        
        # Self-attention
        attn_output = self.attention(normed_input, attention_mask)
        
        # Residual connection with dropout
        output = hidden_states + self.dropout_layer(attn_output)
        
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
        # Standard attention memory
        attention_memory = batch_size * seq_len * seq_len * self.num_heads * 4  # float32
        
        # Activation memory (saved for gradients)
        activation_memory = batch_size * seq_len * self.d_model * 4  # float32
        
        # Reversible memory savings (only store coupling outputs)
        reversible_memory = batch_size * seq_len * self.d_model * 4  # float32
        
        return {
            "standard_memory": attention_memory + activation_memory,
            "reversible_memory": reversible_memory,
            "memory_saved": attention_memory,
            "reduction_ratio": attention_memory / (attention_memory + activation_memory)
        }