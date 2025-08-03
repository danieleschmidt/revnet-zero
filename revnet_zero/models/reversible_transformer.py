"""
Reversible transformer model implementation.

This module implements a complete reversible transformer architecture
with memory-efficient training capabilities for long sequences.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, Dict, Any
import math

from ..layers.reversible_attention import ReversibleAttention
from ..layers.reversible_ffn import ReversibleFFN
from ..layers.coupling_layers import BaseCoupling, AdditiveCoupling
from ..memory.scheduler import MemoryScheduler


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for transformer models.
    
    Supports very long sequences efficiently.
    """
    
    def __init__(self, d_model: int, max_len: int = 1000000, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a parameter)
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input embeddings.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            Input with positional encoding added
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        return self.dropout(x)


class ReversibleTransformerBlock(nn.Module):
    """
    Single reversible transformer block.
    
    Combines reversible attention and feedforward layers with
    coupling functions for memory efficiency.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        coupling: Union[str, BaseCoupling] = "additive",
        dropout: float = 0.1,
        use_flash_attention: bool = False,
        use_rational_attention: bool = False,
        layer_norm_eps: float = 1e-5,
    ):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        
        # Reversible attention layer
        if use_rational_attention:
            from ..layers.rational_attention import RationalFourierAttention
            self.attention = ReversibleAttention(
                d_model=d_model,
                num_heads=num_heads,
                coupling=coupling,
                dropout=dropout,
                use_flash_attention=use_flash_attention,
                layer_norm_eps=layer_norm_eps,
            )
        else:
            self.attention = ReversibleAttention(
                d_model=d_model,
                num_heads=num_heads,
                coupling=coupling,
                dropout=dropout,
                use_flash_attention=use_flash_attention,
                layer_norm_eps=layer_norm_eps,
            )
        
        # Reversible feedforward layer
        self.feed_forward = ReversibleFFN(
            d_model=d_model,
            d_ff=d_ff,
            coupling=coupling,
            dropout=dropout,
            layer_norm_eps=layer_norm_eps,
        )
        
        # Block-level configuration
        self.use_reversible = True
        self.layer_id = None  # Set by parent model
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_reversible: Optional[bool] = None,
    ) -> torch.Tensor:
        """
        Forward pass through reversible transformer block.
        
        Args:
            hidden_states: Input tensor [batch_size, seq_len, d_model]
            attention_mask: Optional attention mask
            use_reversible: Override reversible computation setting
            
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        use_rev = use_reversible if use_reversible is not None else self.use_reversible
        
        # Reversible attention
        hidden_states = self.attention(
            hidden_states, 
            attention_mask=attention_mask, 
            use_reversible=use_rev
        )
        
        # Reversible feedforward
        hidden_states = self.feed_forward(
            hidden_states,
            use_reversible=use_rev
        )
        
        return hidden_states
    
    def set_reversible_mode(self, enabled: bool):
        """
        Enable or disable reversible computation.
        
        Args:
            enabled: Whether to use reversible computation
        """
        self.use_reversible = enabled
    
    def estimate_memory_usage(self, batch_size: int, seq_len: int) -> Dict[str, int]:
        """
        Estimate memory usage for this block.
        
        Args:
            batch_size: Batch size
            seq_len: Sequence length
            
        Returns:
            Dictionary with memory estimates
        """
        attn_memory = self.attention.estimate_memory_usage(batch_size, seq_len)
        ffn_memory = self.feed_forward.estimate_memory_usage(batch_size, seq_len)
        
        return {
            "attention_memory": attn_memory["reversible_memory"],
            "ffn_memory": ffn_memory["reversible_memory"],
            "total_memory": attn_memory["reversible_memory"] + ffn_memory["reversible_memory"],
            "memory_saved": attn_memory["memory_saved"] + ffn_memory["memory_saved"],
        }


class ReversibleTransformer(nn.Module):
    """
    Complete reversible transformer model for long-context training.
    
    This model implements a memory-efficient transformer architecture
    using reversible neural networks to enable training with very long sequences.
    """
    
    def __init__(
        self,
        vocab_size: int = 50257,
        num_layers: int = 12,
        d_model: int = 768,
        num_heads: int = 12,
        d_ff: int = 3072,
        max_seq_len: int = 262144,
        coupling: Union[str, BaseCoupling] = "additive",
        dropout: float = 0.1,
        use_flash_attention: bool = False,
        use_rational_attention: bool = False,
        layer_norm_eps: float = 1e-5,
        tie_weights: bool = True,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.positional_encoding = PositionalEncoding(
            d_model=d_model,
            max_len=max_seq_len,
            dropout=dropout
        )
        
        # Transformer blocks
        self.layers = nn.ModuleList([
            ReversibleTransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                coupling=coupling,
                dropout=dropout,
                use_flash_attention=use_flash_attention,
                use_rational_attention=use_rational_attention,
                layer_norm_eps=layer_norm_eps,
            )
            for _ in range(num_layers)
        ])
        
        # Set layer IDs for memory scheduling
        for i, layer in enumerate(self.layers):
            layer.layer_id = i
        
        # Final layer norm
        self.final_layer_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie weights if requested
        if tie_weights:
            self.output_projection.weight = self.token_embedding.weight
        
        # Memory scheduler (optional)
        self.memory_scheduler: Optional[MemoryScheduler] = None
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_reversible: Optional[bool] = None,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through reversible transformer.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Optional attention mask [batch_size, seq_len]
            labels: Optional labels for language modeling loss
            use_reversible: Override reversible computation setting
            return_dict: Whether to return a dictionary
            
        Returns:
            Model outputs (logits and optionally loss)
        """
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        hidden_states = self.token_embedding(input_ids)
        
        # Add positional encoding
        hidden_states = self.positional_encoding(hidden_states)
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones(
                batch_size, seq_len, device=input_ids.device, dtype=torch.bool
            )
        
        # Expand attention mask for multi-head attention
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.expand(
            batch_size, 1, seq_len, seq_len
        )
        
        # Apply transformer layers
        for i, layer in enumerate(self.layers):
            # Check memory scheduler for recomputation decision
            should_use_reversible = use_reversible
            if should_use_reversible is None and self.memory_scheduler is not None:
                should_use_reversible = not self.memory_scheduler.should_recompute(i)
            
            hidden_states = layer(
                hidden_states,
                attention_mask=extended_attention_mask,
                use_reversible=should_use_reversible,
            )
        
        # Final layer norm
        hidden_states = self.final_layer_norm(hidden_states)
        
        # Output projection
        logits = self.output_projection(hidden_states)
        
        # Calculate loss if labels are provided
        loss = None
        if labels is not None:
            # Shift labels for language modeling
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Calculate cross-entropy loss
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(
                shift_logits.view(-1, self.vocab_size),
                shift_labels.view(-1)
            )
        
        if return_dict:
            return {
                "logits": logits,
                "loss": loss,
                "hidden_states": hidden_states,
            }
        else:
            outputs = (logits,)
            if loss is not None:
                outputs = (loss,) + outputs
            return outputs
    
    def set_memory_scheduler(self, scheduler: MemoryScheduler):
        """
        Set memory scheduler for this model.
        
        Args:
            scheduler: Memory scheduler instance
        """
        self.memory_scheduler = scheduler
    
    def set_reversible_mode(self, enabled: bool):
        """
        Enable or disable reversible computation for all layers.
        
        Args:
            enabled: Whether to use reversible computation
        """
        for layer in self.layers:
            layer.set_reversible_mode(enabled)
    
    def estimate_memory_usage(self, batch_size: int, seq_len: int) -> Dict[str, Any]:
        """
        Estimate total model memory usage.
        
        Args:
            batch_size: Batch size
            seq_len: Sequence length
            
        Returns:
            Dictionary with detailed memory estimates
        """
        # Embedding memory
        embedding_memory = batch_size * seq_len * self.d_model * 4  # float32
        
        # Layer memory
        layer_memories = []
        total_layer_memory = 0
        total_memory_saved = 0
        
        for i, layer in enumerate(self.layers):
            layer_est = layer.estimate_memory_usage(batch_size, seq_len)
            layer_memories.append(layer_est)
            total_layer_memory += layer_est["total_memory"]
            total_memory_saved += layer_est["memory_saved"]
        
        # Output projection memory
        output_memory = batch_size * seq_len * self.vocab_size * 4  # float32
        
        total_memory = embedding_memory + total_layer_memory + output_memory
        
        return {
            "total_memory": total_memory,
            "embedding_memory": embedding_memory,
            "layer_memory": total_layer_memory,
            "output_memory": output_memory,
            "memory_saved": total_memory_saved,
            "reduction_ratio": total_memory_saved / (total_memory + total_memory_saved),
            "layer_breakdown": layer_memories,
            "memory_per_token": total_memory / (batch_size * seq_len),
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive model information.
        
        Returns:
            Dictionary with model configuration and statistics
        """
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "model_type": "ReversibleTransformer",
            "vocab_size": self.vocab_size,
            "num_layers": self.num_layers,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "d_ff": self.d_ff,
            "max_seq_len": self.max_seq_len,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "parameter_size_mb": total_params * 4 / (1024 * 1024),  # Assuming float32
            "memory_scheduler_enabled": self.memory_scheduler is not None,
        }