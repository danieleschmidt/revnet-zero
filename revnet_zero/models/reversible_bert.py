"""
BERT-style reversible transformer implementation.

This module implements a BERT-compatible reversible transformer
for bidirectional language modeling and masked language tasks.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, Dict, Any, List
import math

from .reversible_transformer import ReversibleTransformerBlock, PositionalEncoding
from ..memory.scheduler import MemoryScheduler


class ReversibleBertEmbeddings(nn.Module):
    """
    BERT-style embeddings with token, position, and token type embeddings.
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        max_position_embeddings: int = 512,
        type_vocab_size: int = 2,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-12,
    ):
        super().__init__()
        
        self.word_embeddings = nn.Embedding(vocab_size, d_model)
        self.position_embeddings = nn.Embedding(max_position_embeddings, d_model)
        self.token_type_embeddings = nn.Embedding(type_vocab_size, d_model)
        
        self.layer_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)
        
        # Register position_ids as buffer
        self.register_buffer(
            "position_ids",
            torch.arange(max_position_embeddings).expand((1, -1))
        )
    
    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through BERT embeddings.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            token_type_ids: Token type IDs [batch_size, seq_len]  
            position_ids: Position IDs [batch_size, seq_len]
            
        Returns:
            Embedded representations [batch_size, seq_len, d_model]
        """
        batch_size, seq_len = input_ids.shape
        
        # Position IDs
        if position_ids is None:
            position_ids = self.position_ids[:, :seq_len]
        
        # Token type IDs
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)
        
        # Get embeddings
        word_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        
        # Combine embeddings
        embeddings = word_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)
        
        return embeddings


class ReversibleBertPooler(nn.Module):
    """
    BERT pooler layer for classification tasks.
    """
    
    def __init__(self, d_model: int):
        super().__init__()
        self.dense = nn.Linear(d_model, d_model)
        self.activation = nn.Tanh()
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Pool the hidden states by taking the [CLS] token representation.
        
        Args:
            hidden_states: Hidden states [batch_size, seq_len, d_model]
            
        Returns:
            Pooled representation [batch_size, d_model]
        """
        # Take [CLS] token (first token)
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class ReversibleBert(nn.Module):
    """
    BERT-style reversible transformer model.
    
    This model implements a memory-efficient BERT architecture using
    reversible neural networks for bidirectional language understanding.
    """
    
    def __init__(
        self,
        vocab_size: int = 30522,
        num_layers: int = 12,
        d_model: int = 768,
        num_heads: int = 12,
        d_ff: int = 3072,
        max_position_embeddings: int = 512,
        max_seq_len: int = 262144,
        type_vocab_size: int = 2,
        coupling: Union[str, Any] = "additive",
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        use_flash_attention: bool = False,
        use_rational_attention: bool = False,
        layer_norm_eps: float = 1e-12,
        tie_weights: bool = False,
        add_pooling_layer: bool = True,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_position_embeddings = max_position_embeddings
        self.max_seq_len = max_seq_len
        
        # BERT embeddings
        self.embeddings = ReversibleBertEmbeddings(
            vocab_size=vocab_size,
            d_model=d_model,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            dropout=dropout,
            layer_norm_eps=layer_norm_eps,
        )
        
        # Reversible transformer layers
        self.encoder = nn.ModuleList([
            ReversibleTransformerBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                coupling=coupling,
                dropout=attention_dropout,
                use_flash_attention=use_flash_attention,
                use_rational_attention=use_rational_attention,
                layer_norm_eps=layer_norm_eps,
            )
            for _ in range(num_layers)
        ])
        
        # Set layer IDs
        for i, layer in enumerate(self.encoder):
            layer.layer_id = i
        
        # Pooler for classification tasks
        self.pooler = ReversibleBertPooler(d_model) if add_pooling_layer else None
        
        # Memory scheduler
        self.memory_scheduler: Optional[MemoryScheduler] = None
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_reversible: Optional[bool] = None,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through reversible BERT.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            token_type_ids: Token type IDs [batch_size, seq_len]
            position_ids: Position IDs [batch_size, seq_len]
            labels: Labels for masked language modeling
            use_reversible: Override reversible computation setting
            output_hidden_states: Whether to return all hidden states
            return_dict: Whether to return a dictionary
            
        Returns:
            Model outputs
        """
        batch_size, seq_len = input_ids.shape
        
        # Get embeddings
        hidden_states = self.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )
        
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
        
        # Apply attention mask values
        extended_attention_mask = extended_attention_mask.float()
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        # Store all hidden states if requested
        all_hidden_states = [] if output_hidden_states else None
        
        # Apply encoder layers
        for i, layer in enumerate(self.encoder):
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
            
            # Check memory scheduler
            should_use_reversible = use_reversible
            if should_use_reversible is None and self.memory_scheduler is not None:
                should_use_reversible = not self.memory_scheduler.should_recompute(i)
            
            hidden_states = layer(
                hidden_states,
                attention_mask=extended_attention_mask,
                use_reversible=should_use_reversible,
            )
        
        if output_hidden_states:
            all_hidden_states.append(hidden_states)
        
        # Apply pooler if available
        pooled_output = None
        if self.pooler is not None:
            pooled_output = self.pooler(hidden_states)
        
        # Calculate MLM loss if labels provided
        loss = None
        if labels is not None:
            # This would typically involve a MLM head
            # For now, we'll implement a simple version
            prediction_scores = self._get_mlm_predictions(hidden_states)
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(
                prediction_scores.view(-1, self.vocab_size),
                labels.view(-1)
            )
        
        if return_dict:
            outputs = {
                "last_hidden_state": hidden_states,
                "pooler_output": pooled_output,
                "loss": loss,
            }
            if output_hidden_states:
                outputs["hidden_states"] = all_hidden_states
            return outputs
        else:
            outputs = (hidden_states, pooled_output)
            if output_hidden_states:
                outputs = outputs + (all_hidden_states,)
            if loss is not None:
                outputs = (loss,) + outputs
            return outputs
    
    def _get_mlm_predictions(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Get masked language modeling predictions.
        
        Args:
            hidden_states: Hidden states [batch_size, seq_len, d_model]
            
        Returns:
            Prediction scores [batch_size, seq_len, vocab_size]
        """
        # Simple MLM head - in practice, you'd want a more sophisticated version
        mlm_head = nn.Linear(self.d_model, self.vocab_size)
        if hasattr(self, '_mlm_head'):
            mlm_head = self._mlm_head
        else:
            self._mlm_head = mlm_head.to(hidden_states.device)
        
        return self._mlm_head(hidden_states)
    
    def set_memory_scheduler(self, scheduler: MemoryScheduler):
        """Set memory scheduler for this model."""
        self.memory_scheduler = scheduler
    
    def set_reversible_mode(self, enabled: bool):
        """Enable or disable reversible computation for all layers."""
        for layer in self.encoder:
            layer.set_reversible_mode(enabled)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "model_type": "ReversibleBert",
            "vocab_size": self.vocab_size,
            "num_layers": self.num_layers,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "d_ff": self.d_ff,
            "max_position_embeddings": self.max_position_embeddings,
            "max_seq_len": self.max_seq_len,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "parameter_size_mb": total_params * 4 / (1024 * 1024),
            "memory_scheduler_enabled": self.memory_scheduler is not None,
        }