"""
GPT-style reversible transformer implementation.

This module implements a GPT-compatible reversible transformer
for autoregressive language modeling with memory efficiency.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union, Dict, Any, List
import math

from .reversible_transformer import ReversibleTransformerBlock, PositionalEncoding
from ..memory.scheduler import MemoryScheduler


class ReversibleGPTBlock(ReversibleTransformerBlock):
    """
    GPT-style reversible transformer block with causal attention.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        coupling: Union[str, Any] = "additive",
        dropout: float = 0.1,
        use_flash_attention: bool = False,
        use_rational_attention: bool = False,
        layer_norm_eps: float = 1e-5,
        causal: bool = True,
    ):
        super().__init__(
            d_model=d_model,
            num_heads=num_heads,
            d_ff=d_ff,
            coupling=coupling,
            dropout=dropout,
            use_flash_attention=use_flash_attention,
            use_rational_attention=use_rational_attention,
            layer_norm_eps=layer_norm_eps,
        )
        
        self.causal = causal
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_reversible: Optional[bool] = None,
        past_key_value: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass with optional KV caching for generation.
        """
        use_rev = use_reversible if use_reversible is not None else self.use_reversible
        
        # Create causal mask if needed
        if self.causal and attention_mask is None:
            batch_size, seq_len = hidden_states.shape[:2]
            causal_mask = torch.tril(
                torch.ones(seq_len, seq_len, device=hidden_states.device)
            ).unsqueeze(0).unsqueeze(0)
            attention_mask = causal_mask
        
        # Reversible attention with optional caching
        if use_cache and hasattr(self.attention, 'forward_with_cache'):
            hidden_states, present_key_value = self.attention.forward_with_cache(
                hidden_states,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                use_reversible=use_rev
            )
        else:
            hidden_states = self.attention(
                hidden_states,
                attention_mask=attention_mask,
                use_reversible=use_rev
            )
            present_key_value = None
        
        # Reversible feedforward
        hidden_states = self.feed_forward(
            hidden_states,
            use_reversible=use_rev
        )
        
        if use_cache and present_key_value is not None:
            return hidden_states, present_key_value
        else:
            return hidden_states


class ReversibleGPT(nn.Module):
    """
    GPT-style reversible transformer model.
    
    This model implements a memory-efficient GPT architecture using
    reversible neural networks for autoregressive language modeling.
    """
    
    def __init__(
        self,
        vocab_size: int = 50257,
        num_layers: int = 12,
        d_model: int = 768,
        num_heads: int = 12,
        d_ff: int = 3072,
        max_seq_len: int = 262144,
        coupling: Union[str, Any] = "additive",
        dropout: float = 0.1,
        use_flash_attention: bool = False,
        use_rational_attention: bool = False,
        layer_norm_eps: float = 1e-5,
        tie_weights: bool = True,
        causal: bool = True,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.causal = causal
        
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
            ReversibleGPTBlock(
                d_model=d_model,
                num_heads=num_heads,
                d_ff=d_ff,
                coupling=coupling,
                dropout=dropout,
                use_flash_attention=use_flash_attention,
                use_rational_attention=use_rational_attention,
                layer_norm_eps=layer_norm_eps,
                causal=causal,
            )
            for _ in range(num_layers)
        ])
        
        # Set layer IDs
        for i, layer in enumerate(self.layers):
            layer.layer_id = i
        
        # Final layer norm
        self.final_layer_norm = nn.LayerNorm(d_model, eps=layer_norm_eps)
        
        # Language modeling head
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        
        # Tie weights if requested
        if tie_weights:
            self.lm_head.weight = self.token_embedding.weight
        
        # Memory scheduler
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
        past_key_values: Optional[List[Tuple[torch.Tensor, torch.Tensor]]] = None,
        use_cache: bool = False,
        use_reversible: Optional[bool] = None,
        output_hidden_states: bool = False,
        return_dict: bool = True,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through reversible GPT.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            labels: Labels for language modeling loss
            past_key_values: Cached key/value pairs for generation
            use_cache: Whether to return cached key/value pairs
            use_reversible: Override reversible computation setting
            output_hidden_states: Whether to return all hidden states
            return_dict: Whether to return a dictionary
            
        Returns:
            Model outputs (logits, loss, cached states, etc.)
        """
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        hidden_states = self.token_embedding(input_ids)
        
        # Add positional encoding
        if past_key_values is None:
            # Full sequence
            hidden_states = self.positional_encoding(hidden_states)
        else:
            # Only add positional encoding for new tokens
            position_offset = past_key_values[0][0].size(-2) if past_key_values[0] is not None else 0
            position_ids = torch.arange(
                position_offset, position_offset + seq_len,
                device=input_ids.device
            ).unsqueeze(0)
            
            # Get positional encoding for current positions
            pe = self.positional_encoding.pe[:, position_offset:position_offset + seq_len]
            hidden_states = hidden_states + pe
        
        # Create causal attention mask
        if attention_mask is None and self.causal:
            attention_mask = torch.tril(
                torch.ones(seq_len, seq_len, device=input_ids.device)
            ).unsqueeze(0).unsqueeze(0)
        elif attention_mask is not None:
            # Expand attention mask
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            if self.causal:
                causal_mask = torch.tril(
                    torch.ones(seq_len, seq_len, device=input_ids.device)
                ).unsqueeze(0).unsqueeze(0)
                attention_mask = attention_mask * causal_mask
        
        # Store hidden states and present key/values
        all_hidden_states = [] if output_hidden_states else None
        present_key_values = [] if use_cache else None
        
        # Apply transformer layers
        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states.append(hidden_states)
            
            # Get past key/value for this layer
            past_key_value = past_key_values[i] if past_key_values is not None else None
            
            # Check memory scheduler
            should_use_reversible = use_reversible
            if should_use_reversible is None and self.memory_scheduler is not None:
                should_use_reversible = not self.memory_scheduler.should_recompute(i)
            
            # Forward through layer
            if use_cache:
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    past_key_value=past_key_value,
                    use_cache=use_cache,
                    use_reversible=should_use_reversible,
                )
                if isinstance(layer_outputs, tuple):
                    hidden_states, present_key_value = layer_outputs
                    present_key_values.append(present_key_value)
                else:
                    hidden_states = layer_outputs
                    present_key_values.append(None)
            else:
                hidden_states = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    use_reversible=should_use_reversible,
                )
        
        if output_hidden_states:
            all_hidden_states.append(hidden_states)
        
        # Final layer norm
        hidden_states = self.final_layer_norm(hidden_states)
        
        # Language modeling head
        logits = self.lm_head(hidden_states)
        
        # Calculate loss if labels provided
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
            outputs = {
                "logits": logits,
                "loss": loss,
                "last_hidden_state": hidden_states,
            }
            if use_cache:
                outputs["past_key_values"] = present_key_values
            if output_hidden_states:
                outputs["hidden_states"] = all_hidden_states
            return outputs
        else:
            outputs = (logits,)
            if loss is not None:
                outputs = (loss,) + outputs
            if use_cache:
                outputs = outputs + (present_key_values,)
            if output_hidden_states:
                outputs = outputs + (all_hidden_states,)
            return outputs
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 50,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        do_sample: bool = True,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        use_cache: bool = True,
    ) -> torch.Tensor:
        """
        Generate text autoregressively.
        
        Args:
            input_ids: Input token IDs [batch_size, seq_len]
            max_new_tokens: Maximum number of new tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling
            top_p: Top-p (nucleus) sampling
            do_sample: Whether to sample or use greedy decoding
            pad_token_id: Padding token ID
            eos_token_id: End-of-sequence token ID
            use_cache: Whether to use KV caching
            
        Returns:
            Generated token IDs [batch_size, seq_len + max_new_tokens]
        """
        self.eval()
        
        generated_ids = input_ids.clone()
        past_key_values = None
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Forward pass
                if use_cache and past_key_values is not None:
                    # Only use last token for cached generation
                    model_inputs = generated_ids[:, -1:]
                else:
                    model_inputs = generated_ids
                
                outputs = self.forward(
                    input_ids=model_inputs,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    return_dict=True,
                )
                
                logits = outputs["logits"][:, -1, :]  # Get last token logits
                past_key_values = outputs.get("past_key_values")
                
                # Apply temperature
                if temperature != 1.0:
                    logits = logits / temperature
                
                # Apply top-k filtering
                if top_k is not None:
                    top_k = min(top_k, logits.size(-1))
                    top_k_logits, _ = torch.topk(logits, top_k)
                    logits[logits < top_k_logits[..., [-1]]] = float('-inf')
                
                # Apply top-p filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    logits[:, indices_to_remove] = float('-inf')
                
                # Sample or take most likely token
                if do_sample:
                    probs = F.softmax(logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)
                else:
                    next_token = torch.argmax(logits, dim=-1, keepdim=True)
                
                # Add generated token
                generated_ids = torch.cat([generated_ids, next_token], dim=-1)
                
                # Check for EOS token
                if eos_token_id is not None and (next_token == eos_token_id).all():
                    break
        
        return generated_ids
    
    def set_memory_scheduler(self, scheduler: MemoryScheduler):
        """Set memory scheduler for this model."""
        self.memory_scheduler = scheduler
    
    def set_reversible_mode(self, enabled: bool):
        """Enable or disable reversible computation for all layers."""
        for layer in self.layers:
            layer.set_reversible_mode(enabled)
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get comprehensive model information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "model_type": "ReversibleGPT",
            "vocab_size": self.vocab_size,
            "num_layers": self.num_layers,
            "d_model": self.d_model,
            "num_heads": self.num_heads,
            "d_ff": self.d_ff,
            "max_seq_len": self.max_seq_len,
            "causal": self.causal,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,    
            "parameter_size_mb": total_params * 4 / (1024 * 1024),
            "memory_scheduler_enabled": self.memory_scheduler is not None,
        }