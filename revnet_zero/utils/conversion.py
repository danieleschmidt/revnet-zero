"""
Model conversion utilities for converting standard transformers to reversible.

This module provides tools to convert existing transformer models to
reversible implementations with minimal code changes.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union, List
import copy
import warnings

from ..models.reversible_transformer import ReversibleTransformer, ReversibleTransformerBlock
from ..layers.reversible_attention import ReversibleAttention
from ..layers.reversible_ffn import ReversibleFFN
from ..layers.coupling_layers import AdditiveCoupling


def convert_to_reversible(
    model: nn.Module,
    coupling: str = "additive",
    checkpoint_segments: int = 4,
    preserve_weights: bool = True,
    verify_equivalence: bool = True,
) -> nn.Module:
    """
    Convert a standard transformer model to reversible implementation.
    
    Args:
        model: Original transformer model
        coupling: Type of coupling function ("additive" or "affine")
        checkpoint_segments: Number of segments for gradient checkpointing fallback
        preserve_weights: Whether to preserve original weights
        verify_equivalence: Whether to verify output equivalence
        
    Returns:
        Converted reversible model
    """
    # Detect model type and convert accordingly
    model_type = _detect_model_type(model)
    
    if model_type == "gpt2":
        return _convert_gpt2_to_reversible(
            model, coupling, checkpoint_segments, preserve_weights, verify_equivalence
        )
    elif model_type == "bert":
        return _convert_bert_to_reversible(
            model, coupling, checkpoint_segments, preserve_weights, verify_equivalence
        )
    elif model_type == "generic_transformer":
        return _convert_generic_transformer(
            model, coupling, checkpoint_segments, preserve_weights, verify_equivalence
        )
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def _detect_model_type(model: nn.Module) -> str:
    """
    Detect the type of transformer model.
    
    Args:
        model: Input model
        
    Returns:
        Model type string
    """
    model_name = model.__class__.__name__.lower()
    
    if "gpt" in model_name or "gpt2" in model_name:
        return "gpt2"
    elif "bert" in model_name:
        return "bert"
    elif "transformer" in model_name:
        return "generic_transformer"
    else:
        # Try to infer from module structure
        has_attention = any("attention" in name for name, _ in model.named_modules())
        has_feedforward = any("feedforward" in name or "ffn" in name or "mlp" in name 
                            for name, _ in model.named_modules())
        
        if has_attention and has_feedforward:
            return "generic_transformer"
        else:
            return "unknown"


def _convert_gpt2_to_reversible(
    model: nn.Module,
    coupling: str,
    checkpoint_segments: int,
    preserve_weights: bool,
    verify_equivalence: bool,
) -> nn.Module:
    """Convert GPT-2 style model to reversible."""
    
    # Extract configuration from original model
    config = _extract_gpt2_config(model)
    
    # Create reversible model
    reversible_model = ReversibleTransformer(
        vocab_size=config["vocab_size"],
        num_layers=config["num_layers"],
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        max_seq_len=config.get("max_seq_len", 1024),
        coupling=coupling,
        dropout=config.get("dropout", 0.1),
    )
    
    if preserve_weights:
        _transfer_gpt2_weights(model, reversible_model)
    
    if verify_equivalence:
        _verify_model_equivalence(model, reversible_model, config)
    
    return reversible_model


def _extract_gpt2_config(model: nn.Module) -> Dict[str, Any]:
    """Extract configuration from GPT-2 model."""
    config = {}
    
    # Try to get config from model attributes
    if hasattr(model, "config"):
        model_config = model.config
        config["vocab_size"] = getattr(model_config, "vocab_size", 50257)
        config["num_layers"] = getattr(model_config, "n_layer", 12)
        config["d_model"] = getattr(model_config, "n_embd", 768)
        config["num_heads"] = getattr(model_config, "n_head", 12)
        config["d_ff"] = getattr(model_config, "n_inner", None) or 4 * config["d_model"]
        config["max_seq_len"] = getattr(model_config, "n_positions", 1024)
        config["dropout"] = getattr(model_config, "embd_pdrop", 0.1)
    else:
        # Infer from model structure
        config = _infer_config_from_structure(model)
    
    return config


def _infer_config_from_structure(model: nn.Module) -> Dict[str, Any]:
    """Infer model configuration from module structure."""
    config = {
        "vocab_size": 50257,  # Default
        "num_layers": 12,     # Default
        "d_model": 768,       # Default
        "num_heads": 12,      # Default
        "d_ff": 3072,         # Default
        "max_seq_len": 1024,  # Default
        "dropout": 0.1,       # Default
    }
    
    # Try to infer from embeddings
    for name, module in model.named_modules():
        if isinstance(module, nn.Embedding):
            if "token" in name.lower() or "word" in name.lower():
                config["vocab_size"] = module.num_embeddings
                config["d_model"] = module.embedding_dim
            elif "pos" in name.lower():
                config["max_seq_len"] = module.num_embeddings
                break
    
    # Try to infer number of layers
    transformer_blocks = [
        name for name, module in model.named_modules()
        if "block" in name.lower() or "layer" in name.lower()
    ]
    if transformer_blocks:
        # Count unique layer indices
        layer_indices = set()
        for name in transformer_blocks:
            parts = name.split(".")
            for part in parts:
                if part.isdigit():
                    layer_indices.add(int(part))
        if layer_indices:
            config["num_layers"] = max(layer_indices) + 1
    
    return config


def _transfer_gpt2_weights(original_model: nn.Module, reversible_model: nn.Module):
    """Transfer weights from original GPT-2 to reversible model."""
    
    # Transfer embeddings
    orig_token_emb = _find_module_by_name(original_model, ["wte", "token_embedding", "embeddings.word_embeddings"])
    if orig_token_emb is not None:
        reversible_model.token_embedding.weight.data.copy_(orig_token_emb.weight.data)
    
    # Transfer positional embeddings if available
    orig_pos_emb = _find_module_by_name(original_model, ["wpe", "position_embedding", "embeddings.position_embeddings"])
    if orig_pos_emb is not None:
        # Convert learned positional embeddings to sinusoidal (approximation)
        warnings.warn("Converting learned positional embeddings to sinusoidal encoding")
    
    # Transfer layer weights
    _transfer_transformer_layers(original_model, reversible_model)
    
    # Transfer final layer norm
    orig_ln_f = _find_module_by_name(original_model, ["ln_f", "final_layer_norm", "layernorm"])
    if orig_ln_f is not None:
        reversible_model.final_layer_norm.weight.data.copy_(orig_ln_f.weight.data)
        if orig_ln_f.bias is not None:
            reversible_model.final_layer_norm.bias.data.copy_(orig_ln_f.bias.data)
    
    # Transfer output projection
    orig_lm_head = _find_module_by_name(original_model, ["lm_head", "output_projection", "classifier"])
    if orig_lm_head is not None:
        reversible_model.output_projection.weight.data.copy_(orig_lm_head.weight.data)


def _find_module_by_name(model: nn.Module, possible_names: List[str]) -> Optional[nn.Module]:
    """Find a module by checking multiple possible names."""
    
    # First try exact matches
    for name, module in model.named_modules():
        base_name = name.split(".")[-1]
        if base_name in possible_names:
            return module
    
    # Then try substring matches
    for name, module in model.named_modules():
        for possible_name in possible_names:
            if possible_name in name.lower():
                return module
    
    return None


def _transfer_transformer_layers(original_model: nn.Module, reversible_model: nn.Module):
    """Transfer transformer layer weights."""
    
    # Find transformer blocks in original model
    orig_blocks = []
    for name, module in original_model.named_modules():
        if ("block" in name.lower() or "layer" in name.lower()) and len(name.split(".")) <= 2:
            # This is likely a transformer block
            orig_blocks.append(module)
    
    # Transfer weights to reversible blocks
    for i, (orig_block, rev_block) in enumerate(zip(orig_blocks, reversible_model.layers)):
        try:
            _transfer_single_layer_weights(orig_block, rev_block)
        except Exception as e:
            warnings.warn(f"Failed to transfer weights for layer {i}: {e}")


def _transfer_single_layer_weights(orig_layer: nn.Module, rev_layer: ReversibleTransformerBlock):
    """Transfer weights from a single original layer to reversible layer."""
    
    # Transfer attention weights
    orig_attn = _find_module_by_name(orig_layer, ["attn", "attention", "self_attn"])
    if orig_attn is not None:
        _transfer_attention_weights(orig_attn, rev_layer.attention)
    
    # Transfer FFN weights
    orig_mlp = _find_module_by_name(orig_layer, ["mlp", "ffn", "feed_forward"])
    if orig_mlp is not None:
        _transfer_ffn_weights(orig_mlp, rev_layer.feed_forward)


def _transfer_attention_weights(orig_attn: nn.Module, rev_attn: ReversibleAttention):
    """Transfer attention weights."""
    
    # Find Q, K, V projections
    orig_qkv = _find_module_by_name(orig_attn, ["c_attn", "qkv", "query_key_value"])
    if orig_qkv is not None:
        # Split combined QKV weights
        qkv_weight = orig_qkv.weight.data
        d_model = qkv_weight.size(1)
        
        rev_attn.attention.w_q.weight.data.copy_(qkv_weight[:d_model])
        rev_attn.attention.w_k.weight.data.copy_(qkv_weight[d_model:2*d_model])
        rev_attn.attention.w_v.weight.data.copy_(qkv_weight[2*d_model:])
    else:
        # Separate Q, K, V projections
        orig_q = _find_module_by_name(orig_attn, ["q_proj", "query"])
        orig_k = _find_module_by_name(orig_attn, ["k_proj", "key"])
        orig_v = _find_module_by_name(orig_attn, ["v_proj", "value"])
        
        if orig_q: rev_attn.attention.w_q.weight.data.copy_(orig_q.weight.data)
        if orig_k: rev_attn.attention.w_k.weight.data.copy_(orig_k.weight.data)
        if orig_v: rev_attn.attention.w_v.weight.data.copy_(orig_v.weight.data)
    
    # Transfer output projection
    orig_out = _find_module_by_name(orig_attn, ["c_proj", "out_proj", "output"])
    if orig_out is not None:
        rev_attn.attention.w_o.weight.data.copy_(orig_out.weight.data)


def _transfer_ffn_weights(orig_ffn: nn.Module, rev_ffn: ReversibleFFN):
    """Transfer feedforward network weights."""
    
    # Find first linear layer
    orig_fc1 = _find_module_by_name(orig_ffn, ["c_fc", "fc1", "dense", "w1"])
    if orig_fc1 is not None:
        rev_ffn.ffn.w1.weight.data.copy_(orig_fc1.weight.data)
        if orig_fc1.bias is not None:
            rev_ffn.ffn.w1.bias.data.copy_(orig_fc1.bias.data)
    
    # Find second linear layer
    orig_fc2 = _find_module_by_name(orig_ffn, ["c_proj", "fc2", "output", "w2"])
    if orig_fc2 is not None:
        rev_ffn.ffn.w2.weight.data.copy_(orig_fc2.weight.data)
        if orig_fc2.bias is not None:
            rev_ffn.ffn.w2.bias.data.copy_(orig_fc2.bias.data)


def _verify_model_equivalence(
    original_model: nn.Module,
    reversible_model: nn.Module,
    config: Dict[str, Any],
    tolerance: float = 1e-5
):
    """Verify that converted model produces equivalent outputs."""
    
    # Create test input
    batch_size = 2
    seq_len = min(128, config["max_seq_len"])
    vocab_size = config["vocab_size"]
    
    test_input = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    # Set models to eval mode
    original_model.eval()
    reversible_model.eval()
    
    with torch.no_grad():
        try:
            # Get outputs from both models
            orig_output = original_model(test_input)
            rev_output = reversible_model(test_input, use_reversible=False)
            
            # Extract logits
            if isinstance(orig_output, dict):
                orig_logits = orig_output["logits"]
            elif hasattr(orig_output, "logits"):
                orig_logits = orig_output.logits
            else:
                orig_logits = orig_output[0] if isinstance(orig_output, tuple) else orig_output
            
            if isinstance(rev_output, dict):
                rev_logits = rev_output["logits"]
            else:
                rev_logits = rev_output[0] if isinstance(rev_output, tuple) else rev_output
            
            # Check shapes match
            if orig_logits.shape != rev_logits.shape:
                warnings.warn(f"Output shape mismatch: {orig_logits.shape} vs {rev_logits.shape}")
                return
            
            # Check values are close
            max_diff = torch.max(torch.abs(orig_logits - rev_logits))
            if max_diff > tolerance:
                warnings.warn(f"Model outputs differ by {max_diff:.2e} (tolerance: {tolerance:.2e})")
            else:
                print(f"✓ Model conversion verified (max difference: {max_diff:.2e})")
                
        except Exception as e:
            warnings.warn(f"Could not verify model equivalence: {e}")


def _convert_bert_to_reversible(
    model: nn.Module,
    coupling: str,
    checkpoint_segments: int,
    preserve_weights: bool,
    verify_equivalence: bool,
) -> nn.Module:
    """Convert BERT model to reversible (placeholder)."""
    raise NotImplementedError("BERT conversion not yet implemented")


def _convert_generic_transformer(
    model: nn.Module,
    coupling: str,
    checkpoint_segments: int,
    preserve_weights: bool,
    verify_equivalence: bool,
) -> nn.Module:
    """Convert generic transformer model to reversible."""
    
    # This is a simplified conversion for generic transformers
    config = _infer_config_from_structure(model)
    
    reversible_model = ReversibleTransformer(
        vocab_size=config["vocab_size"],
        num_layers=config["num_layers"],
        d_model=config["d_model"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        coupling=coupling,
    )
    
    if preserve_weights:
        warnings.warn("Weight preservation for generic transformers is limited")
    
    return reversible_model