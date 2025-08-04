"""
Input validation and error handling utilities for RevNet-Zero.

This module provides comprehensive validation functions to ensure
proper input handling and graceful error management.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Union, List, Tuple
import warnings
from functools import wraps


class RevNetValidationError(Exception):
    """Custom exception for RevNet-Zero validation errors."""
    pass


class RevNetConfigurationError(RevNetValidationError):
    """Exception for configuration errors."""
    pass


class RevNetMemoryError(RevNetValidationError):
    """Exception for memory-related errors."""
    pass


def validate_model_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate model configuration parameters.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        Validated and normalized configuration
        
    Raises:
        RevNetConfigurationError: If configuration is invalid
    """
    required_fields = {
        "vocab_size": int,
        "num_layers": int,
        "d_model": int,
        "num_heads": int,
        "d_ff": int,
    }
    
    optional_fields = {
        "max_seq_len": (int, 1024),
        "dropout": (float, 0.1),
        "coupling": (str, "additive"),
        "use_flash_attention": (bool, False),
        "use_rational_attention": (bool, False),
        "layer_norm_eps": (float, 1e-5),
        "tie_weights": (bool, True),
    }
    
    validated_config = {}
    
    # Check required fields
    for field, expected_type in required_fields.items():
        if field not in config:
            raise RevNetConfigurationError(f"Missing required field: {field}")
        
        value = config[field]
        if not isinstance(value, expected_type):
            raise RevNetConfigurationError(
                f"Field '{field}' must be of type {expected_type.__name__}, got {type(value).__name__}"
            )
        
        # Specific validations
        if field == "vocab_size" and value <= 0:
            raise RevNetConfigurationError("vocab_size must be positive")
        
        if field == "num_layers" and value <= 0:
            raise RevNetConfigurationError("num_layers must be positive")
        
        if field == "d_model" and value <= 0:
            raise RevNetConfigurationError("d_model must be positive")
        
        if field == "num_heads" and value <= 0:
            raise RevNetConfigurationError("num_heads must be positive")
        
        if field == "d_ff" and value <= 0:
            raise RevNetConfigurationError("d_ff must be positive")
        
        validated_config[field] = value
    
    # Check d_model divisibility by num_heads
    if validated_config["d_model"] % validated_config["num_heads"] != 0:
        raise RevNetConfigurationError(
            f"d_model ({validated_config['d_model']}) must be divisible by "
            f"num_heads ({validated_config['num_heads']})"
        )
    
    # Add optional fields with defaults
    for field, (expected_type, default_value) in optional_fields.items():
        if field in config:
            value = config[field]
            if not isinstance(value, expected_type):
                raise RevNetConfigurationError(
                    f"Field '{field}' must be of type {expected_type.__name__}, got {type(value).__name__}"
                )
            validated_config[field] = value
        else:
            validated_config[field] = default_value
    
    # Additional validations
    if validated_config["dropout"] < 0 or validated_config["dropout"] >= 1:
        raise RevNetConfigurationError("dropout must be in range [0, 1)")
    
    if validated_config["coupling"] not in ["additive", "affine", "learned"]:
        raise RevNetConfigurationError(
            f"coupling must be one of ['additive', 'affine', 'learned'], got '{validated_config['coupling']}'"
        )
    
    if validated_config["layer_norm_eps"] <= 0:
        raise RevNetConfigurationError("layer_norm_eps must be positive")
    
    return validated_config


def validate_input_tensor(
    tensor: torch.Tensor,
    name: str,
    expected_shape: Optional[Tuple[int, ...]] = None,
    expected_dtype: Optional[torch.dtype] = None,
    min_dim: Optional[int] = None,
    max_dim: Optional[int] = None,
) -> torch.Tensor:
    """
    Validate input tensor properties.
    
    Args:
        tensor: Input tensor to validate
        name: Name of the tensor for error messages
        expected_shape: Expected tensor shape (None values are wildcards)
        expected_dtype: Expected data type
        min_dim: Minimum number of dimensions
        max_dim: Maximum number of dimensions
        
    Returns:
        The validated tensor
        
    Raises:
        RevNetValidationError: If validation fails
    """
    if not isinstance(tensor, torch.Tensor):
        raise RevNetValidationError(f"{name} must be a torch.Tensor, got {type(tensor)}")
    
    # Check dimensions
    if min_dim is not None and tensor.dim() < min_dim:
        raise RevNetValidationError(
            f"{name} must have at least {min_dim} dimensions, got {tensor.dim()}"
        )
    
    if max_dim is not None and tensor.dim() > max_dim:
        raise RevNetValidationError(
            f"{name} must have at most {max_dim} dimensions, got {tensor.dim()}"
        )
    
    # Check shape
    if expected_shape is not None:
        if len(expected_shape) != tensor.dim():
            raise RevNetValidationError(
                f"{name} expected {len(expected_shape)} dimensions, got {tensor.dim()}"
            )
        
        for i, (expected, actual) in enumerate(zip(expected_shape, tensor.shape)):
            if expected is not None and expected != actual:
                raise RevNetValidationError(
                    f"{name} dimension {i} expected size {expected}, got {actual}"
                )
    
    # Check dtype
    if expected_dtype is not None and tensor.dtype != expected_dtype:
        raise RevNetValidationError(
            f"{name} expected dtype {expected_dtype}, got {tensor.dtype}"
        )
    
    # Check for NaN or Inf
    if torch.isnan(tensor).any():
        raise RevNetValidationError(f"{name} contains NaN values")
    
    if torch.isinf(tensor).any():
        raise RevNetValidationError(f"{name} contains infinite values")
    
    return tensor


def validate_sequence_input(
    input_ids: torch.Tensor,
    vocab_size: int,
    max_seq_len: Optional[int] = None,
    attention_mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Validate sequence input for transformer models.
    
    Args:
        input_ids: Input token IDs [batch_size, seq_len]
        vocab_size: Vocabulary size
        max_seq_len: Maximum sequence length
        attention_mask: Optional attention mask [batch_size, seq_len]
        
    Returns:
        Validated (input_ids, attention_mask) tuple
        
    Raises:
        RevNetValidationError: If validation fails
    """
    # Validate input_ids
    input_ids = validate_input_tensor(
        input_ids, "input_ids", 
        min_dim=2, max_dim=2,
        expected_dtype=torch.long
    )
    
    batch_size, seq_len = input_ids.shape
    
    # Check sequence length
    if max_seq_len is not None and seq_len > max_seq_len:
        raise RevNetValidationError(
            f"Sequence length {seq_len} exceeds maximum {max_seq_len}"
        )
    
    # Check token values
    if (input_ids < 0).any():
        raise RevNetValidationError("input_ids contains negative values")
    
    if (input_ids >= vocab_size).any():
        raise RevNetValidationError(
            f"input_ids contains values >= vocab_size ({vocab_size})"
        )
    
    # Validate attention mask if provided
    if attention_mask is not None:
        attention_mask = validate_input_tensor(
            attention_mask, "attention_mask",
            expected_shape=(batch_size, seq_len),
            expected_dtype=torch.bool
        )
    
    return input_ids, attention_mask


def validate_memory_config(
    batch_size: int,
    seq_len: int,
    d_model: int,
    num_layers: int,
    available_memory: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Validate memory configuration and provide warnings.
    
    Args:
        batch_size: Batch size
        seq_len: Sequence length
        d_model: Model dimension
        num_layers: Number of layers
        available_memory: Available memory in bytes
        
    Returns:
        Memory analysis and warnings
        
    Raises:
        RevNetMemoryError: If memory requirements exceed available memory
    """
    if batch_size <= 0:
        raise RevNetValidationError("batch_size must be positive")
    
    if seq_len <= 0:
        raise RevNetValidationError("seq_len must be positive")
    
    if d_model <= 0:
        raise RevNetValidationError("d_model must be positive")
    
    if num_layers <= 0:
        raise RevNetValidationError("num_layers must be positive")
    
    # Estimate memory requirements
    # Rough estimation: attention is O(seq_len^2), activations are O(seq_len * d_model)
    attention_memory = batch_size * seq_len * seq_len * num_layers * 4  # float32
    activation_memory = batch_size * seq_len * d_model * num_layers * 4
    total_estimated = attention_memory + activation_memory
    
    warnings_list = []
    
    # Generate warnings
    if seq_len > 8192:
        warnings_list.append(f"Long sequence length ({seq_len}) may require significant memory")
    
    if batch_size > 16:
        warnings_list.append(f"Large batch size ({batch_size}) may cause memory issues")
    
    if total_estimated > 8 * 1024**3:  # 8GB
        warnings_list.append("Estimated memory usage > 8GB, consider using reversible mode")
    
    # Check against available memory
    if available_memory is not None:
        if total_estimated > available_memory * 0.8:  # 80% threshold
            raise RevNetMemoryError(
                f"Estimated memory ({total_estimated / 1e9:.2f}GB) exceeds "
                f"available memory ({available_memory / 1e9:.2f}GB)"
            )
    
    return {
        "estimated_memory": total_estimated,
        "attention_memory": attention_memory,
        "activation_memory": activation_memory,
        "warnings": warnings_list,
    }


def safe_forward_pass(func):
    """
    Decorator for safe forward pass execution with error handling.
    
    Args:
        func: Forward function to wrap
        
    Returns:
        Wrapped function with error handling
    """
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        try:
            return func(self, *args, **kwargs)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                # Convert OOM error to more helpful message
                raise RevNetMemoryError(
                    f"GPU out of memory during forward pass. "
                    f"Try reducing batch size or sequence length. Original error: {e}"
                )
            else:
                # Re-raise other runtime errors
                raise RevNetValidationError(f"Forward pass failed: {e}")
        except Exception as e:
            # Wrap unexpected errors
            raise RevNetValidationError(f"Unexpected error in forward pass: {e}")
    
    return wrapper


def validate_checkpoint_compatibility(
    checkpoint: Dict[str, Any],
    model_config: Dict[str, Any],
) -> Tuple[bool, List[str]]:
    """
    Validate checkpoint compatibility with model configuration.
    
    Args:
        checkpoint: Model checkpoint dictionary
        model_config: Model configuration
        
    Returns:
        (is_compatible, list_of_issues)
    """
    issues = []
    
    # Check if checkpoint contains required keys
    required_keys = ["model_state_dict"]
    for key in required_keys:
        if key not in checkpoint:
            issues.append(f"Missing required key in checkpoint: {key}")
    
    if "model_state_dict" not in checkpoint:
        return False, issues
    
    state_dict = checkpoint["model_state_dict"]
    
    # Check for key patterns that indicate model architecture
    expected_patterns = [
        "token_embedding.weight",
        "layers.0.attention",
        "layers.0.feed_forward",
        "final_layer_norm.weight",
        "output_projection.weight",
    ]
    
    for pattern in expected_patterns:
        matching_keys = [k for k in state_dict.keys() if pattern in k]
        if not matching_keys:
            issues.append(f"No keys matching pattern '{pattern}' found in checkpoint")
    
    # Check layer count consistency
    layer_keys = [k for k in state_dict.keys() if k.startswith("layers.")]
    if layer_keys:
        max_layer_idx = max(int(k.split(".")[1]) for k in layer_keys if k.split(".")[1].isdigit())
        if max_layer_idx + 1 != model_config.get("num_layers", 0):
            issues.append(
                f"Checkpoint has {max_layer_idx + 1} layers, "
                f"model config specifies {model_config.get('num_layers', 0)}"
            )
    
    # Check embedding dimension
    if "token_embedding.weight" in state_dict:
        emb_shape = state_dict["token_embedding.weight"].shape
        if len(emb_shape) >= 2:
            vocab_size, d_model = emb_shape
            if vocab_size != model_config.get("vocab_size", 0):
                issues.append(
                    f"Checkpoint vocab_size ({vocab_size}) != config vocab_size ({model_config.get('vocab_size', 0)})"
                )
            if d_model != model_config.get("d_model", 0):
                issues.append(
                    f"Checkpoint d_model ({d_model}) != config d_model ({model_config.get('d_model', 0)})"
                )
    
    is_compatible = len(issues) == 0
    return is_compatible, issues


def create_validation_report(
    model_config: Dict[str, Any],
    input_shape: Tuple[int, int],
    device: torch.device,
) -> Dict[str, Any]:
    """
    Create comprehensive validation report.
    
    Args:
        model_config: Model configuration
        input_shape: (batch_size, seq_len)
        device: Target device
        
    Returns:
        Validation report
    """
    batch_size, seq_len = input_shape
    
    report = {
        "timestamp": torch.datetime.now().isoformat() if hasattr(torch, 'datetime') else "unknown",
        "model_config": model_config,
        "input_shape": input_shape,
        "device": str(device),
        "validation_status": "passed",
        "warnings": [],
        "errors": [],
        "recommendations": [],
    }
    
    try:
        # Validate config
        validated_config = validate_model_config(model_config)
        report["validated_config"] = validated_config
        
        # Memory analysis
        available_memory = None
        if device.type == "cuda":
            try:
                available_memory = torch.cuda.get_device_properties(device).total_memory
            except:
                pass
        
        memory_analysis = validate_memory_config(
            batch_size, seq_len, 
            model_config["d_model"], model_config["num_layers"],
            available_memory
        )
        
        report["memory_analysis"] = memory_analysis
        report["warnings"].extend(memory_analysis["warnings"])
        
        # Generate recommendations
        if seq_len > 4096:
            report["recommendations"].append("Consider using gradient checkpointing for long sequences")
        
        if memory_analysis["estimated_memory"] > 4 * 1024**3:  # 4GB
            report["recommendations"].append("Enable reversible mode to reduce memory usage")
        
        if batch_size == 1:
            report["recommendations"].append("Increase batch size for better GPU utilization")
        
    except (RevNetValidationError, RevNetConfigurationError, RevNetMemoryError) as e:
        report["validation_status"] = "failed"
        report["errors"].append(str(e))
    except Exception as e:
        report["validation_status"] = "error"
        report["errors"].append(f"Unexpected validation error: {e}")
    
    return report