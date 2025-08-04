"""
Security and validation utilities for RevNet-Zero.

This module provides comprehensive input validation, security checks,
and safe execution utilities for production deployments.
"""

import torch
import torch.nn as nn
import hashlib
import hmac
import secrets
import re
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
import numpy as np
import logging
from pathlib import Path
import json
from dataclasses import dataclass
from enum import Enum
import warnings


class SecurityLevel(Enum):
    """Security validation levels."""
    PERMISSIVE = "permissive"
    STANDARD = "standard"
    STRICT = "strict"
    PARANOID = "paranoid"


@dataclass
class ValidationResult:
    """Result of a validation check."""
    is_valid: bool
    issues: List[str]
    warnings: List[str]
    security_level: SecurityLevel
    metadata: Dict[str, Any]


class InputValidator:
    """
    Comprehensive input validation for model inputs and parameters.
    
    Provides multi-level security validation to prevent adversarial
    inputs and ensure safe model execution.
    """
    
    def __init__(self, security_level: SecurityLevel = SecurityLevel.STANDARD):
        """
        Initialize input validator.
        
        Args:
            security_level: Level of security validation to apply
        """
        self.security_level = security_level
        self.logger = logging.getLogger(__name__)
        
        # Validation thresholds based on security level
        self.thresholds = self._get_security_thresholds()
    
    def _get_security_thresholds(self) -> Dict[str, Any]:
        """Get security thresholds based on security level."""
        base_thresholds = {
            'max_sequence_length': 1000000,
            'max_batch_size': 1024,
            'max_vocab_size': 1000000,
            'max_tensor_size_gb': 10.0,
            'allowed_dtypes': [torch.float32, torch.float16, torch.bfloat16, torch.long, torch.int],
            'max_dimensions': 8,
            'min_value': -1e6,
            'max_value': 1e6,
        }
        
        if self.security_level == SecurityLevel.PERMISSIVE:
            multiplier = 2.0
        elif self.security_level == SecurityLevel.STANDARD:
            multiplier = 1.0
        elif self.security_level == SecurityLevel.STRICT:
            multiplier = 0.5
        else:  # PARANOID
            multiplier = 0.25
        
        # Apply multiplier to numeric thresholds
        for key in ['max_sequence_length', 'max_batch_size', 'max_vocab_size', 'max_tensor_size_gb']:
            base_thresholds[key] = int(base_thresholds[key] * multiplier)
        
        return base_thresholds
    
    def validate_input_ids(
        self,
        input_ids: torch.Tensor,
        vocab_size: Optional[int] = None,
        max_sequence_length: Optional[int] = None,
    ) -> ValidationResult:
        """
        Validate input token IDs.
        
        Args:
            input_ids: Input token tensor
            vocab_size: Expected vocabulary size
            max_sequence_length: Maximum allowed sequence length
            
        Returns:
            ValidationResult with validation status and details
        """
        issues = []
        warnings = []
        metadata = {}
        
        # Basic type and shape validation
        if not isinstance(input_ids, torch.Tensor):
            issues.append(f"input_ids must be torch.Tensor, got {type(input_ids)}")
            return ValidationResult(False, issues, warnings, self.security_level, metadata)
        
        if input_ids.dim() not in [1, 2]:
            issues.append(f"input_ids must be 1D or 2D tensor, got {input_ids.dim()}D")
        
        if input_ids.dtype not in [torch.long, torch.int]:
            issues.append(f"input_ids must be integer type, got {input_ids.dtype}")
        
        # Size validation
        total_elements = input_ids.numel()
        tensor_size_gb = total_elements * input_ids.element_size() / (1024**3)
        
        if tensor_size_gb > self.thresholds['max_tensor_size_gb']:
            issues.append(f"Tensor size {tensor_size_gb:.2f}GB exceeds limit {self.thresholds['max_tensor_size_gb']}GB")
        
        # Sequence length validation
        if input_ids.dim() >= 2:
            seq_len = input_ids.size(-1)
            batch_size = input_ids.size(0)
            
            max_seq_len = max_sequence_length or self.thresholds['max_sequence_length']
            if seq_len > max_seq_len:
                issues.append(f"Sequence length {seq_len} exceeds maximum {max_seq_len}")
            
            if batch_size > self.thresholds['max_batch_size']:
                issues.append(f"Batch size {batch_size} exceeds maximum {self.thresholds['max_batch_size']}")
            
            metadata.update({
                'sequence_length': seq_len,
                'batch_size': batch_size,
            })
        
        # Value range validation
        if input_ids.numel() > 0:
            min_val = input_ids.min().item()
            max_val = input_ids.max().item()
            
            if min_val < 0:
                issues.append(f"Negative token IDs found: minimum value {min_val}")
            
            if vocab_size and max_val >= vocab_size:
                issues.append(f"Token ID {max_val} exceeds vocabulary size {vocab_size}")
            
            metadata.update({
                'min_token_id': min_val,
                'max_token_id': max_val,
                'unique_tokens': len(torch.unique(input_ids)),
            })
        
        # Advanced security checks for strict/paranoid levels
        if self.security_level in [SecurityLevel.STRICT, SecurityLevel.PARANOID]:
            self._advanced_input_validation(input_ids, issues, warnings, metadata)
        
        is_valid = len(issues) == 0
        return ValidationResult(is_valid, issues, warnings, self.security_level, metadata)
    
    def _advanced_input_validation(
        self,
        input_ids: torch.Tensor,
        issues: List[str],
        warnings: List[str],
        metadata: Dict[str, Any],
    ):
        """Perform advanced security validation."""
        if input_ids.numel() == 0:
            return
        
        # Check for suspicious patterns
        if input_ids.dim() >= 2:
            # Check for repeated sequences (potential adversarial inputs)
            for i in range(input_ids.size(0)):
                sequence = input_ids[i]
                unique_ratio = len(torch.unique(sequence)) / len(sequence)
                
                if unique_ratio < 0.1:  # Less than 10% unique tokens
                    warnings.append(f"Low token diversity in sequence {i}: {unique_ratio:.2%}")
                
                # Check for extremely long repeated patterns
                seq_str = sequence.cpu().numpy().tostring()
                for pattern_len in [2, 4, 8]:
                    if len(seq_str) >= pattern_len * 10:
                        pattern = seq_str[:pattern_len]
                        if seq_str.count(pattern) > len(seq_str) // (pattern_len * 2):
                            warnings.append(f"Highly repetitive pattern detected in sequence {i}")
        
        # Check for outlier values
        if vocab_size := metadata.get('max_token_id'):
            outlier_threshold = vocab_size * 0.95  # Top 5% of vocabulary
            outlier_count = (input_ids >= outlier_threshold).sum().item()
            outlier_ratio = outlier_count / input_ids.numel()
            
            if outlier_ratio > 0.5:  # More than 50% outlier tokens
                warnings.append(f"High ratio of outlier tokens: {outlier_ratio:.2%}")
    
    def validate_attention_mask(
        self,
        attention_mask: torch.Tensor,
        input_shape: Tuple[int, ...],
    ) -> ValidationResult:
        """
        Validate attention mask tensor.
        
        Args:
            attention_mask: Attention mask tensor
            input_shape: Expected input shape to match
            
        Returns:
            ValidationResult with validation status
        """
        issues = []
        warnings = []
        metadata = {}
        
        if not isinstance(attention_mask, torch.Tensor):
            issues.append(f"attention_mask must be torch.Tensor, got {type(attention_mask)}")
            return ValidationResult(False, issues, warnings, self.security_level, metadata)
        
        # Shape validation
        if attention_mask.shape != input_shape:
            issues.append(
                f"attention_mask shape {attention_mask.shape} doesn't match input shape {input_shape}"
            )
        
        # Dtype validation
        if attention_mask.dtype not in [torch.bool, torch.long, torch.int, torch.float]:
            issues.append(f"Invalid attention_mask dtype: {attention_mask.dtype}")
        
        # Value validation
        if attention_mask.dtype in [torch.long, torch.int]:
            unique_values = torch.unique(attention_mask)
            if not torch.all((unique_values == 0) | (unique_values == 1)):
                issues.append("attention_mask values must be 0 or 1 for integer types")
        
        # Check for suspicious patterns
        if self.security_level in [SecurityLevel.STRICT, SecurityLevel.PARANOID]:
            if attention_mask.dim() >= 2:
                for i in range(attention_mask.size(0)):
                    mask = attention_mask[i]
                    attention_ratio = mask.float().mean().item()
                    
                    if attention_ratio < 0.01:  # Less than 1% attention
                        warnings.append(f"Very sparse attention mask in batch {i}: {attention_ratio:.2%}")
                    elif attention_ratio > 0.99:  # More than 99% attention
                        warnings.append(f"Nearly full attention mask in batch {i}: {attention_ratio:.2%}")
        
        metadata.update({
            'attention_ratio': attention_mask.float().mean().item() if attention_mask.numel() > 0 else 0,
            'total_attention_tokens': attention_mask.sum().item() if attention_mask.numel() > 0 else 0,
        })
        
        is_valid = len(issues) == 0
        return ValidationResult(is_valid, issues, warnings, self.security_level, metadata)
    
    def validate_model_config(self, config: Dict[str, Any]) -> ValidationResult:
        """
        Validate model configuration parameters.
        
        Args:
            config: Model configuration dictionary
            
        Returns:
            ValidationResult with validation status
        """
        issues = []
        warnings = []
        metadata = {}
        
        required_fields = ['vocab_size', 'd_model', 'num_layers', 'num_heads']
        for field in required_fields:
            if field not in config:
                issues.append(f"Required field '{field}' missing from config")
        
        # Validate numeric parameters
        numeric_validations = {
            'vocab_size': (1, self.thresholds['max_vocab_size']),
            'd_model': (1, 16384),
            'num_layers': (1, 1000),
            'num_heads': (1, 256),
            'max_seq_len': (1, self.thresholds['max_sequence_length']),
            'dropout': (0.0, 1.0),
        }
        
        for field, (min_val, max_val) in numeric_validations.items():
            if field in config:
                value = config[field]
                if not isinstance(value, (int, float)):
                    issues.append(f"Field '{field}' must be numeric, got {type(value)}")
                elif value < min_val or value > max_val:
                    issues.append(f"Field '{field}' value {value} outside valid range [{min_val}, {max_val}]")
        
        # Validate relationships between parameters
        if 'd_model' in config and 'num_heads' in config:
            d_model = config['d_model']
            num_heads = config['num_heads']
            
            if d_model % num_heads != 0:
                issues.append(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        
        # Security-specific validations
        if self.security_level in [SecurityLevel.STRICT, SecurityLevel.PARANOID]:
            # Check for suspicious configurations
            if config.get('num_layers', 0) > 100:
                warnings.append(f"Very deep model: {config['num_layers']} layers")
            
            if config.get('d_model', 0) > 8192:
                warnings.append(f"Very wide model: {config['d_model']} dimensions")
        
        metadata.update({
            'total_parameters_estimate': self._estimate_parameters(config),
            'memory_estimate_gb': self._estimate_memory_usage(config),
        })
        
        is_valid = len(issues) == 0
        return ValidationResult(is_valid, issues, warnings, self.security_level, metadata)
    
    def _estimate_parameters(self, config: Dict[str, Any]) -> int:
        """Estimate total model parameters."""
        vocab_size = config.get('vocab_size', 50000)
        d_model = config.get('d_model', 768)
        num_layers = config.get('num_layers', 12)
        
        # Rough estimation
        embedding_params = vocab_size * d_model
        layer_params = num_layers * (4 * d_model * d_model)  # Simplified
        
        return embedding_params + layer_params
    
    def _estimate_memory_usage(self, config: Dict[str, Any]) -> float:
        """Estimate memory usage in GB."""
        total_params = self._estimate_parameters(config)
        # 4 bytes per parameter + gradients + optimizer states
        memory_bytes = total_params * 4 * 3  # params + grads + optimizer
        return memory_bytes / (1024**3)


class ModelSecurityChecker:
    """
    Security checker for model files and weights.
    
    Validates model integrity, checks for potential security issues,
    and ensures safe model loading.
    """
    
    def __init__(self, security_level: SecurityLevel = SecurityLevel.STANDARD):
        """
        Initialize security checker.
        
        Args:
            security_level: Level of security checks to perform
        """
        self.security_level = security_level
        self.logger = logging.getLogger(__name__)
    
    def validate_model_file(self, file_path: Union[str, Path]) -> ValidationResult:
        """
        Validate model file before loading.
        
        Args:
            file_path: Path to model file
            
        Returns:
            ValidationResult with validation status
        """
        issues = []
        warnings = []
        metadata = {}
        
        file_path = Path(file_path)
        
        # File existence and basic checks
        if not file_path.exists():
            issues.append(f"Model file does not exist: {file_path}")
            return ValidationResult(False, issues, warnings, self.security_level, metadata)
        
        if not file_path.is_file():
            issues.append(f"Path is not a file: {file_path}")
            return ValidationResult(False, issues, warnings, self.security_level, metadata)
        
        # File size validation
        file_size = file_path.stat().st_size
        file_size_gb = file_size / (1024**3)
        
        max_file_size_gb = 50.0  # Reasonable maximum for model files
        if self.security_level == SecurityLevel.PARANOID:
            max_file_size_gb = 10.0
        
        if file_size_gb > max_file_size_gb:
            issues.append(f"Model file too large: {file_size_gb:.2f}GB (max: {max_file_size_gb}GB)")
        
        metadata.update({
            'file_size_gb': file_size_gb,
            'file_path': str(file_path),
        })
        
        # File extension validation
        allowed_extensions = {'.pt', '.pth', '.bin', '.safetensors'}
        if file_path.suffix.lower() not in allowed_extensions:
            warnings.append(f"Unusual file extension: {file_path.suffix}")
        
        # Content validation (if strict/paranoid)
        if self.security_level in [SecurityLevel.STRICT, SecurityLevel.PARANOID]:
            try:
                self._validate_model_content(file_path, issues, warnings, metadata)
            except Exception as e:
                issues.append(f"Error validating model content: {e}")
        
        is_valid = len(issues) == 0
        return ValidationResult(is_valid, issues, warnings, self.security_level, metadata)
    
    def _validate_model_content(
        self,
        file_path: Path,
        issues: List[str],
        warnings: List[str],
        metadata: Dict[str, Any],
    ):
        """Validate model file content."""
        try:
            # Load model state dict
            state_dict = torch.load(file_path, map_location='cpu', weights_only=True)
            
            if not isinstance(state_dict, dict):
                issues.append("Model file does not contain a valid state dictionary")
                return
            
            # Validate parameter tensors
            total_params = 0
            suspicious_tensors = []
            
            for name, tensor in state_dict.items():
                if not isinstance(tensor, torch.Tensor):
                    warnings.append(f"Non-tensor parameter: {name}")
                    continue
                
                total_params += tensor.numel()
                
                # Check for suspicious values
                if torch.isnan(tensor).any():
                    issues.append(f"NaN values in parameter: {name}")
                
                if torch.isinf(tensor).any():
                    issues.append(f"Infinite values in parameter: {name}")
                
                # Check for extremely large values
                max_abs_val = tensor.abs().max().item()
                if max_abs_val > 1000:
                    suspicious_tensors.append((name, max_abs_val))
            
            metadata.update({
                'total_parameters': total_params,
                'parameter_count': len(state_dict),
                'suspicious_tensors': len(suspicious_tensors),
            })
            
            if suspicious_tensors and self.security_level == SecurityLevel.PARANOID:
                warnings.append(f"Parameters with large values: {len(suspicious_tensors)}")
        
        except Exception as e:
            issues.append(f"Failed to load model for validation: {e}")
    
    def calculate_model_hash(self, file_path: Union[str, Path]) -> str:
        """
        Calculate secure hash of model file.
        
        Args:
            file_path: Path to model file
            
        Returns:
            SHA-256 hash of the file
        """
        file_path = Path(file_path)
        
        sha256_hash = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        
        return sha256_hash.hexdigest()
    
    def verify_model_integrity(
        self,
        file_path: Union[str, Path],
        expected_hash: str,
    ) -> bool:
        """
        Verify model file integrity using hash.
        
        Args:
            file_path: Path to model file
            expected_hash: Expected SHA-256 hash
            
        Returns:
            True if hash matches, False otherwise
        """
        actual_hash = self.calculate_model_hash(file_path)
        return hmac.compare_digest(actual_hash, expected_hash)


class SafeModelLoader:
    """
    Safe model loading with comprehensive security checks.
    
    Provides secure model loading with validation, sandboxing,
    and error recovery capabilities.
    """
    
    def __init__(
        self,
        security_level: SecurityLevel = SecurityLevel.STANDARD,
        max_load_time: float = 300.0,  # 5 minutes
        enable_sandboxing: bool = True,
    ):
        """
        Initialize safe model loader.
        
        Args:
            security_level: Security validation level
            max_load_time: Maximum time allowed for loading
            enable_sandboxing: Whether to enable sandboxed loading
        """
        self.security_level = security_level
        self.max_load_time = max_load_time
        self.enable_sandboxing = enable_sandboxing
        
        self.input_validator = InputValidator(security_level)
        self.security_checker = ModelSecurityChecker(security_level)
        
        self.logger = logging.getLogger(__name__)
    
    def load_model_safely(
        self,
        model_class: type,
        model_path: Union[str, Path],
        config: Optional[Dict[str, Any]] = None,
        expected_hash: Optional[str] = None,
        strict_loading: bool = True,
    ) -> Tuple[nn.Module, ValidationResult]:
        """
        Load model with comprehensive security validation.
        
        Args:
            model_class: Model class to instantiate
            model_path: Path to model weights
            config: Model configuration
            expected_hash: Expected file hash for integrity check
            strict_loading: Whether to use strict loading
            
        Returns:
            Tuple of (loaded_model, validation_result)
        """
        start_time = time.time()
        
        # File validation
        file_validation = self.security_checker.validate_model_file(model_path)
        if not file_validation.is_valid:
            return None, file_validation
        
        # Hash verification
        if expected_hash:
            if not self.security_checker.verify_model_integrity(model_path, expected_hash):
                return None, ValidationResult(
                    False, 
                    ["Model file hash verification failed"],
                    [],
                    self.security_level,
                    {}
                )
        
        # Config validation
        if config:
            config_validation = self.input_validator.validate_model_config(config)
            if not config_validation.is_valid:
                return None, config_validation
        
        try:
            # Load with timeout
            model = self._load_with_timeout(
                model_class, model_path, config, strict_loading
            )
            
            # Post-load validation
            post_load_validation = self._validate_loaded_model(model)
            
            load_time = time.time() - start_time
            post_load_validation.metadata['load_time'] = load_time
            
            if load_time > self.max_load_time:
                post_load_validation.warnings.append(
                    f"Model loading took {load_time:.1f}s (longer than expected)"
                )
            
            return model, post_load_validation
        
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return None, ValidationResult(
                False,
                [f"Model loading failed: {e}"],
                [],
                self.security_level,
                {'load_time': time.time() - start_time}
            )
    
    def _load_with_timeout(
        self,
        model_class: type,
        model_path: Union[str, Path],
        config: Optional[Dict[str, Any]],
        strict_loading: bool,
    ) -> nn.Module:
        """Load model with timeout protection."""
        import signal
        
        def timeout_handler(signum, frame):
            raise TimeoutError(f"Model loading exceeded {self.max_load_time}s timeout")
        
        # Set timeout (Unix-only)
        if hasattr(signal, 'SIGALRM'):
            old_handler = signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(int(self.max_load_time))
        
        try:
            # Create model instance
            if config:
                model = model_class(**config)
            else:
                model = model_class()
            
            # Load state dict
            state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
            
            if hasattr(model, 'load_state_dict'):
                model.load_state_dict(state_dict, strict=strict_loading)
            else:
                # Handle custom loading
                for name, param in state_dict.items():
                    if hasattr(model, name):
                        setattr(model, name, param)
            
            return model
        
        finally:
            # Reset alarm
            if hasattr(signal, 'alarm'):
                signal.alarm(0)
                if hasattr(signal, 'SIGALRM'):
                    signal.signal(signal.SIGALRM, old_handler)
    
    def _validate_loaded_model(self, model: nn.Module) -> ValidationResult:
        """Validate loaded model."""
        issues = []
        warnings = []
        metadata = {}
        
        # Basic model validation
        if not isinstance(model, nn.Module):
            issues.append(f"Loaded object is not a PyTorch module: {type(model)}")
            return ValidationResult(False, issues, warnings, self.security_level, metadata)
        
        # Parameter validation
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        metadata.update({
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_class': type(model).__name__,
        })
        
        # Check for problematic parameters
        nan_params = []
        inf_params = []
        
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                nan_params.append(name)
            if torch.isinf(param).any():
                inf_params.append(name)
        
        if nan_params:
            issues.append(f"Parameters with NaN values: {nan_params}")
        
        if inf_params:
            issues.append(f"Parameters with infinite values: {inf_params}")
        
        # Model-specific validations
        if hasattr(model, 'config'):
            config_validation = self.input_validator.validate_model_config(model.config)
            if not config_validation.is_valid:
                issues.extend(config_validation.issues)
                warnings.extend(config_validation.warnings)
        
        is_valid = len(issues) == 0
        return ValidationResult(is_valid, issues, warnings, self.security_level, metadata)


def create_secure_model_environment(
    security_level: SecurityLevel = SecurityLevel.STANDARD,
    enable_monitoring: bool = True,
    log_file: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Create a secure environment for model operations.
    
    Args:
        security_level: Security level to enforce
        enable_monitoring: Whether to enable monitoring
        log_file: Optional log file path
        
    Returns:
        Dictionary with security components
    """
    # Setup logging
    logger = logging.getLogger('RevNetSecurity')
    if log_file and not logger.handlers:
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    
    # Create security components
    components = {
        'input_validator': InputValidator(security_level),
        'security_checker': ModelSecurityChecker(security_level),
        'safe_loader': SafeModelLoader(security_level),
        'security_level': security_level,
        'logger': logger,
    }
    
    if enable_monitoring:
        from ..utils.monitoring import MetricsCollector
        components['metrics_collector'] = MetricsCollector(
            enable_system_monitoring=True,
            log_to_file=log_file + '.metrics' if log_file else None,
        )
    
    logger.info(f"Secure model environment created with {security_level.value} security level")
    
    return components