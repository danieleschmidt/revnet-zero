"""
Comprehensive error handling and validation for RevNet-Zero.

This module provides robust error handling, custom exceptions,
and validation utilities for all RevNet-Zero components.
"""

import sys
import traceback
import logging
from typing import Any, Dict, Optional, Union, List, Callable
from functools import wraps
from dataclasses import dataclass
from enum import Enum


class RevNetZeroError(Exception):
    """Base exception for all RevNet-Zero errors."""
    pass


class ModelConfigurationError(RevNetZeroError):
    """Raised when model configuration is invalid."""
    pass


class MemoryError(RevNetZeroError):
    """Raised when memory-related operations fail."""
    pass


class ValidationError(RevNetZeroError):
    """Raised when input validation fails."""
    pass


class TrainingError(RevNetZeroError):
    """Raised when training operations fail."""
    pass


class ConvergenceError(RevNetZeroError):
    """Raised when model fails to converge."""
    pass


class HardwareError(RevNetZeroError):
    """Raised when hardware-related issues occur."""
    pass


class ErrorSeverity(Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class ErrorContext:
    """Context information for errors."""
    operation: str
    parameters: Dict[str, Any]
    timestamp: float
    stack_trace: str
    severity: ErrorSeverity
    recovery_suggestions: List[str]


class RevNetZeroErrorHandler:
    """Centralized error handling and recovery system."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or logging.getLogger(__name__)
        self.error_history: List[ErrorContext] = []
        self.recovery_strategies: Dict[type, Callable] = {}
        
    def register_recovery_strategy(self, exception_type: type, strategy: Callable):
        """Register a recovery strategy for a specific exception type."""
        self.recovery_strategies[exception_type] = strategy
    
    def handle_error(self, error: Exception, context: Dict[str, Any], 
                    severity: ErrorSeverity = ErrorSeverity.MEDIUM) -> Optional[Any]:
        """
        Handle an error with appropriate logging and recovery.
        
        Args:
            error: The exception that occurred
            context: Context information about the error
            severity: Severity level of the error
            
        Returns:
            Recovery result if successful, None otherwise
        """
        import time
        
        # Create error context
        error_ctx = ErrorContext(
            operation=context.get('operation', 'unknown'),
            parameters=context.get('parameters', {}),
            timestamp=time.time(),
            stack_trace=traceback.format_exc(),
            severity=severity,
            recovery_suggestions=self._get_recovery_suggestions(error)
        )
        
        # Log error
        self._log_error(error, error_ctx)
        
        # Store in history
        self.error_history.append(error_ctx)
        
        # Attempt recovery
        if type(error) in self.recovery_strategies:
            try:
                return self.recovery_strategies[type(error)](error, context)
            except Exception as recovery_error:
                self.logger.error(f"Recovery failed: {recovery_error}")
        
        # Re-raise if no recovery possible
        if severity == ErrorSeverity.CRITICAL:
            raise error
            
        return None
    
    def _log_error(self, error: Exception, context: ErrorContext):
        """Log error with appropriate level based on severity."""
        log_message = f"Operation '{context.operation}' failed: {error}"
        
        if context.severity == ErrorSeverity.LOW:
            self.logger.info(log_message)
        elif context.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_message)
        elif context.severity == ErrorSeverity.HIGH:
            self.logger.error(log_message)
        else:  # CRITICAL
            self.logger.critical(log_message)
            
        # Log recovery suggestions
        if context.recovery_suggestions:
            self.logger.info("Recovery suggestions:")
            for suggestion in context.recovery_suggestions:
                self.logger.info(f"  - {suggestion}")
    
    def _get_recovery_suggestions(self, error: Exception) -> List[str]:
        """Generate recovery suggestions based on error type."""
        suggestions = []
        
        if isinstance(error, MemoryError):
            suggestions.extend([
                "Try reducing batch size",
                "Use more aggressive memory scheduling",
                "Enable CPU offloading",
                "Reduce sequence length"
            ])
        elif isinstance(error, ModelConfigurationError):
            suggestions.extend([
                "Check model configuration parameters",
                "Ensure d_model is divisible by num_heads",
                "Verify vocab_size is positive",
                "Check max_seq_len is reasonable"
            ])
        elif isinstance(error, ValidationError):
            suggestions.extend([
                "Validate input tensor shapes",
                "Check for NaN or infinite values",
                "Ensure inputs are on correct device",
                "Verify data type compatibility"
            ])
        elif isinstance(error, TrainingError):
            suggestions.extend([
                "Reduce learning rate",
                "Enable gradient clipping",
                "Check loss function implementation",
                "Verify optimizer configuration"
            ])
        elif isinstance(error, HardwareError):
            suggestions.extend([
                "Check CUDA availability",
                "Verify GPU memory",
                "Update PyTorch/CUDA drivers",
                "Try CPU fallback"
            ])
        else:
            suggestions.extend([
                "Check input parameters",
                "Review configuration",
                "Enable debug logging",
                "Consult documentation"
            ])
            
        return suggestions


# Global error handler instance
_global_error_handler = RevNetZeroErrorHandler()


def safe_execute(operation: str, severity: ErrorSeverity = ErrorSeverity.MEDIUM):
    """
    Decorator for safe execution with error handling.
    
    Args:
        operation: Name of the operation for logging
        severity: Error severity level
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                context = {
                    'operation': operation,
                    'parameters': {'args': str(args)[:200], 'kwargs': str(kwargs)[:200]},
                    'function': func.__name__
                }
                
                return _global_error_handler.handle_error(e, context, severity)
                
        return wrapper
    return decorator


def validate_tensor_input(tensor, name: str, expected_shape: Optional[tuple] = None,
                         expected_dtype=None, device=None) -> bool:
    """
    Validate tensor input with comprehensive checks.
    
    Args:
        tensor: Input tensor to validate
        name: Name of tensor for error messages
        expected_shape: Expected tensor shape (None for any shape)
        expected_dtype: Expected data type
        device: Expected device
        
    Returns:
        True if validation passes
        
    Raises:
        ValidationError: If validation fails
    """
    try:
        import torch
        
        # Check if tensor
        if not isinstance(tensor, torch.Tensor):
            raise ValidationError(f"{name} must be a torch.Tensor, got {type(tensor)}")
        
        # Check shape
        if expected_shape is not None:
            if tensor.shape != expected_shape:
                raise ValidationError(
                    f"{name} shape mismatch: expected {expected_shape}, got {tensor.shape}"
                )
        
        # Check data type
        if expected_dtype is not None:
            if tensor.dtype != expected_dtype:
                raise ValidationError(
                    f"{name} dtype mismatch: expected {expected_dtype}, got {tensor.dtype}"
                )
        
        # Check device
        if device is not None:
            if tensor.device != device:
                raise ValidationError(
                    f"{name} device mismatch: expected {device}, got {tensor.device}"
                )
        
        # Check for NaN or infinite values
        if torch.isnan(tensor).any():
            raise ValidationError(f"{name} contains NaN values")
        
        if torch.isinf(tensor).any():
            raise ValidationError(f"{name} contains infinite values")
            
        return True
        
    except ImportError:
        # Mock environment - skip validation
        return True
    except Exception as e:
        raise ValidationError(f"Validation failed for {name}: {e}")


def validate_model_config(config: Dict[str, Any]) -> bool:
    """
    Validate model configuration parameters.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        True if validation passes
        
    Raises:
        ModelConfigurationError: If configuration is invalid
    """
    required_fields = ['d_model', 'num_heads', 'num_layers', 'vocab_size']
    
    # Check required fields
    for field in required_fields:
        if field not in config:
            raise ModelConfigurationError(f"Required field '{field}' missing from config")
    
    # Validate individual fields
    d_model = config['d_model']
    num_heads = config['num_heads']
    num_layers = config['num_layers']
    vocab_size = config['vocab_size']
    
    if not isinstance(d_model, int) or d_model <= 0:
        raise ModelConfigurationError(f"d_model must be positive integer, got {d_model}")
    
    if not isinstance(num_heads, int) or num_heads <= 0:
        raise ModelConfigurationError(f"num_heads must be positive integer, got {num_heads}")
    
    if d_model % num_heads != 0:
        raise ModelConfigurationError(
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        )
    
    if not isinstance(num_layers, int) or num_layers <= 0:
        raise ModelConfigurationError(f"num_layers must be positive integer, got {num_layers}")
    
    if not isinstance(vocab_size, int) or vocab_size <= 0:
        raise ModelConfigurationError(f"vocab_size must be positive integer, got {vocab_size}")
    
    # Optional field validation
    if 'max_seq_len' in config:
        max_seq_len = config['max_seq_len']
        if not isinstance(max_seq_len, int) or max_seq_len <= 0:
            raise ModelConfigurationError(
                f"max_seq_len must be positive integer, got {max_seq_len}"
            )
    
    if 'dropout' in config:
        dropout = config['dropout']
        if not isinstance(dropout, (int, float)) or not (0 <= dropout <= 1):
            raise ModelConfigurationError(
                f"dropout must be float between 0 and 1, got {dropout}"
            )
    
    return True


def memory_safety_check(memory_usage_gb: float, available_memory_gb: float,
                       safety_margin: float = 0.1) -> bool:
    """
    Check if memory usage is within safe limits.
    
    Args:
        memory_usage_gb: Required memory in GB
        available_memory_gb: Available memory in GB  
        safety_margin: Safety margin (0.1 = 10% buffer)
        
    Returns:
        True if memory usage is safe
        
    Raises:
        MemoryError: If memory usage is unsafe
    """
    safe_limit = available_memory_gb * (1 - safety_margin)
    
    if memory_usage_gb > safe_limit:
        raise MemoryError(
            f"Memory usage ({memory_usage_gb:.1f} GB) exceeds safe limit "
            f"({safe_limit:.1f} GB). Available: {available_memory_gb:.1f} GB"
        )
    
    return True


def hardware_compatibility_check() -> Dict[str, Any]:
    """
    Check hardware compatibility and capabilities.
    
    Returns:
        Dictionary with hardware information and capabilities
    """
    hardware_info = {
        'cuda_available': False,
        'cuda_version': None,
        'gpu_count': 0,
        'gpu_memory_gb': 0,
        'cpu_cores': 1,
        'system_memory_gb': 8,
        'warnings': []
    }
    
    try:
        import torch
        
        hardware_info['cuda_available'] = torch.cuda.is_available()
        
        if hardware_info['cuda_available']:
            hardware_info['cuda_version'] = torch.version.cuda
            hardware_info['gpu_count'] = torch.cuda.device_count()
            
            if hardware_info['gpu_count'] > 0:
                hardware_info['gpu_memory_gb'] = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        # CPU information
        try:
            import psutil
            hardware_info['cpu_cores'] = psutil.cpu_count()
            hardware_info['system_memory_gb'] = psutil.virtual_memory().total / 1e9
        except ImportError:
            hardware_info['warnings'].append("psutil not available for detailed system info")
        
    except ImportError:
        hardware_info['warnings'].append("PyTorch not available - using mock environment")
    
    # Add recommendations
    if hardware_info['gpu_memory_gb'] < 8:
        hardware_info['warnings'].append(
            "GPU memory < 8GB: consider aggressive memory scheduling"
        )
    
    if hardware_info['system_memory_gb'] < 16:
        hardware_info['warnings'].append(
            "System memory < 16GB: may limit large model training"
        )
    
    return hardware_info


class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance."""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = 'closed'  # closed, open, half_open
        
    def __call__(self, func: Callable) -> Callable:
        """Decorator to apply circuit breaker pattern."""
        @wraps(func)
        def wrapper(*args, **kwargs):
            import time
            
            current_time = time.time()
            
            # Check if circuit should move from open to half-open
            if (self.state == 'open' and 
                current_time - self.last_failure_time > self.recovery_timeout):
                self.state = 'half_open'
                self.failure_count = 0
            
            # Fail fast if circuit is open
            if self.state == 'open':
                raise HardwareError("Circuit breaker is open - system temporarily unavailable")
            
            try:
                result = func(*args, **kwargs)
                
                # Success - reset failure count if we were in half-open state
                if self.state == 'half_open':
                    self.state = 'closed'
                    self.failure_count = 0
                
                return result
                
            except Exception as e:
                self.failure_count += 1
                self.last_failure_time = current_time
                
                # Open circuit if threshold exceeded
                if self.failure_count >= self.failure_threshold:
                    self.state = 'open'
                
                raise e
        
        return wrapper


# Pre-configured circuit breakers for common operations
memory_operation_breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=30)
model_operation_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
training_operation_breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=120)


def get_error_summary() -> Dict[str, Any]:
    """Get summary of errors encountered during session."""
    return {
        'total_errors': len(_global_error_handler.error_history),
        'error_types': {},
        'severity_counts': {},
        'recent_errors': _global_error_handler.error_history[-5:] if _global_error_handler.error_history else []
    }


# Export main components
__all__ = [
    'RevNetZeroError', 'ModelConfigurationError', 'MemoryError', 'ValidationError',
    'TrainingError', 'ConvergenceError', 'HardwareError', 'ErrorSeverity',
    'RevNetZeroErrorHandler', 'safe_execute', 'validate_tensor_input', 
    'validate_model_config', 'memory_safety_check', 'hardware_compatibility_check',
    'CircuitBreaker', 'get_error_summary'
]