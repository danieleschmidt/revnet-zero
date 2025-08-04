"""
Comprehensive error handling for RevNet-Zero.

This module provides robust error handling, recovery mechanisms,
and detailed error reporting for reversible transformer operations.
"""

import torch
import torch.nn as nn
import traceback
import logging
import warnings
from typing import Optional, Dict, Any, List, Callable, Union
from functools import wraps
import inspect
from contextlib import contextmanager
import sys


class RevNetError(Exception):
    """Base exception class for RevNet-Zero errors."""
    
    def __init__(self, message: str, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.context = context or {}
        self.timestamp = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None


class ReversibilityError(RevNetError):
    """Error in reversible computation."""
    pass


class MemoryError(RevNetError):
    """Memory-related error."""
    pass


class ConfigurationError(RevNetError):
    """Configuration or setup error."""
    pass


class NumericalInstabilityError(RevNetError):
    """Numerical instability detected."""
    pass


class ErrorHandler:
    """
    Comprehensive error handler for RevNet-Zero operations.
    
    Provides error recovery, detailed logging, and diagnostic information.
    """
    
    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        recovery_enabled: bool = True,
        max_retries: int = 3,
        fallback_mode: str = 'standard',
    ):
        """
        Initialize error handler.
        
        Args:
            logger: Logger instance
            recovery_enabled: Whether to attempt error recovery
            max_retries: Maximum number of retry attempts
            fallback_mode: Fallback computation mode ('standard', 'checkpoint')
        """
        self.logger = logger or self._setup_logger()
        self.recovery_enabled = recovery_enabled
        self.max_retries = max_retries
        self.fallback_mode = fallback_mode
        
        # Error tracking
        self.error_counts = {}
        self.recovery_attempts = {}
        self.successful_recoveries = {}
        
        # Recovery strategies
        self.recovery_strategies = {
            'memory_overflow': self._recover_memory_overflow,
            'numerical_instability': self._recover_numerical_instability,
            'reversibility_failure': self._recover_reversibility_failure,
            'gradient_explosion': self._recover_gradient_explosion,
        }
    
    def _setup_logger(self) -> logging.Logger:
        """Setup default logger."""
        logger = logging.getLogger('RevNetErrorHandler')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def handle_error(
        self,
        error: Exception,
        context: Dict[str, Any],
        recovery_fn: Optional[Callable] = None,
    ) -> tuple:
        """
        Handle error with optional recovery.
        
        Args:
            error: Exception that occurred
            context: Error context information
            recovery_fn: Optional recovery function
            
        Returns:
            Tuple of (success, result, error_info)
        """
        error_type = type(error).__name__
        
        # Log error
        self._log_error(error, context)
        
        # Track error statistics
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        # Attempt recovery if enabled
        if self.recovery_enabled and recovery_fn is not None:
            return self._attempt_recovery(error, context, recovery_fn)
        
        # No recovery, return error
        return False, None, {
            'error_type': error_type,
            'message': str(error),
            'context': context,
            'traceback': traceback.format_exc(),
        }
    
    def _log_error(self, error: Exception, context: Dict[str, Any]):
        """Log error with context."""
        error_type = type(error).__name__
        
        self.logger.error(f"Error occurred: {error_type}")
        self.logger.error(f"Message: {str(error)}")
        
        if context:
            self.logger.error("Context:")
            for key, value in context.items():
                self.logger.error(f"  {key}: {value}")
        
        # Log stack trace for debugging
        self.logger.debug("Stack trace:")
        self.logger.debug(traceback.format_exc())
    
    def _attempt_recovery(
        self,
        error: Exception,
        context: Dict[str, Any],
        recovery_fn: Callable,
    ) -> tuple:
        """
        Attempt error recovery.
        
        Args:
            error: Exception to recover from
            context: Error context
            recovery_fn: Recovery function to call
            
        Returns:
            Tuple of (success, result, error_info)
        """
        error_type = type(error).__name__
        
        # Track recovery attempts
        self.recovery_attempts[error_type] = self.recovery_attempts.get(error_type, 0) + 1
        
        for attempt in range(self.max_retries):
            try:
                self.logger.info(f"Attempting recovery {attempt + 1}/{self.max_retries} for {error_type}")
                
                # Apply recovery strategy
                recovery_context = self._apply_recovery_strategy(error, context)
                
                # Retry operation with recovery context
                result = recovery_fn(**recovery_context)
                
                # Recovery successful
                self.successful_recoveries[error_type] = self.successful_recoveries.get(error_type, 0) + 1
                self.logger.info(f"Recovery successful for {error_type}")
                
                return True, result, {
                    'recovered': True,
                    'attempts': attempt + 1,
                    'strategy': recovery_context.get('strategy', 'unknown'),
                }
                
            except Exception as recovery_error:
                self.logger.warning(f"Recovery attempt {attempt + 1} failed: {recovery_error}")
                
                if attempt == self.max_retries - 1:
                    # All recovery attempts failed
                    return False, None, {
                        'error_type': error_type,
                        'original_error': str(error),
                        'recovery_error': str(recovery_error),
                        'recovery_attempts': attempt + 1,
                        'context': context,
                    }
        
        return False, None, {'error': 'Recovery failed'}
    
    def _apply_recovery_strategy(
        self,
        error: Exception,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Apply appropriate recovery strategy."""
        error_type = type(error).__name__
        
        # Detect error category
        if 'memory' in str(error).lower() or 'cuda' in str(error).lower():
            strategy = 'memory_overflow'
        elif 'nan' in str(error).lower() or 'inf' in str(error).lower():
            strategy = 'numerical_instability'
        elif 'gradient' in str(error).lower() and 'norm' in str(error).lower():
            strategy = 'gradient_explosion'
        elif isinstance(error, ReversibilityError):
            strategy = 'reversibility_failure'
        else:
            strategy = 'general'
        
        # Apply strategy
        if strategy in self.recovery_strategies:
            return self.recovery_strategies[strategy](error, context)
        else:
            return self._recover_general(error, context)
    
    def _recover_memory_overflow(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Recover from memory overflow."""
        recovery_context = context.copy()
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Reduce batch size if possible
        if 'batch_size' in context:
            recovery_context['batch_size'] = max(1, context['batch_size'] // 2)
        
        # Enable gradient checkpointing
        recovery_context['use_checkpointing'] = True
        recovery_context['strategy'] = 'memory_overflow_recovery'
        
        return recovery_context
    
    def _recover_numerical_instability(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Recover from numerical instability."""
        recovery_context = context.copy()
        
        # Reduce learning rate
        if 'learning_rate' in context:
            recovery_context['learning_rate'] = context['learning_rate'] * 0.5
        
        # Enable gradient clipping
        recovery_context['max_grad_norm'] = 0.5
        
        # Switch to higher precision if using FP16
        recovery_context['use_fp16'] = False
        recovery_context['strategy'] = 'numerical_stability_recovery'
        
        return recovery_context
    
    def _recover_reversibility_failure(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Recover from reversibility failure."""
        recovery_context = context.copy()
        
        # Disable reversible computation temporarily
        recovery_context['use_reversible'] = False
        recovery_context['strategy'] = 'reversibility_failure_recovery'
        
        return recovery_context
    
    def _recover_gradient_explosion(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """Recover from gradient explosion."""
        recovery_context = context.copy()
        
        # Enable aggressive gradient clipping
        recovery_context['max_grad_norm'] = 0.1
        
        # Reduce learning rate
        if 'learning_rate' in context:
            recovery_context['learning_rate'] = context['learning_rate'] * 0.1
        
        recovery_context['strategy'] = 'gradient_explosion_recovery'
        
        return recovery_context
    
    def _recover_general(self, error: Exception, context: Dict[str, Any]) -> Dict[str, Any]:
        """General recovery strategy."""
        recovery_context = context.copy()
        recovery_context['strategy'] = 'general_recovery'
        
        # Conservative settings
        recovery_context['use_reversible'] = False
        recovery_context['use_fp16'] = False
        
        return recovery_context
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error handling statistics."""
        return {
            'error_counts': self.error_counts.copy(),
            'recovery_attempts': self.recovery_attempts.copy(),
            'successful_recoveries': self.successful_recoveries.copy(),
            'recovery_rate': {
                error_type: (self.successful_recoveries.get(error_type, 0) / 
                           self.recovery_attempts.get(error_type, 1))
                for error_type in self.recovery_attempts
            }
        }


def robust_forward(
    error_handler: Optional[ErrorHandler] = None,
    max_retries: int = 3,
    fallback_mode: str = 'standard',
):
    """
    Decorator for robust forward pass with error handling.
    
    Args:
        error_handler: Error handler instance
        max_retries: Maximum retry attempts
        fallback_mode: Fallback computation mode
    """
    def decorator(forward_fn):
        @wraps(forward_fn)
        def wrapper(self, *args, **kwargs):
            handler = error_handler or ErrorHandler(max_retries=max_retries)
            
            # Prepare context
            context = {
                'function': forward_fn.__name__,
                'class': self.__class__.__name__,
                'args_shapes': [arg.shape if isinstance(arg, torch.Tensor) else type(arg) for arg in args],
                'kwargs': {k: v.shape if isinstance(v, torch.Tensor) else type(v) for k, v in kwargs.items()},
            }
            
            # Define recovery function
            def recovery_fn(**recovery_context):
                # Apply recovery context to kwargs
                modified_kwargs = kwargs.copy()
                for key, value in recovery_context.items():
                    if key in ['use_reversible', 'use_fp16', 'max_grad_norm']:
                        modified_kwargs[key] = value
                
                return forward_fn(self, *args, **modified_kwargs)
            
            try:
                return forward_fn(self, *args, **kwargs)
            except Exception as e:
                success, result, error_info = handler.handle_error(e, context, recovery_fn)
                
                if success:
                    return result
                else:
                    raise RevNetError(f"Forward pass failed: {error_info}")
        
        return wrapper
    return decorator


@contextmanager
def error_recovery_context(
    recovery_enabled: bool = True,
    fallback_mode: str = 'standard',
    logger: Optional[logging.Logger] = None,
):
    """
    Context manager for error recovery operations.
    
    Args:
        recovery_enabled: Whether to enable error recovery
        fallback_mode: Fallback computation mode
        logger: Logger instance
    """
    handler = ErrorHandler(
        logger=logger,
        recovery_enabled=recovery_enabled,
        fallback_mode=fallback_mode,
    )
    
    try:
        yield handler
    except Exception as e:
        if recovery_enabled:
            # Attempt one final recovery
            handler.logger.error(f"Critical error in recovery context: {e}")
        raise


def validate_inputs(
    input_ids: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
    max_seq_len: int = 1000000,
    vocab_size: int = 50257,
) -> Dict[str, Any]:
    """
    Validate model inputs and provide detailed error information.
    
    Args:
        input_ids: Input token IDs
        attention_mask: Attention mask
        labels: Labels for loss computation
        max_seq_len: Maximum sequence length
        vocab_size: Vocabulary size
        
    Returns:
        Validation results
    """
    issues = []
    warnings_list = []
    
    # Validate input_ids
    if not isinstance(input_ids, torch.Tensor):
        issues.append(f"input_ids must be torch.Tensor, got {type(input_ids)}")
    else:
        if input_ids.dim() != 2:
            issues.append(f"input_ids must be 2D (batch_size, seq_len), got shape {input_ids.shape}")
        
        if input_ids.dtype not in [torch.long, torch.int]:
            issues.append(f"input_ids must be integer type, got {input_ids.dtype}")
        
        if torch.any(input_ids < 0) or torch.any(input_ids >= vocab_size):
            issues.append(f"input_ids values must be in range [0, {vocab_size})")
        
        seq_len = input_ids.size(1) if input_ids.dim() >= 2 else 0
        if seq_len > max_seq_len:
            warnings_list.append(f"Sequence length {seq_len} exceeds maximum {max_seq_len}")
        
        if seq_len == 0:
            issues.append("Sequence length cannot be zero")
    
    # Validate attention_mask
    if attention_mask is not None:
        if not isinstance(attention_mask, torch.Tensor):
            issues.append(f"attention_mask must be torch.Tensor, got {type(attention_mask)}")
        else:
            if attention_mask.shape != input_ids.shape:
                issues.append(f"attention_mask shape {attention_mask.shape} doesn't match input_ids shape {input_ids.shape}")
            
            if attention_mask.dtype not in [torch.bool, torch.long, torch.int, torch.float]:
                issues.append(f"attention_mask has invalid dtype {attention_mask.dtype}")
    
    # Validate labels
    if labels is not None:
        if not isinstance(labels, torch.Tensor):
            issues.append(f"labels must be torch.Tensor, got {type(labels)}")
        else:
            if labels.shape != input_ids.shape:
                issues.append(f"labels shape {labels.shape} doesn't match input_ids shape {input_ids.shape}")
            
            if labels.dtype not in [torch.long, torch.int]:
                issues.append(f"labels must be integer type, got {labels.dtype}")
    
    return {
        'valid': len(issues) == 0,
        'issues': issues,
        'warnings': warnings_list,
        'input_shape': input_ids.shape if isinstance(input_ids, torch.Tensor) else None,
        'sequence_length': input_ids.size(1) if isinstance(input_ids, torch.Tensor) and input_ids.dim() >= 2 else 0,
        'batch_size': input_ids.size(0) if isinstance(input_ids, torch.Tensor) and input_ids.dim() >= 2 else 0,
    }


def check_model_health(model: nn.Module) -> Dict[str, Any]:
    """
    Comprehensive model health check.
    
    Args:
        model: Model to check
        
    Returns:
        Health check results
    """
    health_report = {
        'healthy': True,
        'issues': [],
        'warnings': [],
        'statistics': {},
    }
    
    # Check for NaN/Inf parameters
    nan_params = []
    inf_params = []
    zero_params = []
    
    for name, param in model.named_parameters():
        if param is None:
            continue
            
        if torch.isnan(param).any():
            nan_params.append(name)
            health_report['healthy'] = False
        
        if torch.isinf(param).any():
            inf_params.append(name)
            health_report['healthy'] = False
        
        if torch.all(param == 0):
            zero_params.append(name)
    
    if nan_params:
        health_report['issues'].append(f"NaN values in parameters: {nan_params}")
    
    if inf_params:
        health_report['issues'].append(f"Infinite values in parameters: {inf_params}")
    
    if zero_params:
        health_report['warnings'].append(f"Zero parameters (may be uninitialized): {zero_params}")
    
    # Check gradient statistics
    grad_norms = []
    grad_nan_params = []
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = torch.norm(param.grad).item()
            grad_norms.append(grad_norm)
            
            if torch.isnan(param.grad).any():
                grad_nan_params.append(name)
                health_report['healthy'] = False
    
    if grad_nan_params:
        health_report['issues'].append(f"NaN gradients in: {grad_nan_params}")
    
    # Calculate statistics
    health_report['statistics'] = {
        'total_parameters': sum(p.numel() for p in model.parameters()),
        'parameters_with_gradients': sum(1 for p in model.parameters() if p.grad is not None),
        'average_gradient_norm': sum(grad_norms) / len(grad_norms) if grad_norms else 0,
        'max_gradient_norm': max(grad_norms) if grad_norms else 0,
        'nan_parameter_count': len(nan_params),
        'inf_parameter_count': len(inf_params),
        'zero_parameter_count': len(zero_params),
    }
    
    # Check for gradient explosion
    if health_report['statistics']['max_gradient_norm'] > 100:
        health_report['warnings'].append("Large gradient norms detected (possible gradient explosion)")
    
    # Check for vanishing gradients
    if health_report['statistics']['max_gradient_norm'] < 1e-7:
        health_report['warnings'].append("Very small gradient norms detected (possible vanishing gradients)")
    
    return health_report


class SafeModelWrapper:
    """
    Safe wrapper for model operations with comprehensive error handling.
    """
    
    def __init__(
        self,
        model: nn.Module,
        error_handler: Optional[ErrorHandler] = None,
        validate_inputs: bool = True,
        health_check_interval: int = 100,
    ):
        """
        Initialize safe model wrapper.
        
        Args:
            model: Model to wrap
            error_handler: Error handler instance
            validate_inputs: Whether to validate inputs
            health_check_interval: Interval for health checks
        """
        self.model = model
        self.error_handler = error_handler or ErrorHandler()
        self.validate_inputs = validate_inputs
        self.health_check_interval = health_check_interval
        
        self.call_count = 0
        self.last_health_check = 0
        
    def forward(self, *args, **kwargs):
        """Safe forward pass with error handling."""
        self.call_count += 1
        
        # Input validation
        if self.validate_inputs and len(args) > 0:
            validation_result = validate_inputs(args[0], **kwargs)
            if not validation_result['valid']:
                raise ConfigurationError(f"Input validation failed: {validation_result['issues']}")
        
        # Health check
        if self.call_count - self.last_health_check >= self.health_check_interval:
            health_report = check_model_health(self.model)
            if not health_report['healthy']:
                self.error_handler.logger.warning(f"Model health issues: {health_report['issues']}")
            self.last_health_check = self.call_count
        
        # Safe forward pass
        try:
            return self.model(*args, **kwargs)
        except Exception as e:
            context = {
                'call_count': self.call_count,
                'args_info': [type(arg).__name__ for arg in args],
                'kwargs_info': {k: type(v).__name__ for k, v in kwargs.items()},
            }
            
            success, result, error_info = self.error_handler.handle_error(
                e, context, lambda: self.model(*args, **kwargs)
            )
            
            if success:
                return result
            else:
                raise RevNetError(f"Safe forward failed: {error_info}")
    
    def __call__(self, *args, **kwargs):
        """Make wrapper callable."""
        return self.forward(*args, **kwargs)
    
    def __getattr__(self, name):
        """Delegate attribute access to wrapped model."""
        return getattr(self.model, name)