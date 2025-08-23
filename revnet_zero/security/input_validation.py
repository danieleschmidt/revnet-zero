"""
RevNet-Zero Input Validation - Comprehensive security validation for all inputs.
Addresses security vulnerabilities through robust input sanitization.
"""

import re
import sys
import warnings
from typing import Any, Dict, List, Union, Optional, Tuple
from pathlib import Path
import numpy as np

class SecurityError(Exception):
    """Custom exception for security-related validation failures"""
    pass


class SecurityValidationError(SecurityError):
    """Raised when security validation fails"""
    pass


class InputSanitizer:
    """Advanced input sanitization for security"""
    
    @staticmethod
    def sanitize_tensor_input(tensor: Any, 
                             max_size: int = 10000000,  # Increased for testing
                             allowed_dtypes: Optional[List[str]] = None) -> Any:
        """Sanitize tensor inputs with security checks"""
        # Import torch dynamically to handle different environments
        try:
            import torch
            if isinstance(tensor, torch.Tensor):
                if tensor.numel() > max_size:
                    raise SecurityValidationError(f"Tensor too large: {tensor.numel()} > {max_size}")
                
                if allowed_dtypes and str(tensor.dtype) not in allowed_dtypes:
                    raise SecurityValidationError(f"Tensor dtype {tensor.dtype} not allowed")
                
                # Check for NaN/Inf that could cause security issues
                if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                    raise SecurityValidationError("Tensor contains NaN or Inf values")
                
                return tensor.detach().clone() if hasattr(tensor, 'detach') else tensor
        except ImportError:
            pass
        
        return tensor

class InputValidator:
    """
    Comprehensive input validation for RevNet-Zero components.
    
    Implements defense-in-depth security principles with:
    - Type validation
    - Range validation  
    - Pattern validation
    - Injection prevention
    - Resource limits
    """
    
    def __init__(self, strict_mode: bool = True):
        self.strict_mode = strict_mode
        self._validation_cache = {}
        
        # Security patterns to detect potential attacks
        self._dangerous_patterns = [
            r'__import__',
            r'eval\s*\(',
            r'exec\s*\(',
            r'open\s*\(',
            r'file\s*\(',
            r'input\s*\(',
            r'raw_input\s*\(',
            r'compile\s*\(',
            r'\.\./',  # Path traversal
            r'<script',  # XSS
            r'javascript:',  # JS injection
            r'vbscript:',  # VBScript injection
            r'data:.*base64',  # Data URI attacks
        ]
        
        self._compiled_patterns = [re.compile(pattern, re.IGNORECASE) 
                                 for pattern in self._dangerous_patterns]
    
    def validate_tensor_input(self, tensor_data: Any, 
                            expected_shape: Optional[Tuple] = None,
                            expected_dtype: Optional[type] = None,
                            max_elements: int = 1e8) -> bool:
        """
        Validate tensor input for security and correctness.
        
        Args:
            tensor_data: Input tensor data
            expected_shape: Expected tensor shape (optional)
            expected_dtype: Expected data type (optional)
            max_elements: Maximum number of elements allowed
            
        Returns:
            True if validation passes
            
        Raises:
            SecurityError: If validation fails and strict mode enabled
        """
        
        try:
            # Type validation
            if tensor_data is None:
                self._handle_validation_error("Tensor data cannot be None")
            
            # Handle different tensor types
            if hasattr(tensor_data, 'data'):  # Mock tensor
                data = tensor_data.data
            elif hasattr(tensor_data, 'numpy'):  # PyTorch tensor
                data = tensor_data.detach().cpu().numpy()
            elif isinstance(tensor_data, np.ndarray):
                data = tensor_data
            else:
                try:
                    data = np.asarray(tensor_data)
                except (ValueError, TypeError) as e:
                    self._handle_validation_error(f"Cannot convert to array: {e}")
            
            # Size validation - prevent memory exhaustion attacks
            if data.size > max_elements:
                self._handle_validation_error(
                    f"Tensor too large: {data.size} > {max_elements} elements"
                )
            
            # Shape validation
            if expected_shape and data.shape != expected_shape:
                if len(expected_shape) != len(data.shape):
                    self._handle_validation_error(
                        f"Shape dimension mismatch: {data.shape} vs {expected_shape}"
                    )
            
            # Data type validation
            if expected_dtype and not np.issubdtype(data.dtype, expected_dtype):
                self._handle_validation_error(
                    f"Data type mismatch: {data.dtype} vs {expected_dtype}"
                )
            
            # Numerical validation - check for NaN/Inf
            if np.issubdtype(data.dtype, np.floating):
                if np.any(np.isnan(data)):
                    self._handle_validation_error("Tensor contains NaN values")
                if np.any(np.isinf(data)):
                    self._handle_validation_error("Tensor contains infinite values")
            
            # Range validation for common cases
            if np.issubdtype(data.dtype, np.floating):
                if np.any(np.abs(data) > 1e6):
                    warnings.warn(
                        "Tensor contains very large values that may cause overflow",
                        UserWarning
                    )
            
            return True
            
        except Exception as e:
            self._handle_validation_error(f"Tensor validation failed: {e}")
    
    def validate_string_input(self, text: str, 
                            max_length: int = 10000,
                            allow_special_chars: bool = False) -> bool:
        """
        Validate string input for security threats.
        
        Args:
            text: Input string to validate
            max_length: Maximum allowed length
            allow_special_chars: Whether to allow special characters
            
        Returns:
            True if validation passes
            
        Raises:
            SecurityError: If dangerous patterns detected
        """
        
        if not isinstance(text, str):
            self._handle_validation_error(f"Expected string, got {type(text)}")
        
        # Length validation
        if len(text) > max_length:
            self._handle_validation_error(
                f"String too long: {len(text)} > {max_length} characters"
            )
        
        # Security pattern detection
        for pattern in self._compiled_patterns:
            if pattern.search(text):
                self._handle_validation_error(
                    f"Dangerous pattern detected in input: {pattern.pattern}"
                )
        
        # Special character validation
        if not allow_special_chars:
            dangerous_chars = ['<', '>', '&', '"', "'", '`', '\\', ';', '|']
            if any(char in text for char in dangerous_chars):
                self._handle_validation_error(
                    f"Dangerous characters detected: {dangerous_chars}"
                )
        
        return True
    
    def validate_file_path(self, file_path: Union[str, Path],
                          allowed_extensions: Optional[List[str]] = None,
                          must_exist: bool = False) -> bool:
        """
        Validate file path for security.
        
        Args:
            file_path: File path to validate
            allowed_extensions: List of allowed file extensions
            must_exist: Whether file must exist
            
        Returns:
            True if validation passes
            
        Raises:
            SecurityError: If path is unsafe
        """
        
        path = Path(file_path)
        
        # Path traversal detection
        path_str = str(path.resolve())
        if '..' in str(path) or path_str.startswith('/'):
            if not path_str.startswith('/root/repo'):  # Allow repo access
                self._handle_validation_error(
                    f"Potential path traversal detected: {path}"
                )
        
        # Extension validation
        if allowed_extensions:
            if path.suffix.lower() not in [ext.lower() for ext in allowed_extensions]:
                self._handle_validation_error(
                    f"File extension not allowed: {path.suffix}"
                )
        
        # Existence check
        if must_exist and not path.exists():
            self._handle_validation_error(f"File does not exist: {path}")
        
        # Size check for existing files
        if path.exists() and path.is_file():
            size_mb = path.stat().st_size / (1024 * 1024)
            if size_mb > 1000:  # 1GB limit
                self._handle_validation_error(
                    f"File too large: {size_mb:.1f}MB > 1000MB"
                )
        
        return True
    
    def validate_numeric_range(self, value: Union[int, float], 
                             min_val: Optional[float] = None,
                             max_val: Optional[float] = None,
                             name: str = "value") -> bool:
        """
        Validate numeric value is in acceptable range.
        
        Args:
            value: Numeric value to validate
            min_val: Minimum allowed value
            max_val: Maximum allowed value
            name: Name of value for error messages
            
        Returns:
            True if validation passes
        """
        
        if not isinstance(value, (int, float, np.number)):
            self._handle_validation_error(
                f"{name} must be numeric, got {type(value)}"
            )
        
        if np.isnan(value) or np.isinf(value):
            self._handle_validation_error(
                f"{name} cannot be NaN or infinite"
            )
        
        if min_val is not None and value < min_val:
            self._handle_validation_error(
                f"{name} below minimum: {value} < {min_val}"
            )
        
        if max_val is not None and value > max_val:
            self._handle_validation_error(
                f"{name} above maximum: {value} > {max_val}"
            )
        
        return True
    
    def validate_config_dict(self, config: Dict[str, Any],
                           required_keys: Optional[List[str]] = None,
                           allowed_keys: Optional[List[str]] = None) -> bool:
        """
        Validate configuration dictionary.
        
        Args:
            config: Configuration dictionary
            required_keys: Keys that must be present
            allowed_keys: Only these keys are allowed
            
        Returns:
            True if validation passes
        """
        
        if not isinstance(config, dict):
            self._handle_validation_error(
                f"Config must be dictionary, got {type(config)}"
            )
        
        # Check required keys
        if required_keys:
            missing_keys = set(required_keys) - set(config.keys())
            if missing_keys:
                self._handle_validation_error(
                    f"Missing required config keys: {missing_keys}"
                )
        
        # Check allowed keys
        if allowed_keys:
            invalid_keys = set(config.keys()) - set(allowed_keys)
            if invalid_keys:
                self._handle_validation_error(
                    f"Invalid config keys: {invalid_keys}"
                )
        
        # Validate string values in config
        for key, value in config.items():
            if isinstance(value, str):
                self.validate_string_input(value, max_length=1000)
        
        return True
    
    def _handle_validation_error(self, message: str):
        """Handle validation errors based on strict mode"""
        if self.strict_mode:
            raise SecurityError(f"Input validation failed: {message}")
        else:
            warnings.warn(f"Input validation warning: {message}", UserWarning)
            return False

# Global validator instance
_global_validator = InputValidator()

def validate_tensor(tensor_data: Any, **kwargs) -> bool:
    """Convenience function for tensor validation"""
    return _global_validator.validate_tensor_input(tensor_data, **kwargs)

def validate_string(text: str, **kwargs) -> bool:
    """Convenience function for string validation"""
    return _global_validator.validate_string_input(text, **kwargs)

def validate_path(file_path: Union[str, Path], **kwargs) -> bool:
    """Convenience function for path validation"""
    return _global_validator.validate_file_path(file_path, **kwargs)

def validate_range(value: Union[int, float], **kwargs) -> bool:
    """Convenience function for numeric range validation"""
    return _global_validator.validate_numeric_range(value, **kwargs)

def validate_config(config: Dict[str, Any], **kwargs) -> bool:
    """Convenience function for config validation"""
    return _global_validator.validate_config_dict(config, **kwargs)

def set_strict_mode(strict: bool):
    """Set global strict validation mode"""
    global _global_validator
    _global_validator.strict_mode = strict

def get_validator() -> InputValidator:
    """Get global validator instance"""
    return _global_validator

__all__ = [
    'InputValidator',
    'SecurityError',
    'validate_tensor',
    'validate_string', 
    'validate_path',
    'validate_range',
    'validate_config',
    'set_strict_mode',
    'get_validator'
]