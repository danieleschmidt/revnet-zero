"""
Security module for RevNet-Zero.

This module provides comprehensive security features including
input validation, model integrity checking, and safe execution.
"""

from .validation import (
    SecurityLevel,
    ValidationResult,
    InputValidator,
    ModelSecurityChecker,
    SafeModelLoader,
    create_secure_model_environment,
)

from .secure_validation import (
    SecureInputValidator,
    SecureModelLoader as EnhancedSecureModelLoader,
    SecurityAuditLogger,
    validate_input,
    secure_file_load,
    get_security_summary,
)

from .input_validation import (
    InputValidator as ComprehensiveInputValidator,
    SecurityError,
    validate_tensor,
    validate_string,
    validate_path,
    validate_range,
    validate_config as validate_config_secure,
    set_strict_mode,
    get_validator,
)

__all__ = [
    # Core security components
    'SecurityLevel',
    'ValidationResult',
    'InputValidator',
    'ModelSecurityChecker',
    'SafeModelLoader',
    'create_secure_model_environment',
    # Enhanced validation
    'SecureInputValidator',
    'EnhancedSecureModelLoader',
    'SecurityAuditLogger',
    'validate_input',
    'secure_file_load',
    'get_security_summary',
    # Comprehensive input validation
    'ComprehensiveInputValidator',
    'SecurityError',
    'validate_tensor',
    'validate_string',
    'validate_path',
    'validate_range',
    'validate_config_secure',
    'set_strict_mode',
    'get_validator',
]