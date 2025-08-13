"""
Advanced Security Validation and Input Sanitization System
"""

import torch
import torch.nn as nn
import numpy as np
import hashlib
import hmac
import secrets
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
import json
import re
from pathlib import Path

@dataclass
class SecurityViolation:
    """Represents a security violation detected during validation"""
    violation_type: str
    severity: str  # low, medium, high, critical
    description: str
    context: Dict[str, Any]
    suggested_fix: str
    timestamp: float

class InputSanitizer:
    """Advanced input sanitization for model inputs and configurations"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.violations: List[SecurityViolation] = []
        
        # Validation rules
        self.tensor_validators = {
            'shape': self._validate_tensor_shape,
            'dtype': self._validate_tensor_dtype,
            'range': self._validate_tensor_range,
            'finite': self._validate_tensor_finite,
            'memory': self._validate_tensor_memory
        }
        
        self.config_validators = {
            'parameters': self._validate_config_parameters,
            'paths': self._validate_file_paths,
            'types': self._validate_config_types,
            'ranges': self._validate_config_ranges
        }
        
        self.logger = logging.getLogger(__name__)
        
    def sanitize_tensor_input(self, tensor: torch.Tensor, 
                            constraints: Optional[Dict] = None) -> Tuple[torch.Tensor, List[SecurityViolation]]:
        """Sanitize tensor inputs with comprehensive validation"""
        
        violations = []
        sanitized_tensor = tensor.clone()
        
        constraints = constraints or {}
        
        # Basic tensor validation
        for validator_name, validator_func in self.tensor_validators.items():
            try:
                is_valid, violation = validator_func(sanitized_tensor, constraints.get(validator_name))
                if not is_valid and violation:
                    violations.append(violation)
                    
                    # Apply automatic fixes where possible
                    sanitized_tensor = self._apply_tensor_fix(sanitized_tensor, violation)
                    
            except Exception as e:
                violation = SecurityViolation(
                    violation_type='validation_error',
                    severity='medium',
                    description=f"Validation error in {validator_name}: {str(e)}",
                    context={'validator': validator_name, 'error': str(e)},
                    suggested_fix="Check tensor format and validation constraints",
                    timestamp=torch.cuda.Event(enable_timing=True).elapsed_time(torch.cuda.Event(enable_timing=True))
                )
                violations.append(violation)
                
        return sanitized_tensor, violations
        
    def sanitize_config(self, config: Dict[str, Any]) -> Tuple[Dict[str, Any], List[SecurityViolation]]:
        """Sanitize configuration with security validation"""
        
        violations = []
        sanitized_config = config.copy()
        
        # Apply config validators
        for validator_name, validator_func in self.config_validators.items():
            try:
                is_valid, config_violations = validator_func(sanitized_config)
                if not is_valid:
                    violations.extend(config_violations)
                    
                    # Apply fixes
                    sanitized_config = self._apply_config_fixes(sanitized_config, config_violations)
                    
            except Exception as e:
                violation = SecurityViolation(
                    violation_type='config_validation_error',
                    severity='medium', 
                    description=f"Config validation error in {validator_name}: {str(e)}",
                    context={'validator': validator_name, 'error': str(e)},
                    suggested_fix="Review configuration format and values",
                    timestamp=torch.cuda.Event(enable_timing=True).elapsed_time(torch.cuda.Event(enable_timing=True))
                )
                violations.append(violation)
                
        # Additional security checks
        security_violations = self._perform_security_checks(sanitized_config)
        violations.extend(security_violations)
        
        return sanitized_config, violations
        
    def _validate_tensor_shape(self, tensor: torch.Tensor, constraints: Optional[Dict]) -> Tuple[bool, Optional[SecurityViolation]]:
        """Validate tensor shape constraints"""
        
        if constraints is None:
            return True, None
            
        shape = tensor.shape
        
        # Check minimum dimensions
        if 'min_dims' in constraints and len(shape) < constraints['min_dims']:
            return False, SecurityViolation(
                violation_type='tensor_shape',
                severity='high',
                description=f"Tensor has {len(shape)} dimensions, minimum required: {constraints['min_dims']}",
                context={'actual_shape': shape, 'constraints': constraints},
                suggested_fix="Ensure input tensor has correct number of dimensions",
                timestamp=torch.cuda.Event(enable_timing=True).elapsed_time(torch.cuda.Event(enable_timing=True))
            )
            
        # Check maximum dimensions
        if 'max_dims' in constraints and len(shape) > constraints['max_dims']:
            return False, SecurityViolation(
                violation_type='tensor_shape',
                severity='medium',
                description=f"Tensor has {len(shape)} dimensions, maximum allowed: {constraints['max_dims']}",
                context={'actual_shape': shape, 'constraints': constraints},
                suggested_fix="Reshape or reduce tensor dimensions",
                timestamp=torch.cuda.Event(enable_timing=True).elapsed_time(torch.cuda.Event(enable_timing=True))
            )
            
        # Check shape bounds
        if 'max_size' in constraints:
            total_elements = tensor.numel()
            if total_elements > constraints['max_size']:
                return False, SecurityViolation(
                    violation_type='tensor_size',
                    severity='critical',
                    description=f"Tensor has {total_elements} elements, maximum allowed: {constraints['max_size']}",
                    context={'total_elements': total_elements, 'constraints': constraints},
                    suggested_fix="Reduce tensor size or increase memory limits",
                    timestamp=torch.cuda.Event(enable_timing=True).elapsed_time(torch.cuda.Event(enable_timing=True))
                )
                
        return True, None
        
    def _validate_tensor_dtype(self, tensor: torch.Tensor, constraints: Optional[Dict]) -> Tuple[bool, Optional[SecurityViolation]]:
        """Validate tensor data type"""
        
        if constraints is None:
            return True, None
            
        allowed_dtypes = constraints.get('allowed_dtypes', [])
        if allowed_dtypes and tensor.dtype not in allowed_dtypes:
            return False, SecurityViolation(
                violation_type='tensor_dtype',
                severity='medium',
                description=f"Tensor dtype {tensor.dtype} not in allowed types: {allowed_dtypes}",
                context={'actual_dtype': tensor.dtype, 'allowed': allowed_dtypes},
                suggested_fix="Convert tensor to allowed dtype",
                timestamp=torch.cuda.Event(enable_timing=True).elapsed_time(torch.cuda.Event(enable_timing=True))
            )
            
        return True, None
        
    def _validate_tensor_range(self, tensor: torch.Tensor, constraints: Optional[Dict]) -> Tuple[bool, Optional[SecurityViolation]]:
        """Validate tensor value ranges"""
        
        if constraints is None:
            return True, None
            
        # Check minimum values
        if 'min_value' in constraints:
            min_val = tensor.min().item()
            if min_val < constraints['min_value']:
                return False, SecurityViolation(
                    violation_type='tensor_range',
                    severity='medium',
                    description=f"Tensor minimum value {min_val} below threshold {constraints['min_value']}",
                    context={'min_value': min_val, 'threshold': constraints['min_value']},
                    suggested_fix="Clamp tensor values to valid range",
                    timestamp=torch.cuda.Event(enable_timing=True).elapsed_time(torch.cuda.Event(enable_timing=True))
                )
                
        # Check maximum values
        if 'max_value' in constraints:
            max_val = tensor.max().item()
            if max_val > constraints['max_value']:
                return False, SecurityViolation(
                    violation_type='tensor_range',
                    severity='medium',
                    description=f"Tensor maximum value {max_val} above threshold {constraints['max_value']}",
                    context={'max_value': max_val, 'threshold': constraints['max_value']},
                    suggested_fix="Clamp tensor values to valid range",
                    timestamp=torch.cuda.Event(enable_timing=True).elapsed_time(torch.cuda.Event(enable_timing=True))
                )
                
        return True, None
        
    def _validate_tensor_finite(self, tensor: torch.Tensor, constraints: Optional[Dict]) -> Tuple[bool, Optional[SecurityViolation]]:
        """Validate tensor contains finite values"""
        
        if not torch.isfinite(tensor).all():
            return False, SecurityViolation(
                violation_type='tensor_finite',
                severity='high',
                description="Tensor contains non-finite values (NaN or Inf)",
                context={'has_nan': torch.isnan(tensor).any().item(), 'has_inf': torch.isinf(tensor).any().item()},
                suggested_fix="Replace NaN/Inf values with valid numbers",
                timestamp=torch.cuda.Event(enable_timing=True).elapsed_time(torch.cuda.Event(enable_timing=True))
            )
            
        return True, None
        
    def _validate_tensor_memory(self, tensor: torch.Tensor, constraints: Optional[Dict]) -> Tuple[bool, Optional[SecurityViolation]]:
        """Validate tensor memory usage"""
        
        if constraints is None:
            return True, None
            
        memory_bytes = tensor.numel() * tensor.element_size()
        max_memory = constraints.get('max_memory_mb', 1000) * 1024 * 1024  # Default 1GB
        
        if memory_bytes > max_memory:
            return False, SecurityViolation(
                violation_type='tensor_memory',
                severity='critical',
                description=f"Tensor memory usage {memory_bytes / 1024**2:.1f}MB exceeds limit {max_memory / 1024**2:.1f}MB",
                context={'memory_mb': memory_bytes / 1024**2, 'limit_mb': max_memory / 1024**2},
                suggested_fix="Reduce tensor size or increase memory limits",
                timestamp=torch.cuda.Event(enable_timing=True).elapsed_time(torch.cuda.Event(enable_timing=True))
            )
            
        return True, None
        
    def _validate_config_parameters(self, config: Dict[str, Any]) -> Tuple[bool, List[SecurityViolation]]:
        """Validate configuration parameters"""
        
        violations = []
        
        # Check for dangerous parameters
        dangerous_params = ['__import__', 'eval', 'exec', 'open', 'file']
        for key, value in config.items():
            if isinstance(value, str):
                for dangerous in dangerous_params:
                    if dangerous in value:
                        violations.append(SecurityViolation(
                            violation_type='dangerous_parameter',
                            severity='critical',
                            description=f"Potentially dangerous parameter detected: {dangerous} in {key}",
                            context={'key': key, 'value': value, 'dangerous_pattern': dangerous},
                            suggested_fix="Remove or sanitize dangerous parameters",
                            timestamp=torch.cuda.Event(enable_timing=True).elapsed_time(torch.cuda.Event(enable_timing=True))
                        ))
                        
        return len(violations) == 0, violations
        
    def _validate_file_paths(self, config: Dict[str, Any]) -> Tuple[bool, List[SecurityViolation]]:
        """Validate file paths for security"""
        
        violations = []
        
        path_keys = ['model_path', 'data_path', 'output_path', 'config_path', 'checkpoint_path']
        
        for key in path_keys:
            if key in config:
                path_str = str(config[key])
                
                # Check for path traversal attempts
                if '..' in path_str or path_str.startswith('/'):
                    violations.append(SecurityViolation(
                        violation_type='path_traversal',
                        severity='critical',
                        description=f"Potential path traversal in {key}: {path_str}",
                        context={'key': key, 'path': path_str},
                        suggested_fix="Use relative paths and validate path safety",
                        timestamp=torch.cuda.Event(enable_timing=True).elapsed_time(torch.cuda.Event(enable_timing=True))
                    ))
                    
                # Check for system paths
                system_paths = ['/etc', '/var', '/usr', '/sys', '/proc']
                if any(path_str.startswith(sys_path) for sys_path in system_paths):
                    violations.append(SecurityViolation(
                        violation_type='system_path_access',
                        severity='high',
                        description=f"Attempt to access system path in {key}: {path_str}",
                        context={'key': key, 'path': path_str},
                        suggested_fix="Use application-specific directories",
                        timestamp=torch.cuda.Event(enable_timing=True).elapsed_time(torch.cuda.Event(enable_timing=True))
                    ))
                    
        return len(violations) == 0, violations
        
    def _validate_config_types(self, config: Dict[str, Any]) -> Tuple[bool, List[SecurityViolation]]:
        """Validate configuration value types"""
        
        violations = []
        
        # Define expected types for common parameters
        expected_types = {
            'batch_size': int,
            'learning_rate': (int, float),
            'num_epochs': int,
            'max_length': int,
            'num_layers': int,
            'hidden_size': int,
            'dropout': float,
            'gradient_clip_value': (int, float)
        }
        
        for key, expected_type in expected_types.items():
            if key in config:
                value = config[key]
                if not isinstance(value, expected_type):
                    violations.append(SecurityViolation(
                        violation_type='config_type_mismatch',
                        severity='medium',
                        description=f"Parameter {key} has type {type(value)}, expected {expected_type}",
                        context={'key': key, 'actual_type': type(value), 'expected_type': expected_type},
                        suggested_fix="Convert parameter to correct type",
                        timestamp=torch.cuda.Event(enable_timing=True).elapsed_time(torch.cuda.Event(enable_timing=True))
                    ))
                    
        return len(violations) == 0, violations
        
    def _validate_config_ranges(self, config: Dict[str, Any]) -> Tuple[bool, List[SecurityViolation]]:
        """Validate configuration value ranges"""
        
        violations = []
        
        # Define acceptable ranges
        ranges = {
            'batch_size': (1, 1024),
            'learning_rate': (1e-8, 1.0),
            'num_epochs': (1, 10000),
            'max_length': (1, 1000000),
            'num_layers': (1, 1000),
            'hidden_size': (1, 100000),
            'dropout': (0.0, 1.0)
        }
        
        for key, (min_val, max_val) in ranges.items():
            if key in config:
                value = config[key]
                if isinstance(value, (int, float)):
                    if value < min_val or value > max_val:
                        violations.append(SecurityViolation(
                            violation_type='config_range_violation',
                            severity='medium',
                            description=f"Parameter {key} value {value} outside range [{min_val}, {max_val}]",
                            context={'key': key, 'value': value, 'min': min_val, 'max': max_val},
                            suggested_fix=f"Set {key} to value within [{min_val}, {max_val}]",
                            timestamp=torch.cuda.Event(enable_timing=True).elapsed_time(torch.cuda.Event(enable_timing=True))
                        ))
                        
        return len(violations) == 0, violations
        
    def _perform_security_checks(self, config: Dict[str, Any]) -> List[SecurityViolation]:
        """Perform additional security checks"""
        
        violations = []
        
        # Check for code injection attempts
        code_patterns = [
            r'__import__\s*\(',
            r'eval\s*\(',
            r'exec\s*\(',
            r'subprocess\.',
            r'os\.system',
            r'open\s*\('
        ]
        
        for key, value in config.items():
            if isinstance(value, str):
                for pattern in code_patterns:
                    if re.search(pattern, value):
                        violations.append(SecurityViolation(
                            violation_type='code_injection',
                            severity='critical',
                            description=f"Potential code injection pattern in {key}: {pattern}",
                            context={'key': key, 'value': value, 'pattern': pattern},
                            suggested_fix="Remove or sanitize potentially dangerous code patterns",
                            timestamp=torch.cuda.Event(enable_timing=True).elapsed_time(torch.cuda.Event(enable_timing=True))
                        ))
                        
        return violations
        
    def _apply_tensor_fix(self, tensor: torch.Tensor, violation: SecurityViolation) -> torch.Tensor:
        """Apply automatic fixes to tensor violations"""
        
        if violation.violation_type == 'tensor_range':
            # Clamp values to valid range
            if 'min_value' in violation.context:
                tensor = torch.clamp(tensor, min=violation.context['threshold'])
            if 'max_value' in violation.context:
                tensor = torch.clamp(tensor, max=violation.context['threshold'])
                
        elif violation.violation_type == 'tensor_finite':
            # Replace NaN/Inf with zeros
            tensor = torch.where(torch.isfinite(tensor), tensor, torch.zeros_like(tensor))
            
        return tensor
        
    def _apply_config_fixes(self, config: Dict[str, Any], violations: List[SecurityViolation]) -> Dict[str, Any]:
        """Apply automatic fixes to configuration violations"""
        
        for violation in violations:
            if violation.violation_type == 'config_range_violation':
                key = violation.context['key']
                min_val = violation.context['min']
                max_val = violation.context['max']
                value = violation.context['value']
                
                # Clamp to range
                config[key] = max(min_val, min(max_val, value))
                
            elif violation.violation_type == 'dangerous_parameter':
                # Remove dangerous parameters
                key = violation.context['key']
                self.logger.warning(f"Removing dangerous parameter: {key}")
                config.pop(key, None)
                
        return config

class SecurityAuditLogger:
    """Advanced security audit logging"""
    
    def __init__(self, log_file: Optional[str] = None):
        self.log_file = Path(log_file) if log_file else Path("security_audit.log")
        self.violations_count = 0
        self.session_id = secrets.token_hex(16)
        
        # Setup logging
        self.logger = logging.getLogger('security_audit')
        handler = logging.FileHandler(self.log_file)
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
    def log_violation(self, violation: SecurityViolation) -> None:
        """Log a security violation"""
        
        self.violations_count += 1
        
        log_entry = {
            'session_id': self.session_id,
            'violation_id': self.violations_count,
            'type': violation.violation_type,
            'severity': violation.severity,
            'description': violation.description,
            'context': violation.context,
            'suggested_fix': violation.suggested_fix,
            'timestamp': violation.timestamp
        }
        
        self.logger.warning(f"SECURITY_VIOLATION: {json.dumps(log_entry)}")
        
    def generate_security_report(self) -> str:
        """Generate comprehensive security report"""
        
        report_lines = [
            "# Security Audit Report",
            f"Session ID: {self.session_id}",
            f"Total Violations: {self.violations_count}",
            f"Report Generated: {torch.cuda.Event(enable_timing=True).elapsed_time(torch.cuda.Event(enable_timing=True))}",
            ""
        ]
        
        # Read and analyze log file
        if self.log_file.exists():
            with open(self.log_file, 'r') as f:
                log_content = f.read()
                
            # Count violation types
            violation_counts = {}
            for line in log_content.split('\n'):
                if 'SECURITY_VIOLATION' in line:
                    try:
                        violation_data = json.loads(line.split('SECURITY_VIOLATION: ')[1])
                        v_type = violation_data.get('type', 'unknown')
                        violation_counts[v_type] = violation_counts.get(v_type, 0) + 1
                    except:
                        pass
                        
            report_lines.append("## Violation Summary")
            for v_type, count in violation_counts.items():
                report_lines.append(f"- {v_type}: {count}")
                
        return "\n".join(report_lines)