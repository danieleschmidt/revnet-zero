"""
Advanced Security Validation for RevNet-Zero.

Implements comprehensive security measures:
- Input sanitization and validation
- Model integrity verification
- Secure computation protocols
- Privacy-preserving techniques
- Attack detection and mitigation
"""

import hashlib
import hmac
import logging
import secrets
import time
from typing import Dict, List, Tuple, Optional, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
from datetime import datetime, timedelta
import threading
from collections import defaultdict, deque
import numpy as np
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    import base64
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False


class SecurityLevel(Enum):
    """Security levels for different deployment scenarios."""
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"
    HIGH_SECURITY = "high_security"
    GOVERNMENT = "government"


class ThreatLevel(Enum):
    """Threat level classification."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class SecurityEvent:
    """Security event for audit logging."""
    event_type: str
    threat_level: ThreatLevel
    description: str
    source: str
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    remediation_taken: Optional[str] = None


@dataclass
class ValidationResult:
    """Result of security validation."""
    valid: bool
    confidence: float
    threats_detected: List[str]
    recommendations: List[str]
    risk_score: float
    processing_time_ms: float


class InputSanitizer:
    """Advanced input sanitization for ML inputs."""
    
    def __init__(self, security_level: SecurityLevel = SecurityLevel.PRODUCTION):
        self.security_level = security_level
        self.logger = logging.getLogger(f"{__name__}.InputSanitizer")
        
        # Security thresholds based on level
        self.thresholds = self._get_security_thresholds()
        
        # Pattern detection for adversarial inputs
        self.adversarial_detector = AdversarialInputDetector()
        
        # Rate limiting
        self.rate_limiter = RateLimiter()
        
    def _get_security_thresholds(self) -> Dict[str, Any]:
        """Get security thresholds based on security level."""
        thresholds = {
            SecurityLevel.DEVELOPMENT: {
                "max_input_size": 1e8,  # 100MB
                "max_sequence_length": 1048576,  # 1M tokens
                "input_validation_timeout": 30.0,
                "adversarial_threshold": 0.7
            },
            SecurityLevel.TESTING: {
                "max_input_size": 5e7,  # 50MB
                "max_sequence_length": 524288,  # 512K tokens
                "input_validation_timeout": 20.0,
                "adversarial_threshold": 0.6
            },
            SecurityLevel.PRODUCTION: {
                "max_input_size": 1e7,  # 10MB
                "max_sequence_length": 262144,  # 256K tokens
                "input_validation_timeout": 10.0,
                "adversarial_threshold": 0.5
            },
            SecurityLevel.HIGH_SECURITY: {
                "max_input_size": 5e6,  # 5MB
                "max_sequence_length": 131072,  # 128K tokens
                "input_validation_timeout": 5.0,
                "adversarial_threshold": 0.3
            },
            SecurityLevel.GOVERNMENT: {
                "max_input_size": 1e6,  # 1MB
                "max_sequence_length": 65536,  # 64K tokens
                "input_validation_timeout": 2.0,
                "adversarial_threshold": 0.2
            }
        }
        return thresholds[self.security_level]
    
    def validate_input(self, 
                      input_data: Any,
                      client_id: str = "unknown",
                      context: Dict[str, Any] = None) -> ValidationResult:
        """Comprehensive input validation with security checks."""
        start_time = time.time()
        context = context or {}
        
        threats_detected = []
        recommendations = []
        risk_score = 0.0
        
        try:
            # Rate limiting check
            if not self.rate_limiter.check_rate_limit(client_id):
                threats_detected.append("rate_limit_exceeded")
                risk_score += 0.3
                
                return ValidationResult(
                    valid=False,
                    confidence=1.0,
                    threats_detected=threats_detected,
                    recommendations=["Reduce request rate"],
                    risk_score=risk_score,
                    processing_time_ms=(time.time() - start_time) * 1000
                )
            
            # Basic size validation
            input_size = self._estimate_input_size(input_data)
            if input_size > self.thresholds["max_input_size"]:
                threats_detected.append("oversized_input")
                risk_score += 0.4
            
            # Sequence length validation (for text inputs)
            if hasattr(input_data, 'shape') and len(input_data.shape) >= 2:
                seq_length = input_data.shape[-1]
                if seq_length > self.thresholds["max_sequence_length"]:
                    threats_detected.append("excessive_sequence_length")
                    risk_score += 0.3
            
            # Data type validation
            if TORCH_AVAILABLE and isinstance(input_data, torch.Tensor):
                # Check for unusual data types
                if input_data.dtype not in [torch.float32, torch.float16, torch.int64, torch.int32]:
                    threats_detected.append("unusual_data_type")
                    risk_score += 0.2
                
                # Check for NaN/Inf values
                if torch.isnan(input_data).any() or torch.isinf(input_data).any():
                    threats_detected.append("invalid_numeric_values")
                    risk_score += 0.3
            
            # Adversarial input detection
            adversarial_score = self.adversarial_detector.detect(input_data)
            if adversarial_score > self.thresholds["adversarial_threshold"]:
                threats_detected.append("potential_adversarial_input")
                risk_score += 0.5
                recommendations.append(f"Adversarial confidence: {adversarial_score:.2f}")
            
            # Pattern-based threat detection
            pattern_threats = self._detect_input_patterns(input_data)
            threats_detected.extend(pattern_threats)
            risk_score += len(pattern_threats) * 0.1
            
            # Privacy leak detection
            privacy_threats = self._detect_privacy_leaks(input_data, context)
            threats_detected.extend(privacy_threats)
            risk_score += len(privacy_threats) * 0.2
            
            # Generate recommendations
            if risk_score > 0.5:
                recommendations.append("High risk input detected - consider additional validation")
            if "oversized_input" in threats_detected:
                recommendations.append("Implement input chunking or compression")
            if "rate_limit_exceeded" in threats_detected:
                recommendations.append("Implement exponential backoff")
            
            # Determine validity
            is_valid = risk_score < 0.7 and len(threats_detected) < 3
            confidence = max(0.0, 1.0 - risk_score)
            
            # Log security event if threats detected
            if threats_detected:
                threat_level = ThreatLevel.HIGH if risk_score > 0.7 else ThreatLevel.MEDIUM
                self._log_security_event(
                    SecurityEvent(
                        event_type="input_validation",
                        threat_level=threat_level,
                        description=f"Threats detected: {', '.join(threats_detected)}",
                        source=client_id,
                        metadata={
                            "risk_score": risk_score,
                            "input_size": input_size,
                            "threats": threats_detected
                        }
                    )
                )
            
            return ValidationResult(
                valid=is_valid,
                confidence=confidence,
                threats_detected=threats_detected,
                recommendations=recommendations,
                risk_score=risk_score,
                processing_time_ms=(time.time() - start_time) * 1000
            )
            
        except Exception as e:
            self.logger.error(f"Input validation error: {e}")
            return ValidationResult(
                valid=False,
                confidence=0.0,
                threats_detected=["validation_error"],
                recommendations=["Contact system administrator"],
                risk_score=1.0,
                processing_time_ms=(time.time() - start_time) * 1000
            )
    
    def _estimate_input_size(self, input_data: Any) -> int:
        """Estimate memory size of input data."""
        if TORCH_AVAILABLE and isinstance(input_data, torch.Tensor):
            return input_data.element_size() * input_data.nelement()
        elif isinstance(input_data, np.ndarray):
            return input_data.nbytes
        elif isinstance(input_data, (str, bytes)):
            return len(input_data)
        elif hasattr(input_data, '__sizeof__'):
            return input_data.__sizeof__()
        else:
            return len(str(input_data))
    
    def _detect_input_patterns(self, input_data: Any) -> List[str]:
        """Detect suspicious patterns in input data."""
        threats = []
        
        if TORCH_AVAILABLE and isinstance(input_data, torch.Tensor):
            # Check for unusual value distributions
            if len(input_data.shape) > 0:
                # Extremely high values
                if torch.max(torch.abs(input_data)) > 1e6:
                    threats.append("extreme_values")
                
                # All zeros or ones (potential dummy data)
                if torch.all(input_data == 0) or torch.all(input_data == 1):
                    threats.append("uniform_data_pattern")
                
                # Repeated patterns
                if len(input_data.shape) >= 2 and input_data.shape[0] > 1:
                    first_row = input_data[0]
                    if torch.all(torch.stack([torch.allclose(row, first_row) for row in input_data])):
                        threats.append("repeated_data_pattern")
        
        elif isinstance(input_data, str):
            # SQL injection patterns
            sql_patterns = ['DROP ', 'DELETE ', 'INSERT ', 'UPDATE ', 'SELECT ', 'UNION ']
            if any(pattern.lower() in input_data.lower() for pattern in sql_patterns):
                threats.append("sql_injection_pattern")
            
            # Script injection patterns
            script_patterns = ['<script', 'javascript:', 'eval(', 'exec(']
            if any(pattern.lower() in input_data.lower() for pattern in script_patterns):
                threats.append("script_injection_pattern")
        
        return threats
    
    def _detect_privacy_leaks(self, input_data: Any, context: Dict[str, Any]) -> List[str]:
        """Detect potential privacy leaks in input data."""
        threats = []
        
        if isinstance(input_data, str):
            # Email patterns
            import re
            email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
            if re.search(email_pattern, input_data):
                threats.append("email_address_detected")
            
            # Phone number patterns
            phone_pattern = r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b'
            if re.search(phone_pattern, input_data):
                threats.append("phone_number_detected")
            
            # SSN patterns (US)
            ssn_pattern = r'\b\d{3}-\d{2}-\d{4}\b'
            if re.search(ssn_pattern, input_data):
                threats.append("ssn_pattern_detected")
        
        return threats
    
    def _log_security_event(self, event: SecurityEvent) -> None:
        """Log security event for audit trail."""
        self.logger.warning(f"Security Event: {event.event_type} - {event.description}")
        # In production, would send to SIEM or security monitoring system


class AdversarialInputDetector:
    """Detector for adversarial inputs using statistical analysis."""
    
    def __init__(self):
        self.baseline_stats: Dict[str, Any] = {}
        self.detection_history: deque = deque(maxlen=1000)
        
    def detect(self, input_data: Any) -> float:
        """Detect adversarial patterns in input data."""
        if not TORCH_AVAILABLE or not isinstance(input_data, torch.Tensor):
            return 0.0
        
        try:
            # Calculate statistical features
            features = self._extract_statistical_features(input_data)
            
            # Compare with baseline (if available)
            adversarial_score = self._calculate_adversarial_score(features)
            
            # Update detection history
            self.detection_history.append({
                "timestamp": datetime.now(),
                "score": adversarial_score,
                "features": features
            })
            
            return adversarial_score
            
        except Exception:
            return 0.0  # Default to no threat if detection fails
    
    def _extract_statistical_features(self, tensor: torch.Tensor) -> Dict[str, float]:
        """Extract statistical features for adversarial detection."""
        flattened = tensor.flatten().float()
        
        return {
            "mean": torch.mean(flattened).item(),
            "std": torch.std(flattened).item(),
            "min": torch.min(flattened).item(),
            "max": torch.max(flattened).item(),
            "l2_norm": torch.norm(flattened, p=2).item(),
            "l1_norm": torch.norm(flattened, p=1).item(),
            "entropy": self._calculate_entropy(flattened),
            "sparsity": (flattened == 0).float().mean().item()
        }
    
    def _calculate_entropy(self, tensor: torch.Tensor) -> float:
        """Calculate entropy of tensor values."""
        # Discretize values for entropy calculation
        _, counts = torch.unique(torch.round(tensor * 1000), return_counts=True)
        probs = counts.float() / len(tensor)
        entropy = -torch.sum(probs * torch.log2(probs + 1e-10))
        return entropy.item()
    
    def _calculate_adversarial_score(self, features: Dict[str, float]) -> float:
        """Calculate adversarial score based on statistical features."""
        score = 0.0
        
        # Check for unusual statistical properties
        if features["std"] > 100 or features["std"] < 0.001:
            score += 0.3
        
        if features["l2_norm"] > 1000:
            score += 0.2
        
        if features["entropy"] < 1.0:  # Very low entropy
            score += 0.3
        
        if features["sparsity"] > 0.9:  # Very sparse
            score += 0.2
        
        return min(1.0, score)


class RateLimiter:
    """Rate limiter with adaptive thresholds."""
    
    def __init__(self, 
                 default_rate: int = 100,  # requests per minute
                 burst_rate: int = 10,     # requests per second
                 window_size: int = 60):   # seconds
        self.default_rate = default_rate
        self.burst_rate = burst_rate
        self.window_size = window_size
        
        # Client tracking
        self.client_history: Dict[str, deque] = defaultdict(lambda: deque())
        self.client_reputation: Dict[str, float] = defaultdict(lambda: 1.0)
        
        self._lock = threading.Lock()
    
    def check_rate_limit(self, client_id: str) -> bool:
        """Check if client is within rate limits."""
        with self._lock:
            current_time = time.time()
            
            # Clean old entries
            client_requests = self.client_history[client_id]
            while client_requests and current_time - client_requests[0] > self.window_size:
                client_requests.popleft()
            
            # Check burst rate (last second)
            recent_requests = sum(1 for t in client_requests if current_time - t <= 1.0)
            if recent_requests >= self.burst_rate:
                self._update_reputation(client_id, -0.1)
                return False
            
            # Check sustained rate (per minute)
            total_requests = len(client_requests)
            rate_limit = self._get_adaptive_rate_limit(client_id)
            
            if total_requests >= rate_limit:
                self._update_reputation(client_id, -0.05)
                return False
            
            # Accept request
            client_requests.append(current_time)
            self._update_reputation(client_id, 0.01)
            return True
    
    def _get_adaptive_rate_limit(self, client_id: str) -> int:
        """Get adaptive rate limit based on client reputation."""
        reputation = self.client_reputation[client_id]
        # Higher reputation = higher rate limit
        return int(self.default_rate * reputation)
    
    def _update_reputation(self, client_id: str, delta: float) -> None:
        """Update client reputation score."""
        current = self.client_reputation[client_id]
        self.client_reputation[client_id] = max(0.1, min(2.0, current + delta))


class ModelIntegrityVerifier:
    """Verifies model integrity and detects tampering."""
    
    def __init__(self, security_level: SecurityLevel = SecurityLevel.PRODUCTION):
        self.security_level = security_level
        self.known_checksums: Dict[str, str] = {}
        self.verification_log: List[Dict[str, Any]] = []
        
        self.logger = logging.getLogger(f"{__name__}.ModelIntegrityVerifier")
    
    def register_model(self, model_name: str, model_data: Any) -> str:
        """Register a model and compute its integrity checksum."""
        checksum = self._compute_checksum(model_data)
        self.known_checksums[model_name] = checksum
        
        self.logger.info(f"Registered model {model_name} with checksum {checksum[:16]}...")
        return checksum
    
    def verify_model(self, model_name: str, model_data: Any) -> bool:
        """Verify model integrity against known checksum."""
        if model_name not in self.known_checksums:
            self.logger.warning(f"No checksum registered for model {model_name}")
            return False
        
        current_checksum = self._compute_checksum(model_data)
        expected_checksum = self.known_checksums[model_name]
        
        is_valid = current_checksum == expected_checksum
        
        verification_entry = {
            "timestamp": datetime.now().isoformat(),
            "model_name": model_name,
            "expected_checksum": expected_checksum,
            "actual_checksum": current_checksum,
            "valid": is_valid
        }
        
        self.verification_log.append(verification_entry)
        
        if not is_valid:
            self.logger.error(f"Model integrity verification failed for {model_name}")
            self.logger.error(f"Expected: {expected_checksum}")
            self.logger.error(f"Actual: {current_checksum}")
        
        return is_valid
    
    def _compute_checksum(self, model_data: Any) -> str:
        """Compute SHA-256 checksum of model data."""
        if TORCH_AVAILABLE and isinstance(model_data, torch.nn.Module):
            # Serialize model state dict
            state_dict = model_data.state_dict()
            serialized = json.dumps(
                {k: v.cpu().numpy().tolist() for k, v in state_dict.items()},
                sort_keys=True
            )
            data_bytes = serialized.encode('utf-8')
        elif hasattr(model_data, 'encode'):
            data_bytes = model_data.encode('utf-8')
        else:
            data_bytes = str(model_data).encode('utf-8')
        
        return hashlib.sha256(data_bytes).hexdigest()


class SecureComputationManager:
    """Manager for secure computation protocols."""
    
    def __init__(self, security_level: SecurityLevel = SecurityLevel.PRODUCTION):
        self.security_level = security_level
        self.encryption_enabled = CRYPTO_AVAILABLE and security_level in [
            SecurityLevel.HIGH_SECURITY, SecurityLevel.GOVERNMENT
        ]
        
        if self.encryption_enabled:
            self._setup_encryption()
        
        self.logger = logging.getLogger(f"{__name__}.SecureComputationManager")
    
    def _setup_encryption(self) -> None:
        """Setup encryption for secure computation."""
        if not CRYPTO_AVAILABLE:
            self.logger.warning("Cryptography library not available, encryption disabled")
            self.encryption_enabled = False
            return
        
        # Generate encryption key
        password = secrets.token_bytes(32)
        salt = secrets.token_bytes(16)
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password))
        self.cipher = Fernet(key)
        
        self.logger.info("Encryption setup completed for secure computation")
    
    def encrypt_data(self, data: bytes) -> bytes:
        """Encrypt sensitive data."""
        if not self.encryption_enabled:
            return data
        
        return self.cipher.encrypt(data)
    
    def decrypt_data(self, encrypted_data: bytes) -> bytes:
        """Decrypt sensitive data."""
        if not self.encryption_enabled:
            return encrypted_data
        
        return self.cipher.decrypt(encrypted_data)
    
    def secure_computation(self, 
                          computation_fn: Callable,
                          sensitive_data: Any,
                          **kwargs) -> Any:
        """Perform computation with security measures."""
        start_time = time.time()
        
        try:
            # Encrypt sensitive data if needed
            if self.encryption_enabled and isinstance(sensitive_data, (str, bytes)):
                if isinstance(sensitive_data, str):
                    sensitive_data = sensitive_data.encode('utf-8')
                encrypted_data = self.encrypt_data(sensitive_data)
                
                # In a real implementation, computation would work on encrypted data
                # For demo, we decrypt for computation
                decrypted_data = self.decrypt_data(encrypted_data)
                result = computation_fn(decrypted_data, **kwargs)
            else:
                result = computation_fn(sensitive_data, **kwargs)
            
            computation_time = time.time() - start_time
            
            self.logger.info(f"Secure computation completed in {computation_time:.3f}s")
            return result
            
        except Exception as e:
            self.logger.error(f"Secure computation failed: {e}")
            raise


class SecurityAuditManager:
    """Comprehensive security audit and monitoring."""
    
    def __init__(self, audit_log_path: str = "security_audit.log"):
        self.audit_log_path = Path(audit_log_path)
        self.security_events: deque = deque(maxlen=10000)
        self.threat_patterns: Dict[str, int] = defaultdict(int)
        
        self.logger = logging.getLogger(f"{__name__}.SecurityAuditManager")
        
        # Setup audit logging
        self._setup_audit_logging()
    
    def _setup_audit_logging(self) -> None:
        """Setup dedicated audit logging."""
        audit_handler = logging.FileHandler(self.audit_log_path)
        audit_formatter = logging.Formatter(
            '%(asctime)s - AUDIT - %(levelname)s - %(message)s'
        )
        audit_handler.setFormatter(audit_formatter)
        
        audit_logger = logging.getLogger('security_audit')
        audit_logger.addHandler(audit_handler)
        audit_logger.setLevel(logging.INFO)
        
        self.audit_logger = audit_logger
    
    def log_security_event(self, event: SecurityEvent) -> None:
        """Log security event for audit trail."""
        self.security_events.append(event)
        self.threat_patterns[event.event_type] += 1
        
        # Log to audit trail
        self.audit_logger.info(
            f"Event: {event.event_type} | Threat: {event.threat_level.value} | "
            f"Source: {event.source} | Description: {event.description}"
        )
        
        # Check for threat patterns
        self._analyze_threat_patterns()
    
    def _analyze_threat_patterns(self) -> None:
        """Analyze patterns in security events."""
        recent_events = [e for e in self.security_events 
                        if (datetime.now() - e.timestamp).total_seconds() < 3600]  # Last hour
        
        if len(recent_events) > 50:  # High volume of events
            self.audit_logger.warning(f"High volume of security events: {len(recent_events)} in last hour")
        
        # Check for repeated threat types
        threat_counts = defaultdict(int)
        for event in recent_events:
            threat_counts[event.event_type] += 1
        
        for threat_type, count in threat_counts.items():
            if count > 10:  # Repeated threats
                self.audit_logger.warning(f"Repeated threat pattern: {threat_type} occurred {count} times")
    
    def generate_security_report(self, 
                               hours_back: int = 24) -> Dict[str, Any]:
        """Generate comprehensive security report."""
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        recent_events = [e for e in self.security_events if e.timestamp > cutoff_time]
        
        # Analyze events by threat level
        threat_level_counts = defaultdict(int)
        event_type_counts = defaultdict(int)
        
        for event in recent_events:
            threat_level_counts[event.threat_level.value] += 1
            event_type_counts[event.event_type] += 1
        
        # Calculate risk metrics
        total_events = len(recent_events)
        high_risk_events = sum(1 for e in recent_events 
                              if e.threat_level in [ThreatLevel.HIGH, ThreatLevel.CRITICAL])
        
        risk_score = (high_risk_events / max(total_events, 1)) * 100
        
        return {
            "report_period_hours": hours_back,
            "total_events": total_events,
            "high_risk_events": high_risk_events,
            "risk_score_percentage": risk_score,
            "threat_level_distribution": dict(threat_level_counts),
            "event_type_distribution": dict(event_type_counts),
            "top_threat_sources": self._get_top_threat_sources(recent_events),
            "recommendations": self._generate_security_recommendations(recent_events),
            "timestamp": datetime.now().isoformat()
        }
    
    def _get_top_threat_sources(self, events: List[SecurityEvent]) -> List[Tuple[str, int]]:
        """Get top threat sources from events."""
        source_counts = defaultdict(int)
        for event in events:
            source_counts[event.source] += 1
        
        return sorted(source_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    
    def _generate_security_recommendations(self, events: List[SecurityEvent]) -> List[str]:
        """Generate security recommendations based on events."""
        recommendations = []
        
        # Analyze patterns and generate recommendations
        event_types = [e.event_type for e in events]
        
        if event_types.count("rate_limit_exceeded") > 10:
            recommendations.append("Consider implementing more aggressive rate limiting")
        
        if event_types.count("potential_adversarial_input") > 5:
            recommendations.append("Enhance adversarial input detection mechanisms")
        
        if event_types.count("oversized_input") > 5:
            recommendations.append("Implement stricter input size validation")
        
        high_risk_count = sum(1 for e in events if e.threat_level == ThreatLevel.HIGH)
        if high_risk_count > 10:
            recommendations.append("Review and strengthen security policies")
        
        return recommendations


class ComprehensiveSecurityManager:
    """Central manager for all security components."""
    
    def __init__(self, 
                 security_level: SecurityLevel = SecurityLevel.PRODUCTION,
                 config_path: Optional[str] = None):
        self.security_level = security_level
        self.config = self._load_security_config(config_path)
        
        # Initialize security components
        self.input_sanitizer = InputSanitizer(security_level)
        self.model_verifier = ModelIntegrityVerifier(security_level)
        self.secure_computation = SecureComputationManager(security_level)
        self.audit_manager = SecurityAuditManager()
        
        self.logger = logging.getLogger(f"{__name__}.ComprehensiveSecurityManager")
        
        # Start security monitoring
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._security_monitor_loop, daemon=True)
        self._monitor_thread.start()
        
        self.logger.info(f"Security manager initialized with level: {security_level.value}")
    
    def _load_security_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load security configuration."""
        default_config = {
            "audit_enabled": True,
            "encryption_enabled": self.security_level in [SecurityLevel.HIGH_SECURITY, SecurityLevel.GOVERNMENT],
            "monitoring_interval_seconds": 60,
            "alert_thresholds": {
                "high_risk_events_per_hour": 20,
                "total_events_per_hour": 100
            }
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    file_config = json.load(f)
                default_config.update(file_config)
            except Exception as e:
                self.logger.warning(f"Failed to load security config: {e}")
        
        return default_config
    
    def validate_and_process(self, 
                           input_data: Any,
                           processing_fn: Callable,
                           client_id: str = "unknown",
                           **kwargs) -> Tuple[bool, Any, ValidationResult]:
        """Comprehensive validation and secure processing."""
        # Input validation
        validation_result = self.input_sanitizer.validate_input(
            input_data, client_id, kwargs
        )
        
        if not validation_result.valid:
            # Log security event
            self.audit_manager.log_security_event(
                SecurityEvent(
                    event_type="input_validation_failed",
                    threat_level=ThreatLevel.HIGH,
                    description=f"Input validation failed: {', '.join(validation_result.threats_detected)}",
                    source=client_id,
                    metadata={"risk_score": validation_result.risk_score}
                )
            )
            return False, None, validation_result
        
        # Secure processing
        try:
            result = self.secure_computation.secure_computation(
                processing_fn, input_data, **kwargs
            )
            
            # Log successful processing
            self.audit_manager.log_security_event(
                SecurityEvent(
                    event_type="secure_processing_success",
                    threat_level=ThreatLevel.LOW,
                    description="Input processed successfully with security measures",
                    source=client_id,
                    metadata={"processing_time_ms": validation_result.processing_time_ms}
                )
            )
            
            return True, result, validation_result
            
        except Exception as e:
            # Log processing error
            self.audit_manager.log_security_event(
                SecurityEvent(
                    event_type="processing_error",
                    threat_level=ThreatLevel.MEDIUM,
                    description=f"Processing failed: {str(e)}",
                    source=client_id,
                    metadata={"error_type": type(e).__name__}
                )
            )
            return False, None, validation_result
    
    def _security_monitor_loop(self) -> None:
        """Security monitoring loop."""
        while self._monitoring:
            try:
                self._perform_security_monitoring()
                time.sleep(self.config.get("monitoring_interval_seconds", 60))
            except Exception as e:
                self.logger.error(f"Security monitoring error: {e}")
                time.sleep(30)
    
    def _perform_security_monitoring(self) -> None:
        """Perform periodic security monitoring."""
        # Generate security report
        report = self.audit_manager.generate_security_report(hours_back=1)
        
        # Check alert thresholds
        thresholds = self.config.get("alert_thresholds", {})
        
        if report["high_risk_events"] > thresholds.get("high_risk_events_per_hour", 20):
            self.logger.critical(f"HIGH RISK ALERT: {report['high_risk_events']} high-risk events in last hour")
        
        if report["total_events"] > thresholds.get("total_events_per_hour", 100):
            self.logger.warning(f"High event volume: {report['total_events']} events in last hour")
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status."""
        return {
            "security_level": self.security_level.value,
            "encryption_enabled": self.secure_computation.encryption_enabled,
            "monitoring_active": self._monitoring,
            "recent_report": self.audit_manager.generate_security_report(hours_back=24),
            "input_sanitizer_stats": {
                "rate_limiter_active": True,
                "adversarial_detection_active": True,
                "thresholds": self.input_sanitizer.thresholds
            },
            "model_verification_stats": {
                "registered_models": len(self.model_verifier.known_checksums),
                "verifications_performed": len(self.model_verifier.verification_log)
            }
        }
    
    def shutdown(self) -> None:
        """Shutdown security manager gracefully."""
        self._monitoring = False
        
        if hasattr(self, '_monitor_thread'):
            self._monitor_thread.join(timeout=5)
        
        self.logger.info("Security manager shutdown complete")


# Export key classes
__all__ = [
    "ComprehensiveSecurityManager",
    "InputSanitizer",
    "ModelIntegrityVerifier",
    "SecureComputationManager",
    "SecurityAuditManager",
    "SecurityLevel",
    "ThreatLevel",
    "SecurityEvent",
    "ValidationResult"
]
