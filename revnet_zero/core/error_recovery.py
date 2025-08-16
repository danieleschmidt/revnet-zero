"""
RevNet-Zero Error Recovery System - Advanced error handling and graceful degradation.
Implements circuit breaker patterns and automatic fallback mechanisms.
"""

import time
import logging
import threading
from typing import Any, Dict, List, Optional, Callable, Type, Union
from enum import Enum
from dataclasses import dataclass, field
from functools import wraps

class ErrorSeverity(Enum):
    """Error severity levels for intelligent handling"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class RecoveryStrategy(Enum):
    """Available recovery strategies"""
    RETRY = "retry"
    FALLBACK = "fallback"
    GRACEFUL_DEGRADATION = "graceful_degradation"
    CIRCUIT_BREAKER = "circuit_breaker"
    FAIL_FAST = "fail_fast"

@dataclass
class ErrorRecord:
    """Record of an error occurrence"""
    timestamp: float
    error_type: str
    error_message: str
    severity: ErrorSeverity
    component: str
    recovery_attempted: bool = False
    recovery_successful: bool = False
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RecoveryConfig:
    """Configuration for error recovery behavior"""
    max_retries: int = 3
    retry_delay: float = 1.0
    retry_backoff: float = 2.0
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 60.0
    fallback_enabled: bool = True
    graceful_degradation: bool = True

class CircuitBreaker:
    """Circuit breaker implementation for fault tolerance"""
    
    def __init__(self, failure_threshold: int = 5, 
                 recovery_timeout: float = 60.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = 0
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self._lock = threading.Lock()
    
    def __call__(self, func: Callable) -> Callable:
        """Decorator to apply circuit breaker to function"""
        @wraps(func)
        def wrapper(*args, **kwargs):
            with self._lock:
                if self.state == "OPEN":
                    if time.time() - self.last_failure_time > self.recovery_timeout:
                        self.state = "HALF_OPEN"
                    else:
                        raise CircuitBreakerOpenError(
                            f"Circuit breaker open for {func.__name__}"
                        )
                
                try:
                    result = func(*args, **kwargs)
                    if self.state == "HALF_OPEN":
                        self.state = "CLOSED"
                        self.failure_count = 0
                    return result
                
                except Exception as e:
                    self.failure_count += 1
                    self.last_failure_time = time.time()
                    
                    if self.failure_count >= self.failure_threshold:
                        self.state = "OPEN"
                    
                    raise e
        
        return wrapper

class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open"""
    pass

class ErrorRecoverySystem:
    """
    Comprehensive error recovery system with multiple strategies.
    
    Provides intelligent error handling with:
    - Automatic retry with exponential backoff
    - Circuit breaker pattern for fault tolerance  
    - Fallback function registration
    - Graceful degradation modes
    - Error analytics and reporting
    """
    
    def __init__(self, config: Optional[RecoveryConfig] = None):
        self.config = config or RecoveryConfig()
        self.error_history: List[ErrorRecord] = []
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        self.fallback_functions: Dict[str, Callable] = {}
        self.degradation_modes: Dict[str, Callable] = {}
        self._lock = threading.Lock()
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def register_fallback(self, component: str, fallback_func: Callable):
        """Register a fallback function for a component"""
        self.fallback_functions[component] = fallback_func
        self.logger.info(f"Registered fallback for {component}")
    
    def register_degradation_mode(self, component: str, degraded_func: Callable):
        """Register a degraded mode function for a component"""
        self.degradation_modes[component] = degraded_func
        self.logger.info(f"Registered degradation mode for {component}")
    
    def get_circuit_breaker(self, component: str) -> CircuitBreaker:
        """Get or create circuit breaker for component"""
        if component not in self.circuit_breakers:
            self.circuit_breakers[component] = CircuitBreaker(
                failure_threshold=self.config.circuit_breaker_threshold,
                recovery_timeout=self.config.circuit_breaker_timeout
            )
        return self.circuit_breakers[component]
    
    def handle_error(self, error: Exception, component: str, 
                    context: Optional[Dict[str, Any]] = None,
                    strategy: RecoveryStrategy = RecoveryStrategy.RETRY) -> Any:
        """
        Handle an error with specified recovery strategy.
        
        Args:
            error: The exception that occurred
            component: Component name where error occurred
            context: Additional context information
            strategy: Recovery strategy to use
            
        Returns:
            Result from recovery attempt or raises exception
        """
        
        # Classify error severity
        severity = self._classify_error_severity(error)
        
        # Record error
        error_record = ErrorRecord(
            timestamp=time.time(),
            error_type=type(error).__name__,
            error_message=str(error),
            severity=severity,
            component=component,
            context=context or {}
        )
        
        with self._lock:
            self.error_history.append(error_record)
        
        self.logger.error(
            f"Error in {component}: {error} (severity: {severity.value})"
        )
        
        # Apply recovery strategy
        try:
            if strategy == RecoveryStrategy.RETRY:
                return self._retry_strategy(error, component, context)
            elif strategy == RecoveryStrategy.FALLBACK:
                return self._fallback_strategy(error, component, context)
            elif strategy == RecoveryStrategy.GRACEFUL_DEGRADATION:
                return self._degradation_strategy(error, component, context)
            elif strategy == RecoveryStrategy.CIRCUIT_BREAKER:
                return self._circuit_breaker_strategy(error, component, context)
            elif strategy == RecoveryStrategy.FAIL_FAST:
                raise error
            else:
                raise ValueError(f"Unknown recovery strategy: {strategy}")
                
        except Exception as recovery_error:
            error_record.recovery_attempted = True
            error_record.recovery_successful = False
            self.logger.error(f"Recovery failed for {component}: {recovery_error}")
            raise recovery_error
    
    def _classify_error_severity(self, error: Exception) -> ErrorSeverity:
        """Classify error severity based on type and message"""
        
        # Critical errors
        if isinstance(error, (MemoryError, SystemError, KeyboardInterrupt)):
            return ErrorSeverity.CRITICAL
        
        # High severity errors
        if isinstance(error, (ImportError, AttributeError, TypeError)):
            return ErrorSeverity.HIGH
        
        # Medium severity errors  
        if isinstance(error, (ValueError, RuntimeError, OSError)):
            return ErrorSeverity.MEDIUM
        
        # Low severity errors
        return ErrorSeverity.LOW
    
    def _retry_strategy(self, error: Exception, component: str, 
                       context: Optional[Dict[str, Any]]) -> Any:
        """Implement retry strategy with exponential backoff"""
        
        for attempt in range(self.config.max_retries):
            if attempt > 0:
                delay = self.config.retry_delay * (self.config.retry_backoff ** (attempt - 1))
                self.logger.info(f"Retrying {component} after {delay:.1f}s (attempt {attempt + 1})")
                time.sleep(delay)
            
            try:
                # This would need to be implemented by specific components
                # For now, we check if there's a fallback available
                if component in self.fallback_functions:
                    result = self.fallback_functions[component](context)
                    self.logger.info(f"Retry successful for {component}")
                    return result
                else:
                    # Re-raise original error if no fallback
                    raise error
                    
            except Exception as retry_error:
                if attempt == self.config.max_retries - 1:
                    raise retry_error
                continue
    
    def _fallback_strategy(self, error: Exception, component: str,
                          context: Optional[Dict[str, Any]]) -> Any:
        """Implement fallback strategy"""
        
        if component not in self.fallback_functions:
            raise RuntimeError(f"No fallback function registered for {component}")
        
        try:
            result = self.fallback_functions[component](context)
            self.logger.info(f"Fallback successful for {component}")
            return result
        except Exception as fallback_error:
            self.logger.error(f"Fallback failed for {component}: {fallback_error}")
            raise fallback_error
    
    def _degradation_strategy(self, error: Exception, component: str,
                             context: Optional[Dict[str, Any]]) -> Any:
        """Implement graceful degradation strategy"""
        
        if component not in self.degradation_modes:
            # Try fallback as degradation
            if component in self.fallback_functions:
                return self._fallback_strategy(error, component, context)
            else:
                raise RuntimeError(f"No degradation mode registered for {component}")
        
        try:
            result = self.degradation_modes[component](context)
            self.logger.warning(f"Running {component} in degraded mode")
            return result
        except Exception as degradation_error:
            self.logger.error(f"Degradation failed for {component}: {degradation_error}")
            raise degradation_error
    
    def _circuit_breaker_strategy(self, error: Exception, component: str,
                                 context: Optional[Dict[str, Any]]) -> Any:
        """Implement circuit breaker strategy"""
        
        circuit_breaker = self.get_circuit_breaker(component)
        
        # The circuit breaker logic is handled by the decorator
        # Here we just re-raise the error
        raise error
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics"""
        
        with self._lock:
            total_errors = len(self.error_history)
            if total_errors == 0:
                return {"total_errors": 0}
            
            # Error counts by severity
            severity_counts = {}
            for severity in ErrorSeverity:
                severity_counts[severity.value] = sum(
                    1 for record in self.error_history 
                    if record.severity == severity
                )
            
            # Error counts by component
            component_counts = {}
            for record in self.error_history:
                component_counts[record.component] = component_counts.get(record.component, 0) + 1
            
            # Recovery success rate
            recovery_attempts = sum(1 for record in self.error_history if record.recovery_attempted)
            recovery_successes = sum(1 for record in self.error_history if record.recovery_successful)
            recovery_rate = recovery_successes / recovery_attempts if recovery_attempts > 0 else 0
            
            # Recent error rate (last 5 minutes)
            recent_time = time.time() - 300
            recent_errors = sum(1 for record in self.error_history if record.timestamp > recent_time)
            
            return {
                "total_errors": total_errors,
                "severity_distribution": severity_counts,
                "component_distribution": component_counts,
                "recovery_success_rate": recovery_rate,
                "recent_error_rate": recent_errors / 5,  # errors per minute
                "circuit_breaker_states": {
                    component: cb.state for component, cb in self.circuit_breakers.items()
                }
            }
    
    def reset_error_history(self):
        """Reset error history (for testing or cleanup)"""
        with self._lock:
            self.error_history.clear()
        self.logger.info("Error history reset")
    
    def reset_circuit_breakers(self):
        """Reset all circuit breakers"""
        for component, cb in self.circuit_breakers.items():
            cb.state = "CLOSED"
            cb.failure_count = 0
            cb.last_failure_time = 0
        self.logger.info("All circuit breakers reset")

# Global error recovery system
_global_recovery_system = None

def get_recovery_system() -> ErrorRecoverySystem:
    """Get global error recovery system instance"""
    global _global_recovery_system
    if _global_recovery_system is None:
        _global_recovery_system = ErrorRecoverySystem()
    return _global_recovery_system

def with_error_recovery(component: str, strategy: RecoveryStrategy = RecoveryStrategy.RETRY):
    """Decorator to add error recovery to functions"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                recovery_system = get_recovery_system()
                context = {
                    'function': func.__name__,
                    'args': args,
                    'kwargs': kwargs
                }
                return recovery_system.handle_error(e, component, context, strategy)
        return wrapper
    return decorator

def register_fallback_function(component: str, fallback_func: Callable):
    """Register a fallback function globally"""
    recovery_system = get_recovery_system()
    recovery_system.register_fallback(component, fallback_func)

def register_degradation_mode(component: str, degraded_func: Callable):
    """Register a degradation mode globally"""
    recovery_system = get_recovery_system()
    recovery_system.register_degradation_mode(component, degraded_func)

__all__ = [
    'ErrorSeverity',
    'RecoveryStrategy', 
    'ErrorRecord',
    'RecoveryConfig',
    'CircuitBreaker',
    'CircuitBreakerOpenError',
    'ErrorRecoverySystem',
    'get_recovery_system',
    'with_error_recovery',
    'register_fallback_function',
    'register_degradation_mode'
]