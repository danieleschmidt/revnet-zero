"""
Resilient Error Handling and Recovery System
"""

import torch
import torch.nn as nn
import logging
import traceback
import time
import threading
from typing import Dict, List, Optional, Tuple, Any, Callable, Type
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import json
import pickle
from pathlib import Path
from contextlib import contextmanager
import signal
import psutil

class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class RecoveryAction(Enum):
    """Available recovery actions"""
    RETRY = "retry"
    FALLBACK = "fallback"
    RESTART = "restart"
    ABORT = "abort"
    IGNORE = "ignore"

@dataclass
class ErrorContext:
    """Context information for error handling"""
    error_type: str
    error_message: str
    stack_trace: str
    timestamp: float
    module: str
    function: str
    severity: ErrorSeverity
    system_state: Dict[str, Any] = field(default_factory=dict)
    recovery_attempts: int = 0
    max_recovery_attempts: int = 3

@dataclass
class RecoveryStrategy:
    """Recovery strategy for handling errors"""
    error_pattern: str
    actions: List[RecoveryAction]
    max_attempts: int = 3
    backoff_factor: float = 2.0
    timeout: float = 300.0  # 5 minutes
    custom_handler: Optional[Callable] = None

class ErrorHandler(ABC):
    """Base class for error handlers"""
    
    def __init__(self, name: str):
        self.name = name
        
    @abstractmethod
    def can_handle(self, error_context: ErrorContext) -> bool:
        """Check if this handler can handle the error"""
        pass
        
    @abstractmethod
    def handle(self, error_context: ErrorContext) -> Tuple[bool, Any]:
        """Handle the error, return (success, result)"""
        pass

class MemoryErrorHandler(ErrorHandler):
    """Handler for CUDA out of memory errors"""
    
    def __init__(self):
        super().__init__("memory_error_handler")
        
    def can_handle(self, error_context: ErrorContext) -> bool:
        """Check if this is a memory error"""
        memory_indicators = [
            "CUDA out of memory",
            "RuntimeError: out of memory", 
            "OutOfMemoryError",
            "cuda runtime error"
        ]
        
        return any(indicator in error_context.error_message for indicator in memory_indicators)
        
    def handle(self, error_context: ErrorContext) -> Tuple[bool, Any]:
        """Handle memory error with progressive recovery strategies"""
        
        recovery_steps = [
            self._clear_gpu_cache,
            self._reduce_batch_size,
            self._enable_gradient_checkpointing,
            self._move_to_cpu_fallback
        ]
        
        for i, step in enumerate(recovery_steps):
            try:
                result = step(error_context)
                if result:
                    logging.info(f"Memory error resolved using step {i+1}: {step.__name__}")
                    return True, result
            except Exception as e:
                logging.warning(f"Recovery step {i+1} failed: {e}")
                
        return False, None
        
    def _clear_gpu_cache(self, error_context: ErrorContext) -> bool:
        """Clear GPU memory cache"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            return True
        return False
        
    def _reduce_batch_size(self, error_context: ErrorContext) -> bool:
        """Reduce batch size in system state"""
        system_state = error_context.system_state
        
        if 'batch_size' in system_state:
            old_batch_size = system_state['batch_size']
            new_batch_size = max(1, old_batch_size // 2)
            system_state['batch_size'] = new_batch_size
            
            logging.info(f"Reduced batch size from {old_batch_size} to {new_batch_size}")
            return True
            
        return False
        
    def _enable_gradient_checkpointing(self, error_context: ErrorContext) -> bool:
        """Enable gradient checkpointing"""
        system_state = error_context.system_state
        
        if 'model' in system_state:
            model = system_state['model']
            if hasattr(model, 'gradient_checkpointing'):
                model.gradient_checkpointing = True
                logging.info("Enabled gradient checkpointing")
                return True
                
        return False
        
    def _move_to_cpu_fallback(self, error_context: ErrorContext) -> bool:
        """Move computation to CPU as fallback"""
        system_state = error_context.system_state
        
        if 'model' in system_state and 'device' in system_state:
            model = system_state['model']
            model.cpu()
            system_state['device'] = 'cpu'
            
            logging.warning("Moved model to CPU due to memory constraints")
            return True
            
        return False

class NetworkErrorHandler(ErrorHandler):
    """Handler for network-related errors"""
    
    def __init__(self):
        super().__init__("network_error_handler")
        
    def can_handle(self, error_context: ErrorContext) -> bool:
        """Check if this is a network error"""
        network_indicators = [
            "ConnectionError",
            "TimeoutError", 
            "URLError",
            "HTTPError",
            "socket.timeout",
            "requests.exceptions"
        ]
        
        return any(indicator in error_context.error_message for indicator in network_indicators)
        
    def handle(self, error_context: ErrorContext) -> Tuple[bool, Any]:
        """Handle network error with retries and exponential backoff"""
        
        max_retries = 5
        base_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                delay = base_delay * (2 ** attempt)
                time.sleep(delay)
                
                # Re-attempt the operation (this would be passed in context)
                if 'retry_function' in error_context.system_state:
                    retry_function = error_context.system_state['retry_function']
                    result = retry_function()
                    return True, result
                    
                return True, None
                
            except Exception as e:
                logging.warning(f"Network retry attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    return False, None
                    
        return False, None

class ModelStateCorruptionHandler(ErrorHandler):
    """Handler for model state corruption errors"""
    
    def __init__(self):
        super().__init__("model_corruption_handler")
        
    def can_handle(self, error_context: ErrorContext) -> bool:
        """Check if this is a model corruption error"""
        corruption_indicators = [
            "nan",
            "inf",
            "gradient explosion",
            "invalid tensor",
            "corrupted",
            "RuntimeError: Function"
        ]
        
        return any(indicator.lower() in error_context.error_message.lower() 
                  for indicator in corruption_indicators)
                  
    def handle(self, error_context: ErrorContext) -> Tuple[bool, Any]:
        """Handle model corruption with state restoration"""
        
        try:
            # Try to restore from last known good state
            if 'model' in error_context.system_state:
                model = error_context.system_state['model']
                
                # Check for saved checkpoint
                checkpoint_path = error_context.system_state.get('checkpoint_path')
                if checkpoint_path and Path(checkpoint_path).exists():
                    checkpoint = torch.load(checkpoint_path, map_location='cpu')
                    model.load_state_dict(checkpoint['model_state_dict'])
                    
                    logging.info(f"Restored model from checkpoint: {checkpoint_path}")
                    return True, model
                    
                # Fallback: reinitialize problematic parameters
                self._reinitialize_nan_parameters(model)
                return True, model
                
        except Exception as e:
            logging.error(f"Failed to restore model state: {e}")
            
        return False, None
        
    def _reinitialize_nan_parameters(self, model: nn.Module) -> None:
        """Reinitialize parameters that contain NaN or Inf"""
        
        for name, param in model.named_parameters():
            if torch.isnan(param).any() or torch.isinf(param).any():
                # Reinitialize using Xavier uniform
                if param.dim() >= 2:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.uniform_(param, -0.1, 0.1)
                    
                logging.warning(f"Reinitialized parameter {name} due to NaN/Inf values")

class ResilientErrorHandlingSystem:
    """Comprehensive error handling and recovery system"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        
        # Error handlers
        self.error_handlers: List[ErrorHandler] = []
        self._setup_default_handlers()
        
        # Recovery strategies
        self.recovery_strategies: Dict[str, RecoveryStrategy] = {}
        self._setup_default_strategies()
        
        # Error history and statistics
        self.error_history: List[ErrorContext] = []
        self.error_stats: Dict[str, int] = {}
        self.recovery_stats: Dict[str, Dict[str, int]] = {}
        
        # State management
        self.system_checkpoints: Dict[str, Any] = {}
        self.last_known_good_state: Optional[Dict] = None
        
        # Monitoring
        self.error_monitoring_active = True
        self.monitoring_thread: Optional[threading.Thread] = None
        
        self.logger = logging.getLogger(__name__)
        
        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()
        
    def _setup_default_handlers(self) -> None:
        """Setup default error handlers"""
        
        self.error_handlers.extend([
            MemoryErrorHandler(),
            NetworkErrorHandler(),
            ModelStateCorruptionHandler()
        ])
        
    def _setup_default_strategies(self) -> None:
        """Setup default recovery strategies"""
        
        # Memory error strategy
        self.recovery_strategies['memory_error'] = RecoveryStrategy(
            error_pattern='.*out of memory.*',
            actions=[RecoveryAction.RETRY, RecoveryAction.FALLBACK],
            max_attempts=3,
            backoff_factor=1.5,
            timeout=180.0
        )
        
        # Network error strategy
        self.recovery_strategies['network_error'] = RecoveryStrategy(
            error_pattern='.*Connection.*|.*Timeout.*|.*HTTP.*',
            actions=[RecoveryAction.RETRY],
            max_attempts=5,
            backoff_factor=2.0,
            timeout=300.0
        )
        
        # Model corruption strategy
        self.recovery_strategies['model_corruption'] = RecoveryStrategy(
            error_pattern='.*nan.*|.*inf.*|.*gradient.*explosion.*',
            actions=[RecoveryAction.FALLBACK, RecoveryAction.RESTART],
            max_attempts=2,
            timeout=120.0
        )
        
    def _setup_signal_handlers(self) -> None:
        """Setup signal handlers for graceful error handling"""
        
        def signal_handler(signum, frame):
            self.logger.info(f"Received signal {signum}, initiating graceful shutdown")
            self.save_error_statistics()
            
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
    def add_error_handler(self, handler: ErrorHandler) -> None:
        """Add a custom error handler"""
        self.error_handlers.append(handler)
        
    def add_recovery_strategy(self, name: str, strategy: RecoveryStrategy) -> None:
        """Add a custom recovery strategy"""
        self.recovery_strategies[name] = strategy
        
    @contextmanager
    def resilient_execution(self, 
                          operation_name: str,
                          system_state: Optional[Dict] = None,
                          max_attempts: int = 3):
        """Context manager for resilient execution with automatic error handling"""
        
        system_state = system_state or {}
        attempt = 0
        
        while attempt < max_attempts:
            try:
                # Create checkpoint before execution
                self.create_checkpoint(f"{operation_name}_checkpoint")
                
                yield
                
                # Operation succeeded
                self.logger.debug(f"Operation {operation_name} completed successfully")
                return
                
            except Exception as e:
                attempt += 1
                
                # Create error context
                error_context = ErrorContext(
                    error_type=type(e).__name__,
                    error_message=str(e),
                    stack_trace=traceback.format_exc(),
                    timestamp=time.time(),
                    module=self.__class__.__module__,
                    function=operation_name,
                    severity=self._determine_error_severity(e),
                    system_state=system_state,
                    recovery_attempts=attempt,
                    max_recovery_attempts=max_attempts
                )
                
                # Handle the error
                handled, result = self.handle_error(error_context)
                
                if handled and attempt < max_attempts:
                    self.logger.info(f"Error handled, retrying operation {operation_name} (attempt {attempt + 1})")
                    continue
                else:
                    # All recovery attempts failed
                    self.logger.error(f"All recovery attempts failed for operation {operation_name}")
                    raise e
                    
    def handle_error(self, error_context: ErrorContext) -> Tuple[bool, Any]:
        """Handle an error using appropriate handlers and strategies"""
        
        # Log the error
        self._log_error(error_context)
        
        # Update statistics
        self._update_error_stats(error_context)
        
        # Try to find a suitable handler
        for handler in self.error_handlers:
            if handler.can_handle(error_context):
                try:
                    success, result = handler.handle(error_context)
                    
                    # Update recovery statistics
                    self._update_recovery_stats(handler.name, success)
                    
                    if success:
                        self.logger.info(f"Error handled successfully by {handler.name}")
                        return True, result
                        
                except Exception as handler_error:
                    self.logger.error(f"Error handler {handler.name} failed: {handler_error}")
                    
        # No handler could resolve the error
        self.logger.warning(f"No handler could resolve error: {error_context.error_type}")
        
        # Try fallback strategies
        return self._apply_fallback_strategies(error_context)
        
    def _apply_fallback_strategies(self, error_context: ErrorContext) -> Tuple[bool, Any]:
        """Apply fallback recovery strategies"""
        
        # Find matching recovery strategy
        for strategy_name, strategy in self.recovery_strategies.items():
            if self._matches_pattern(error_context.error_message, strategy.error_pattern):
                return self._execute_recovery_strategy(error_context, strategy)
                
        # No strategy found, try generic recovery
        return self._generic_recovery(error_context)
        
    def _execute_recovery_strategy(self, 
                                 error_context: ErrorContext,
                                 strategy: RecoveryStrategy) -> Tuple[bool, Any]:
        """Execute a recovery strategy"""
        
        for action in strategy.actions:
            try:
                if action == RecoveryAction.RETRY:
                    # Implement retry with backoff
                    if error_context.recovery_attempts < strategy.max_attempts:
                        delay = strategy.backoff_factor ** error_context.recovery_attempts
                        time.sleep(delay)
                        return True, None
                        
                elif action == RecoveryAction.FALLBACK:
                    # Try to restore from checkpoint
                    restored_state = self.restore_checkpoint()
                    if restored_state:
                        return True, restored_state
                        
                elif action == RecoveryAction.RESTART:
                    # Restart the affected component
                    return self._restart_component(error_context)
                    
                elif action == RecoveryAction.IGNORE:
                    # Log and continue
                    self.logger.warning(f"Ignoring error as per strategy: {error_context.error_message}")
                    return True, None
                    
            except Exception as e:
                self.logger.error(f"Recovery action {action} failed: {e}")
                
        return False, None
        
    def _restart_component(self, error_context: ErrorContext) -> Tuple[bool, Any]:
        """Restart the affected component"""
        
        # This would implement component restart logic
        # For now, we'll just log the restart attempt
        self.logger.info(f"Restarting component for error: {error_context.error_type}")
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return True, None
        
    def _generic_recovery(self, error_context: ErrorContext) -> Tuple[bool, Any]:
        """Generic recovery when no specific strategy is available"""
        
        # Try system-level recovery actions
        recovery_actions = [
            self._clear_system_memory,
            self._reduce_system_load,
            self._reset_cuda_context
        ]
        
        for action in recovery_actions:
            try:
                if action():
                    return True, None
            except Exception as e:
                self.logger.error(f"Generic recovery action failed: {e}")
                
        return False, None
        
    def _clear_system_memory(self) -> bool:
        """Clear system memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        return True
        
    def _reduce_system_load(self) -> bool:
        """Reduce system load"""
        # This could implement load reduction strategies
        # For now, just sleep briefly
        time.sleep(1.0)
        return True
        
    def _reset_cuda_context(self) -> bool:
        """Reset CUDA context"""
        if torch.cuda.is_available():
            try:
                # This is a drastic measure - only for critical situations
                torch.cuda.empty_cache()
                return True
            except:
                pass
        return False
        
    def create_checkpoint(self, checkpoint_name: str, state: Optional[Dict] = None) -> None:
        """Create a system checkpoint"""
        
        if state is None:
            state = {
                'timestamp': time.time(),
                'memory_usage': torch.cuda.memory_allocated() if torch.cuda.is_available() else 0,
                'system_metrics': {
                    'cpu_percent': psutil.cpu_percent(),
                    'memory_percent': psutil.virtual_memory().percent
                }
            }
            
        self.system_checkpoints[checkpoint_name] = state
        self.last_known_good_state = state.copy()
        
    def restore_checkpoint(self, checkpoint_name: Optional[str] = None) -> Optional[Dict]:
        """Restore from a checkpoint"""
        
        if checkpoint_name and checkpoint_name in self.system_checkpoints:
            return self.system_checkpoints[checkpoint_name]
        elif self.last_known_good_state:
            return self.last_known_good_state
            
        return None
        
    def _determine_error_severity(self, error: Exception) -> ErrorSeverity:
        """Determine error severity based on error type"""
        
        critical_errors = [
            'SystemExit',
            'KeyboardInterrupt', 
            'MemoryError',
            'RecursionError'
        ]
        
        high_errors = [
            'RuntimeError',
            'ValueError',
            'TypeError'
        ]
        
        error_name = type(error).__name__
        
        if error_name in critical_errors:
            return ErrorSeverity.CRITICAL
        elif error_name in high_errors:
            return ErrorSeverity.HIGH
        elif 'warning' in str(error).lower():
            return ErrorSeverity.LOW
        else:
            return ErrorSeverity.MEDIUM
            
    def _log_error(self, error_context: ErrorContext) -> None:
        """Log error with context"""
        
        log_data = {
            'timestamp': error_context.timestamp,
            'error_type': error_context.error_type,
            'error_message': error_context.error_message,
            'severity': error_context.severity.value,
            'module': error_context.module,
            'function': error_context.function,
            'recovery_attempts': error_context.recovery_attempts
        }
        
        if error_context.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(f"CRITICAL_ERROR: {json.dumps(log_data)}")
        elif error_context.severity == ErrorSeverity.HIGH:
            self.logger.error(f"ERROR: {json.dumps(log_data)}")
        else:
            self.logger.warning(f"WARNING: {json.dumps(log_data)}")
            
        # Store in history
        self.error_history.append(error_context)
        
        # Keep only last 1000 errors
        if len(self.error_history) > 1000:
            self.error_history = self.error_history[-1000:]
            
    def _update_error_stats(self, error_context: ErrorContext) -> None:
        """Update error statistics"""
        
        error_type = error_context.error_type
        self.error_stats[error_type] = self.error_stats.get(error_type, 0) + 1
        
    def _update_recovery_stats(self, handler_name: str, success: bool) -> None:
        """Update recovery statistics"""
        
        if handler_name not in self.recovery_stats:
            self.recovery_stats[handler_name] = {'success': 0, 'failure': 0}
            
        if success:
            self.recovery_stats[handler_name]['success'] += 1
        else:
            self.recovery_stats[handler_name]['failure'] += 1
            
    def _matches_pattern(self, text: str, pattern: str) -> bool:
        """Check if text matches pattern"""
        import re
        try:
            return bool(re.search(pattern, text, re.IGNORECASE))
        except:
            return pattern.lower() in text.lower()
            
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics"""
        
        total_errors = len(self.error_history)
        recent_errors = [e for e in self.error_history if time.time() - e.timestamp < 3600]  # Last hour
        
        severity_counts = {}
        for severity in ErrorSeverity:
            severity_counts[severity.value] = sum(1 for e in self.error_history if e.severity == severity)
            
        return {
            'total_errors': total_errors,
            'recent_errors_1h': len(recent_errors),
            'error_types': dict(self.error_stats),
            'severity_breakdown': severity_counts,
            'recovery_statistics': dict(self.recovery_stats),
            'active_checkpoints': len(self.system_checkpoints),
            'last_error': self.error_history[-1].timestamp if self.error_history else None
        }
        
    def save_error_statistics(self, filepath: Optional[str] = None) -> None:
        """Save error statistics to file"""
        
        if filepath is None:
            filepath = f"error_statistics_{int(time.time())}.json"
            
        stats = self.get_error_statistics()
        
        # Add error history (limited)
        stats['recent_error_history'] = [
            {
                'timestamp': e.timestamp,
                'error_type': e.error_type,
                'error_message': e.error_message[:200],  # Truncate long messages
                'severity': e.severity.value,
                'recovery_attempts': e.recovery_attempts
            }
            for e in self.error_history[-100:]  # Last 100 errors
        ]
        
        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=2)
            
        self.logger.info(f"Error statistics saved to {filepath}")

# Global instance for easy access
resilient_error_handler = ResilientErrorHandlingSystem()