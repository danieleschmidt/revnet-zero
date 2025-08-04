"""
Comprehensive logging system for RevNet-Zero.

This module provides structured logging with different levels,
performance tracking, and memory monitoring capabilities.
"""

import logging
import sys
import time
import json
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from contextlib import contextmanager
from functools import wraps
import threading
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class LogEntry:
    """Structured log entry."""
    timestamp: str
    level: str
    message: str
    module: str
    function: str
    extra: Dict[str, Any]


class RevNetLogger:
    """
    Enhanced logger for RevNet-Zero with structured logging and performance tracking.
    """
    
    def __init__(
        self,
        name: str = "revnet_zero",
        level: str = "INFO",
        log_file: Optional[Path] = None,
        structured_logging: bool = True,
        console_output: bool = True,
    ):
        """
        Initialize RevNet logger.
        
        Args:
            name: Logger name
            level: Logging level
            log_file: Optional log file path
            structured_logging: Whether to use structured JSON logging
            console_output: Whether to output to console
        """
        self.name = name
        self.structured_logging = structured_logging
        self.log_entries: List[LogEntry] = []
        self._lock = threading.Lock()
        
        # Setup Python logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(getattr(logging, level.upper()))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Console handler
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_formatter = self._create_formatter(structured=False)
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
        
        # File handler
        if log_file is not None:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file)
            
            if structured_logging:
                file_formatter = self._create_formatter(structured=True)
            else:
                file_formatter = self._create_formatter(structured=False)
            
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
        
        # Performance tracking
        self.performance_data = {}
        self.memory_snapshots = []
    
    def _create_formatter(self, structured: bool = False) -> logging.Formatter:
        """Create appropriate formatter."""
        if structured:
            return StructuredFormatter()
        else:
            return logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
    
    def _log_entry(self, level: str, message: str, extra: Dict[str, Any] = None):
        """Create and store log entry."""
        import inspect
        
        # Get caller information
        frame = inspect.currentframe()
        caller_frame = frame.f_back.f_back  # Go up two frames to get actual caller
        module = caller_frame.f_globals.get('__name__', 'unknown')
        function = caller_frame.f_code.co_name
        
        entry = LogEntry(
            timestamp=datetime.now().isoformat(),
            level=level,
            message=message,
            module=module,
            function=function,
            extra=extra or {}
        )
        
        with self._lock:
            self.log_entries.append(entry)
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self.logger.debug(message, extra=kwargs)
        self._log_entry("DEBUG", message, kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self.logger.info(message, extra=kwargs)
        self._log_entry("INFO", message, kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message."""
        self.logger.warning(message, extra=kwargs)
        self._log_entry("WARNING", message, kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        self.logger.error(message, extra=kwargs)
        self._log_entry("ERROR", message, kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message."""
        self.logger.critical(message, extra=kwargs)
        self._log_entry("CRITICAL", message, kwargs)
    
    def log_model_info(self, model_info: Dict[str, Any]):
        """Log model information."""
        self.info("Model configuration", model_info=model_info)
    
    def log_training_step(
        self,
        step: int,
        loss: float,
        learning_rate: float,
        memory_usage: Optional[int] = None,
        **metrics
    ):
        """Log training step information."""
        log_data = {
            "step": step,
            "loss": loss,
            "learning_rate": learning_rate,
        }
        
        if memory_usage is not None:
            log_data["memory_usage_gb"] = memory_usage / 1e9
        
        log_data.update(metrics)
        
        self.info(f"Training step {step}", **log_data)
    
    def log_memory_usage(self, operation: str, memory_before: int, memory_after: int):
        """Log memory usage for an operation."""
        memory_delta = memory_after - memory_before
        
        self.debug(
            f"Memory usage for {operation}",
            operation=operation,
            memory_before_gb=memory_before / 1e9,
            memory_after_gb=memory_after / 1e9,
            memory_delta_gb=memory_delta / 1e9
        )
    
    def log_performance_metrics(self, operation: str, duration: float, **metrics):
        """Log performance metrics."""
        self.info(
            f"Performance: {operation}",
            operation=operation,
            duration_seconds=duration,
            **metrics
        )
    
    def log_error_with_context(self, error: Exception, context: Dict[str, Any]):
        """Log error with additional context."""
        self.error(
            f"Error occurred: {str(error)}",
            error_type=type(error).__name__,
            error_message=str(error),
            context=context
        )
    
    def get_log_summary(self) -> Dict[str, Any]:
        """Get summary of log entries."""
        with self._lock:
            level_counts = {}
            for entry in self.log_entries:
                level_counts[entry.level] = level_counts.get(entry.level, 0) + 1
            
            return {
                "total_entries": len(self.log_entries),
                "level_counts": level_counts,
                "first_entry": self.log_entries[0].timestamp if self.log_entries else None,
                "last_entry": self.log_entries[-1].timestamp if self.log_entries else None,
            }
    
    def export_logs(self, file_path: Path):
        """Export logs to JSON file."""
        with self._lock:
            log_data = [asdict(entry) for entry in self.log_entries]
        
        with open(file_path, 'w') as f:
            json.dump(log_data, f, indent=2)
        
        self.info(f"Logs exported to {file_path}")


class StructuredFormatter(logging.Formatter):
    """Custom formatter for structured JSON logging."""
    
    def format(self, record):
        log_entry = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "funcName": record.funcName,
            "lineno": record.lineno,
        }
        
        # Add extra fields
        if hasattr(record, 'extra'):
            log_entry.update(record.extra)
        
        return json.dumps(log_entry)


class PerformanceTracker:
    """Performance tracking utilities."""
    
    def __init__(self, logger: RevNetLogger):
        self.logger = logger
        self.active_timers = {}
        self._lock = threading.Lock()
    
    @contextmanager
    def track_operation(self, operation_name: str, **context):
        """Context manager to track operation performance."""
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        try:
            yield
        finally:
            end_time = time.time()
            end_memory = self._get_memory_usage()
            duration = end_time - start_time
            
            # Log performance
            self.logger.log_performance_metrics(
                operation_name,
                duration,
                start_memory_gb=start_memory / 1e9 if start_memory else None,
                end_memory_gb=end_memory / 1e9 if end_memory else None,
                memory_delta_gb=(end_memory - start_memory) / 1e9 if start_memory and end_memory else None,
                **context
            )
    
    def start_timer(self, name: str):
        """Start a named timer."""
        with self._lock:
            self.active_timers[name] = time.time()
    
    def end_timer(self, name: str, **context) -> float:
        """End a named timer and return duration."""
        with self._lock:
            if name not in self.active_timers:
                self.logger.warning(f"Timer '{name}' was not started")
                return 0.0
            
            duration = time.time() - self.active_timers[name]
            del self.active_timers[name]
            
            self.logger.log_performance_metrics(name, duration, **context)
            return duration
    
    def _get_memory_usage(self) -> Optional[int]:
        """Get current memory usage."""
        try:
            import torch
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated()
        except ImportError:
            pass
        return None


def logged_method(operation_name: Optional[str] = None):
    """Decorator to automatically log method calls."""
    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            # Get logger from self or create default
            logger = getattr(self, 'logger', None)
            if logger is None:
                logger = get_logger(self.__class__.__name__)
            
            op_name = operation_name or f"{self.__class__.__name__}.{func.__name__}"
            
            # Create performance tracker
            tracker = PerformanceTracker(logger)
            
            try:
                with tracker.track_operation(op_name):
                    result = func(self, *args, **kwargs)
                
                logger.debug(f"Successfully completed {op_name}")
                return result
                
            except Exception as e:
                logger.log_error_with_context(
                    e, 
                    {
                        "operation": op_name,
                        "args_count": len(args),
                        "kwargs_keys": list(kwargs.keys())
                    }
                )
                raise
        
        return wrapper
    return decorator


def get_logger(
    name: str = "revnet_zero",
    level: str = "INFO",
    log_file: Optional[Union[str, Path]] = None,
) -> RevNetLogger:
    """
    Get or create a RevNet logger.
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Optional log file path
        
    Returns:
        RevNet logger instance
    """
    # Simple singleton pattern
    if not hasattr(get_logger, '_loggers'):
        get_logger._loggers = {}
    
    if name not in get_logger._loggers:
        log_path = Path(log_file) if log_file else None
        get_logger._loggers[name] = RevNetLogger(
            name=name,
            level=level,
            log_file=log_path
        )
    
    return get_logger._loggers[name]


def setup_logging(
    level: str = "INFO",
    log_dir: Optional[Union[str, Path]] = None,
    structured: bool = True,
) -> RevNetLogger:
    """
    Setup global logging configuration.
    
    Args:
        level: Global logging level
        log_dir: Directory for log files
        structured: Whether to use structured logging
        
    Returns:
        Main logger instance
    """
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "revnet_zero.log"
    else:
        log_file = None
    
    logger = RevNetLogger(
        name="revnet_zero",
        level=level,
        log_file=log_file,
        structured_logging=structured
    )
    
    logger.info("Logging system initialized", level=level, structured=structured)
    return logger


# Global logger instance
_global_logger = None


def get_global_logger() -> RevNetLogger:
    """Get the global logger instance."""
    global _global_logger
    if _global_logger is None:
        _global_logger = setup_logging()
    return _global_logger