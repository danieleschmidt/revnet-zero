"""
Comprehensive monitoring and observability for RevNet-Zero.

This module provides real-time monitoring, metrics collection,
health checks, and observability features for production deployments.
"""

import time
import threading
import logging
import json
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
import statistics


class MetricType(Enum):
    """Types of metrics we can collect."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    TIMER = "timer"


@dataclass
class Metric:
    """Individual metric data point."""
    name: str
    value: Union[float, int]
    metric_type: MetricType
    timestamp: float
    tags: Dict[str, str]
    unit: Optional[str] = None


@dataclass
class HealthStatus:
    """Health check result."""
    component: str
    status: str  # healthy, degraded, unhealthy
    message: str
    timestamp: float
    details: Dict[str, Any]


class MetricsCollector:
    """Thread-safe metrics collection system."""
    
    def __init__(self, retention_seconds: int = 3600):
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.retention_seconds = retention_seconds
        self._lock = threading.Lock()
        self._cleanup_thread = threading.Thread(target=self._cleanup_old_metrics, daemon=True)
        self._cleanup_thread.start()
        
    def record_counter(self, name: str, value: Union[float, int] = 1, 
                      tags: Optional[Dict[str, str]] = None):
        """Record a counter metric (monotonic increasing)."""
        self._record_metric(name, value, MetricType.COUNTER, tags or {})
    
    def record_gauge(self, name: str, value: Union[float, int], 
                    tags: Optional[Dict[str, str]] = None):
        """Record a gauge metric (point-in-time value)."""
        self._record_metric(name, value, MetricType.GAUGE, tags or {})
    
    def record_histogram(self, name: str, value: Union[float, int], 
                        tags: Optional[Dict[str, str]] = None):
        """Record a histogram metric (distribution of values)."""
        self._record_metric(name, value, MetricType.HISTOGRAM, tags or {})
    
    def record_timer(self, name: str, duration_seconds: float, 
                    tags: Optional[Dict[str, str]] = None):
        """Record a timer metric (duration measurement)."""
        self._record_metric(name, duration_seconds, MetricType.TIMER, tags or {})
    
    def _record_metric(self, name: str, value: Union[float, int], 
                      metric_type: MetricType, tags: Dict[str, str]):
        """Internal method to record metrics."""
        metric = Metric(
            name=name,
            value=value,
            metric_type=metric_type,
            timestamp=time.time(),
            tags=tags
        )
        
        with self._lock:
            self.metrics[name].append(metric)
    
    def get_metrics(self, name: Optional[str] = None, 
                   since_timestamp: Optional[float] = None) -> List[Metric]:
        """Get metrics, optionally filtered by name and time."""
        with self._lock:
            if name:
                metrics = list(self.metrics[name])
            else:
                metrics = []
                for metric_deque in self.metrics.values():
                    metrics.extend(metric_deque)
            
            if since_timestamp:
                metrics = [m for m in metrics if m.timestamp >= since_timestamp]
            
            return sorted(metrics, key=lambda m: m.timestamp)
    
    def get_metric_summary(self, name: str, window_seconds: int = 60) -> Dict[str, Any]:
        """Get statistical summary of a metric over a time window."""
        cutoff_time = time.time() - window_seconds
        recent_metrics = [m for m in self.get_metrics(name, cutoff_time)]
        
        if not recent_metrics:
            return {'count': 0, 'message': 'No data in time window'}
        
        values = [m.value for m in recent_metrics]
        
        return {
            'count': len(values),
            'sum': sum(values),
            'min': min(values),
            'max': max(values),
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'std': statistics.stdev(values) if len(values) > 1 else 0,
            'p95': statistics.quantiles(values, n=20)[18] if len(values) >= 20 else max(values),
            'p99': statistics.quantiles(values, n=100)[98] if len(values) >= 100 else max(values),
            'window_seconds': window_seconds
        }
    
    def _cleanup_old_metrics(self):
        """Background thread to clean up old metrics."""
        while True:
            try:
                time.sleep(300)  # Cleanup every 5 minutes
                cutoff_time = time.time() - self.retention_seconds
                
                with self._lock:
                    for name, metric_deque in self.metrics.items():
                        # Remove old metrics
                        while metric_deque and metric_deque[0].timestamp < cutoff_time:
                            metric_deque.popleft()
                            
            except Exception as e:
                logging.error(f"Error in metrics cleanup: {e}")


class PerformanceMonitor:
    """Monitor performance of RevNet-Zero operations."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        self.metrics = metrics_collector
        self.active_timers: Dict[str, float] = {}
    
    def time_operation(self, operation_name: str):
        """Context manager for timing operations."""
        return self.OperationTimer(self, operation_name)
    
    class OperationTimer:
        """Context manager for operation timing."""
        
        def __init__(self, monitor: 'PerformanceMonitor', operation_name: str):
            self.monitor = monitor
            self.operation_name = operation_name
            self.start_time = None
        
        def __enter__(self):
            self.start_time = time.time()
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            if self.start_time:
                duration = time.time() - self.start_time
                self.monitor.metrics.record_timer(
                    f"operation_duration",
                    duration,
                    {"operation": self.operation_name}
                )
                
                # Record success/failure
                status = "success" if exc_type is None else "failure"
                self.monitor.metrics.record_counter(
                    f"operation_count",
                    1,
                    {"operation": self.operation_name, "status": status}
                )
    
    def record_memory_usage(self, component: str, memory_bytes: int):
        """Record memory usage for a component."""
        self.metrics.record_gauge(
            "memory_usage_bytes",
            memory_bytes,
            {"component": component}
        )
    
    def record_throughput(self, operation: str, items_processed: int, 
                         duration_seconds: float):
        """Record throughput metrics."""
        throughput = items_processed / duration_seconds if duration_seconds > 0 else 0
        
        self.metrics.record_gauge(
            "throughput_items_per_second",
            throughput,
            {"operation": operation}
        )
        
        self.metrics.record_counter(
            "items_processed_total",
            items_processed,
            {"operation": operation}
        )


class HealthChecker:
    """Comprehensive health checking system."""
    
    def __init__(self):
        self.health_checks: Dict[str, Callable[[], HealthStatus]] = {}
        self.last_results: Dict[str, HealthStatus] = {}
        
    def register_health_check(self, component: str, check_func: Callable[[], HealthStatus]):
        """Register a health check function for a component."""
        self.health_checks[component] = check_func
    
    def check_component_health(self, component: str) -> HealthStatus:
        """Check health of a specific component."""
        if component not in self.health_checks:
            return HealthStatus(
                component=component,
                status="unknown",
                message=f"No health check registered for {component}",
                timestamp=time.time(),
                details={}
            )
        
        try:
            result = self.health_checks[component]()
            self.last_results[component] = result
            return result
        except Exception as e:
            error_status = HealthStatus(
                component=component,
                status="unhealthy",
                message=f"Health check failed: {e}",
                timestamp=time.time(),
                details={"error": str(e)}
            )
            self.last_results[component] = error_status
            return error_status
    
    def check_all_health(self) -> Dict[str, HealthStatus]:
        """Check health of all registered components."""
        results = {}
        for component in self.health_checks:
            results[component] = self.check_component_health(component)
        return results
    
    def get_overall_health(self) -> str:
        """Get overall system health status."""
        all_health = self.check_all_health()
        
        if not all_health:
            return "unknown"
        
        statuses = [status.status for status in all_health.values()]
        
        if "unhealthy" in statuses:
            return "unhealthy"
        elif "degraded" in statuses:
            return "degraded"
        elif all(s == "healthy" for s in statuses):
            return "healthy"
        else:
            return "unknown"


def default_health_checks() -> Dict[str, Callable[[], HealthStatus]]:
    """Get default health checks for RevNet-Zero components."""
    
    def check_pytorch_health() -> HealthStatus:
        """Check PyTorch availability and basic functionality."""
        try:
            import torch
            
            # Basic tensor operations
            x = torch.tensor([1.0, 2.0, 3.0])
            y = x * 2
            
            # CUDA check
            cuda_available = torch.cuda.is_available()
            
            return HealthStatus(
                component="pytorch",
                status="healthy",
                message="PyTorch is working correctly",
                timestamp=time.time(),
                details={
                    "version": torch.__version__,
                    "cuda_available": cuda_available,
                    "basic_ops_working": True
                }
            )
        except ImportError:
            return HealthStatus(
                component="pytorch",
                status="unhealthy", 
                message="PyTorch not available",
                timestamp=time.time(),
                details={"error": "import_error"}
            )
        except Exception as e:
            return HealthStatus(
                component="pytorch",
                status="degraded",
                message=f"PyTorch partially working: {e}",
                timestamp=time.time(),
                details={"error": str(e)}
            )
    
    def check_memory_health() -> HealthStatus:
        """Check system memory health."""
        try:
            import psutil
            
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            if memory_percent > 90:
                status = "unhealthy"
                message = f"Memory usage critical: {memory_percent:.1f}%"
            elif memory_percent > 75:
                status = "degraded"
                message = f"Memory usage high: {memory_percent:.1f}%"
            else:
                status = "healthy"
                message = f"Memory usage normal: {memory_percent:.1f}%"
            
            return HealthStatus(
                component="memory",
                status=status,
                message=message,
                timestamp=time.time(),
                details={
                    "total_gb": memory.total / 1e9,
                    "available_gb": memory.available / 1e9,
                    "percent_used": memory_percent
                }
            )
        except ImportError:
            return HealthStatus(
                component="memory",
                status="unknown",
                message="psutil not available for memory monitoring",
                timestamp=time.time(),
                details={}
            )
    
    def check_revnet_components() -> HealthStatus:
        """Check RevNet-Zero components."""
        try:
            from revnet_zero import ReversibleTransformer, MemoryScheduler
            
            # Test basic component instantiation
            model_config = {
                'vocab_size': 1000,
                'd_model': 128,
                'num_heads': 8,
                'num_layers': 2,
                'max_seq_len': 512
            }
            
            model = ReversibleTransformer(**model_config)
            scheduler = MemoryScheduler()
            
            return HealthStatus(
                component="revnet_components",
                status="healthy", 
                message="All RevNet-Zero components working",
                timestamp=time.time(),
                details={
                    "model_created": True,
                    "scheduler_created": True
                }
            )
        except Exception as e:
            return HealthStatus(
                component="revnet_components",
                status="unhealthy",
                message=f"RevNet components failed: {e}",
                timestamp=time.time(),
                details={"error": str(e)}
            )
    
    return {
        "pytorch": check_pytorch_health,
        "memory": check_memory_health,
        "revnet_components": check_revnet_components
    }


class RevNetZeroMonitor:
    """Main monitoring system for RevNet-Zero."""
    
    def __init__(self):
        self.metrics = MetricsCollector()
        self.performance = PerformanceMonitor(self.metrics)
        self.health = HealthChecker()
        
        # Register default health checks
        for name, check_func in default_health_checks().items():
            self.health.register_health_check(name, check_func)
        
        self.logger = logging.getLogger(__name__)
        
    def start_monitoring(self):
        """Start the monitoring system."""
        self.logger.info("RevNet-Zero monitoring started")
        
        # Log initial health status
        health_status = self.health.get_overall_health()
        self.logger.info(f"Initial health status: {health_status}")
        
    def get_monitoring_dashboard(self) -> Dict[str, Any]:
        """Get comprehensive monitoring dashboard data."""
        dashboard = {
            'timestamp': time.time(),
            'overall_health': self.health.get_overall_health(),
            'component_health': {},
            'metrics_summary': {},
            'performance_stats': {}
        }
        
        # Health information
        for component, status in self.health.check_all_health().items():
            dashboard['component_health'][component] = asdict(status)
        
        # Key metrics summaries
        key_metrics = [
            'operation_duration',
            'memory_usage_bytes', 
            'throughput_items_per_second',
            'operation_count'
        ]
        
        for metric_name in key_metrics:
            summary = self.metrics.get_metric_summary(metric_name)
            if summary['count'] > 0:
                dashboard['metrics_summary'][metric_name] = summary
        
        return dashboard
    
    def export_metrics_prometheus(self) -> str:
        """Export metrics in Prometheus format."""
        lines = []
        lines.append("# RevNet-Zero Metrics")
        
        # Get all metrics from the last hour
        cutoff_time = time.time() - 3600
        metrics = self.metrics.get_metrics(since_timestamp=cutoff_time)
        
        # Group by metric name
        metric_groups = defaultdict(list)
        for metric in metrics:
            metric_groups[metric.name].append(metric)
        
        # Format for Prometheus
        for name, metric_list in metric_groups.items():
            # Use the latest value for gauges, sum for counters
            if metric_list[0].metric_type == MetricType.GAUGE:
                latest_metric = max(metric_list, key=lambda m: m.timestamp)
                tags_str = ",".join([f'{k}="{v}"' for k, v in latest_metric.tags.items()])
                if tags_str:
                    line = f'{name}{{{tags_str}}} {latest_metric.value}'
                else:
                    line = f'{name} {latest_metric.value}'
                lines.append(line)
            elif metric_list[0].metric_type == MetricType.COUNTER:
                # Sum counter values
                total_value = sum(m.value for m in metric_list)
                tags_str = ",".join([f'{k}="{v}"' for k, v in metric_list[0].tags.items()])
                if tags_str:
                    line = f'{name}_total{{{tags_str}}} {total_value}'
                else:
                    line = f'{name}_total {total_value}'
                lines.append(line)
        
        return "\n".join(lines)


# Global monitoring instance
_global_monitor = RevNetZeroMonitor()


def get_monitor() -> RevNetZeroMonitor:
    """Get the global monitoring instance."""
    return _global_monitor


def monitor_operation(operation_name: str):
    """Decorator to monitor operation performance."""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            with _global_monitor.performance.time_operation(operation_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator


# Example usage functions
def example_monitoring_setup():
    """Example of how to set up monitoring in an application."""
    monitor = get_monitor()
    monitor.start_monitoring()
    
    # Custom health check
    def check_custom_component():
        return HealthStatus(
            component="custom_component",
            status="healthy",
            message="Custom component working",
            timestamp=time.time(),
            details={}
        )
    
    monitor.health.register_health_check("custom_component", check_custom_component)
    
    # Record some metrics
    monitor.metrics.record_counter("requests_total", tags={"endpoint": "/api/v1/predict"})
    monitor.metrics.record_gauge("active_connections", 42)
    monitor.performance.record_memory_usage("model", 2.5 * 1e9)  # 2.5 GB
    
    # Get dashboard
    dashboard = monitor.get_monitoring_dashboard()
    print("Monitoring Dashboard:", json.dumps(dashboard, indent=2, default=str))


# Export main components
__all__ = [
    'MetricType', 'Metric', 'HealthStatus', 'MetricsCollector', 
    'PerformanceMonitor', 'HealthChecker', 'RevNetZeroMonitor',
    'get_monitor', 'monitor_operation', 'default_health_checks'
]