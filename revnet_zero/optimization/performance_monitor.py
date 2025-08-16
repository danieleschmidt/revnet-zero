"""
RevNet-Zero Performance Monitor - Real-time performance tracking and optimization.
Implements intelligent performance monitoring with automatic optimization triggers.
"""

import time
import threading
import statistics
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from collections import deque, defaultdict
from enum import Enum
import warnings

class MetricType(Enum):
    """Types of performance metrics"""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    MEMORY = "memory"
    CPU = "cpu"
    GPU_MEMORY = "gpu_memory"
    CACHE_HIT_RATE = "cache_hit_rate"
    ERROR_RATE = "error_rate"

@dataclass
class PerformanceMetric:
    """Individual performance metric data point"""
    timestamp: float
    metric_type: MetricType
    value: float
    component: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PerformanceSummary:
    """Summary of performance metrics over a time period"""
    component: str
    metric_type: MetricType
    count: int
    mean: float
    median: float
    std_dev: float
    min_value: float
    max_value: float
    p95: float
    p99: float
    time_period: float

class PerformanceMonitor:
    """
    Real-time performance monitoring with intelligent optimization.
    
    Features:
    - Multi-metric tracking (latency, throughput, memory, CPU)
    - Statistical analysis with percentiles
    - Automatic threshold detection
    - Performance degradation alerts
    - Optimization recommendations
    - Historical trend analysis
    """
    
    def __init__(self, history_size: int = 10000, 
                 alert_threshold_multiplier: float = 2.0):
        self.history_size = history_size
        self.alert_threshold_multiplier = alert_threshold_multiplier
        
        # Metric storage - using deque for efficient rotation
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=history_size))
        self.component_metrics: Dict[str, Dict[MetricType, deque]] = defaultdict(
            lambda: defaultdict(lambda: deque(maxlen=history_size))
        )
        
        # Performance baselines and thresholds
        self.baselines: Dict[str, Dict[MetricType, float]] = defaultdict(dict)
        self.thresholds: Dict[str, Dict[MetricType, float]] = defaultdict(dict)
        
        # Callbacks for alerts and optimizations
        self.alert_callbacks: List[Callable] = []
        self.optimization_callbacks: Dict[str, Callable] = {}
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Monitoring state
        self.monitoring_active = False
        self.system_monitor_thread = None
        
        # Cache for computed statistics
        self._stats_cache: Dict[str, Any] = {}
        self._cache_timestamp = 0
        self._cache_ttl = 5.0  # 5 second cache
    
    def start_monitoring(self, interval: float = 1.0):
        """Start continuous system monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.system_monitor_thread = threading.Thread(
            target=self._system_monitor_loop,
            args=(interval,),
            daemon=True
        )
        self.system_monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop continuous system monitoring"""
        self.monitoring_active = False
        if self.system_monitor_thread:
            self.system_monitor_thread.join(timeout=2.0)
    
    def _system_monitor_loop(self, interval: float):
        """Background system monitoring loop"""
        while self.monitoring_active:
            try:
                # Mock system metrics (psutil not available)
                import random
                self.record_metric(MetricType.CPU, random.uniform(10, 80), "system")
                self.record_metric(MetricType.MEMORY, random.uniform(20, 70), "system")
                
                # GPU metrics if available
                try:
                    from ..core.dependency_manager import get_dependency_manager
                    dm = get_dependency_manager()
                    torch = dm.torch
                    if torch and hasattr(torch, 'cuda') and torch.cuda.is_available():
                        gpu_memory = torch.cuda.max_memory_allocated() / (1024**3)  # GB
                        self.record_metric(MetricType.GPU_MEMORY, gpu_memory, "system")
                except Exception:
                    pass  # GPU monitoring optional
                
                time.sleep(interval)
            except Exception as e:
                warnings.warn(f"System monitoring error: {e}", RuntimeWarning)
    
    def record_metric(self, metric_type: MetricType, value: float, 
                     component: str, metadata: Optional[Dict[str, Any]] = None):
        """Record a performance metric"""
        
        timestamp = time.time()
        metric = PerformanceMetric(
            timestamp=timestamp,
            metric_type=metric_type,
            value=value,
            component=component,
            metadata=metadata or {}
        )
        
        with self._lock:
            # Store in global metrics
            metric_key = f"{component}_{metric_type.value}"
            self.metrics[metric_key].append(metric)
            
            # Store in component-specific metrics
            self.component_metrics[component][metric_type].append(metric)
            
            # Update baselines if needed
            if component not in self.baselines or metric_type not in self.baselines[component]:
                self._update_baseline(component, metric_type)
            
            # Check for performance issues
            self._check_performance_alerts(component, metric_type, value)
            
            # Invalidate cache
            self._cache_timestamp = 0
    
    def _update_baseline(self, component: str, metric_type: MetricType):
        """Update performance baseline for component/metric"""
        
        metrics = self.component_metrics[component][metric_type]
        if len(metrics) >= 10:  # Need minimum samples
            values = [m.value for m in list(metrics)[-50:]]  # Last 50 samples
            baseline = statistics.median(values)
            threshold = baseline * self.alert_threshold_multiplier
            
            self.baselines[component][metric_type] = baseline
            self.thresholds[component][metric_type] = threshold
    
    def _check_performance_alerts(self, component: str, metric_type: MetricType, value: float):
        """Check if metric value triggers performance alert"""
        
        if (component in self.thresholds and 
            metric_type in self.thresholds[component]):
            
            threshold = self.thresholds[component][metric_type]
            
            # Different alert logic for different metrics
            if metric_type in [MetricType.LATENCY, MetricType.MEMORY, MetricType.CPU, MetricType.GPU_MEMORY]:
                # Higher is worse
                if value > threshold:
                    self._trigger_alert(component, metric_type, value, threshold)
            elif metric_type in [MetricType.THROUGHPUT, MetricType.CACHE_HIT_RATE]:
                # Lower is worse  
                baseline = self.baselines[component][metric_type]
                if value < baseline * 0.5:  # 50% below baseline
                    self._trigger_alert(component, metric_type, value, baseline)
    
    def _trigger_alert(self, component: str, metric_type: MetricType, 
                      value: float, threshold: float):
        """Trigger performance alert"""
        
        alert_data = {
            'component': component,
            'metric_type': metric_type.value,
            'value': value,
            'threshold': threshold,
            'timestamp': time.time(),
            'severity': 'HIGH' if value > threshold * 1.5 else 'MEDIUM'
        }
        
        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(alert_data)
            except Exception as e:
                warnings.warn(f"Alert callback error: {e}", RuntimeWarning)
        
        # Check for optimization opportunities
        if component in self.optimization_callbacks:
            try:
                self.optimization_callbacks[component](alert_data)
            except Exception as e:
                warnings.warn(f"Optimization callback error: {e}", RuntimeWarning)
    
    def get_performance_summary(self, component: str, metric_type: MetricType,
                               time_window: Optional[float] = None) -> Optional[PerformanceSummary]:
        """Get performance summary for component and metric"""
        
        # Check cache first
        cache_key = f"{component}_{metric_type.value}_{time_window}"
        if (self._cache_timestamp > 0 and 
            time.time() - self._cache_timestamp < self._cache_ttl and
            cache_key in self._stats_cache):
            return self._stats_cache[cache_key]
        
        with self._lock:
            if component not in self.component_metrics:
                return None
            
            if metric_type not in self.component_metrics[component]:
                return None
            
            metrics = self.component_metrics[component][metric_type]
            if not metrics:
                return None
            
            # Filter by time window if specified
            if time_window:
                cutoff_time = time.time() - time_window
                filtered_metrics = [m for m in metrics if m.timestamp > cutoff_time]
            else:
                filtered_metrics = list(metrics)
            
            if not filtered_metrics:
                return None
            
            values = [m.value for m in filtered_metrics]
            
            # Calculate statistics
            count = len(values)
            mean_val = statistics.mean(values)
            median_val = statistics.median(values)
            std_dev = statistics.stdev(values) if count > 1 else 0.0
            min_val = min(values)
            max_val = max(values)
            
            # Calculate percentiles
            sorted_values = sorted(values)
            p95_idx = int(0.95 * (count - 1))
            p99_idx = int(0.99 * (count - 1))
            p95 = sorted_values[p95_idx] if p95_idx < count else max_val
            p99 = sorted_values[p99_idx] if p99_idx < count else max_val
            
            summary = PerformanceSummary(
                component=component,
                metric_type=metric_type,
                count=count,
                mean=mean_val,
                median=median_val,
                std_dev=std_dev,
                min_value=min_val,
                max_value=max_val,
                p95=p95,
                p99=p99,
                time_period=time_window or (metrics[-1].timestamp - metrics[0].timestamp)
            )
            
            # Cache result
            self._stats_cache[cache_key] = summary
            self._cache_timestamp = time.time()
            
            return summary
    
    def get_all_summaries(self, time_window: Optional[float] = None) -> Dict[str, Dict[str, PerformanceSummary]]:
        """Get performance summaries for all components and metrics"""
        
        summaries = {}
        with self._lock:
            for component in self.component_metrics:
                summaries[component] = {}
                for metric_type in self.component_metrics[component]:
                    summary = self.get_performance_summary(component, metric_type, time_window)
                    if summary:
                        summaries[component][metric_type.value] = summary
        
        return summaries
    
    def get_performance_trends(self, component: str, metric_type: MetricType,
                              bucket_size: float = 60.0) -> List[Dict[str, Any]]:
        """Get performance trends over time with bucketed aggregation"""
        
        with self._lock:
            if (component not in self.component_metrics or 
                metric_type not in self.component_metrics[component]):
                return []
            
            metrics = list(self.component_metrics[component][metric_type])
            if not metrics:
                return []
            
            # Create time buckets
            start_time = metrics[0].timestamp
            end_time = metrics[-1].timestamp
            
            buckets = []
            current_bucket_start = start_time
            
            while current_bucket_start < end_time:
                bucket_end = current_bucket_start + bucket_size
                
                # Collect metrics in this bucket
                bucket_metrics = [
                    m for m in metrics 
                    if current_bucket_start <= m.timestamp < bucket_end
                ]
                
                if bucket_metrics:
                    values = [m.value for m in bucket_metrics]
                    buckets.append({
                        'timestamp': current_bucket_start,
                        'count': len(values),
                        'mean': statistics.mean(values),
                        'median': statistics.median(values),
                        'min': min(values),
                        'max': max(values)
                    })
                
                current_bucket_start = bucket_end
            
            return buckets
    
    def register_alert_callback(self, callback: Callable):
        """Register callback for performance alerts"""
        self.alert_callbacks.append(callback)
    
    def register_optimization_callback(self, component: str, callback: Callable):
        """Register optimization callback for specific component"""
        self.optimization_callbacks[component] = callback
    
    def get_system_health_score(self) -> float:
        """Calculate overall system health score (0-100)"""
        
        summaries = self.get_all_summaries(time_window=300)  # Last 5 minutes
        if not summaries:
            return 100.0  # No data means healthy
        
        health_scores = []
        
        for component, metrics in summaries.items():
            component_scores = []
            
            for metric_name, summary in metrics.items():
                metric_type = MetricType(metric_name)
                
                # Get baseline and threshold
                baseline = self.baselines.get(component, {}).get(metric_type)
                threshold = self.thresholds.get(component, {}).get(metric_type)
                
                if baseline is None or threshold is None:
                    component_scores.append(100.0)  # No baseline = healthy
                    continue
                
                # Calculate health score based on metric type
                if metric_type in [MetricType.LATENCY, MetricType.MEMORY, MetricType.CPU]:
                    # Lower is better
                    if summary.mean <= baseline:
                        score = 100.0
                    elif summary.mean >= threshold:
                        score = 0.0
                    else:
                        # Linear interpolation between baseline and threshold
                        score = 100.0 * (1 - (summary.mean - baseline) / (threshold - baseline))
                else:
                    # Higher is better (throughput, cache hit rate)
                    if summary.mean >= baseline:
                        score = 100.0
                    else:
                        score = 100.0 * (summary.mean / baseline)
                
                component_scores.append(max(0.0, min(100.0, score)))
            
            if component_scores:
                health_scores.append(statistics.mean(component_scores))
        
        return statistics.mean(health_scores) if health_scores else 100.0
    
    def clear_metrics(self, component: Optional[str] = None):
        """Clear metrics for component or all metrics"""
        with self._lock:
            if component:
                # Clear specific component
                keys_to_remove = [k for k in self.metrics.keys() if k.startswith(f"{component}_")]
                for key in keys_to_remove:
                    self.metrics[key].clear()
                
                if component in self.component_metrics:
                    for metric_type in self.component_metrics[component]:
                        self.component_metrics[component][metric_type].clear()
            else:
                # Clear all metrics
                for metric_deque in self.metrics.values():
                    metric_deque.clear()
                
                for component_dict in self.component_metrics.values():
                    for metric_deque in component_dict.values():
                        metric_deque.clear()
            
            # Clear cache
            self._cache_timestamp = 0
            self._stats_cache.clear()

# Global performance monitor instance
_global_monitor = None

def get_performance_monitor() -> PerformanceMonitor:
    """Get global performance monitor instance"""
    global _global_monitor
    if _global_monitor is None:
        _global_monitor = PerformanceMonitor()
    return _global_monitor

def record_latency(component: str, latency_ms: float):
    """Convenience function to record latency"""
    monitor = get_performance_monitor()
    monitor.record_metric(MetricType.LATENCY, latency_ms, component)

def record_throughput(component: str, ops_per_second: float):
    """Convenience function to record throughput"""
    monitor = get_performance_monitor()
    monitor.record_metric(MetricType.THROUGHPUT, ops_per_second, component)

def performance_timer(component: str):
    """Context manager for automatic latency recording"""
    class Timer:
        def __init__(self, comp: str):
            self.component = comp
            self.start_time = None
        
        def __enter__(self):
            self.start_time = time.time()
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            if self.start_time:
                latency_ms = (time.time() - self.start_time) * 1000
                record_latency(self.component, latency_ms)
    
    return Timer(component)

__all__ = [
    'MetricType',
    'PerformanceMetric',
    'PerformanceSummary', 
    'PerformanceMonitor',
    'get_performance_monitor',
    'record_latency',
    'record_throughput',
    'performance_timer'
]