"""
Monitoring and observability utilities for RevNet-Zero.

This module provides comprehensive monitoring, metrics collection,
and observability tools for reversible transformer training.
"""

import torch
import torch.nn as nn
import time
import psutil
import json
import os
from typing import Dict, List, Optional, Any, Callable, Union
from collections import defaultdict, deque
from contextlib import contextmanager
import threading
import logging
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class MetricSnapshot:
    """Snapshot of system metrics at a point in time."""
    timestamp: float
    step: int
    loss: Optional[float] = None
    learning_rate: Optional[float] = None
    gradient_norm: Optional[float] = None
    memory_allocated_gb: Optional[float] = None
    memory_cached_gb: Optional[float] = None
    cpu_percent: Optional[float] = None
    gpu_utilization: Optional[float] = None
    throughput_tokens_per_sec: Optional[float] = None
    model_flops: Optional[float] = None


class MetricsCollector:
    """
    Comprehensive metrics collector for training monitoring.
    
    Collects system metrics, training metrics, and performance data
    with configurable collection intervals and storage backends.
    """
    
    def __init__(
        self,
        collection_interval: float = 1.0,
        max_history: int = 10000,
        enable_gpu_monitoring: bool = True,
        enable_system_monitoring: bool = True,
        log_to_file: Optional[str] = None,
    ):
        """
        Initialize metrics collector.
        
        Args:
            collection_interval: Interval between metric collection in seconds
            max_history: Maximum number of metric snapshots to keep
            enable_gpu_monitoring: Whether to monitor GPU metrics
            enable_system_monitoring: Whether to monitor system metrics
            log_to_file: Optional file path to log metrics
        """
        self.collection_interval = collection_interval
        self.max_history = max_history
        self.enable_gpu_monitoring = enable_gpu_monitoring and torch.cuda.is_available()
        self.enable_system_monitoring = enable_system_monitoring
        self.log_to_file = log_to_file
        
        # Metrics storage
        self.metrics_history: deque = deque(maxlen=max_history)
        self.aggregated_metrics: Dict[str, List[float]] = defaultdict(list)
        
        # Collection state
        self.collecting = False
        self.collection_thread: Optional[threading.Thread] = None
        self.last_collection_time = 0
        self.current_step = 0
        
        # Current values (for external updates)
        self.current_loss: Optional[float] = None
        self.current_lr: Optional[float] = None
        self.current_grad_norm: Optional[float] = None
        self.current_throughput: Optional[float] = None
        
        # Logger
        self.logger = logging.getLogger(__name__)
        
        # File logging
        if self.log_to_file:
            self.file_handler = open(self.log_to_file, 'w')
        else:
            self.file_handler = None
    
    def start_collection(self, background: bool = True):
        """
        Start metrics collection.
        
        Args:
            background: Whether to collect metrics in background thread
        """
        if self.collecting:
            self.logger.warning("Metrics collection already started")
            return
        
        self.collecting = True
        self.logger.info("Starting metrics collection")
        
        if background:
            self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
            self.collection_thread.start()
        else:
            self._collection_loop()
    
    def stop_collection(self):
        """Stop metrics collection."""
        if not self.collecting:
            return
        
        self.collecting = False
        self.logger.info("Stopping metrics collection")
        
        if self.collection_thread and self.collection_thread.is_alive():
            self.collection_thread.join(timeout=5.0)
        
        if self.file_handler:
            self.file_handler.close()
            self.file_handler = None
    
    def _collection_loop(self):
        """Main collection loop."""
        while self.collecting:
            current_time = time.time()
            
            if current_time - self.last_collection_time >= self.collection_interval:
                try:
                    self.collect_snapshot()
                    self.last_collection_time = current_time
                except Exception as e:
                    self.logger.error(f"Error collecting metrics: {e}")
            
            time.sleep(0.1)  # Small sleep to prevent busy waiting
    
    def collect_snapshot(self) -> MetricSnapshot:
        """
        Collect a complete metrics snapshot.
        
        Returns:
            MetricSnapshot with current metrics
        """
        snapshot = MetricSnapshot(
            timestamp=time.time(),
            step=self.current_step,
            loss=self.current_loss,
            learning_rate=self.current_lr,
            gradient_norm=self.current_grad_norm,
            throughput_tokens_per_sec=self.current_throughput,
        )
        
        # Collect GPU metrics
        if self.enable_gpu_monitoring:
            try:
                snapshot.memory_allocated_gb = torch.cuda.memory_allocated() / (1024**3)
                snapshot.memory_cached_gb = torch.cuda.memory_reserved() / (1024**3)
                
                # GPU utilization (requires nvidia-ml-py)
                try:
                    import pynvml
                    pynvml.nvmlInit()
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    snapshot.gpu_utilization = utilization.gpu
                except ImportError:
                    pass  # pynvml not available
                
            except Exception as e:
                self.logger.debug(f"GPU metrics collection failed: {e}")
        
        # Collect system metrics
        if self.enable_system_monitoring:
            try:
                snapshot.cpu_percent = psutil.cpu_percent()
            except Exception as e:
                self.logger.debug(f"System metrics collection failed: {e}")
        
        # Store snapshot
        self.metrics_history.append(snapshot)
        
        # Update aggregated metrics
        self._update_aggregated_metrics(snapshot)
        
        # Log to file if enabled
        if self.file_handler:
            self.file_handler.write(json.dumps(asdict(snapshot)) + '\n')
            self.file_handler.flush()
        
        return snapshot
    
    def _update_aggregated_metrics(self, snapshot: MetricSnapshot):
        """Update aggregated metrics with new snapshot."""
        if snapshot.loss is not None:
            self.aggregated_metrics['loss'].append(snapshot.loss)
        if snapshot.learning_rate is not None:
            self.aggregated_metrics['learning_rate'].append(snapshot.learning_rate)
        if snapshot.gradient_norm is not None:
            self.aggregated_metrics['gradient_norm'].append(snapshot.gradient_norm)
        if snapshot.memory_allocated_gb is not None:
            self.aggregated_metrics['memory_allocated_gb'].append(snapshot.memory_allocated_gb)
        if snapshot.cpu_percent is not None:
            self.aggregated_metrics['cpu_percent'].append(snapshot.cpu_percent)
        if snapshot.throughput_tokens_per_sec is not None:
            self.aggregated_metrics['throughput_tokens_per_sec'].append(snapshot.throughput_tokens_per_sec)
    
    def update_training_metrics(
        self,
        step: int,
        loss: Optional[float] = None,
        learning_rate: Optional[float] = None,
        gradient_norm: Optional[float] = None,
        throughput: Optional[float] = None,
    ):
        """
        Update current training metrics.
        
        Args:
            step: Training step
            loss: Current loss value
            learning_rate: Current learning rate
            gradient_norm: Current gradient norm
            throughput: Current throughput in tokens/sec
        """
        self.current_step = step
        if loss is not None:
            self.current_loss = loss
        if learning_rate is not None:
            self.current_lr = learning_rate
        if gradient_norm is not None:
            self.current_grad_norm = gradient_norm
        if throughput is not None:
            self.current_throughput = throughput
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current metrics summary."""
        if not self.metrics_history:
            return {}
        
        latest = self.metrics_history[-1]
        return {
            'step': latest.step,
            'loss': latest.loss,
            'learning_rate': latest.learning_rate,
            'gradient_norm': latest.gradient_norm,
            'memory_allocated_gb': latest.memory_allocated_gb,
            'memory_cached_gb': latest.memory_cached_gb,
            'cpu_percent': latest.cpu_percent,
            'gpu_utilization': latest.gpu_utilization,
            'throughput_tokens_per_sec': latest.throughput_tokens_per_sec,
        }
    
    def get_statistics(self, window_size: Optional[int] = None) -> Dict[str, Dict[str, float]]:
        """
        Get statistical summary of metrics.
        
        Args:
            window_size: Size of sliding window for statistics
            
        Returns:
            Dictionary of metric statistics
        """
        if not self.metrics_history:
            return {}
        
        # Use recent window or all data
        if window_size:
            recent_data = list(self.metrics_history)[-window_size:]
        else:
            recent_data = list(self.metrics_history)
        
        stats = {}
        
        # Calculate statistics for each metric
        metrics_to_analyze = [
            'loss', 'learning_rate', 'gradient_norm', 'memory_allocated_gb',
            'cpu_percent', 'gpu_utilization', 'throughput_tokens_per_sec'
        ]
        
        for metric in metrics_to_analyze:
            values = [getattr(snapshot, metric) for snapshot in recent_data 
                     if getattr(snapshot, metric) is not None]
            
            if values:
                stats[metric] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'median': np.median(values),
                    'count': len(values),
                }
        
        return stats
    
    def plot_metrics(
        self,
        metrics: List[str] = None,
        save_path: Optional[str] = None,
        show_plot: bool = True,
        window_size: Optional[int] = None,
    ):
        """
        Plot collected metrics.
        
        Args:
            metrics: List of metrics to plot
            save_path: Path to save plot
            show_plot: Whether to display plot
            window_size: Size of sliding window for plotting
        """
        if not self.metrics_history:
            self.logger.warning("No metrics data to plot")
            return
        
        if metrics is None:
            metrics = ['loss', 'memory_allocated_gb', 'gradient_norm', 'throughput_tokens_per_sec']
        
        # Filter available metrics
        available_metrics = []
        for metric in metrics:
            values = [getattr(snapshot, metric) for snapshot in self.metrics_history 
                     if getattr(snapshot, metric) is not None]
            if values:
                available_metrics.append(metric)
        
        if not available_metrics:
            self.logger.warning("No plottable metrics found")
            return
        
        # Create subplots
        n_metrics = len(available_metrics)
        fig, axes = plt.subplots(n_metrics, 1, figsize=(12, 4 * n_metrics))
        if n_metrics == 1:
            axes = [axes]
        
        # Use recent window or all data
        if window_size:
            plot_data = list(self.metrics_history)[-window_size:]
        else:
            plot_data = list(self.metrics_history)
        
        for i, metric in enumerate(available_metrics):
            # Extract data
            timestamps = [snapshot.timestamp for snapshot in plot_data]
            values = [getattr(snapshot, metric) for snapshot in plot_data]
            steps = [snapshot.step for snapshot in plot_data]
            
            # Filter None values
            valid_data = [(t, v, s) for t, v, s in zip(timestamps, values, steps) if v is not None]
            if not valid_data:
                continue
            
            timestamps, values, steps = zip(*valid_data)
            
            # Plot
            ax = axes[i]
            ax.plot(steps, values, label=metric, linewidth=1.5)
            ax.set_xlabel('Training Step')
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.set_title(f'{metric.replace("_", " ").title()} over Training')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Metrics plot saved to {save_path}")
        
        if show_plot:
            plt.show()
        else:
            plt.close()
    
    def export_metrics(self, file_path: str, format: str = 'json'):
        """
        Export collected metrics to file.
        
        Args:
            file_path: Output file path
            format: Export format ('json', 'csv')
        """
        if not self.metrics_history:
            self.logger.warning("No metrics to export")
            return
        
        if format == 'json':
            data = [asdict(snapshot) for snapshot in self.metrics_history]
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
        
        elif format == 'csv':
            import pandas as pd
            data = [asdict(snapshot) for snapshot in self.metrics_history]
            df = pd.DataFrame(data)
            df.to_csv(file_path, index=False)
        
        else:
            raise ValueError(f"Unsupported export format: {format}")
        
        self.logger.info(f"Metrics exported to {file_path}")


class PerformanceMonitor:
    """
    Monitor model performance and detect anomalies.
    
    Tracks training progress, detects performance issues,
    and provides recommendations for optimization.
    """
    
    def __init__(
        self,
        alert_thresholds: Optional[Dict[str, float]] = None,
        detection_window: int = 100,
        min_improvement_rate: float = 0.001,
    ):
        """
        Initialize performance monitor.
        
        Args:
            alert_thresholds: Thresholds for performance alerts
            detection_window: Window size for anomaly detection
            min_improvement_rate: Minimum expected improvement rate
        """
        self.alert_thresholds = alert_thresholds or {
            'loss_spike_factor': 2.0,
            'gradient_norm_threshold': 100.0,
            'memory_utilization_threshold': 0.95,
            'throughput_drop_factor': 0.5,
        }
        self.detection_window = detection_window
        self.min_improvement_rate = min_improvement_rate
        
        # Performance tracking
        self.loss_history: deque = deque(maxlen=detection_window)
        self.gradient_norm_history: deque = deque(maxlen=detection_window)
        self.throughput_history: deque = deque(maxlen=detection_window)
        
        # Anomaly detection state
        self.alerts: List[Dict[str, Any]] = []
        self.last_alert_time = {}
        
        # Logger
        self.logger = logging.getLogger(__name__)
    
    def update(
        self,
        loss: Optional[float] = None,
        gradient_norm: Optional[float] = None,
        throughput: Optional[float] = None,
        memory_utilization: Optional[float] = None,
        step: int = 0,
    ):
        """
        Update performance metrics and check for anomalies.
        
        Args:
            loss: Current loss value
            gradient_norm: Current gradient norm
            throughput: Current throughput
            memory_utilization: Current memory utilization
            step: Training step
        """
        current_time = time.time()
        
        # Update history
        if loss is not None:
            self.loss_history.append((step, loss, current_time))
        if gradient_norm is not None:
            self.gradient_norm_history.append((step, gradient_norm, current_time))
        if throughput is not None:
            self.throughput_history.append((step, throughput, current_time))
        
        # Check for anomalies
        self._check_loss_anomalies(step, current_time)
        self._check_gradient_anomalies(step, current_time)
        self._check_throughput_anomalies(step, current_time)
        
        if memory_utilization is not None:
            self._check_memory_anomalies(memory_utilization, step, current_time)
    
    def _check_loss_anomalies(self, step: int, current_time: float):
        """Check for loss-related anomalies."""
        if len(self.loss_history) < 10:
            return
        
        recent_losses = [loss for _, loss, _ in list(self.loss_history)[-10:]]
        current_loss = recent_losses[-1]
        
        # Check for loss spikes
        if len(recent_losses) >= 2:
            avg_recent_loss = np.mean(recent_losses[:-1])
            if current_loss > avg_recent_loss * self.alert_thresholds['loss_spike_factor']:
                self._create_alert(
                    'loss_spike',
                    f"Loss spike detected: {current_loss:.4f} vs average {avg_recent_loss:.4f}",
                    step,
                    current_time,
                )
        
        # Check for loss stagnation
        if len(self.loss_history) >= self.detection_window:
            early_losses = [loss for _, loss, _ in list(self.loss_history)[:20]]
            recent_losses = [loss for _, loss, _ in list(self.loss_history)[-20:]]
            
            early_avg = np.mean(early_losses)
            recent_avg = np.mean(recent_losses)
            
            improvement_rate = (early_avg - recent_avg) / early_avg if early_avg > 0 else 0
            
            if improvement_rate < self.min_improvement_rate:
                self._create_alert(
                    'loss_stagnation',
                    f"Loss improvement rate too low: {improvement_rate:.4f}",
                    step,
                    current_time,
                )
    
    def _check_gradient_anomalies(self, step: int, current_time: float):
        """Check for gradient-related anomalies."""
        if len(self.gradient_norm_history) < 5:
            return
        
        recent_norms = [norm for _, norm, _ in list(self.gradient_norm_history)[-5:]]
        current_norm = recent_norms[-1]
        
        # Check for gradient explosion
        if current_norm > self.alert_thresholds['gradient_norm_threshold']:
            self._create_alert(
                'gradient_explosion',
                f"Large gradient norm detected: {current_norm:.4f}",
                step,
                current_time,
            )
        
        # Check for vanishing gradients
        if current_norm < 1e-7:
            self._create_alert(
                'vanishing_gradients',
                f"Very small gradient norm detected: {current_norm:.2e}",
                step,
                current_time,
            )
    
    def _check_throughput_anomalies(self, step: int, current_time: float):
        """Check for throughput-related anomalies."""
        if len(self.throughput_history) < 10:
            return
        
        recent_throughputs = [tp for _, tp, _ in list(self.throughput_history)[-10:]]
        current_throughput = recent_throughputs[-1]
        
        if len(recent_throughputs) >= 2:
            avg_recent_throughput = np.mean(recent_throughputs[:-1])
            
            if current_throughput < avg_recent_throughput * self.alert_thresholds['throughput_drop_factor']:
                self._create_alert(
                    'throughput_drop',
                    f"Throughput drop detected: {current_throughput:.2f} vs average {avg_recent_throughput:.2f}",
                    step,
                    current_time,
                )
    
    def _check_memory_anomalies(self, memory_utilization: float, step: int, current_time: float):
        """Check for memory-related anomalies."""
        if memory_utilization > self.alert_thresholds['memory_utilization_threshold']:
            self._create_alert(
                'high_memory_usage',
                f"High memory utilization: {memory_utilization:.2%}",
                step,
                current_time,
            )
    
    def _create_alert(self, alert_type: str, message: str, step: int, timestamp: float):
        """Create a performance alert."""
        # Rate limiting: don't create same alert type too frequently
        if alert_type in self.last_alert_time:
            if timestamp - self.last_alert_time[alert_type] < 60:  # 1 minute cooldown
                return
        
        alert = {
            'type': alert_type,
            'message': message,
            'step': step,
            'timestamp': timestamp,
        }
        
        self.alerts.append(alert)
        self.last_alert_time[alert_type] = timestamp
        
        self.logger.warning(f"Performance alert: {message} at step {step}")
    
    def get_alerts(
        self,
        alert_types: Optional[List[str]] = None,
        recent_only: bool = True,
        time_window: float = 3600,  # 1 hour
    ) -> List[Dict[str, Any]]:
        """
        Get performance alerts.
        
        Args:
            alert_types: Filter by alert types
            recent_only: Only return recent alerts
            time_window: Time window for recent alerts in seconds
            
        Returns:
            List of alerts
        """
        alerts = self.alerts.copy()
        
        # Filter by time
        if recent_only:
            current_time = time.time()
            alerts = [alert for alert in alerts 
                     if current_time - alert['timestamp'] <= time_window]
        
        # Filter by type
        if alert_types:
            alerts = [alert for alert in alerts if alert['type'] in alert_types]
        
        return alerts
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        summary = {
            'total_alerts': len(self.alerts),
            'recent_alerts': len(self.get_alerts(recent_only=True)),
            'alert_types': {},
        }
        
        # Count alerts by type
        for alert in self.alerts:
            alert_type = alert['type']
            summary['alert_types'][alert_type] = summary['alert_types'].get(alert_type, 0) + 1
        
        # Performance trends
        if self.loss_history:
            recent_losses = [loss for _, loss, _ in list(self.loss_history)[-20:]]
            if len(recent_losses) >= 2:
                summary['loss_trend'] = 'decreasing' if recent_losses[-1] < recent_losses[0] else 'increasing'
                summary['recent_loss_change'] = recent_losses[-1] - recent_losses[0]
        
        if self.throughput_history:
            recent_throughputs = [tp for _, tp, _ in list(self.throughput_history)[-20:]]
            if recent_throughputs:
                summary['avg_recent_throughput'] = np.mean(recent_throughputs)
        
        return summary


@contextmanager
def performance_monitoring(
    metrics_collector: Optional[MetricsCollector] = None,
    performance_monitor: Optional[PerformanceMonitor] = None,
    enable_background_collection: bool = True,
):
    """
    Context manager for comprehensive performance monitoring.
    
    Args:
        metrics_collector: Metrics collector instance
        performance_monitor: Performance monitor instance
        enable_background_collection: Whether to collect metrics in background
    """
    # Initialize components if not provided
    if metrics_collector is None:
        metrics_collector = MetricsCollector()
    
    if performance_monitor is None:
        performance_monitor = PerformanceMonitor()
    
    # Start monitoring
    metrics_collector.start_collection(background=enable_background_collection)
    
    try:
        yield metrics_collector, performance_monitor
    finally:
        # Stop monitoring
        metrics_collector.stop_collection()
        
        # Log final summary
        logger = logging.getLogger(__name__)
        performance_summary = performance_monitor.get_performance_summary()
        logger.info(f"Monitoring session completed. Summary: {performance_summary}")


class ModelProfiler:
    """
    Detailed model profiling for performance optimization.
    
    Profiles model execution to identify bottlenecks and
    optimization opportunities.
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.profile_data = {}
        self.layer_times = defaultdict(list)
        self.hooks = []
    
    def start_profiling(self):
        """Start detailed model profiling."""
        self._clear_hooks()
        
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                hook = module.register_forward_hook(self._create_timing_hook(name))
                self.hooks.append(hook)
    
    def stop_profiling(self):
        """Stop profiling and cleanup hooks."""
        self._clear_hooks()
    
    def _clear_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
    
    def _create_timing_hook(self, name: str):
        """Create timing hook for a module."""
        def hook(module, input, output):
            torch.cuda.synchronize()  # Ensure accurate timing
            end_time = time.perf_counter()
            
            # Get start time from thread-local storage
            import threading
            thread_local = getattr(threading.current_thread(), 'revnet_timing', None)
            if thread_local and hasattr(thread_local, 'start_time'):
                execution_time = end_time - thread_local.start_time
                self.layer_times[name].append(execution_time)
            
            thread_local.start_time = end_time
        
        return hook
    
    def profile_forward_pass(
        self,
        input_data: torch.Tensor,
        num_iterations: int = 10,
    ) -> Dict[str, Any]:
        """
        Profile forward pass execution.
        
        Args:
            input_data: Input tensor for profiling
            num_iterations: Number of iterations to average
            
        Returns:
            Profiling results
        """
        self.start_profiling()
        
        # Initialize thread-local storage
        import threading
        thread_local = threading.local()
        thread_local.start_time = time.perf_counter()
        threading.current_thread().revnet_timing = thread_local
        
        total_times = []
        
        for i in range(num_iterations):
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            with torch.no_grad():
                _ = self.model(input_data)
            
            torch.cuda.synchronize()
            total_time = time.perf_counter() - start_time
            total_times.append(total_time)
        
        self.stop_profiling()
        
        # Analyze results
        results = {
            'total_time_stats': {
                'mean': np.mean(total_times),
                'std': np.std(total_times),
                'min': np.min(total_times),
                'max': np.max(total_times),
            },
            'layer_timings': {},
            'bottleneck_layers': [],
        }
        
        # Analyze layer timings
        total_layer_time = 0
        for layer_name, times in self.layer_times.items():
            if times:
                mean_time = np.mean(times)
                results['layer_timings'][layer_name] = {
                    'mean_time': mean_time,
                    'percentage': 0,  # Will be calculated below
                    'std': np.std(times),
                }
                total_layer_time += mean_time
        
        # Calculate percentages and identify bottlenecks
        for layer_name, stats in results['layer_timings'].items():
            if total_layer_time > 0:
                percentage = (stats['mean_time'] / total_layer_time) * 100
                stats['percentage'] = percentage
                
                # Identify bottlenecks (layers taking >10% of time)
                if percentage > 10:
                    results['bottleneck_layers'].append({
                        'layer': layer_name,
                        'time': stats['mean_time'],
                        'percentage': percentage,
                    })
        
        # Sort bottlenecks by time
        results['bottleneck_layers'].sort(key=lambda x: x['time'], reverse=True)
        
        return results
    
    def generate_optimization_recommendations(
        self,
        profile_results: Dict[str, Any]
    ) -> List[str]:
        """
        Generate optimization recommendations based on profiling results.
        
        Args:
            profile_results: Results from profile_forward_pass
            
        Returns:
            List of optimization recommendations
        """
        recommendations = []
        
        # Check for bottleneck layers
        bottlenecks = profile_results.get('bottleneck_layers', [])
        if bottlenecks:
            top_bottleneck = bottlenecks[0]
            recommendations.append(
                f"Optimize '{top_bottleneck['layer']}' layer which takes "
                f"{top_bottleneck['percentage']:.1f}% of execution time"
            )
        
        # Check for attention layers
        attention_layers = [layer for layer in profile_results['layer_timings'] 
                          if 'attention' in layer.lower()]
        if attention_layers:
            recommendations.append(
                "Consider using Flash Attention or other optimized attention implementations"
            )
        
        # Check for linear layers
        linear_layers = [layer for layer in profile_results['layer_timings'] 
                        if 'linear' in layer.lower() or 'dense' in layer.lower()]
        if len(linear_layers) > 10:
            recommendations.append(
                "Consider fusing consecutive linear layers for better performance"
            )
        
        # General recommendations
        recommendations.extend([
            "Enable mixed precision training (FP16/BF16) if not already enabled",
            "Consider gradient checkpointing for memory-time tradeoff",
            "Use torch.compile() for PyTorch 2.0+ optimization",
        ])
        
        return recommendations