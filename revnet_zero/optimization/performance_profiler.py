"""
Advanced performance profiler for RevNet-Zero.

Provides comprehensive profiling of model performance including
GPU utilization, memory patterns, and computation bottlenecks.
"""

import torch
import time
import psutil
from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import threading
import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np


@dataclass
class ProfilerConfig:
    """Configuration for performance profiler."""
    enable_gpu_profiling: bool = True
    enable_memory_profiling: bool = True
    enable_cpu_profiling: bool = True
    sampling_interval: float = 0.1  # seconds
    max_samples: int = 10000
    profile_kernels: bool = False
    export_traces: bool = True
    trace_dir: str = "./profiling_traces"


@dataclass
class PerformanceMetrics:
    """Performance metrics for a single operation."""
    operation_name: str
    duration_ms: float
    gpu_memory_before: int = 0
    gpu_memory_after: int = 0
    gpu_memory_peak: int = 0
    gpu_utilization: float = 0.0
    cpu_utilization: float = 0.0
    throughput_tokens_per_sec: float = 0.0
    flops: int = 0
    memory_bandwidth_gb_per_sec: float = 0.0
    start_time: float = field(default_factory=time.time)
    end_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemMetrics:
    """System-level performance metrics."""
    timestamp: float
    gpu_memory_used: int
    gpu_memory_total: int
    gpu_utilization: float
    cpu_utilization: float
    memory_used: int
    memory_total: int
    temperature: float = 0.0
    power_draw: float = 0.0


class GPUMonitor:
    """GPU monitoring utilities."""
    
    def __init__(self):
        self.has_nvidia_ml = False
        try:
            import pynvml
            pynvml.nvmlInit()
            self.has_nvidia_ml = True
            self.pynvml = pynvml
        except ImportError:
            pass
    
    def get_gpu_info(self, device_id: int = 0) -> Dict[str, Any]:
        """Get GPU information."""
        info = {}
        
        if torch.cuda.is_available():
            info.update({
                "name": torch.cuda.get_device_name(device_id),
                "memory_allocated": torch.cuda.memory_allocated(device_id),
                "memory_reserved": torch.cuda.memory_reserved(device_id),
                "max_memory_allocated": torch.cuda.max_memory_allocated(device_id),
            })
            
            # Get total memory
            if hasattr(torch.cuda, 'get_device_properties'):
                props = torch.cuda.get_device_properties(device_id)
                info["memory_total"] = props.total_memory
        
        if self.has_nvidia_ml:
            try:
                handle = self.pynvml.nvmlDeviceGetHandleByIndex(device_id)
                
                # GPU utilization
                util = self.pynvml.nvmlDeviceGetUtilizationRates(handle)
                info["gpu_utilization"] = util.gpu
                info["memory_utilization"] = util.memory
                
                # Temperature
                temp = self.pynvml.nvmlDeviceGetTemperature(handle, self.pynvml.NVML_TEMPERATURE_GPU)
                info["temperature"] = temp
                
                # Power
                power = self.pynvml.nvmlDeviceGetPowerUsage(handle)
                info["power_draw"] = power / 1000.0  # Convert to watts
                
            except Exception:
                pass
        
        return info
    
    def get_memory_info(self, device_id: int = 0) -> Tuple[int, int, int]:
        """Get GPU memory information."""
        if not torch.cuda.is_available():
            return 0, 0, 0
        
        allocated = torch.cuda.memory_allocated(device_id)
        reserved = torch.cuda.memory_reserved(device_id)
        
        total = 0
        if hasattr(torch.cuda, 'get_device_properties'):
            props = torch.cuda.get_device_properties(device_id)
            total = props.total_memory
        
        return allocated, reserved, total


class PerformanceProfiler:
    """
    Advanced performance profiler for RevNet-Zero models.
    
    Provides comprehensive profiling including GPU utilization,
    memory usage patterns, and operation-level timing.
    """
    
    def __init__(self, config: ProfilerConfig = None):
        """
        Initialize performance profiler.
        
        Args:
            config: Profiler configuration
        """
        self.config = config or ProfilerConfig()
        
        # Monitoring components
        self.gpu_monitor = GPUMonitor()
        self.system_metrics: deque = deque(maxlen=self.config.max_samples)
        self.operation_metrics: List[PerformanceMetrics] = []
        
        # Profiling state
        self.profiling_active = False
        self.monitoring_thread = None
        self.lock = threading.RLock()
        
        # Operation timing
        self.active_operations: Dict[str, PerformanceMetrics] = {}
        
        # Trace output
        self.trace_dir = Path(self.config.trace_dir)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
    
    def start_profiling(self):
        """Start performance profiling."""
        with self.lock:
            if self.profiling_active:
                return
            
            self.profiling_active = True
            self.system_metrics.clear()
            self.operation_metrics.clear()
            
            # Start monitoring thread
            if self.config.enable_gpu_profiling or self.config.enable_memory_profiling:
                self.monitoring_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
                self.monitoring_thread.start()
    
    def stop_profiling(self):
        """Stop performance profiling."""
        with self.lock:
            self.profiling_active = False
            
            if self.monitoring_thread and self.monitoring_thread.is_alive():
                self.monitoring_thread.join(timeout=1.0)
    
    def _monitoring_loop(self):
        """Background monitoring loop."""
        while self.profiling_active:
            try:
                metrics = self._collect_system_metrics()
                self.system_metrics.append(metrics)
                
                time.sleep(self.config.sampling_interval)
            except Exception as e:
                print(f"Monitoring error: {e}")
                break
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        gpu_info = self.gpu_monitor.get_gpu_info()
        
        # CPU metrics
        cpu_percent = psutil.cpu_percent()
        
        # Memory metrics
        memory_info = psutil.virtual_memory()
        
        return SystemMetrics(
            timestamp=time.time(),
            gpu_memory_used=gpu_info.get("memory_allocated", 0),
            gpu_memory_total=gpu_info.get("memory_total", 0),
            gpu_utilization=gpu_info.get("gpu_utilization", 0.0),
            cpu_utilization=cpu_percent,
            memory_used=memory_info.used,
            memory_total=memory_info.total,
            temperature=gpu_info.get("temperature", 0.0),
            power_draw=gpu_info.get("power_draw", 0.0),
        )
    
    def start_operation(self, operation_name: str, **metadata) -> str:
        """
        Start timing an operation.
        
        Args:
            operation_name: Name of the operation
            **metadata: Additional metadata
            
        Returns:
            Operation ID for ending the operation
        """
        operation_id = f"{operation_name}_{time.time()}_{id(threading.current_thread())}"
        
        # Get initial GPU state
        gpu_memory_before = 0
        if torch.cuda.is_available():
            gpu_memory_before = torch.cuda.memory_allocated()
        
        metrics = PerformanceMetrics(
            operation_name=operation_name,
            duration_ms=0.0,
            gpu_memory_before=gpu_memory_before,
            start_time=time.time(),
            metadata=metadata,
        )
        
        with self.lock:
            self.active_operations[operation_id] = metrics
        
        return operation_id
    
    def end_operation(self, operation_id: str, **additional_metadata):
        """
        End timing an operation.
        
        Args:
            operation_id: Operation ID from start_operation
            **additional_metadata: Additional metadata
        """
        end_time = time.time()
        
        # Get final GPU state
        gpu_memory_after = 0
        gpu_memory_peak = 0
        if torch.cuda.is_available():
            gpu_memory_after = torch.cuda.memory_allocated()
            gpu_memory_peak = torch.cuda.max_memory_allocated()
        
        with self.lock:
            if operation_id not in self.active_operations:
                return
            
            metrics = self.active_operations[operation_id]
            del self.active_operations[operation_id]
        
        # Update metrics
        metrics.end_time = end_time
        metrics.duration_ms = (end_time - metrics.start_time) * 1000
        metrics.gpu_memory_after = gpu_memory_after
        metrics.gpu_memory_peak = gpu_memory_peak
        metrics.metadata.update(additional_metadata)
        
        # Calculate derived metrics
        self._calculate_derived_metrics(metrics)
        
        self.operation_metrics.append(metrics)
    
    def _calculate_derived_metrics(self, metrics: PerformanceMetrics):
        """Calculate derived performance metrics."""
        # Throughput calculation
        if "tokens" in metrics.metadata:
            tokens = metrics.metadata["tokens"]
            metrics.throughput_tokens_per_sec = tokens / (metrics.duration_ms / 1000)
        
        # Memory bandwidth (rough estimate)
        memory_transferred = metrics.gpu_memory_after - metrics.gpu_memory_before
        if memory_transferred > 0 and metrics.duration_ms > 0:
            metrics.memory_bandwidth_gb_per_sec = (memory_transferred / 1e9) / (metrics.duration_ms / 1000)
        
        # FLOPS estimation (if provided)
        if "flops" in metrics.metadata:
            metrics.flops = metrics.metadata["flops"]
    
    def profile_function(self, func: Callable, operation_name: str = None, **metadata):
        """
        Profile a function call.
        
        Args:
            func: Function to profile
            operation_name: Name for the operation
            **metadata: Additional metadata
            
        Returns:
            Function result
        """
        op_name = operation_name or func.__name__
        op_id = self.start_operation(op_name, **metadata)
        
        try:
            result = func()
            return result
        finally:
            self.end_operation(op_id)
    
    def profile_forward_pass(self, model: torch.nn.Module, inputs: torch.Tensor, **metadata):
        """
        Profile a forward pass through a model.
        
        Args:
            model: Model to profile
            inputs: Input tensors
            **metadata: Additional metadata
            
        Returns:
            Model outputs
        """
        # Add input metadata
        metadata.update({
            "model_name": model.__class__.__name__,
            "input_shape": list(inputs.shape),
            "tokens": inputs.numel(),
        })
        
        def forward_func():
            return model(inputs)
        
        return self.profile_function(forward_func, "forward_pass", **metadata)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance profiling summary."""
        if not self.operation_metrics:
            return {"error": "No profiling data available"}
        
        # Operation-level statistics
        operation_stats = defaultdict(list)
        for metrics in self.operation_metrics:
            operation_stats[metrics.operation_name].append(metrics)
        
        summary = {
            "total_operations": len(self.operation_metrics),
            "unique_operations": len(operation_stats),
            "profiling_duration": self._get_profiling_duration(),
            "operations": {},
        }
        
        # Per-operation statistics
        for op_name, op_metrics in operation_stats.items():
            durations = [m.duration_ms for m in op_metrics]
            throughputs = [m.throughput_tokens_per_sec for m in op_metrics if m.throughput_tokens_per_sec > 0]
            
            summary["operations"][op_name] = {
                "count": len(op_metrics),
                "total_time_ms": sum(durations),
                "avg_time_ms": np.mean(durations),
                "min_time_ms": min(durations),
                "max_time_ms": max(durations),
                "std_time_ms": np.std(durations),
            }
            
            if throughputs:
                summary["operations"][op_name].update({
                    "avg_throughput": np.mean(throughputs),
                    "max_throughput": max(throughputs),
                })
        
        # System-level statistics
        if self.system_metrics:
            gpu_utils = [m.gpu_utilization for m in self.system_metrics]
            cpu_utils = [m.cpu_utilization for m in self.system_metrics]
            memory_usage = [m.gpu_memory_used for m in self.system_metrics]
            
            summary["system"] = {
                "avg_gpu_utilization": np.mean(gpu_utils),
                "max_gpu_utilization": max(gpu_utils),
                "avg_cpu_utilization": np.mean(cpu_utils),
                "max_cpu_utilization": max(cpu_utils),
                "peak_gpu_memory": max(memory_usage) if memory_usage else 0,
                "avg_gpu_memory": np.mean(memory_usage) if memory_usage else 0,
            }
        
        return summary
    
    def _get_profiling_duration(self) -> float:
        """Get total profiling duration."""
        if not self.operation_metrics:
            return 0.0
        
        start_time = min(m.start_time for m in self.operation_metrics)
        end_time = max(m.end_time for m in self.operation_metrics)
        
        return end_time - start_time
    
    def export_trace(self, filename: str = None):
        """Export profiling trace to file."""
        if filename is None:
            filename = f"trace_{int(time.time())}.json"
        
        trace_path = self.trace_dir / filename
        
        # Prepare trace data
        trace_data = {
            "traceEvents": [],
            "displayTimeUnit": "ms",
            "systemMetrics": [],
        }
        
        # Add operation events
        for metrics in self.operation_metrics:
            event = {
                "name": metrics.operation_name,
                "cat": "operation",
                "ph": "X",  # Complete event
                "ts": int(metrics.start_time * 1000000),  # microseconds
                "dur": int(metrics.duration_ms * 1000),  # microseconds
                "pid": 1,
                "tid": 1,
                "args": {
                    "gpu_memory_delta": metrics.gpu_memory_after - metrics.gpu_memory_before,
                    "throughput": metrics.throughput_tokens_per_sec,
                    **metrics.metadata,
                }
            }
            trace_data["traceEvents"].append(event)
        
        # Add system metrics
        for metrics in list(self.system_metrics):
            trace_data["systemMetrics"].append({
                "timestamp": metrics.timestamp,
                "gpu_utilization": metrics.gpu_utilization,
                "cpu_utilization": metrics.cpu_utilization,
                "gpu_memory_used": metrics.gpu_memory_used,
                "temperature": metrics.temperature,
                "power_draw": metrics.power_draw,
            })
        
        # Save trace
        with open(trace_path, 'w') as f:
            json.dump(trace_data, f, indent=2)
        
        print(f"Profiling trace saved to {trace_path}")
    
    def plot_performance_metrics(self, save_path: str = None):
        """Plot performance metrics."""
        if not self.operation_metrics and not self.system_metrics:
            print("No profiling data to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Operation timing histogram
        if self.operation_metrics:
            durations = [m.duration_ms for m in self.operation_metrics]
            axes[0, 0].hist(durations, bins=30, alpha=0.7)
            axes[0, 0].set_xlabel('Duration (ms)')
            axes[0, 0].set_ylabel('Count')
            axes[0, 0].set_title('Operation Duration Distribution')
        
        # GPU utilization over time
        if self.system_metrics:
            timestamps = [m.timestamp for m in self.system_metrics]
            gpu_utils = [m.gpu_utilization for m in self.system_metrics]
            
            if timestamps and gpu_utils:
                # Convert to relative time
                start_time = min(timestamps)
                rel_times = [(t - start_time) / 60 for t in timestamps]  # minutes
                
                axes[0, 1].plot(rel_times, gpu_utils)
                axes[0, 1].set_xlabel('Time (minutes)')
                axes[0, 1].set_ylabel('GPU Utilization (%)')
                axes[0, 1].set_title('GPU Utilization Over Time')
        
        # Memory usage over time
        if self.system_metrics:
            memory_usage = [m.gpu_memory_used / 1e9 for m in self.system_metrics]  # GB
            
            if timestamps and memory_usage:
                axes[1, 0].plot(rel_times, memory_usage)
                axes[1, 0].set_xlabel('Time (minutes)')
                axes[1, 0].set_ylabel('GPU Memory (GB)')
                axes[1, 0].set_title('GPU Memory Usage Over Time')
        
        # Operation throughput
        if self.operation_metrics:
            throughputs = [m.throughput_tokens_per_sec for m in self.operation_metrics 
                          if m.throughput_tokens_per_sec > 0]
            
            if throughputs:
                axes[1, 1].hist(throughputs, bins=20, alpha=0.7)
                axes[1, 1].set_xlabel('Throughput (tokens/sec)')
                axes[1, 1].set_ylabel('Count')
                axes[1, 1].set_title('Throughput Distribution')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Performance plots saved to {save_path}")
        
        plt.show()


# Context manager for easy profiling
class ProfileContext:
    """Context manager for performance profiling."""
    
    def __init__(self, profiler: PerformanceProfiler, operation_name: str, **metadata):
        self.profiler = profiler
        self.operation_name = operation_name
        self.metadata = metadata
        self.operation_id = None
    
    def __enter__(self):
        self.operation_id = self.profiler.start_operation(self.operation_name, **self.metadata)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.operation_id:
            self.profiler.end_operation(self.operation_id)