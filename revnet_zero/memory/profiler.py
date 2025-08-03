"""
Memory profiling utilities for reversible neural networks.

This module provides tools for profiling memory usage patterns and
generating optimization recommendations.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import time
import json
import matplotlib.pyplot as plt
from pathlib import Path


@dataclass
class MemorySnapshot:
    """Single memory usage snapshot."""
    timestamp: float
    allocated: int
    reserved: int
    active: int
    step: str
    layer_id: Optional[int] = None


@dataclass
class LayerProfile:
    """Memory profile for a single layer."""
    layer_id: int
    layer_name: str
    forward_memory: int
    backward_memory: int
    peak_memory: int
    forward_time: float
    backward_time: float
    activations_size: int
    parameters_size: int
    gradients_size: int


class MemoryProfiler:
    """
    Basic memory profiler for reversible networks.
    
    Tracks memory usage during forward and backward passes to identify
    memory bottlenecks and optimization opportunities.
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.snapshots: List[MemorySnapshot] = []
        self.layer_profiles: Dict[int, LayerProfile] = {}
        self.profiling_enabled = False
        
        # Baseline memory
        self.baseline_memory = 0
        
    def start_profiling(self):
        """Start memory profiling."""
        self.profiling_enabled = True
        self.snapshots.clear()
        self.layer_profiles.clear()
        
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
            torch.cuda.empty_cache()
            self.baseline_memory = torch.cuda.memory_allocated(self.device)
        
        self._take_snapshot("start")
    
    def stop_profiling(self):
        """Stop memory profiling."""
        self.profiling_enabled = False
        self._take_snapshot("end")
    
    def _take_snapshot(self, step: str, layer_id: Optional[int] = None):
        """Take a memory usage snapshot."""
        if not self.profiling_enabled:
            return
        
        if self.device.type == "cuda":
            allocated = torch.cuda.memory_allocated(self.device)
            reserved = torch.cuda.memory_reserved(self.device)
            active = allocated
        else:
            # Approximate for CPU
            allocated = reserved = active = 0
        
        snapshot = MemorySnapshot(
            timestamp=time.time(),
            allocated=allocated,
            reserved=reserved,
            active=active,
            step=step,
            layer_id=layer_id,
        )
        
        self.snapshots.append(snapshot)
    
    def profile_layer(self, layer_id: int, layer_name: str, layer: nn.Module, 
                     input_tensor: torch.Tensor) -> LayerProfile:
        """
        Profile memory usage for a single layer.
        
        Args:
            layer_id: Layer identifier
            layer_name: Layer name
            layer: The layer module
            input_tensor: Input tensor for profiling
            
        Returns:
            Memory profile for the layer
        """
        if not self.profiling_enabled:
            raise RuntimeError("Profiling not started. Call start_profiling() first.")
        
        # Baseline memory before layer
        self._take_snapshot(f"layer_{layer_id}_start", layer_id)
        start_memory = self._get_current_memory()
        
        # Forward pass timing
        start_time = time.time()
        with torch.no_grad():
            output = layer(input_tensor)
        forward_time = time.time() - start_time
        
        # Memory after forward pass
        self._take_snapshot(f"layer_{layer_id}_forward", layer_id)
        forward_memory = self._get_current_memory() - start_memory
        
        # Backward pass (if applicable)
        if input_tensor.requires_grad:
            # Enable gradients for backward pass
            input_tensor.requires_grad_(True)
            output = layer(input_tensor)
            
            # Create dummy loss
            loss = output.sum()
            
            start_time = time.time()
            loss.backward()
            backward_time = time.time() - start_time
            
            self._take_snapshot(f"layer_{layer_id}_backward", layer_id)
            backward_memory = self._get_current_memory() - start_memory - forward_memory
        else:
            backward_time = 0.0
            backward_memory = 0
        
        # Calculate component sizes
        parameters_size = sum(p.numel() * p.element_size() for p in layer.parameters())
        gradients_size = sum(
            p.grad.numel() * p.grad.element_size() 
            for p in layer.parameters() 
            if p.grad is not None
        )
        activations_size = output.numel() * output.element_size()
        
        # Peak memory during layer computation
        layer_snapshots = [
            s for s in self.snapshots 
            if s.layer_id == layer_id and s.timestamp >= self.snapshots[-3].timestamp
        ]
        peak_memory = max(s.allocated for s in layer_snapshots) - start_memory
        
        profile = LayerProfile(
            layer_id=layer_id,
            layer_name=layer_name,
            forward_memory=forward_memory,
            backward_memory=backward_memory,
            peak_memory=peak_memory,
            forward_time=forward_time,
            backward_time=backward_time,
            activations_size=activations_size,
            parameters_size=parameters_size,
            gradients_size=gradients_size,
        )
        
        self.layer_profiles[layer_id] = profile
        return profile
    
    def _get_current_memory(self) -> int:
        """Get current memory usage."""
        if self.device.type == "cuda":
            return torch.cuda.memory_allocated(self.device)
        else:
            return 0
    
    def get_memory_timeline(self) -> List[Tuple[float, int]]:
        """
        Get memory usage timeline.
        
        Returns:
            List of (timestamp, memory_usage) tuples
        """
        return [(s.timestamp, s.allocated) for s in self.snapshots]
    
    def get_peak_memory(self) -> int:
        """Get peak memory usage during profiling."""
        if not self.snapshots:
            return 0
        return max(s.allocated for s in self.snapshots)
    
    def get_memory_reduction_estimate(self, reversible_layers: List[int]) -> Dict[str, int]:
        """
        Estimate memory reduction from using reversible layers.
        
        Args:
            reversible_layers: List of layer IDs to make reversible
            
        Returns:
            Dictionary with memory reduction estimates
        """
        total_activation_memory = sum(
            profile.activations_size 
            for layer_id, profile in self.layer_profiles.items()
            if layer_id in reversible_layers
        )
        
        # Reversible layers only store coupling outputs (much smaller)
        reversible_memory = total_activation_memory * 0.1  # Approximate
        
        return {
            "standard_memory": total_activation_memory,
            "reversible_memory": reversible_memory,
            "memory_saved": total_activation_memory - reversible_memory,
            "reduction_percentage": (total_activation_memory - reversible_memory) / total_activation_memory * 100,
        }


class DetailedMemoryProfiler(MemoryProfiler):
    """
    Advanced memory profiler with detailed analysis and visualization.
    
    Provides comprehensive memory analysis including bottleneck identification
    and optimization recommendations.
    """
    
    def __init__(self, device: Optional[torch.device] = None, 
                 output_dir: Optional[Path] = None):
        super().__init__(device)
        self.output_dir = output_dir or Path("./memory_profiles")
        self.output_dir.mkdir(exist_ok=True)
        
        # Additional tracking
        self.operation_profiles: Dict[str, List[float]] = {}
        self.memory_events: List[Dict[str, Any]] = []
    
    def profile_operation(self, operation_name: str, operation_fn, *args, **kwargs):
        """
        Profile a specific operation.
        
        Args:
            operation_name: Name of the operation
            operation_fn: Function to profile
            *args, **kwargs: Arguments for the function
            
        Returns:
            Result of the operation
        """
        if operation_name not in self.operation_profiles:
            self.operation_profiles[operation_name] = []
        
        start_memory = self._get_current_memory()
        start_time = time.time()
        
        result = operation_fn(*args, **kwargs)
        
        end_time = time.time()
        end_memory = self._get_current_memory()
        
        self.operation_profiles[operation_name].append({
            "memory_delta": end_memory - start_memory,
            "execution_time": end_time - start_time,
            "timestamp": start_time,
        })
        
        return result
    
    def generate_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive memory profiling report.
        
        Returns:
            Detailed profiling report
        """
        report = {
            "summary": {
                "total_snapshots": len(self.snapshots),
                "profiling_duration": self.snapshots[-1].timestamp - self.snapshots[0].timestamp if self.snapshots else 0,
                "peak_memory": self.get_peak_memory(),
                "baseline_memory": self.baseline_memory,
                "memory_overhead": self.get_peak_memory() - self.baseline_memory,
            },
            "layer_profiles": {},
            "memory_timeline": self.get_memory_timeline(),
            "operation_profiles": self.operation_profiles,
            "recommendations": self._generate_recommendations(),
        }
        
        # Add layer profile details
        for layer_id, profile in self.layer_profiles.items():
            report["layer_profiles"][layer_id] = {
                "layer_name": profile.layer_name,
                "forward_memory": profile.forward_memory,
                "backward_memory": profile.backward_memory,
                "peak_memory": profile.peak_memory,
                "forward_time": profile.forward_time,
                "backward_time": profile.backward_time,
                "memory_efficiency": profile.forward_memory / max(profile.forward_time, 1e-6),
                "recompute_cost_ratio": profile.backward_time / max(profile.forward_time, 1e-6),
            }
        
        return report
    
    def _generate_recommendations(self) -> List[Dict[str, Any]]:
        """Generate optimization recommendations."""
        recommendations = []
        
        if not self.layer_profiles:
            return recommendations
        
        # Identify memory-heavy layers
        memory_threshold = 1024 * 1024 * 10  # 10MB
        memory_heavy_layers = [
            (layer_id, profile) 
            for layer_id, profile in self.layer_profiles.items()
            if profile.peak_memory > memory_threshold
        ]
        
        if memory_heavy_layers:
            recommendations.append({
                "type": "high_memory_layers",
                "priority": "high",
                "description": f"Found {len(memory_heavy_layers)} layers with high memory usage",
                "affected_layers": [layer_id for layer_id, _ in memory_heavy_layers],
                "suggestion": "Consider making these layers reversible to reduce memory usage",
                "expected_savings": sum(profile.peak_memory for _, profile in memory_heavy_layers) * 0.7,
            })
        
        # Identify slow layers
        time_threshold = 0.1  # 100ms
        slow_layers = [
            (layer_id, profile)
            for layer_id, profile in self.layer_profiles.items()
            if profile.forward_time > time_threshold
        ]
        
        if slow_layers:
            recommendations.append({
                "type": "slow_layers",
                "priority": "medium", 
                "description": f"Found {len(slow_layers)} layers with slow execution",
                "affected_layers": [layer_id for layer_id, _ in slow_layers],
                "suggestion": "Consider kernel optimization or different implementation",
                "expected_improvement": "10-30% speedup possible",
            })
        
        # Memory fragmentation detection
        peak_memory = self.get_peak_memory()
        if peak_memory > self.baseline_memory * 3:
            recommendations.append({
                "type": "memory_fragmentation",
                "priority": "medium",
                "description": "High memory usage detected, possible fragmentation",
                "suggestion": "Enable memory pooling or adjust batch size",
                "expected_improvement": "20-40% memory reduction",
            })
        
        return recommendations
    
    def plot_memory_timeline(self, save_path: Optional[Path] = None):
        """
        Plot memory usage timeline.
        
        Args:
            save_path: Optional path to save the plot
        """
        if not self.snapshots:
            print("No profiling data available")
            return
        
        timestamps = [s.timestamp - self.snapshots[0].timestamp for s in self.snapshots]
        memory_usage = [s.allocated / (1024**3) for s in self.snapshots]  # GB
        
        plt.figure(figsize=(12, 6))
        plt.plot(timestamps, memory_usage, 'b-', linewidth=2)
        plt.xlabel('Time (seconds)')
        plt.ylabel('Memory Usage (GB)')
        plt.title('Memory Usage Timeline')
        plt.grid(True, alpha=0.3)
        
        # Annotate layer boundaries
        layer_boundaries = [
            (s.timestamp - self.snapshots[0].timestamp, s.allocated / (1024**3))
            for s in self.snapshots
            if s.layer_id is not None and "start" in s.step
        ]
        
        for time_point, memory_point in layer_boundaries:
            plt.axvline(x=time_point, color='r', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.savefig(self.output_dir / "memory_timeline.png", dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def export_chrome_trace(self, save_path: Optional[Path] = None):
        """
        Export profiling data in Chrome trace format.
        
        Args:
            save_path: Optional path to save the trace
        """
        trace_events = []
        
        for i, snapshot in enumerate(self.snapshots):
            if i == 0:
                continue
            
            prev_snapshot = self.snapshots[i - 1]
            
            event = {
                "name": snapshot.step,
                "cat": "memory",
                "ph": "X",  # Complete event
                "ts": int(prev_snapshot.timestamp * 1e6),  # microseconds
                "dur": int((snapshot.timestamp - prev_snapshot.timestamp) * 1e6),
                "pid": 1,
                "tid": 1,
                "args": {
                    "allocated": snapshot.allocated,
                    "reserved": snapshot.reserved,
                    "layer_id": snapshot.layer_id,
                }
            }
            trace_events.append(event)
        
        trace_data = {
            "traceEvents": trace_events,
            "displayTimeUnit": "ms",
        }
        
        output_path = save_path or self.output_dir / "memory_trace.json"
        with open(output_path, 'w') as f:
            json.dump(trace_data, f, indent=2)
        
        print(f"Chrome trace saved to {output_path}")
        print("Open chrome://tracing and load this file for visualization")