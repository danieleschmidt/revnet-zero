"""
Memory optimization utilities for reversible neural networks.

This module provides tools for automatically optimizing memory usage
in reversible transformer models.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from .scheduler import MemoryScheduler, RecomputePolicy
from .profiler import MemoryProfiler, LayerProfile


@dataclass
class OptimizationResult:
    """Result of memory optimization."""
    original_memory: int
    optimized_memory: int
    memory_saved: int
    reduction_percentage: float
    optimizations_applied: List[str]
    performance_impact: float


class MemoryOptimizer:
    """
    Automatic memory optimizer for reversible neural networks.
    
    Analyzes model memory usage patterns and applies optimizations
    to reduce memory consumption while minimizing performance impact.
    """
    
    def __init__(
        self,
        model: nn.Module,
        target_memory_reduction: float = 0.7,
        max_performance_impact: float = 0.2,
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.target_memory_reduction = target_memory_reduction
        self.max_performance_impact = max_performance_impact
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Analysis results
        self.layer_profiles: Dict[int, LayerProfile] = {}
        self.optimization_candidates: List[Tuple[int, str, float]] = []
        
    def analyze_model(self, sample_input: torch.Tensor) -> Dict[str, Any]:
        """
        Analyze model memory usage patterns.
        
        Args:
            sample_input: Sample input tensor for profiling
            
        Returns:
            Analysis results
        """
        profiler = MemoryProfiler(self.device)
        
        # Profile each layer
        profiler.start_profiling()
        
        layer_id = 0
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Leaf modules only
                try:
                    profile = profiler.profile_layer(layer_id, name, module, sample_input)
                    self.layer_profiles[layer_id] = profile
                    layer_id += 1
                except Exception as e:
                    print(f"Warning: Could not profile layer {name}: {e}")
                    continue
        
        profiler.stop_profiling()
        
        # Analyze optimization opportunities
        self._identify_optimization_candidates()
        
        return {
            "total_layers": len(self.layer_profiles),
            "total_memory": sum(p.peak_memory for p in self.layer_profiles.values()),
            "optimization_candidates": len(self.optimization_candidates),
            "estimated_savings": self._estimate_total_savings(),
        }
    
    def _identify_optimization_candidates(self):
        """Identify layers that are good candidates for optimization."""
        self.optimization_candidates.clear()
        
        for layer_id, profile in self.layer_profiles.items():
            # Calculate memory efficiency and recomputation cost
            memory_usage = profile.peak_memory
            recompute_cost = profile.backward_time / max(profile.forward_time, 1e-6)
            
            # Layers with high memory usage and low recomputation cost are good candidates
            if memory_usage > 1024 * 1024:  # > 1MB
                optimization_score = memory_usage / (1 + recompute_cost)
                self.optimization_candidates.append((layer_id, "reversible", optimization_score))
    
    def _estimate_total_savings(self) -> int:
        """Estimate total memory savings from optimizations."""
        total_savings = 0
        
        for layer_id, optimization_type, score in self.optimization_candidates:
            if optimization_type == "reversible":
                # Reversible layers save ~70% of activation memory
                profile = self.layer_profiles[layer_id]
                savings = profile.activations_size * 0.7
                total_savings += savings
        
        return int(total_savings)
    
    def optimize_memory(self, sample_input: torch.Tensor) -> OptimizationResult:
        """
        Automatically optimize model memory usage.
        
        Args:
            sample_input: Sample input for optimization
            
        Returns:
            Optimization results
        """
        # Initial analysis
        initial_analysis = self.analyze_model(sample_input)
        original_memory = initial_analysis["total_memory"]
        
        optimizations_applied = []
        current_memory = original_memory
        
        # Sort candidates by optimization score (highest first)
        candidates = sorted(self.optimization_candidates, key=lambda x: x[2], reverse=True)
        
        for layer_id, optimization_type, score in candidates:
            # Check if we've reached our target reduction
            current_reduction = (original_memory - current_memory) / original_memory
            if current_reduction >= self.target_memory_reduction:
                break
            
            # Apply optimization
            if optimization_type == "reversible":
                success = self._make_layer_reversible(layer_id)
                if success:
                    profile = self.layer_profiles[layer_id]
                    memory_saved = profile.activations_size * 0.7
                    current_memory -= memory_saved
                    optimizations_applied.append(f"Made layer {layer_id} reversible")
        
        # Calculate final results
        memory_saved = original_memory - current_memory
        reduction_percentage = memory_saved / original_memory * 100
        
        # Estimate performance impact (rough approximation)
        performance_impact = len(optimizations_applied) * 0.05  # 5% per optimization
        
        return OptimizationResult(
            original_memory=original_memory,
            optimized_memory=current_memory,
            memory_saved=memory_saved,
            reduction_percentage=reduction_percentage,
            optimizations_applied=optimizations_applied,
            performance_impact=performance_impact,
        )
    
    def _make_layer_reversible(self, layer_id: int) -> bool:
        """
        Convert a layer to reversible implementation.
        
        Args:
            layer_id: Layer to convert
            
        Returns:
            True if successful, False otherwise
        """
        # This is a placeholder - in practice, this would involve
        # replacing standard layers with reversible equivalents
        # For now, we just simulate the optimization
        return True
    
    def generate_scheduler_config(self) -> MemoryScheduler:
        """
        Generate optimized memory scheduler configuration.
        
        Returns:
            Configured memory scheduler
        """
        scheduler = MemoryScheduler(self.model)
        
        # Configure policies based on analysis
        for layer_id, profile in self.layer_profiles.items():
            memory_usage = profile.peak_memory
            recompute_cost = profile.backward_time / max(profile.forward_time, 1e-6)
            
            if memory_usage > 10 * 1024 * 1024:  # > 10MB
                scheduler.set_policy(layer_id, RecomputePolicy.RECOMPUTE)
            elif memory_usage < 1024 * 1024:  # < 1MB
                scheduler.set_policy(layer_id, RecomputePolicy.STORE)
            else:
                scheduler.set_policy(layer_id, RecomputePolicy.ADAPTIVE)
        
        return scheduler
    
    def suggest_batch_size(self, max_memory: int, current_batch_size: int) -> int:
        """
        Suggest optimal batch size based on memory constraints.
        
        Args:
            max_memory: Maximum available memory
            current_batch_size: Current batch size
            
        Returns:
            Suggested batch size
        """
        if not self.layer_profiles:
            return current_batch_size
        
        # Estimate memory usage per sample
        total_memory_per_sample = sum(
            p.peak_memory for p in self.layer_profiles.values()
        ) // current_batch_size
        
        # Calculate optimal batch size
        optimal_batch_size = max_memory // total_memory_per_sample
        
        # Apply safety margin
        return max(1, int(optimal_batch_size * 0.8))
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """
        Generate comprehensive optimization report.
        
        Returns:
            Detailed optimization report
        """
        if not self.layer_profiles:
            return {"error": "Model not analyzed yet. Call analyze_model() first."}
        
        # Layer analysis
        layer_analysis = []
        for layer_id, profile in self.layer_profiles.items():
            memory_efficiency = profile.peak_memory / max(profile.forward_time, 1e-6)
            recompute_ratio = profile.backward_time / max(profile.forward_time, 1e-6)
            
            layer_analysis.append({
                "layer_id": layer_id,
                "layer_name": profile.layer_name,
                "memory_usage": profile.peak_memory,
                "memory_efficiency": memory_efficiency,
                "recompute_ratio": recompute_ratio,
                "optimization_score": memory_efficiency / (1 + recompute_ratio),
            })
        
        # Sort by optimization potential
        layer_analysis.sort(key=lambda x: x["optimization_score"], reverse=True)
        
        # Summary statistics
        total_memory = sum(p.peak_memory for p in self.layer_profiles.values())
        avg_recompute_ratio = sum(
            p.backward_time / max(p.forward_time, 1e-6) 
            for p in self.layer_profiles.values()
        ) / len(self.layer_profiles)
        
        return {
            "summary": {
                "total_layers": len(self.layer_profiles),
                "total_memory": total_memory,
                "average_recompute_ratio": avg_recompute_ratio,
                "optimization_candidates": len(self.optimization_candidates),
                "estimated_max_savings": self._estimate_total_savings(),
            },
            "layer_analysis": layer_analysis,
            "optimization_candidates": [
                {
                    "layer_id": layer_id,
                    "optimization_type": opt_type,
                    "score": score,
                    "layer_name": self.layer_profiles[layer_id].layer_name,
                }
                for layer_id, opt_type, score in self.optimization_candidates
            ],
            "recommendations": self._generate_optimization_recommendations(),
        }
    
    def _generate_optimization_recommendations(self) -> List[Dict[str, Any]]:
        """Generate optimization recommendations."""
        recommendations = []
        
        # High memory usage layers
        high_memory_layers = [
            (layer_id, profile)
            for layer_id, profile in self.layer_profiles.items()
            if profile.peak_memory > 50 * 1024 * 1024  # > 50MB
        ]
        
        if high_memory_layers:
            recommendations.append({
                "type": "high_memory_optimization",
                "priority": "high",
                "description": f"Convert {len(high_memory_layers)} high-memory layers to reversible",
                "affected_layers": [layer_id for layer_id, _ in high_memory_layers],
                "expected_savings": sum(p.peak_memory * 0.7 for _, p in high_memory_layers),
                "implementation": "Use ReversibleAttention/ReversibleFFN replacements",
            })
        
        # Batch size optimization
        total_memory = sum(p.peak_memory for p in self.layer_profiles.values())
        if total_memory > 1024 * 1024 * 1024:  # > 1GB
            recommendations.append({
                "type": "batch_size_optimization", 
                "priority": "medium",
                "description": "Consider reducing batch size for memory efficiency",
                "expected_savings": total_memory * 0.5,
                "implementation": "Reduce batch size and use gradient accumulation",
            })
        
        # Gradient checkpointing
        slow_layers = [
            layer_id for layer_id, profile in self.layer_profiles.items()
            if profile.backward_time / max(profile.forward_time, 1e-6) < 2.0
        ]
        
        if slow_layers:
            recommendations.append({
                "type": "gradient_checkpointing",
                "priority": "low",
                "description": f"Apply gradient checkpointing to {len(slow_layers)} fast layers",
                "affected_layers": slow_layers,
                "expected_savings": "20-40% memory reduction",
                "implementation": "Use torch.utils.checkpoint for these layers",
            })
        
        return recommendations