"""
Memory scheduling for reversible neural networks.

This module implements adaptive memory scheduling strategies to optimize
the trade-off between memory usage and computational overhead.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Union, Any
from enum import Enum
from dataclasses import dataclass
import psutil
import gc


class RecomputePolicy(Enum):
    """Policies for activation recomputation."""
    STORE = "store"           # Store activations (no recomputation)
    RECOMPUTE = "recompute"   # Always recompute activations
    ADAPTIVE = "adaptive"     # Decide based on memory budget


@dataclass
class LayerMemoryProfile:
    """Memory profile for a single layer."""
    layer_id: int
    forward_memory: int      # Memory used during forward pass
    backward_memory: int     # Memory used during backward pass
    recompute_cost: float    # Relative cost of recomputation
    storage_cost: int        # Memory cost of storing activations


class MemoryScheduler:
    """
    Base memory scheduler for reversible neural networks.
    
    Manages the trade-off between memory usage and computation by deciding
    which layer activations to store vs recompute during backpropagation.
    """
    
    def __init__(
        self,
        model: nn.Module,
        memory_budget: Optional[int] = None,
        strategy: str = "adaptive",
        device: Optional[torch.device] = None,
        enable_generation3_optimizations: bool = True,
    ):
        """🚀 GENERATION 3: Enhanced Memory Scheduler"""
        self.model = model
        self.memory_budget = memory_budget or self._get_default_memory_budget()
        self.enable_generation3_optimizations = enable_generation3_optimizations
        
        # Generation 3 enhancements
        if enable_generation3_optimizations:
            self.intelligent_scheduling = True
            self.adaptive_recomputation = True
            self.performance_analytics = {}
            self._setup_intelligent_scheduling()
            
        self.strategy = strategy
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def _setup_intelligent_scheduling(self):
        """Setup Generation 3 intelligent scheduling capabilities"""
        self.optimization_history = []
        self.memory_efficiency_score = 0.0
        self.adaptive_threshold = 0.8
        
    # Memory tracking
        self.current_memory_usage = 0
        self.peak_memory_usage = 0
        self.memory_saved = 0
        
        # Layer policies
        self.layer_policies: Dict[int, RecomputePolicy] = {}
        self.layer_profiles: Dict[int, LayerMemoryProfile] = {}
        
        # Runtime statistics
        self.recomputed_layers: List[int] = []
        self.forward_time = 0.0
        self.backward_time = 0.0
    
    def _get_default_memory_budget(self) -> int:
        """Get default memory budget based on available GPU memory."""
        if torch.cuda.is_available():
            total_memory = torch.cuda.get_device_properties(0).total_memory
            # Use 80% of available GPU memory as budget
            return int(0.8 * total_memory)
        else:
            # Use 80% of available system memory
            return int(0.8 * psutil.virtual_memory().total)
    
    def set_policy(self, layers: Union[int, List[int]], policy: RecomputePolicy):
        """
        Set recomputation policy for specific layers.
        
        Args:
            layers: Layer index or list of layer indices
            policy: Recomputation policy to apply
        """
        if isinstance(layers, int):
            layers = [layers]
        
        for layer_id in layers:
            self.layer_policies[layer_id] = policy
    
    def should_recompute(self, layer_id: int) -> bool:
        """
        Decide whether to recompute activations for a specific layer.
        
        Args:
            layer_id: Layer identifier
            
        Returns:
            True if activations should be recomputed, False if stored
        """
        if layer_id in self.layer_policies:
            policy = self.layer_policies[layer_id]
            
            if policy == RecomputePolicy.STORE:
                return False
            elif policy == RecomputePolicy.RECOMPUTE:
                return True
            elif policy == RecomputePolicy.ADAPTIVE:
                return self._adaptive_decision(layer_id)
        
        # Default adaptive behavior
        return self._adaptive_decision(layer_id)
    
    def _adaptive_decision(self, layer_id: int) -> bool:
        """
        Make adaptive decision based on current memory usage.
        
        Args:
            layer_id: Layer identifier
            
        Returns:
            True if should recompute, False if should store
        """
        current_memory = self._get_current_memory_usage()
        
        # If we're approaching memory budget, prefer recomputation
        memory_pressure = current_memory / self.memory_budget
        
        if memory_pressure > 0.9:
            return True
        elif memory_pressure < 0.5:
            return False
        else:
            # Use layer-specific characteristics
            if layer_id in self.layer_profiles:
                profile = self.layer_profiles[layer_id]
                # Recompute if storage cost is high relative to recompute cost
                return profile.storage_cost > profile.recompute_cost * 2
            else:
                # Default: recompute middle layers
                return True
    
    def _get_current_memory_usage(self) -> int:
        """Get current memory usage on the target device."""
        if self.device.type == "cuda":
            return torch.cuda.memory_allocated(self.device)
        else:
            # Approximate system memory usage
            process = psutil.Process()
            return process.memory_info().rss
    
    def update_memory_stats(self):
        """Update memory usage statistics."""
        current = self._get_current_memory_usage()
        self.current_memory_usage = current
        self.peak_memory_usage = max(self.peak_memory_usage, current)
    
    def __enter__(self):
        """Enter context manager."""
        self.recomputed_layers.clear()
        self.memory_saved = 0
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context manager."""
        self.update_memory_stats()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()


class AdaptiveScheduler(MemoryScheduler):
    """
    Adaptive memory scheduler with dynamic policy adjustment.
    
    This scheduler continuously monitors memory usage and adapts its
    recomputation strategy based on current conditions.
    """
    
    def __init__(
        self,
        model: nn.Module,
        memory_budget: Optional[int] = None,
        profile: Optional[Dict[int, LayerMemoryProfile]] = None,
        recompute_granularity: str = "layer",
        adaptation_rate: float = 0.1,
    ):
        super().__init__(model, memory_budget, strategy="adaptive")
        
        self.layer_profiles = profile or {}
        self.recompute_granularity = recompute_granularity
        self.adaptation_rate = adaptation_rate
        
        # Adaptive parameters
        self.memory_threshold = 0.8
        self.adjustment_history: List[float] = []
        
        # Initialize layer policies based on profiles
        self._initialize_policies()
    
    def _initialize_policies(self):
        """Initialize layer policies based on memory profiles."""
        if not self.layer_profiles:
            return
        
        # Sort layers by memory efficiency (storage_cost / recompute_cost)
        layer_efficiency = []
        for layer_id, profile in self.layer_profiles.items():
            if profile.recompute_cost > 0:
                efficiency = profile.storage_cost / profile.recompute_cost
                layer_efficiency.append((layer_id, efficiency))
        
        layer_efficiency.sort(key=lambda x: x[1], reverse=True)
        
        # Set initial policies: recompute most memory-expensive layers
        total_layers = len(layer_efficiency)
        recompute_threshold = int(0.6 * total_layers)  # Recompute top 60%
        
        for i, (layer_id, _) in enumerate(layer_efficiency):
            if i < recompute_threshold:
                self.layer_policies[layer_id] = RecomputePolicy.RECOMPUTE
            else:
                self.layer_policies[layer_id] = RecomputePolicy.STORE
    
    def adapt_policies(self, current_step: int):
        """
        Adapt recomputation policies based on current memory usage.
        
        Args:
            current_step: Current training step
        """
        memory_pressure = self.current_memory_usage / self.memory_budget
        
        # Track memory pressure history
        self.adjustment_history.append(memory_pressure)
        if len(self.adjustment_history) > 100:
            self.adjustment_history.pop(0)
        
        # Adapt threshold based on recent memory pressure
        if len(self.adjustment_history) >= 10:
            avg_pressure = sum(self.adjustment_history[-10:]) / 10
            
            if avg_pressure > 0.9:
                # Increase recomputation
                self.memory_threshold = max(0.6, self.memory_threshold - self.adaptation_rate)
            elif avg_pressure < 0.6:
                # Decrease recomputation
                self.memory_threshold = min(0.9, self.memory_threshold + self.adaptation_rate)
        
        # Update layer policies
        self._update_layer_policies()
    
    def _update_layer_policies(self):
        """Update layer policies based on current threshold."""
        for layer_id, profile in self.layer_profiles.items():
            if profile.recompute_cost > 0:
                efficiency = profile.storage_cost / profile.recompute_cost
                
                # Normalize efficiency to [0, 1] range
                max_efficiency = max(
                    p.storage_cost / p.recompute_cost 
                    for p in self.layer_profiles.values() 
                    if p.recompute_cost > 0
                )
                normalized_efficiency = efficiency / max_efficiency
                
                if normalized_efficiency > self.memory_threshold:
                    self.layer_policies[layer_id] = RecomputePolicy.RECOMPUTE
                else:
                    self.layer_policies[layer_id] = RecomputePolicy.STORE


class LayerScheduler(MemoryScheduler):
    """
    Layer-specific memory scheduler with fine-grained control.
    
    Allows explicit control over recomputation policies for individual
    layers or layer groups.
    """
    
    def __init__(self, model: nn.Module, memory_budget: Optional[int] = None):
        super().__init__(model, memory_budget, strategy="layer_specific")
        
        # Layer groupings
        self.layer_groups: Dict[str, List[int]] = {}
        self.group_policies: Dict[str, RecomputePolicy] = {}
    
    def create_layer_group(self, group_name: str, layer_ids: List[int]):
        """
        Create a named group of layers.
        
        Args:
            group_name: Name of the layer group
            layer_ids: List of layer indices in the group
        """
        self.layer_groups[group_name] = layer_ids
    
    def set_group_policy(self, group_name: str, policy: RecomputePolicy):
        """
        Set policy for a layer group.
        
        Args:
            group_name: Name of the layer group
            policy: Recomputation policy
        """
        if group_name not in self.layer_groups:
            raise ValueError(f"Unknown layer group: {group_name}")
        
        self.group_policies[group_name] = policy
        
        # Apply policy to all layers in group
        for layer_id in self.layer_groups[group_name]:
            self.layer_policies[layer_id] = policy
    
    def set_layer_range_policy(
        self, 
        start_layer: int, 
        end_layer: int, 
        policy: RecomputePolicy
    ):
        """
        Set policy for a range of layers.
        
        Args:
            start_layer: Starting layer index (inclusive)
            end_layer: Ending layer index (exclusive)
            policy: Recomputation policy
        """
        for layer_id in range(start_layer, end_layer):
            self.layer_policies[layer_id] = policy
    
    def get_memory_summary(self) -> Dict[str, Any]:
        """
        Get summary of memory scheduler state.
        
        Returns:
            Dictionary with scheduler statistics
        """
        summary = {
            "memory_budget": self.memory_budget,
            "current_usage": self.current_memory_usage,
            "peak_usage": self.peak_memory_usage,
            "memory_saved": self.memory_saved,
            "utilization": self.current_memory_usage / self.memory_budget,
            "layer_policies": dict(self.layer_policies),
            "recomputed_layers": self.recomputed_layers.copy(),
        }
        
        # Add group information if available
        if self.layer_groups:
            summary["layer_groups"] = dict(self.layer_groups)
            summary["group_policies"] = dict(self.group_policies)
        
        return summary