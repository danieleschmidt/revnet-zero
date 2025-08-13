"""
Adaptive Learning Engine for autonomous model evolution
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging
from collections import defaultdict
import numpy as np

@dataclass
class LearningMetrics:
    """Metrics for tracking learning progress"""
    loss: float
    memory_usage: float
    throughput: float
    gradient_norm: float
    activation_stats: Dict[str, float]
    timestamp: float

class AdaptiveLearningEngine:
    """
    Intelligent engine that adapts model architecture and training based on performance
    """
    
    def __init__(self, model, config: Optional[Dict] = None):
        self.model = model
        self.config = config or {}
        
        # Learning history
        self.metrics_history: List[LearningMetrics] = []
        self.adaptation_history: List[Dict] = []
        
        # Adaptation parameters
        self.adaptation_threshold = self.config.get('adaptation_threshold', 0.01)
        self.memory_budget = self.config.get('memory_budget', 40 * 1024**3)
        
        # Performance tracking
        self.performance_tracker = PerformanceTracker()
        
        # Logger
        self.logger = logging.getLogger(__name__)
        
    def record_metrics(self, 
                      loss: float,
                      memory_usage: float, 
                      throughput: float,
                      model_state: Dict) -> None:
        """Record performance metrics for adaptation analysis"""
        
        # Calculate gradient norm
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        gradient_norm = total_norm ** 0.5
        
        # Activation statistics
        activation_stats = self._compute_activation_stats(model_state)
        
        metrics = LearningMetrics(
            loss=loss,
            memory_usage=memory_usage,
            throughput=throughput,
            gradient_norm=gradient_norm,
            activation_stats=activation_stats,
            timestamp=torch.cuda.Event(enable_timing=True).elapsed_time(torch.cuda.Event(enable_timing=True))
        )
        
        self.metrics_history.append(metrics)
        
        # Trigger adaptation if needed
        if len(self.metrics_history) > 10:
            self._evaluate_adaptation_need()
    
    def _compute_activation_stats(self, model_state: Dict) -> Dict[str, float]:
        """Compute statistics on model activations"""
        stats = {}
        for name, tensor in model_state.items():
            if isinstance(tensor, torch.Tensor) and tensor.numel() > 0:
                stats[f'{name}_mean'] = tensor.mean().item()
                stats[f'{name}_std'] = tensor.std().item()
                stats[f'{name}_max'] = tensor.max().item()
                stats[f'{name}_min'] = tensor.min().item()
        return stats
    
    def _evaluate_adaptation_need(self) -> None:
        """Evaluate if model needs adaptation based on recent metrics"""
        recent_metrics = self.metrics_history[-10:]
        
        # Check for performance degradation
        loss_trend = self._compute_trend([m.loss for m in recent_metrics])
        memory_trend = self._compute_trend([m.memory_usage for m in recent_metrics])
        throughput_trend = self._compute_trend([m.throughput for m in recent_metrics])
        
        adaptations_needed = []
        
        # Loss is increasing - need optimization
        if loss_trend > self.adaptation_threshold:
            adaptations_needed.append('loss_optimization')
            
        # Memory usage increasing - need memory optimization
        if memory_trend > 0.1 or recent_metrics[-1].memory_usage > self.memory_budget * 0.9:
            adaptations_needed.append('memory_optimization')
            
        # Throughput decreasing - need performance optimization
        if throughput_trend < -self.adaptation_threshold:
            adaptations_needed.append('performance_optimization')
            
        # Gradient explosion detection
        if recent_metrics[-1].gradient_norm > 10.0:
            adaptations_needed.append('gradient_stabilization')
            
        if adaptations_needed:
            self._apply_adaptations(adaptations_needed)
    
    def _compute_trend(self, values: List[float]) -> float:
        """Compute trend (slope) of recent values"""
        if len(values) < 2:
            return 0.0
            
        x = np.arange(len(values))
        y = np.array(values)
        
        # Linear regression
        slope = np.polyfit(x, y, 1)[0]
        return slope
    
    def _apply_adaptations(self, adaptations: List[str]) -> None:
        """Apply necessary adaptations to the model"""
        adaptation_record = {
            'timestamp': torch.cuda.Event(enable_timing=True).elapsed_time(torch.cuda.Event(enable_timing=True)),
            'adaptations': adaptations,
            'metrics_before': self.metrics_history[-1]
        }
        
        for adaptation in adaptations:
            if adaptation == 'memory_optimization':
                self._adapt_memory_strategy()
            elif adaptation == 'performance_optimization':
                self._adapt_performance_strategy()
            elif adaptation == 'loss_optimization':
                self._adapt_learning_strategy()
            elif adaptation == 'gradient_stabilization':
                self._adapt_gradient_strategy()
        
        self.adaptation_history.append(adaptation_record)
        self.logger.info(f"Applied adaptations: {adaptations}")
    
    def _adapt_memory_strategy(self) -> None:
        """Adapt memory usage strategy"""
        # Increase recomputation granularity
        if hasattr(self.model, 'memory_scheduler'):
            current_policy = self.model.memory_scheduler.recompute_granularity
            if current_policy == 'attention':
                self.model.memory_scheduler.recompute_granularity = 'layer'
            elif current_policy == 'layer':
                self.model.memory_scheduler.recompute_granularity = 'block'
                
        self.logger.info("Adapted memory strategy for better efficiency")
    
    def _adapt_performance_strategy(self) -> None:
        """Adapt performance optimization strategy"""
        # Enable more aggressive caching
        if hasattr(self.model, 'cache_manager'):
            self.model.cache_manager.set_strategy('aggressive')
            
        # Optimize attention patterns
        if hasattr(self.model, 'attention_layers'):
            for layer in self.model.attention_layers:
                if hasattr(layer, 'use_flash_attention'):
                    layer.use_flash_attention = True
                    
        self.logger.info("Adapted performance strategy for better throughput")
    
    def _adapt_learning_strategy(self) -> None:
        """Adapt learning rate and optimization strategy"""
        # This would typically interact with the optimizer
        # For now, we log the need for learning rate adjustment
        self.logger.info("Learning strategy adaptation needed - consider learning rate adjustment")
    
    def _adapt_gradient_strategy(self) -> None:
        """Adapt gradient handling for stability"""
        # Enable gradient clipping
        if hasattr(self.model, 'gradient_clip_val'):
            self.model.gradient_clip_val = min(1.0, self.model.gradient_clip_val * 0.8)
            
        self.logger.info("Adapted gradient strategy for stability")
    
    def get_optimization_recommendations(self) -> List[str]:
        """Get current optimization recommendations based on metrics"""
        if not self.metrics_history:
            return []
            
        recommendations = []
        recent_metrics = self.metrics_history[-5:]
        
        # Memory recommendations
        avg_memory = np.mean([m.memory_usage for m in recent_metrics])
        if avg_memory > self.memory_budget * 0.8:
            recommendations.append("Consider increasing recomputation granularity")
            
        # Performance recommendations  
        avg_throughput = np.mean([m.throughput for m in recent_metrics])
        if avg_throughput < 100:  # tokens/sec threshold
            recommendations.append("Enable flash attention and kernel optimizations")
            
        # Stability recommendations
        avg_grad_norm = np.mean([m.gradient_norm for m in recent_metrics])
        if avg_grad_norm > 5.0:
            recommendations.append("Apply gradient clipping for training stability")
            
        return recommendations

class PerformanceTracker:
    """Tracks detailed performance metrics"""
    
    def __init__(self):
        self.metrics = defaultdict(list)
        
    def record(self, name: str, value: float) -> None:
        """Record a metric value"""
        self.metrics[name].append(value)
        
    def get_stats(self, name: str) -> Dict[str, float]:
        """Get statistics for a metric"""
        if name not in self.metrics:
            return {}
            
        values = self.metrics[name]
        return {
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'count': len(values)
        }