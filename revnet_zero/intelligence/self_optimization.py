"""
Self-Optimizing System for autonomous model evolution and improvement
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod
import logging
import json
import time
from collections import deque, defaultdict
from pathlib import Path

@dataclass
class OptimizationRule:
    """Rule for self-optimization based on conditions"""
    name: str
    condition: Callable[[Dict], bool]
    action: Callable[[Any], None]
    priority: int
    cooldown: int = 300  # 5 minutes between applications
    last_applied: float = 0.0
    success_rate: float = 0.0
    application_count: int = 0
    
class SelfOptimizingSystem:
    """System that automatically optimizes model performance based on observed patterns"""
    
    def __init__(self, model, config: Optional[Dict] = None):
        self.model = model
        self.config = config or {}
        
        # Optimization rules
        self.optimization_rules: List[OptimizationRule] = []
        self._initialize_default_rules()
        
        # Performance tracking
        self.performance_history = deque(maxlen=1000)
        self.optimization_history: List[Dict] = []
        
        # Learning components
        self.pattern_detector = PerformancePatternDetector()
        self.strategy_learner = OptimizationStrategyLearner()
        
        # State tracking
        self.current_metrics: Dict[str, float] = {}
        self.baseline_metrics: Dict[str, float] = {}
        
        self.logger = logging.getLogger(__name__)
        
    def _initialize_default_rules(self) -> None:
        """Initialize default optimization rules"""
        
        # Memory optimization rules
        self.add_optimization_rule(
            name="reduce_memory_on_pressure",
            condition=lambda metrics: metrics.get('memory_usage_percent', 0) > 85,
            action=self._optimize_memory_usage,
            priority=1
        )
        
        # Throughput optimization rules
        self.add_optimization_rule(
            name="improve_throughput_on_degradation",
            condition=lambda metrics: (
                metrics.get('throughput_drop_percent', 0) > 20 and
                metrics.get('gradient_norm', 0) < 5.0  # Ensure stability
            ),
            action=self._optimize_throughput,
            priority=2
        )
        
        # Stability optimization rules
        self.add_optimization_rule(
            name="stabilize_on_gradient_explosion",
            condition=lambda metrics: metrics.get('gradient_norm', 0) > 10.0,
            action=self._optimize_stability,
            priority=0  # Highest priority
        )
        
        # Adaptive learning rules
        self.add_optimization_rule(
            name="adapt_learning_on_plateau",
            condition=lambda metrics: (
                metrics.get('loss_plateau_steps', 0) > 100 and
                metrics.get('loss_variance', 1.0) < 0.01
            ),
            action=self._adapt_learning_strategy,
            priority=3
        )
        
        # Resource optimization rules
        self.add_optimization_rule(
            name="balance_resources",
            condition=lambda metrics: (
                metrics.get('cpu_utilization', 0) > 80 and
                metrics.get('gpu_utilization', 100) < 70
            ),
            action=self._balance_resource_usage,
            priority=4
        )
        
    def add_optimization_rule(self,
                            name: str,
                            condition: Callable[[Dict], bool],
                            action: Callable[[Any], None],
                            priority: int,
                            cooldown: int = 300) -> None:
        """Add a new optimization rule"""
        
        rule = OptimizationRule(
            name=name,
            condition=condition,
            action=action,
            priority=priority,
            cooldown=cooldown
        )
        
        self.optimization_rules.append(rule)
        # Sort by priority (lower number = higher priority)
        self.optimization_rules.sort(key=lambda r: r.priority)
        
        self.logger.info(f"Added optimization rule: {name}")
        
    def update_metrics(self, metrics: Dict[str, float]) -> None:
        """Update current metrics and trigger optimization if needed"""
        
        # Store metrics
        self.current_metrics = metrics.copy()
        self.current_metrics['timestamp'] = time.time()
        self.performance_history.append(self.current_metrics.copy())
        
        # Update baseline if not set
        if not self.baseline_metrics:
            self.baseline_metrics = metrics.copy()
            
        # Add derived metrics
        self._compute_derived_metrics()
        
        # Detect patterns
        patterns = self.pattern_detector.detect_patterns(self.performance_history)
        
        # Update strategy learning
        self.strategy_learner.update_performance(metrics)
        
        # Check and apply optimization rules
        self._evaluate_and_apply_rules()
        
    def _compute_derived_metrics(self) -> None:
        """Compute derived metrics from raw performance data"""
        
        if len(self.performance_history) < 10:
            return
            
        recent_history = list(self.performance_history)[-10:]
        
        # Throughput trend
        throughputs = [h.get('throughput', 0) for h in recent_history]
        if throughputs and self.baseline_metrics.get('throughput'):
            avg_throughput = np.mean(throughputs)
            baseline_throughput = self.baseline_metrics['throughput']
            throughput_drop = ((baseline_throughput - avg_throughput) / baseline_throughput) * 100
            self.current_metrics['throughput_drop_percent'] = max(0, throughput_drop)
            
        # Loss plateau detection
        losses = [h.get('loss', 0) for h in recent_history if h.get('loss')]
        if len(losses) >= 5:
            loss_variance = np.var(losses)
            self.current_metrics['loss_variance'] = loss_variance
            
            # Count steps since significant loss improvement
            if len(self.performance_history) >= 100:
                last_100 = list(self.performance_history)[-100:]
                loss_improvements = []
                for i in range(1, len(last_100)):
                    if last_100[i].get('loss', 0) < last_100[i-1].get('loss', 0) * 0.99:  # 1% improvement
                        loss_improvements.append(i)
                        
                steps_since_improvement = len(last_100) - (max(loss_improvements) if loss_improvements else 0)
                self.current_metrics['loss_plateau_steps'] = steps_since_improvement
                
        # Memory usage percentage
        if 'memory_used' in self.current_metrics and 'memory_total' in self.current_metrics:
            memory_percent = (self.current_metrics['memory_used'] / self.current_metrics['memory_total']) * 100
            self.current_metrics['memory_usage_percent'] = memory_percent
            
    def _evaluate_and_apply_rules(self) -> None:
        """Evaluate optimization rules and apply applicable ones"""
        
        current_time = time.time()
        applied_rules = []
        
        for rule in self.optimization_rules:
            # Check cooldown
            if current_time - rule.last_applied < rule.cooldown:
                continue
                
            # Check condition
            if rule.condition(self.current_metrics):
                try:
                    # Record metrics before optimization
                    metrics_before = self.current_metrics.copy()
                    
                    # Apply optimization
                    rule.action(self.model)
                    rule.last_applied = current_time
                    rule.application_count += 1
                    
                    # Record optimization
                    optimization_record = {
                        'rule_name': rule.name,
                        'timestamp': current_time,
                        'metrics_before': metrics_before,
                        'trigger_condition': rule.condition.__name__ if hasattr(rule.condition, '__name__') else str(rule.condition)
                    }
                    
                    self.optimization_history.append(optimization_record)
                    applied_rules.append(rule.name)
                    
                    self.logger.info(f"Applied optimization rule: {rule.name}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to apply optimization rule {rule.name}: {e}")
                    
        if applied_rules:
            self.logger.info(f"Applied {len(applied_rules)} optimization rules: {applied_rules}")
            
    def _optimize_memory_usage(self, model) -> None:
        """Optimize memory usage"""
        
        optimizations = []
        
        # Increase gradient accumulation
        if hasattr(model, 'gradient_accumulation_steps'):
            old_steps = model.gradient_accumulation_steps
            model.gradient_accumulation_steps = min(old_steps * 2, 64)
            optimizations.append(f'gradient_accumulation: {old_steps} -> {model.gradient_accumulation_steps}')
            
        # Enable more aggressive memory scheduling
        if hasattr(model, 'memory_scheduler'):
            if hasattr(model.memory_scheduler, 'set_strategy'):
                model.memory_scheduler.set_strategy('aggressive')
                optimizations.append('memory_scheduler: aggressive')
                
            # Increase recomputation granularity
            if hasattr(model.memory_scheduler, 'recompute_granularity'):
                if model.memory_scheduler.recompute_granularity == 'attention':
                    model.memory_scheduler.recompute_granularity = 'layer'
                    optimizations.append('recompute_granularity: layer')
                elif model.memory_scheduler.recompute_granularity == 'layer':
                    model.memory_scheduler.recompute_granularity = 'block'
                    optimizations.append('recompute_granularity: block')
                    
        # Enable activation compression if available
        if hasattr(model, 'enable_activation_compression'):
            model.enable_activation_compression(True)
            optimizations.append('activation_compression: enabled')
            
        self.logger.info(f"Memory optimizations: {optimizations}")
        
    def _optimize_throughput(self, model) -> None:
        """Optimize throughput performance"""
        
        optimizations = []
        
        # Enable flash attention
        if hasattr(model, 'layers'):
            for layer in model.layers:
                if hasattr(layer, 'attention') and hasattr(layer.attention, 'use_flash_attention'):
                    if not layer.attention.use_flash_attention:
                        layer.attention.use_flash_attention = True
                        optimizations.append('flash_attention: enabled')
                        
        # Optimize kernel selection
        if hasattr(model, 'kernel_manager'):
            if hasattr(model.kernel_manager, 'select_optimal_kernels'):
                model.kernel_manager.select_optimal_kernels()
                optimizations.append('kernel_optimization: optimal')
                
        # Enable tensor fusion
        if hasattr(model, 'enable_tensor_fusion'):
            model.enable_tensor_fusion(True)
            optimizations.append('tensor_fusion: enabled')
            
        # Optimize cache settings
        if hasattr(model, 'cache_manager'):
            if hasattr(model.cache_manager, 'optimize_for_throughput'):
                model.cache_manager.optimize_for_throughput()
                optimizations.append('cache_optimization: throughput')
                
        self.logger.info(f"Throughput optimizations: {optimizations}")
        
    def _optimize_stability(self, model) -> None:
        """Optimize training stability"""
        
        optimizations = []
        
        # Enable gradient clipping
        if hasattr(model, 'gradient_clip_value'):
            if model.gradient_clip_value is None or model.gradient_clip_value > 1.0:
                model.gradient_clip_value = 1.0
                optimizations.append('gradient_clipping: 1.0')
        elif hasattr(model, 'enable_gradient_clipping'):
            model.enable_gradient_clipping(1.0)
            optimizations.append('gradient_clipping: enabled')
            
        # Reduce learning rate temporarily
        if hasattr(model, 'adjust_learning_rate'):
            model.adjust_learning_rate(0.5)  # Reduce by half
            optimizations.append('learning_rate: reduced')
            
        # Enable mixed precision stability features
        if hasattr(model, 'enable_loss_scaling'):
            model.enable_loss_scaling(True)
            optimizations.append('loss_scaling: enabled')
            
        self.logger.info(f"Stability optimizations: {optimizations}")
        
    def _adapt_learning_strategy(self, model) -> None:
        """Adapt learning strategy for better convergence"""
        
        optimizations = []
        
        # Learning rate scheduling
        if hasattr(model, 'learning_rate_scheduler'):
            if hasattr(model.learning_rate_scheduler, 'step'):
                model.learning_rate_scheduler.step()
                optimizations.append('lr_scheduler: stepped')
                
        # Optimizer switching or parameter adjustment
        if hasattr(model, 'optimizer_manager'):
            if hasattr(model.optimizer_manager, 'adapt_optimizer'):
                model.optimizer_manager.adapt_optimizer()
                optimizations.append('optimizer: adapted')
                
        # Enable adaptive attention patterns
        if hasattr(model, 'enable_adaptive_attention'):
            model.enable_adaptive_attention(True)
            optimizations.append('adaptive_attention: enabled')
            
        self.logger.info(f"Learning adaptations: {optimizations}")
        
    def _balance_resource_usage(self, model) -> None:
        """Balance CPU and GPU resource usage"""
        
        optimizations = []
        
        # Adjust data loading parallelism
        if hasattr(model, 'data_loader_workers'):
            # Reduce CPU-intensive data loading
            if model.data_loader_workers > 2:
                model.data_loader_workers = max(2, model.data_loader_workers // 2)
                optimizations.append(f'data_workers: reduced to {model.data_loader_workers}')
                
        # Move more computation to GPU
        if hasattr(model, 'move_preprocessing_to_gpu'):
            model.move_preprocessing_to_gpu(True)
            optimizations.append('preprocessing: moved to GPU')
            
        self.logger.info(f"Resource balancing: {optimizations}")
        
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization activities"""
        
        rule_stats = {}
        for rule in self.optimization_rules:
            rule_stats[rule.name] = {
                'applications': rule.application_count,
                'success_rate': rule.success_rate,
                'last_applied': rule.last_applied,
                'priority': rule.priority
            }
            
        return {
            'total_optimizations': len(self.optimization_history),
            'active_rules': len(self.optimization_rules),
            'rule_statistics': rule_stats,
            'recent_optimizations': self.optimization_history[-10:] if self.optimization_history else [],
            'performance_trends': self._compute_performance_trends()
        }
        
    def _compute_performance_trends(self) -> Dict[str, float]:
        """Compute performance trends over time"""
        
        if len(self.performance_history) < 20:
            return {}
            
        recent_history = list(self.performance_history)[-20:]
        trends = {}
        
        for metric in ['throughput', 'memory_usage_percent', 'loss']:
            values = [h.get(metric) for h in recent_history if h.get(metric) is not None]
            if len(values) >= 10:
                # Simple linear trend
                x = np.arange(len(values))
                slope = np.polyfit(x, values, 1)[0]
                trends[f'{metric}_trend'] = slope
                
        return trends
        
    def save_optimization_state(self, filepath: str) -> None:
        """Save optimization state to file"""
        
        state = {
            'optimization_history': self.optimization_history,
            'rule_statistics': {
                rule.name: {
                    'application_count': rule.application_count,
                    'success_rate': rule.success_rate,
                    'last_applied': rule.last_applied
                }
                for rule in self.optimization_rules
            },
            'baseline_metrics': self.baseline_metrics,
            'performance_summary': self.get_optimization_summary()
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
            
        self.logger.info(f"Saved optimization state to {filepath}")

class PerformancePatternDetector:
    """Detects patterns in performance metrics for proactive optimization"""
    
    def __init__(self):
        self.detected_patterns: List[Dict] = []
        
    def detect_patterns(self, history: deque) -> List[Dict]:
        """Detect patterns in performance history"""
        
        if len(history) < 20:
            return []
            
        patterns = []
        
        # Detect memory pressure patterns
        memory_pattern = self._detect_memory_pressure_pattern(history)
        if memory_pattern:
            patterns.append(memory_pattern)
            
        # Detect throughput degradation patterns
        throughput_pattern = self._detect_throughput_pattern(history)
        if throughput_pattern:
            patterns.append(throughput_pattern)
            
        return patterns
        
    def _detect_memory_pressure_pattern(self, history: deque) -> Optional[Dict]:
        """Detect memory pressure building up over time"""
        
        recent_memory = [h.get('memory_usage_percent', 0) for h in list(history)[-10:]]
        
        if len(recent_memory) >= 5:
            # Check for increasing trend
            slope = np.polyfit(range(len(recent_memory)), recent_memory, 1)[0]
            
            if slope > 2.0 and np.mean(recent_memory) > 70:  # Increasing by 2% per step, above 70%
                return {
                    'type': 'memory_pressure_buildup',
                    'severity': 'medium',
                    'trend_slope': slope,
                    'current_usage': recent_memory[-1]
                }
                
        return None
        
    def _detect_throughput_pattern(self, history: deque) -> Optional[Dict]:
        """Detect throughput degradation patterns"""
        
        recent_throughput = [h.get('throughput', 0) for h in list(history)[-10:] if h.get('throughput', 0) > 0]
        
        if len(recent_throughput) >= 5:
            # Check for decreasing trend
            slope = np.polyfit(range(len(recent_throughput)), recent_throughput, 1)[0]
            
            if slope < -5.0:  # Decreasing by 5 tokens/sec per step
                return {
                    'type': 'throughput_degradation',
                    'severity': 'medium',
                    'trend_slope': slope,
                    'current_throughput': recent_throughput[-1]
                }
                
        return None

class OptimizationStrategyLearner:
    """Learns which optimization strategies work best in different situations"""
    
    def __init__(self):
        self.strategy_effectiveness: Dict[str, List[float]] = defaultdict(list)
        self.context_patterns: List[Dict] = []
        
    def update_performance(self, metrics: Dict[str, float]) -> None:
        """Update performance learning from current metrics"""
        
        # This would implement more sophisticated learning
        # For now, we just track basic effectiveness
        pass
        
    def get_best_strategy(self, context: Dict[str, float]) -> Optional[str]:
        """Get the best optimization strategy for current context"""
        
        # Simplified strategy selection
        if context.get('memory_usage_percent', 0) > 85:
            return 'memory_optimization'
        elif context.get('throughput_drop_percent', 0) > 20:
            return 'throughput_optimization'
        elif context.get('gradient_norm', 0) > 10:
            return 'stability_optimization'
            
        return None