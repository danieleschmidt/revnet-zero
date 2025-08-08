"""
Automatic performance optimization system for RevNet-Zero.

This module implements intelligent performance optimization that automatically:
- Analyzes model architecture and workload patterns
- Optimizes memory scheduling strategies
- Tunes batch sizes and sequence lengths
- Manages GPU/CPU resource allocation
- Adapts to changing workload patterns
- Provides optimization recommendations
"""

import time
import threading
import logging
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
import statistics


class OptimizationTarget(Enum):
    """Optimization objectives."""
    THROUGHPUT = "throughput"          # Maximize tokens/second
    LATENCY = "latency"                # Minimize response time
    MEMORY = "memory"                  # Minimize memory usage
    ENERGY = "energy"                  # Minimize energy consumption
    BALANCED = "balanced"              # Balance all objectives


class OptimizationStrategy(Enum):
    """Optimization strategies."""
    CONSERVATIVE = "conservative"      # Safe, proven optimizations
    AGGRESSIVE = "aggressive"         # Maximum performance optimizations
    ADAPTIVE = "adaptive"             # Learns and adapts over time
    CUSTOM = "custom"                 # User-defined strategy


@dataclass
class PerformanceMetrics:
    """Performance measurement data."""
    timestamp: float
    throughput_tokens_per_sec: float
    latency_p50_ms: float
    latency_p95_ms: float
    memory_usage_gb: float
    gpu_utilization: float
    cpu_utilization: float
    energy_consumption_watts: float
    batch_size: int
    sequence_length: int
    
    def __post_init__(self):
        if self.timestamp == 0:
            self.timestamp = time.time()


@dataclass
class OptimizationConfig:
    """Configuration for automatic optimization."""
    target: OptimizationTarget = OptimizationTarget.BALANCED
    strategy: OptimizationStrategy = OptimizationStrategy.ADAPTIVE
    measurement_window_minutes: int = 5
    optimization_interval_minutes: int = 15
    min_samples_for_optimization: int = 10
    performance_threshold: float = 0.05  # 5% improvement threshold
    safety_margin: float = 0.9  # Use 90% of resources as limit
    enable_auto_scaling: bool = True
    enable_memory_optimization: bool = True
    enable_batch_optimization: bool = True
    max_optimization_iterations: int = 5


@dataclass
class OptimizationRecommendation:
    """Optimization recommendation."""
    parameter: str
    current_value: Any
    recommended_value: Any
    expected_improvement: float
    confidence: float
    reasoning: str
    risk_level: str  # low, medium, high


class WorkloadAnalyzer:
    """Analyzes workload patterns and characteristics."""
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        self.request_history: List[Dict[str, Any]] = []
        self.performance_history: List[PerformanceMetrics] = []
        self._lock = threading.Lock()
    
    def record_request(self, request_data: Dict[str, Any]):
        """Record request characteristics."""
        with self._lock:
            request_info = {
                'timestamp': time.time(),
                'sequence_length': request_data.get('sequence_length', 0),
                'batch_size': request_data.get('batch_size', 1),
                'model_size': request_data.get('model_size', 'unknown'),
                'processing_time': request_data.get('processing_time', 0),
                'memory_used': request_data.get('memory_used', 0)
            }
            
            self.request_history.append(request_info)
            
            # Keep history bounded
            if len(self.request_history) > self.history_size:
                self.request_history = self.request_history[-self.history_size//2:]
    
    def record_performance(self, metrics: PerformanceMetrics):
        """Record performance metrics."""
        with self._lock:
            self.performance_history.append(metrics)
            
            if len(self.performance_history) > self.history_size:
                self.performance_history = self.performance_history[-self.history_size//2:]
    
    def get_workload_characteristics(self) -> Dict[str, Any]:
        """Analyze workload characteristics."""
        with self._lock:
            if not self.request_history:
                return {}
            
            recent_requests = self.request_history[-100:]  # Last 100 requests
            
            sequence_lengths = [r['sequence_length'] for r in recent_requests if r['sequence_length'] > 0]
            batch_sizes = [r['batch_size'] for r in recent_requests if r['batch_size'] > 0]
            processing_times = [r['processing_time'] for r in recent_requests if r['processing_time'] > 0]
            
            characteristics = {
                'request_count': len(recent_requests),
                'avg_requests_per_minute': len(recent_requests) / max(1, (time.time() - recent_requests[0]['timestamp']) / 60),
                'sequence_length_stats': self._calculate_stats(sequence_lengths),
                'batch_size_stats': self._calculate_stats(batch_sizes),
                'processing_time_stats': self._calculate_stats(processing_times),
                'workload_type': self._classify_workload(sequence_lengths, batch_sizes),
                'peak_hours': self._identify_peak_hours(),
                'resource_utilization_pattern': self._analyze_resource_pattern()
            }
            
            return characteristics
    
    def _calculate_stats(self, values: List[float]) -> Dict[str, float]:
        """Calculate statistical summary."""
        if not values:
            return {}
        
        return {
            'min': min(values),
            'max': max(values),
            'mean': statistics.mean(values),
            'median': statistics.median(values),
            'std': statistics.stdev(values) if len(values) > 1 else 0,
            'p95': statistics.quantiles(values, n=20)[18] if len(values) >= 20 else max(values)
        }
    
    def _classify_workload(self, seq_lengths: List[int], batch_sizes: List[int]) -> str:
        """Classify workload type based on patterns."""
        if not seq_lengths or not batch_sizes:
            return "unknown"
        
        avg_seq_len = statistics.mean(seq_lengths)
        avg_batch_size = statistics.mean(batch_sizes)
        
        if avg_seq_len > 32000:
            return "long_context"
        elif avg_seq_len > 8000:
            return "medium_context"
        elif avg_batch_size > 16:
            return "high_throughput_batch"
        elif avg_batch_size < 4:
            return "low_latency_single"
        else:
            return "balanced_general"
    
    def _identify_peak_hours(self) -> List[int]:
        """Identify peak usage hours."""
        hour_counts = [0] * 24
        
        for request in self.request_history:
            hour = int((request['timestamp'] % 86400) // 3600)
            hour_counts[hour] += 1
        
        avg_count = sum(hour_counts) / 24
        peak_hours = [i for i, count in enumerate(hour_counts) if count > avg_count * 1.5]
        
        return peak_hours
    
    def _analyze_resource_pattern(self) -> Dict[str, str]:
        """Analyze resource utilization patterns."""
        if not self.performance_history:
            return {}
        
        recent_metrics = self.performance_history[-50:]
        
        memory_usage = [m.memory_usage_gb for m in recent_metrics]
        gpu_util = [m.gpu_utilization for m in recent_metrics]
        
        patterns = {}
        
        if memory_usage:
            avg_memory = statistics.mean(memory_usage)
            if avg_memory > 0.8:
                patterns['memory'] = 'memory_constrained'
            elif avg_memory < 0.3:
                patterns['memory'] = 'memory_underutilized'
            else:
                patterns['memory'] = 'memory_balanced'
        
        if gpu_util:
            avg_gpu = statistics.mean(gpu_util)
            if avg_gpu > 0.8:
                patterns['gpu'] = 'gpu_saturated'
            elif avg_gpu < 0.3:
                patterns['gpu'] = 'gpu_underutilized'
            else:
                patterns['gpu'] = 'gpu_balanced'
        
        return patterns


class OptimizationEngine:
    """Core optimization engine that makes intelligent decisions."""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.analyzer = WorkloadAnalyzer()
        self.optimization_history: List[Dict[str, Any]] = []
        self.current_parameters: Dict[str, Any] = {}
        self._lock = threading.Lock()
        
        # Optimization models (simplified)
        self.parameter_bounds = {
            'batch_size': (1, 32),
            'sequence_length': (512, 65536),
            'memory_budget_gb': (1, 64),
            'recompute_granularity': ['layer', 'block', 'attention'],
            'scheduling_strategy': ['conservative', 'adaptive', 'aggressive']
        }
    
    def analyze_and_recommend(self) -> List[OptimizationRecommendation]:
        """Analyze current state and generate recommendations."""
        with self._lock:
            workload_chars = self.analyzer.get_workload_characteristics()
            
            if not workload_chars or workload_chars.get('request_count', 0) < self.config.min_samples_for_optimization:
                return []
            
            recommendations = []
            
            # Memory optimization
            if self.config.enable_memory_optimization:
                memory_recs = self._optimize_memory_usage(workload_chars)
                recommendations.extend(memory_recs)
            
            # Batch optimization
            if self.config.enable_batch_optimization:
                batch_recs = self._optimize_batch_parameters(workload_chars)
                recommendations.extend(batch_recs)
            
            # Scheduling optimization
            scheduling_recs = self._optimize_scheduling_strategy(workload_chars)
            recommendations.extend(scheduling_recs)
            
            # Resource allocation optimization
            resource_recs = self._optimize_resource_allocation(workload_chars)
            recommendations.extend(resource_recs)
            
            # Filter recommendations by confidence and expected improvement
            filtered_recs = [
                rec for rec in recommendations
                if rec.confidence > 0.7 and rec.expected_improvement > self.config.performance_threshold
            ]
            
            # Sort by expected improvement
            filtered_recs.sort(key=lambda x: x.expected_improvement, reverse=True)
            
            return filtered_recs[:self.config.max_optimization_iterations]
    
    def _optimize_memory_usage(self, workload_chars: Dict[str, Any]) -> List[OptimizationRecommendation]:
        """Generate memory optimization recommendations."""
        recommendations = []
        
        resource_pattern = workload_chars.get('resource_utilization_pattern', {})
        memory_pattern = resource_pattern.get('memory', 'unknown')
        
        if memory_pattern == 'memory_constrained':
            # Recommend more aggressive recomputation
            recommendations.append(OptimizationRecommendation(
                parameter='scheduling_strategy',
                current_value=self.current_parameters.get('scheduling_strategy', 'adaptive'),
                recommended_value='aggressive',
                expected_improvement=0.15,
                confidence=0.8,
                reasoning='High memory usage detected, aggressive scheduling will reduce memory footprint',
                risk_level='low'
            ))
            
            # Recommend smaller batch sizes
            current_batch = workload_chars.get('batch_size_stats', {}).get('mean', 8)
            if current_batch > 4:
                recommendations.append(OptimizationRecommendation(
                    parameter='max_batch_size',
                    current_value=current_batch,
                    recommended_value=max(2, int(current_batch * 0.7)),
                    expected_improvement=0.12,
                    confidence=0.75,
                    reasoning='Reduce batch size to lower memory pressure',
                    risk_level='low'
                ))
        
        elif memory_pattern == 'memory_underutilized':
            # Recommend larger batch sizes for better throughput
            current_batch = workload_chars.get('batch_size_stats', {}).get('mean', 8)
            if current_batch < 16:
                recommendations.append(OptimizationRecommendation(
                    parameter='max_batch_size',
                    current_value=current_batch,
                    recommended_value=min(32, int(current_batch * 1.5)),
                    expected_improvement=0.20,
                    confidence=0.85,
                    reasoning='Memory underutilized, increase batch size for better throughput',
                    risk_level='low'
                ))
        
        return recommendations
    
    def _optimize_batch_parameters(self, workload_chars: Dict[str, Any]) -> List[OptimizationRecommendation]:
        """Generate batch optimization recommendations."""
        recommendations = []
        
        workload_type = workload_chars.get('workload_type', 'unknown')
        
        if workload_type == 'high_throughput_batch':
            # Already optimized for batching, may recommend batch timeout adjustments
            recommendations.append(OptimizationRecommendation(
                parameter='batch_timeout_ms',
                current_value=self.current_parameters.get('batch_timeout_ms', 50),
                recommended_value=100,  # Longer timeout for better batching
                expected_improvement=0.08,
                confidence=0.7,
                reasoning='High throughput workload benefits from longer batch collection time',
                risk_level='medium'
            ))
        
        elif workload_type == 'low_latency_single':
            # Optimize for latency
            recommendations.append(OptimizationRecommendation(
                parameter='batch_timeout_ms',
                current_value=self.current_parameters.get('batch_timeout_ms', 50),
                recommended_value=10,  # Shorter timeout for lower latency
                expected_improvement=0.15,
                confidence=0.8,
                reasoning='Low latency workload benefits from immediate processing',
                risk_level='low'
            ))
        
        elif workload_type == 'long_context':
            # Long context sequences need different optimization
            recommendations.append(OptimizationRecommendation(
                parameter='sequence_batching_enabled',
                current_value=self.current_parameters.get('sequence_batching_enabled', True),
                recommended_value=False,  # Process individually for memory efficiency
                expected_improvement=0.25,
                confidence=0.85,
                reasoning='Long context sequences should be processed individually to manage memory',
                risk_level='low'
            ))
        
        return recommendations
    
    def _optimize_scheduling_strategy(self, workload_chars: Dict[str, Any]) -> List[OptimizationRecommendation]:
        """Generate scheduling optimization recommendations."""
        recommendations = []
        
        workload_type = workload_chars.get('workload_type', 'unknown')
        resource_pattern = workload_chars.get('resource_utilization_pattern', {})
        
        current_strategy = self.current_parameters.get('scheduling_strategy', 'adaptive')
        
        # Recommend strategy based on workload characteristics
        if workload_type == 'long_context' and current_strategy != 'aggressive':
            recommendations.append(OptimizationRecommendation(
                parameter='scheduling_strategy',
                current_value=current_strategy,
                recommended_value='aggressive',
                expected_improvement=0.30,
                confidence=0.9,
                reasoning='Long context workloads benefit significantly from aggressive memory scheduling',
                risk_level='low'
            ))
        
        elif (workload_type in ['high_throughput_batch', 'balanced_general'] 
              and resource_pattern.get('memory') == 'memory_balanced' 
              and current_strategy == 'aggressive'):
            recommendations.append(OptimizationRecommendation(
                parameter='scheduling_strategy',
                current_value=current_strategy,
                recommended_value='adaptive',
                expected_improvement=0.10,
                confidence=0.75,
                reasoning='Balanced workload with sufficient memory can use adaptive scheduling for better throughput',
                risk_level='low'
            ))
        
        return recommendations
    
    def _optimize_resource_allocation(self, workload_chars: Dict[str, Any]) -> List[OptimizationRecommendation]:
        """Generate resource allocation recommendations."""
        recommendations = []
        
        resource_pattern = workload_chars.get('resource_utilization_pattern', {})
        peak_hours = workload_chars.get('peak_hours', [])
        
        # Auto-scaling recommendations
        if self.config.enable_auto_scaling and len(peak_hours) > 0:
            recommendations.append(OptimizationRecommendation(
                parameter='auto_scale_schedule',
                current_value=self.current_parameters.get('auto_scale_schedule', {}),
                recommended_value={'peak_hours': peak_hours, 'scale_factor': 1.5},
                expected_improvement=0.20,
                confidence=0.8,
                reasoning=f'Identified peak hours {peak_hours}, proactive scaling will improve performance',
                risk_level='medium'
            ))
        
        # GPU utilization optimization
        gpu_pattern = resource_pattern.get('gpu', 'unknown')
        if gpu_pattern == 'gpu_underutilized':
            recommendations.append(OptimizationRecommendation(
                parameter='mixed_precision_enabled',
                current_value=self.current_parameters.get('mixed_precision_enabled', False),
                recommended_value=True,
                expected_improvement=0.25,
                confidence=0.85,
                reasoning='GPU underutilized, mixed precision can improve throughput',
                risk_level='low'
            ))
        
        return recommendations
    
    def apply_recommendation(self, recommendation: OptimizationRecommendation) -> bool:
        """Apply an optimization recommendation."""
        try:
            # Record the optimization attempt
            optimization_record = {
                'timestamp': time.time(),
                'parameter': recommendation.parameter,
                'old_value': recommendation.current_value,
                'new_value': recommendation.recommended_value,
                'expected_improvement': recommendation.expected_improvement,
                'status': 'applied'
            }
            
            # Update current parameters
            self.current_parameters[recommendation.parameter] = recommendation.recommended_value
            
            # Record in history
            with self._lock:
                self.optimization_history.append(optimization_record)
            
            logging.info(f"Applied optimization: {recommendation.parameter} = {recommendation.recommended_value}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to apply optimization {recommendation.parameter}: {e}")
            return False
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Generate comprehensive optimization report."""
        workload_chars = self.analyzer.get_workload_characteristics()
        recommendations = self.analyze_and_recommend()
        
        report = {
            'timestamp': time.time(),
            'workload_analysis': workload_chars,
            'current_parameters': self.current_parameters.copy(),
            'recommendations': [
                {
                    'parameter': rec.parameter,
                    'current': rec.current_value,
                    'recommended': rec.recommended_value,
                    'improvement': f"{rec.expected_improvement:.1%}",
                    'confidence': f"{rec.confidence:.1%}",
                    'reasoning': rec.reasoning,
                    'risk': rec.risk_level
                }
                for rec in recommendations
            ],
            'optimization_history': self.optimization_history[-10:],  # Last 10 optimizations
            'performance_summary': self._get_performance_summary()
        }
        
        return report
    
    def _get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary from recent metrics."""
        recent_metrics = self.analyzer.performance_history[-20:]
        
        if not recent_metrics:
            return {}
        
        return {
            'avg_throughput': statistics.mean(m.throughput_tokens_per_sec for m in recent_metrics),
            'avg_latency_p95': statistics.mean(m.latency_p95_ms for m in recent_metrics),
            'avg_memory_usage': statistics.mean(m.memory_usage_gb for m in recent_metrics),
            'avg_gpu_utilization': statistics.mean(m.gpu_utilization for m in recent_metrics),
            'performance_trend': self._calculate_performance_trend(recent_metrics)
        }
    
    def _calculate_performance_trend(self, metrics: List[PerformanceMetrics]) -> str:
        """Calculate overall performance trend."""
        if len(metrics) < 5:
            return 'insufficient_data'
        
        # Simple trend calculation based on throughput
        first_half = metrics[:len(metrics)//2]
        second_half = metrics[len(metrics)//2:]
        
        first_avg = statistics.mean(m.throughput_tokens_per_sec for m in first_half)
        second_avg = statistics.mean(m.throughput_tokens_per_sec for m in second_half)
        
        change = (second_avg - first_avg) / first_avg
        
        if change > 0.05:
            return 'improving'
        elif change < -0.05:
            return 'degrading'
        else:
            return 'stable'


class AutoOptimizer:
    """Main auto-optimizer that coordinates optimization activities."""
    
    def __init__(self, config: OptimizationConfig = None):
        self.config = config or OptimizationConfig()
        self.engine = OptimizationEngine(self.config)
        self._optimization_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._running = False
        
        self.logger = logging.getLogger(__name__)
    
    def start(self):
        """Start the auto-optimization process."""
        if self._running:
            return
        
        self._running = True
        self._stop_event.clear()
        
        self._optimization_thread = threading.Thread(
            target=self._optimization_loop,
            name="RevNetZero-AutoOptimizer",
            daemon=True
        )
        self._optimization_thread.start()
        
        self.logger.info("Auto-optimizer started")
    
    def stop(self):
        """Stop the auto-optimization process."""
        if not self._running:
            return
        
        self._running = False
        self._stop_event.set()
        
        if self._optimization_thread:
            self._optimization_thread.join(timeout=10)
        
        self.logger.info("Auto-optimizer stopped")
    
    def _optimization_loop(self):
        """Main optimization loop."""
        while not self._stop_event.wait(self.config.optimization_interval_minutes * 60):
            try:
                self._run_optimization_cycle()
            except Exception as e:
                self.logger.error(f"Optimization cycle failed: {e}")
    
    def _run_optimization_cycle(self):
        """Run a single optimization cycle."""
        self.logger.info("Starting optimization cycle")
        
        # Get recommendations
        recommendations = self.engine.analyze_and_recommend()
        
        if not recommendations:
            self.logger.info("No optimization recommendations at this time")
            return
        
        self.logger.info(f"Generated {len(recommendations)} optimization recommendations")
        
        # Apply top recommendations based on strategy
        applied_count = 0
        
        for recommendation in recommendations:
            if self.config.strategy == OptimizationStrategy.CONSERVATIVE and recommendation.risk_level == 'high':
                continue
            
            if self.engine.apply_recommendation(recommendation):
                applied_count += 1
                
                # Wait between optimizations to observe effects
                if applied_count < len(recommendations):
                    time.sleep(30)
        
        self.logger.info(f"Applied {applied_count} optimizations")
        
        # Generate and log optimization report
        if applied_count > 0:
            report = self.engine.get_optimization_report()
            self.logger.info(f"Optimization report: {json.dumps(report, indent=2, default=str)}")
    
    def record_request(self, request_data: Dict[str, Any]):
        """Record request for analysis."""
        self.engine.analyzer.record_request(request_data)
    
    def record_performance(self, metrics: PerformanceMetrics):
        """Record performance metrics."""
        self.engine.analyzer.record_performance(metrics)
    
    def get_current_recommendations(self) -> List[OptimizationRecommendation]:
        """Get current optimization recommendations."""
        return self.engine.analyze_and_recommend()
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """Get comprehensive optimization report."""
        return self.engine.get_optimization_report()
    
    def manual_optimization(self, parameter: str, value: Any) -> bool:
        """Manually apply an optimization."""
        recommendation = OptimizationRecommendation(
            parameter=parameter,
            current_value=self.engine.current_parameters.get(parameter),
            recommended_value=value,
            expected_improvement=0.0,
            confidence=1.0,
            reasoning="Manual optimization",
            risk_level="unknown"
        )
        
        return self.engine.apply_recommendation(recommendation)


# Global auto-optimizer instance
_global_optimizer: Optional[AutoOptimizer] = None


def get_auto_optimizer() -> AutoOptimizer:
    """Get global auto-optimizer instance."""
    global _global_optimizer
    if _global_optimizer is None:
        _global_optimizer = AutoOptimizer()
    return _global_optimizer


def optimize_model_automatically(
    model_config: Dict[str, Any],
    optimization_target: OptimizationTarget = OptimizationTarget.BALANCED
) -> Dict[str, Any]:
    """
    Automatically optimize model configuration.
    
    Args:
        model_config: Current model configuration
        optimization_target: Optimization objective
        
    Returns:
        Optimized configuration
    """
    config = OptimizationConfig(target=optimization_target)
    optimizer = AutoOptimizer(config)
    
    # Simulate some workload data for optimization
    for i in range(20):
        optimizer.record_request({
            'sequence_length': model_config.get('max_seq_len', 2048),
            'batch_size': 4,
            'processing_time': 0.1 + i * 0.01,
            'memory_used': 2.0 + i * 0.1
        })
    
    # Get recommendations
    recommendations = optimizer.get_current_recommendations()
    
    # Apply recommendations to config
    optimized_config = model_config.copy()
    for rec in recommendations:
        if rec.parameter in optimized_config:
            optimized_config[rec.parameter] = rec.recommended_value
    
    return optimized_config


__all__ = [
    'AutoOptimizer', 'OptimizationConfig', 'OptimizationTarget', 'OptimizationStrategy',
    'PerformanceMetrics', 'OptimizationRecommendation', 'WorkloadAnalyzer',
    'OptimizationEngine', 'get_auto_optimizer', 'optimize_model_automatically'
]