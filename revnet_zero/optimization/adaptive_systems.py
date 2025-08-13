"""
Adaptive Learning Systems for RevNet-Zero.

Implements self-improving patterns that learn and evolve automatically:
- Adaptive caching based on access patterns
- Auto-scaling triggers based on load
- Self-healing with circuit breakers
- Performance optimization from metrics
"""

import time
import threading
import logging
from typing import Dict, List, Tuple, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import numpy as np
import json
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor
import asyncio
from enum import Enum
import statistics
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class AdaptationStrategy(Enum):
    """Strategies for adaptive behavior."""
    CONSERVATIVE = "conservative"  # Slow adaptation, high stability
    BALANCED = "balanced"         # Moderate adaptation
    AGGRESSIVE = "aggressive"     # Fast adaptation, may be unstable
    HYBRID = "hybrid"             # Context-dependent strategy


@dataclass
class PerformanceMetric:
    """Performance metric with trend analysis."""
    name: str
    value: float
    timestamp: datetime
    context: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not isinstance(self.timestamp, datetime):
            self.timestamp = datetime.now()


@dataclass
class AdaptationDecision:
    """Decision made by adaptive system."""
    component: str
    action: str
    reason: str
    confidence: float
    parameters: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    impact: Optional[float] = None
    success: Optional[bool] = None


class AdaptiveCacheManager:
    """Adaptive caching system that learns access patterns."""
    
    def __init__(self, 
                 max_cache_size: int = 1000,
                 adaptation_window: int = 100,
                 learning_rate: float = 0.1):
        self.max_cache_size = max_cache_size
        self.adaptation_window = adaptation_window
        self.learning_rate = learning_rate
        
        # Cache storage
        self.cache: Dict[str, Any] = {}
        self.cache_metadata: Dict[str, Dict] = {}
        
        # Access pattern learning
        self.access_history: deque = deque(maxlen=adaptation_window * 10)
        self.access_patterns: Dict[str, List[float]] = defaultdict(list)
        self.temporal_patterns: Dict[str, List[datetime]] = defaultdict(list)
        
        # Adaptive parameters
        self.cache_weights: Dict[str, float] = defaultdict(lambda: 1.0)
        self.eviction_scores: Dict[str, float] = defaultdict(lambda: 0.0)
        
        self.logger = logging.getLogger(f"{__name__}.AdaptiveCacheManager")
        self._lock = threading.RLock()
    
    def get(self, key: str, compute_fn: Optional[Callable] = None) -> Any:
        """Get item from cache with adaptive learning."""
        with self._lock:
            current_time = datetime.now()
            
            # Record access
            self.access_history.append((key, current_time, 'get'))
            self.temporal_patterns[key].append(current_time)
            
            if key in self.cache:
                # Update access metadata
                metadata = self.cache_metadata[key]
                metadata['access_count'] = metadata.get('access_count', 0) + 1
                metadata['last_access'] = current_time
                
                # Learn from successful hit
                self._update_cache_weights(key, hit=True)
                
                self.logger.debug(f"Cache hit for {key}")
                return self.cache[key]
            
            # Cache miss
            self.logger.debug(f"Cache miss for {key}")
            self._update_cache_weights(key, hit=False)
            
            # Compute value if function provided
            if compute_fn:
                value = compute_fn()
                self.put(key, value)
                return value
            
            return None
    
    def put(self, key: str, value: Any) -> None:
        """Put item in cache with adaptive eviction."""
        with self._lock:
            current_time = datetime.now()
            
            # Record put operation
            self.access_history.append((key, current_time, 'put'))
            
            # Check if eviction needed
            if len(self.cache) >= self.max_cache_size and key not in self.cache:
                self._adaptive_eviction()
            
            # Store in cache
            self.cache[key] = value
            self.cache_metadata[key] = {
                'created': current_time,
                'last_access': current_time,
                'access_count': 1,
                'size': self._estimate_size(value),
                'importance_score': self._calculate_importance(key)
            }
            
            # Learn from patterns periodically
            if len(self.access_history) % self.adaptation_window == 0:
                self._adapt_cache_strategy()
    
    def _update_cache_weights(self, key: str, hit: bool) -> None:
        """Update cache weights based on hit/miss patterns."""
        # Simple reinforcement learning update
        reward = 1.0 if hit else -0.5
        current_weight = self.cache_weights[key]
        
        # Update weight with learning rate
        self.cache_weights[key] = current_weight + self.learning_rate * reward
        
        # Bound weights
        self.cache_weights[key] = max(0.1, min(10.0, self.cache_weights[key]))
    
    def _adaptive_eviction(self) -> None:
        """Evict items using learned patterns."""
        if not self.cache:
            return
        
        # Calculate eviction scores for all items
        scores = {}
        current_time = datetime.now()
        
        for key, metadata in self.cache_metadata.items():
            # Base score from metadata
            time_since_access = (current_time - metadata['last_access']).total_seconds()
            access_frequency = metadata['access_count'] / max(1, 
                (current_time - metadata['created']).total_seconds() / 3600)  # per hour
            
            # Incorporate learned weights
            importance = metadata.get('importance_score', 1.0)
            weight = self.cache_weights[key]
            
            # Calculate composite score (lower = more likely to evict)
            scores[key] = (access_frequency * weight * importance) / (time_since_access + 1)
        
        # Evict lowest scoring item
        evict_key = min(scores.keys(), key=lambda k: scores[k])
        del self.cache[evict_key]
        del self.cache_metadata[evict_key]
        
        self.logger.info(f"Adaptively evicted {evict_key} (score: {scores[evict_key]:.3f})")
    
    def _adapt_cache_strategy(self) -> None:
        """Adapt caching strategy based on learned patterns."""
        # Analyze temporal patterns
        self._analyze_temporal_patterns()
        
        # Adjust cache size if needed
        self._adapt_cache_size()
        
        # Update importance scoring
        self._update_importance_scoring()
        
        self.logger.info("Adapted cache strategy based on learned patterns")
    
    def _analyze_temporal_patterns(self) -> None:
        """Analyze temporal access patterns to predict future needs."""
        for key, timestamps in self.temporal_patterns.items():
            if len(timestamps) < 3:
                continue
            
            # Calculate access intervals
            intervals = []
            for i in range(1, len(timestamps)):
                interval = (timestamps[i] - timestamps[i-1]).total_seconds()
                intervals.append(interval)
            
            # Predict next access time
            if intervals:
                mean_interval = statistics.mean(intervals)
                last_access = timestamps[-1]
                predicted_next = last_access + timedelta(seconds=mean_interval)
                
                # Adjust weight based on prediction
                time_to_next = (predicted_next - datetime.now()).total_seconds()
                if time_to_next < 3600:  # Next access within an hour
                    self.cache_weights[key] *= 1.2  # Increase importance
    
    def _adapt_cache_size(self) -> None:
        """Adapt cache size based on hit rates and memory pressure."""
        recent_accesses = list(self.access_history)[-self.adaptation_window:]
        if not recent_accesses:
            return
        
        # Calculate hit rate
        hits = sum(1 for _, _, op in recent_accesses if op == 'get' and _ in self.cache)
        total_gets = sum(1 for _, _, op in recent_accesses if op == 'get')
        
        if total_gets > 0:
            hit_rate = hits / total_gets
            
            # Adjust cache size based on hit rate
            if hit_rate < 0.5 and self.max_cache_size > 100:
                # Low hit rate, reduce cache size
                self.max_cache_size = int(self.max_cache_size * 0.9)
            elif hit_rate > 0.8 and self.max_cache_size < 10000:
                # High hit rate, increase cache size
                self.max_cache_size = int(self.max_cache_size * 1.1)
    
    def _update_importance_scoring(self) -> None:
        """Update importance scoring based on learned patterns."""
        for key in self.cache_metadata:
            metadata = self.cache_metadata[key]
            
            # Combine multiple factors
            access_recency = (datetime.now() - metadata['last_access']).total_seconds()
            access_frequency = metadata['access_count']
            learned_weight = self.cache_weights[key]
            
            # Calculate new importance score
            importance = (access_frequency * learned_weight) / (access_recency + 1)
            metadata['importance_score'] = importance
    
    def _calculate_importance(self, key: str) -> float:
        """Calculate initial importance score for new items."""
        # Use learned weights if available
        return self.cache_weights.get(key, 1.0)
    
    def _estimate_size(self, value: Any) -> int:
        """Estimate memory size of cached value."""
        if hasattr(value, '__sizeof__'):
            return value.__sizeof__()
        return len(str(value))
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics and learned patterns."""
        recent_accesses = list(self.access_history)[-100:]
        hits = sum(1 for key, _, op in recent_accesses if op == 'get' and key in self.cache)
        total_gets = sum(1 for _, _, op in recent_accesses if op == 'get')
        
        return {
            'cache_size': len(self.cache),
            'max_cache_size': self.max_cache_size,
            'hit_rate': hits / max(1, total_gets),
            'learned_weights_count': len(self.cache_weights),
            'top_weighted_keys': sorted(self.cache_weights.items(), 
                                      key=lambda x: x[1], reverse=True)[:5],
            'temporal_patterns_count': len(self.temporal_patterns)
        }


class AutoScalingManager:
    """Auto-scaling system that adapts to load patterns."""
    
    def __init__(self, 
                 min_resources: int = 1,
                 max_resources: int = 10,
                 scale_up_threshold: float = 0.8,
                 scale_down_threshold: float = 0.3,
                 observation_window: int = 60):
        self.min_resources = min_resources
        self.max_resources = max_resources
        self.scale_up_threshold = scale_up_threshold
        self.scale_down_threshold = scale_down_threshold
        self.observation_window = observation_window
        
        # Current state
        self.current_resources = min_resources
        self.resource_utilization: deque = deque(maxlen=observation_window)
        self.scaling_history: List[AdaptationDecision] = []
        
        # Adaptive parameters
        self.load_predictor = LoadPredictor()
        self.scaling_strategy = AdaptationStrategy.BALANCED
        
        # Metrics
        self.metrics_history: deque = deque(maxlen=1000)
        
        self.logger = logging.getLogger(f"{__name__}.AutoScalingManager")
        self._lock = threading.Lock()
        
        # Start monitoring thread
        self._monitoring = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()
    
    def report_load(self, utilization: float, context: Dict[str, Any] = None) -> None:
        """Report current resource utilization."""
        with self._lock:
            metric = PerformanceMetric(
                name="resource_utilization",
                value=utilization,
                timestamp=datetime.now(),
                context=context or {}
            )
            
            self.resource_utilization.append(utilization)
            self.metrics_history.append(metric)
            
            # Learn from load patterns
            self.load_predictor.update(utilization, context)
    
    def _monitor_loop(self) -> None:
        """Main monitoring loop for auto-scaling decisions."""
        while self._monitoring:
            try:
                self._evaluate_scaling_decision()
                time.sleep(10)  # Check every 10 seconds
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(5)
    
    def _evaluate_scaling_decision(self) -> None:
        """Evaluate whether scaling action is needed."""
        with self._lock:
            if len(self.resource_utilization) < 10:
                return  # Need enough data points
            
            recent_utilization = list(self.resource_utilization)[-10:]
            avg_utilization = statistics.mean(recent_utilization)
            trend = self._calculate_trend(recent_utilization)
            
            # Predict future load
            predicted_load = self.load_predictor.predict(steps_ahead=3)
            
            # Decide on scaling action
            decision = self._make_scaling_decision(
                avg_utilization, trend, predicted_load
            )
            
            if decision:
                self._execute_scaling_decision(decision)
    
    def _make_scaling_decision(self, 
                             avg_utilization: float,
                             trend: float,
                             predicted_load: float) -> Optional[AdaptationDecision]:
        """Make intelligent scaling decision."""
        
        # Adjust thresholds based on strategy
        if self.scaling_strategy == AdaptationStrategy.AGGRESSIVE:
            scale_up_thresh = self.scale_up_threshold * 0.8
            scale_down_thresh = self.scale_down_threshold * 1.2
        elif self.scaling_strategy == AdaptationStrategy.CONSERVATIVE:
            scale_up_thresh = self.scale_up_threshold * 1.2
            scale_down_thresh = self.scale_down_threshold * 0.8
        else:  # BALANCED
            scale_up_thresh = self.scale_up_threshold
            scale_down_thresh = self.scale_down_threshold
        
        # Consider current utilization, trend, and prediction
        combined_signal = (avg_utilization * 0.5 + 
                          trend * 0.3 + 
                          predicted_load * 0.2)
        
        # Scale up decision
        if (combined_signal > scale_up_thresh and 
            self.current_resources < self.max_resources):
            
            confidence = min(1.0, (combined_signal - scale_up_thresh) / 0.2)
            
            return AdaptationDecision(
                component="auto_scaler",
                action="scale_up",
                reason=f"High utilization: {avg_utilization:.2f}, trend: {trend:.2f}, predicted: {predicted_load:.2f}",
                confidence=confidence,
                parameters={
                    "from_resources": self.current_resources,
                    "to_resources": min(self.max_resources, self.current_resources + 1),
                    "utilization": avg_utilization,
                    "trend": trend,
                    "predicted_load": predicted_load
                }
            )
        
        # Scale down decision
        elif (combined_signal < scale_down_thresh and 
              self.current_resources > self.min_resources):
            
            confidence = min(1.0, (scale_down_thresh - combined_signal) / 0.2)
            
            return AdaptationDecision(
                component="auto_scaler",
                action="scale_down",
                reason=f"Low utilization: {avg_utilization:.2f}, trend: {trend:.2f}, predicted: {predicted_load:.2f}",
                confidence=confidence,
                parameters={
                    "from_resources": self.current_resources,
                    "to_resources": max(self.min_resources, self.current_resources - 1),
                    "utilization": avg_utilization,
                    "trend": trend,
                    "predicted_load": predicted_load
                }
            )
        
        return None
    
    def _execute_scaling_decision(self, decision: AdaptationDecision) -> None:
        """Execute scaling decision and measure impact."""
        old_resources = self.current_resources
        new_resources = decision.parameters["to_resources"]
        
        self.logger.info(f"Executing scaling decision: {decision.action} from {old_resources} to {new_resources}")
        
        # Record pre-change metrics
        pre_metrics = self._get_current_metrics()
        
        # Execute scaling (placeholder - would call actual scaling functions)
        self.current_resources = new_resources
        decision.impact = self._calculate_scaling_impact(old_resources, new_resources)
        decision.success = True  # Assume success for now
        
        # Record decision
        self.scaling_history.append(decision)
        
        # Learn from outcome
        self._learn_from_scaling_outcome(decision, pre_metrics)
    
    def _calculate_trend(self, values: List[float]) -> float:
        """Calculate trend in utilization values."""
        if len(values) < 2:
            return 0.0
        
        # Simple linear regression slope
        n = len(values)
        x = list(range(n))
        y = values
        
        x_mean = statistics.mean(x)
        y_mean = statistics.mean(y)
        
        numerator = sum((x[i] - x_mean) * (y[i] - y_mean) for i in range(n))
        denominator = sum((x[i] - x_mean) ** 2 for i in range(n))
        
        return numerator / max(denominator, 1e-10)
    
    def _calculate_scaling_impact(self, old_resources: int, new_resources: int) -> float:
        """Calculate impact of scaling decision."""
        # Simplified impact calculation
        resource_change = (new_resources - old_resources) / old_resources
        return resource_change
    
    def _learn_from_scaling_outcome(self, 
                                   decision: AdaptationDecision,
                                   pre_metrics: Dict[str, float]) -> None:
        """Learn from scaling decision outcomes."""
        # Wait a bit for metrics to stabilize
        time.sleep(30)
        
        post_metrics = self._get_current_metrics()
        
        # Calculate improvement
        improvement = 0.0
        if 'avg_utilization' in pre_metrics and 'avg_utilization' in post_metrics:
            if decision.action == "scale_up":
                # For scale up, improvement is reduction in utilization
                improvement = pre_metrics['avg_utilization'] - post_metrics['avg_utilization']
            else:
                # For scale down, improvement is maintaining similar utilization with fewer resources
                utilization_change = abs(post_metrics['avg_utilization'] - pre_metrics['avg_utilization'])
                improvement = 1.0 - utilization_change  # Less change is better
        
        # Adapt strategy based on outcome
        if improvement > 0.1:  # Good outcome
            if decision.confidence < 0.8:
                # Successful low-confidence decision -> be more aggressive
                if self.scaling_strategy == AdaptationStrategy.CONSERVATIVE:
                    self.scaling_strategy = AdaptationStrategy.BALANCED
                elif self.scaling_strategy == AdaptationStrategy.BALANCED:
                    self.scaling_strategy = AdaptationStrategy.AGGRESSIVE
        elif improvement < -0.1:  # Poor outcome
            # Unsuccessful decision -> be more conservative
            if self.scaling_strategy == AdaptationStrategy.AGGRESSIVE:
                self.scaling_strategy = AdaptationStrategy.BALANCED
            elif self.scaling_strategy == AdaptationStrategy.BALANCED:
                self.scaling_strategy = AdaptationStrategy.CONSERVATIVE
        
        self.logger.info(f"Learned from scaling outcome: improvement={improvement:.3f}, new_strategy={self.scaling_strategy}")
    
    def _get_current_metrics(self) -> Dict[str, float]:
        """Get current performance metrics."""
        if not self.resource_utilization:
            return {}
        
        recent = list(self.resource_utilization)[-10:]
        return {
            'avg_utilization': statistics.mean(recent),
            'max_utilization': max(recent),
            'min_utilization': min(recent),
            'current_resources': self.current_resources
        }
    
    def get_scaling_statistics(self) -> Dict[str, Any]:
        """Get scaling statistics and learned patterns."""
        return {
            'current_resources': self.current_resources,
            'scaling_strategy': self.scaling_strategy.value,
            'scaling_decisions_count': len(self.scaling_history),
            'recent_utilization': list(self.resource_utilization)[-10:],
            'successful_scalings': sum(1 for d in self.scaling_history if d.success),
            'load_prediction_accuracy': self.load_predictor.get_accuracy()
        }


class LoadPredictor:
    """Simple load predictor using moving averages and trends."""
    
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.load_history: deque = deque(maxlen=window_size)
        self.context_history: deque = deque(maxlen=window_size)
        
        # Simple prediction state
        self.moving_average = 0.0
        self.trend = 0.0
        
        # Accuracy tracking
        self.predictions: deque = deque(maxlen=100)
        self.actual_values: deque = deque(maxlen=100)
    
    def update(self, load: float, context: Dict[str, Any] = None) -> None:
        """Update predictor with new load observation."""
        self.load_history.append(load)
        self.context_history.append(context or {})
        
        # Update moving average
        if self.load_history:
            self.moving_average = statistics.mean(self.load_history)
        
        # Update trend
        if len(self.load_history) >= 2:
            recent = list(self.load_history)[-10:]
            if len(recent) >= 2:
                self.trend = (recent[-1] - recent[0]) / len(recent)
    
    def predict(self, steps_ahead: int = 1) -> float:
        """Predict load N steps ahead."""
        if not self.load_history:
            return 0.5  # Default prediction
        
        # Simple linear prediction
        prediction = self.moving_average + (self.trend * steps_ahead)
        
        # Bound prediction
        prediction = max(0.0, min(1.0, prediction))
        
        # Store for accuracy calculation
        self.predictions.append(prediction)
        
        return prediction
    
    def get_accuracy(self) -> float:
        """Calculate prediction accuracy."""
        if len(self.predictions) < 10 or len(self.actual_values) < 10:
            return 0.0
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        errors = []
        min_len = min(len(self.predictions), len(self.actual_values))
        
        for i in range(min_len):
            if self.actual_values[i] != 0:
                error = abs(self.predictions[i] - self.actual_values[i]) / self.actual_values[i]
                errors.append(error)
        
        if errors:
            mape = statistics.mean(errors)
            return max(0.0, 1.0 - mape)  # Convert to accuracy
        
        return 0.0


class SelfHealingCircuitBreaker:
    """Circuit breaker that learns failure patterns and adapts."""
    
    def __init__(self, 
                 failure_threshold: int = 5,
                 timeout_seconds: int = 60,
                 success_threshold: int = 3):
        self.failure_threshold = failure_threshold
        self.timeout_seconds = timeout_seconds
        self.success_threshold = success_threshold
        
        # State
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time = None
        
        # Learning components
        self.failure_patterns: Dict[str, int] = defaultdict(int)
        self.recovery_times: List[float] = []
        self.adaptive_timeout = timeout_seconds
        
        self.logger = logging.getLogger(f"{__name__}.SelfHealingCircuitBreaker")
        self._lock = threading.Lock()
    
    def call(self, func: Callable, *args, **kwargs) -> Any:
        """Execute function with circuit breaker protection."""
        with self._lock:
            if self.state == "OPEN":
                if self._should_attempt_reset():
                    self.state = "HALF_OPEN"
                    self.logger.info("Circuit breaker moving to HALF_OPEN state")
                else:
                    raise Exception("Circuit breaker is OPEN")
            
            try:
                result = func(*args, **kwargs)
                self._on_success()
                return result
            except Exception as e:
                self._on_failure(str(e))
                raise
    
    def _on_success(self) -> None:
        """Handle successful call."""
        if self.state == "HALF_OPEN":
            self.success_count += 1
            if self.success_count >= self.success_threshold:
                self.state = "CLOSED"
                self.failure_count = 0
                self.success_count = 0
                
                # Learn from recovery
                if self.last_failure_time:
                    recovery_time = time.time() - self.last_failure_time
                    self.recovery_times.append(recovery_time)
                    self._adapt_timeout()
                
                self.logger.info("Circuit breaker reset to CLOSED state")
        elif self.state == "CLOSED":
            self.failure_count = max(0, self.failure_count - 1)  # Gradually reduce failure count
    
    def _on_failure(self, error_message: str) -> None:
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        # Learn from failure patterns
        error_type = self._categorize_error(error_message)
        self.failure_patterns[error_type] += 1
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            self.logger.warning(f"Circuit breaker tripped to OPEN state after {self.failure_count} failures")
            
            # Adapt threshold based on failure patterns
            self._adapt_failure_threshold()
    
    def _should_attempt_reset(self) -> bool:
        """Determine if circuit breaker should attempt reset."""
        if not self.last_failure_time:
            return True
        
        time_since_failure = time.time() - self.last_failure_time
        return time_since_failure >= self.adaptive_timeout
    
    def _categorize_error(self, error_message: str) -> str:
        """Categorize error for pattern learning."""
        error_lower = error_message.lower()
        
        if "timeout" in error_lower or "connection" in error_lower:
            return "network_error"
        elif "memory" in error_lower or "out of" in error_lower:
            return "resource_error"
        elif "permission" in error_lower or "access" in error_lower:
            return "auth_error"
        else:
            return "unknown_error"
    
    def _adapt_timeout(self) -> None:
        """Adapt timeout based on learned recovery patterns."""
        if len(self.recovery_times) >= 5:
            # Use 90th percentile of recovery times
            sorted_times = sorted(self.recovery_times)
            percentile_90 = sorted_times[int(0.9 * len(sorted_times))]
            
            # Adapt timeout (with bounds)
            self.adaptive_timeout = max(10, min(300, percentile_90 * 1.5))
            
            self.logger.info(f"Adapted circuit breaker timeout to {self.adaptive_timeout:.1f}s")
    
    def _adapt_failure_threshold(self) -> None:
        """Adapt failure threshold based on failure patterns."""
        total_failures = sum(self.failure_patterns.values())
        
        if total_failures >= 20:  # Enough data to adapt
            # If mostly transient errors, be more tolerant
            transient_ratio = (self.failure_patterns["network_error"] / total_failures)
            
            if transient_ratio > 0.7:
                self.failure_threshold = min(10, self.failure_threshold + 1)
            elif transient_ratio < 0.3:
                self.failure_threshold = max(3, self.failure_threshold - 1)
            
            self.logger.info(f"Adapted failure threshold to {self.failure_threshold}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get circuit breaker statistics."""
        return {
            'state': self.state,
            'failure_count': self.failure_count,
            'success_count': self.success_count,
            'adaptive_timeout': self.adaptive_timeout,
            'failure_threshold': self.failure_threshold,
            'failure_patterns': dict(self.failure_patterns),
            'recovery_times_count': len(self.recovery_times),
            'avg_recovery_time': statistics.mean(self.recovery_times) if self.recovery_times else 0
        }


class AdaptiveSystemManager:
    """Central manager for all adaptive systems."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        
        # Initialize adaptive components
        self.cache_manager = AdaptiveCacheManager(
            max_cache_size=self.config.get('cache_size', 1000),
            learning_rate=self.config.get('cache_learning_rate', 0.1)
        )
        
        self.scaling_manager = AutoScalingManager(
            min_resources=self.config.get('min_resources', 1),
            max_resources=self.config.get('max_resources', 10)
        )
        
        self.circuit_breaker = SelfHealingCircuitBreaker(
            failure_threshold=self.config.get('failure_threshold', 5),
            timeout_seconds=self.config.get('circuit_timeout', 60)
        )
        
        # Global metrics and learning
        self.global_metrics: deque = deque(maxlen=10000)
        self.adaptation_history: List[AdaptationDecision] = []
        
        self.logger = logging.getLogger(f"{__name__}.AdaptiveSystemManager")
        
        # Start global coordination thread
        self._coordinating = True
        self._coordination_thread = threading.Thread(target=self._coordination_loop, daemon=True)
        self._coordination_thread.start()
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        default_config = {
            'cache_size': 1000,
            'cache_learning_rate': 0.1,
            'min_resources': 1,
            'max_resources': 10,
            'failure_threshold': 5,
            'circuit_timeout': 60
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    file_config = json.load(f)
                default_config.update(file_config)
            except Exception as e:
                self.logger.warning(f"Failed to load config from {config_path}: {e}")
        
        return default_config
    
    def _coordination_loop(self) -> None:
        """Global coordination loop for adaptive systems."""
        while self._coordinating:
            try:
                self._coordinate_adaptations()
                time.sleep(60)  # Coordinate every minute
            except Exception as e:
                self.logger.error(f"Error in coordination loop: {e}")
                time.sleep(30)
    
    def _coordinate_adaptations(self) -> None:
        """Coordinate adaptations across all systems."""
        # Collect metrics from all systems
        cache_stats = self.cache_manager.get_statistics()
        scaling_stats = self.scaling_manager.get_scaling_statistics()
        circuit_stats = self.circuit_breaker.get_statistics()
        
        # Global optimization decisions
        if cache_stats['hit_rate'] < 0.3 and scaling_stats['current_resources'] < scaling_stats.get('max_resources', 10):
            # Low cache hit rate might benefit from more resources
            self.scaling_manager.report_load(0.9, {'reason': 'low_cache_performance'})
        
        if circuit_stats['state'] == 'OPEN' and scaling_stats['current_resources'] > 1:
            # Circuit breaker open might indicate overload
            self.scaling_manager.report_load(0.95, {'reason': 'circuit_breaker_open'})
        
        # Log coordinated insights
        self.logger.info(f"Global coordination: cache_hit={cache_stats['hit_rate']:.2f}, "
                        f"resources={scaling_stats['current_resources']}, "
                        f"circuit_state={circuit_stats['state']}")
    
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health and adaptation status."""
        return {
            'cache': self.cache_manager.get_statistics(),
            'scaling': self.scaling_manager.get_scaling_statistics(),
            'circuit_breaker': self.circuit_breaker.get_statistics(),
            'adaptations_total': len(self.adaptation_history),
            'timestamp': datetime.now().isoformat()
        }
    
    def shutdown(self) -> None:
        """Shutdown adaptive systems gracefully."""
        self._coordinating = False
        self.scaling_manager._monitoring = False
        
        # Wait for threads to finish
        if hasattr(self, '_coordination_thread'):
            self._coordination_thread.join(timeout=5)
        
        self.logger.info("Adaptive systems shutdown complete")


# Export key classes
__all__ = [
    "AdaptiveSystemManager",
    "AdaptiveCacheManager",
    "AutoScalingManager", 
    "SelfHealingCircuitBreaker",
    "AdaptationStrategy",
    "PerformanceMetric",
    "AdaptationDecision"
]
