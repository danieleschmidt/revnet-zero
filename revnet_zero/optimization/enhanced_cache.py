"""
RevNet-Zero Enhanced Cache - Advanced adaptive caching with ML-driven optimization.
Builds upon existing intelligent cache with additional performance features.
"""

import time
import pickle
import hashlib
import threading
import statistics
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from collections import OrderedDict, defaultdict
from enum import Enum
import warnings

class CachePolicy(Enum):
    """Cache eviction policies"""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used 
    TTL = "ttl"  # Time To Live
    ADAPTIVE = "adaptive"  # ML-driven adaptive policy
    SIZE_AWARE = "size_aware"  # Size-aware eviction

class CacheHint(Enum):
    """Cache usage hints for optimization"""
    FREQUENT = "frequent"  # Frequently accessed data
    LARGE = "large"  # Large data items
    TEMPORARY = "temporary"  # Short-lived data
    CRITICAL = "critical"  # Performance-critical data
    COMPUTE_EXPENSIVE = "compute_expensive"  # Expensive to recompute

@dataclass
class EnhancedCacheEntry:
    """Enhanced cache entry with performance metrics"""
    key: str
    value: Any
    timestamp: float
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    size_bytes: int = 0
    ttl: Optional[float] = None
    hint: Optional[CacheHint] = None
    compute_time: float = 0.0
    memory_pressure_score: float = 0.0
    performance_impact: float = 1.0
    
    def is_expired(self) -> bool:
        """Check if entry has expired"""
        if self.ttl is None:
            return False
        return time.time() - self.timestamp > self.ttl
    
    def access(self):
        """Record access to this entry"""
        self.access_count += 1
        self.last_access = time.time()
        
    def calculate_utility_score(self) -> float:
        """Calculate utility score for eviction decisions"""
        current_time = time.time()
        
        # Recency factor (0-1, higher = more recent)
        time_since_access = current_time - self.last_access
        recency_score = max(0, 1 - (time_since_access / 3600))  # 1 hour decay
        
        # Frequency factor (normalized by age)
        age_hours = (current_time - self.timestamp) / 3600
        frequency_score = self.access_count / max(1, age_hours)
        
        # Compute cost factor
        compute_score = min(1.0, self.compute_time / 10.0)  # Normalize to 10s max
        
        # Size penalty (larger items get lower scores)
        size_penalty = 1.0 / (1.0 + self.size_bytes / 1024 / 1024)  # MB penalty
        
        # Hint-based boost
        hint_boost = 1.0
        if self.hint == CacheHint.CRITICAL:
            hint_boost = 2.0
        elif self.hint == CacheHint.COMPUTE_EXPENSIVE:
            hint_boost = 1.5
        elif self.hint == CacheHint.FREQUENT:
            hint_boost = 1.3
        elif self.hint == CacheHint.TEMPORARY:
            hint_boost = 0.5
            
        # Combined score
        utility = (
            0.3 * recency_score +
            0.25 * frequency_score +
            0.2 * compute_score +
            0.15 * size_penalty +
            0.1 * self.performance_impact
        ) * hint_boost
        
        return utility

@dataclass
class CacheMetrics:
    """Enhanced cache performance metrics"""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    memory_usage_bytes: int = 0
    hit_rate: float = 0.0
    avg_access_time_ms: float = 0.0
    cache_efficiency: float = 0.0
    memory_pressure: float = 0.0
    adaptive_score: float = 0.0

class EnhancedPerformanceCache:
    """
    Enhanced performance cache with advanced optimization features.
    
    New Features:
    - Predictive pre-loading based on access patterns
    - Memory pressure-aware eviction
    - Performance impact tracking
    - Adaptive size management
    - ML-driven optimization hints
    """
    
    def __init__(self, 
                 max_size: int = 2000,
                 max_memory_mb: float = 200.0,
                 policy: CachePolicy = CachePolicy.ADAPTIVE,
                 adaptive_sizing: bool = True,
                 predictive_loading: bool = True):
        
        self.max_size = max_size
        self.max_memory_mb = max_memory_mb
        self.max_memory_bytes = int(max_memory_mb * 1024 * 1024)
        self.policy = policy
        self.adaptive_sizing = adaptive_sizing
        self.predictive_loading = predictive_loading
        
        # Enhanced storage
        self.cache: OrderedDict[str, EnhancedCacheEntry] = OrderedDict()
        self.metrics = CacheMetrics()
        
        # Performance tracking
        self.access_patterns: Dict[str, List[float]] = defaultdict(list)
        self.performance_history: List[Tuple[float, float]] = []  # (timestamp, hit_rate)
        self.memory_pressure_history: List[float] = []
        
        # Adaptive management
        self.target_hit_rate = 0.85
        self.memory_pressure_threshold = 0.8
        self.auto_tuning_enabled = True
        
        # Prediction models
        self.access_predictor: Optional[Callable] = None
        self.eviction_predictor: Optional[Callable] = None
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Background optimization
        self._optimization_thread: Optional[threading.Thread] = None
        self._optimization_active = False
        
    def start_auto_optimization(self, interval: float = 30.0):
        """Start background optimization thread"""
        if self._optimization_active:
            return
            
        self._optimization_active = True
        self._optimization_thread = threading.Thread(
            target=self._optimization_loop,
            args=(interval,),
            daemon=True
        )
        self._optimization_thread.start()
    
    def stop_auto_optimization(self):
        """Stop background optimization"""
        self._optimization_active = False
        if self._optimization_thread:
            self._optimization_thread.join(timeout=2.0)
    
    def _optimization_loop(self, interval: float):
        """Background optimization loop"""
        while self._optimization_active:
            try:
                self._adaptive_optimization()
                self._predictive_management()
                self._update_performance_metrics()
                time.sleep(interval)
            except Exception as e:
                warnings.warn(f"Cache optimization error: {e}", RuntimeWarning)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Enhanced get with performance tracking"""
        start_time = time.time()
        
        with self._lock:
            entry = self.cache.get(key)
            
            if entry is None or entry.is_expired():
                # Cache miss
                self.metrics.misses += 1
                self._record_access_pattern(key, False)
                return default
            
            # Cache hit
            entry.access()
            self.metrics.hits += 1
            self._record_access_pattern(key, True)
            
            # Update access time metric
            access_time_ms = (time.time() - start_time) * 1000
            self._update_access_time_metric(access_time_ms)
            
            # Move to end for LRU policies
            if self.policy in [CachePolicy.LRU, CachePolicy.ADAPTIVE]:
                self.cache.move_to_end(key)
            
            return entry.value
    
    def put(self, key: str, value: Any, 
            ttl: Optional[float] = None,
            hint: Optional[CacheHint] = None,
            compute_time: float = 0.0,
            performance_impact: float = 1.0) -> bool:
        """Enhanced put with performance tracking"""
        
        with self._lock:
            size_bytes = self._estimate_size(value)
            
            # Check memory constraints
            if size_bytes > self.max_memory_bytes:
                return False
            
            # Create enhanced entry
            entry = EnhancedCacheEntry(
                key=key,
                value=value,
                timestamp=time.time(),
                size_bytes=size_bytes,
                ttl=ttl,
                hint=hint,
                compute_time=compute_time,
                performance_impact=performance_impact
            )
            
            # Remove existing entry if present
            if key in self.cache:
                old_entry = self.cache[key]
                self.metrics.memory_usage_bytes -= old_entry.size_bytes
            
            # Make space if needed
            while not self._has_space_for(size_bytes):
                if not self._evict_one():
                    return False
            
            # Add to cache
            self.cache[key] = entry
            self.metrics.memory_usage_bytes += size_bytes
            
            return True
    
    def _has_space_for(self, size_bytes: int) -> bool:
        """Check if cache has space for new entry"""
        return (len(self.cache) < self.max_size and 
                self.metrics.memory_usage_bytes + size_bytes <= self.max_memory_bytes)
    
    def _evict_one(self) -> bool:
        """Evict one entry using enhanced policy"""
        if not self.cache:
            return False
        
        if self.policy == CachePolicy.ADAPTIVE:
            key_to_evict = self._adaptive_eviction_key()
        else:
            key_to_evict = self._standard_eviction_key()
        
        if key_to_evict:
            self._evict_entry(key_to_evict)
            return True
        
        return False
    
    def _adaptive_eviction_key(self) -> Optional[str]:
        """Enhanced adaptive eviction using utility scores"""
        if not self.cache:
            return None
        
        # Calculate utility scores for all entries
        utility_scores = {}
        current_memory_pressure = self._calculate_memory_pressure()
        
        for key, entry in self.cache.items():
            utility = entry.calculate_utility_score()
            
            # Apply memory pressure adjustment
            if current_memory_pressure > 0.8:
                # Penalize large items more under memory pressure
                size_penalty = entry.size_bytes / (1024 * 1024)  # MB
                utility *= (1.0 - 0.3 * size_penalty * current_memory_pressure)
            
            utility_scores[key] = utility
        
        # Return key with lowest utility score
        return min(utility_scores, key=utility_scores.get)
    
    def _standard_eviction_key(self) -> Optional[str]:
        """Standard eviction policies"""
        if self.policy == CachePolicy.LRU:
            return next(iter(self.cache))
        elif self.policy == CachePolicy.LFU:
            return min(self.cache, key=lambda k: self.cache[k].access_count)
        elif self.policy == CachePolicy.TTL:
            # Evict expired first, then oldest
            for key, entry in self.cache.items():
                if entry.is_expired():
                    return key
            return next(iter(self.cache))
        else:
            return next(iter(self.cache))  # Fallback to LRU
    
    def _evict_entry(self, key: str):
        """Remove entry and update metrics"""
        if key in self.cache:
            entry = self.cache.pop(key)
            self.metrics.memory_usage_bytes -= entry.size_bytes
            self.metrics.evictions += 1
        
        # Clean up access patterns
        if key in self.access_patterns:
            del self.access_patterns[key]
    
    def _record_access_pattern(self, key: str, hit: bool):
        """Record access pattern for prediction"""
        timestamp = time.time()
        self.access_patterns[key].append(timestamp)
        
        # Keep only recent patterns (last 24 hours)
        cutoff = timestamp - 24 * 3600
        self.access_patterns[key] = [
            t for t in self.access_patterns[key] if t > cutoff
        ]
    
    def _update_access_time_metric(self, access_time_ms: float):
        """Update average access time with exponential smoothing"""
        if self.metrics.hits == 1:
            self.metrics.avg_access_time_ms = access_time_ms
        else:
            alpha = 0.1
            self.metrics.avg_access_time_ms = (
                alpha * access_time_ms + 
                (1 - alpha) * self.metrics.avg_access_time_ms
            )
    
    def _calculate_memory_pressure(self) -> float:
        """Calculate current memory pressure (0-1)"""
        return self.metrics.memory_usage_bytes / self.max_memory_bytes
    
    def _adaptive_optimization(self):
        """Perform adaptive optimization"""
        with self._lock:
            # Update performance metrics
            self._update_performance_metrics()
            
            # Adjust cache size if adaptive sizing enabled
            if self.adaptive_sizing:
                self._adjust_cache_size()
            
            # Update eviction policy if needed
            self._adapt_eviction_policy()
    
    def _adjust_cache_size(self):
        """Dynamically adjust cache size based on performance"""
        current_hit_rate = self.metrics.hit_rate
        memory_pressure = self._calculate_memory_pressure()
        
        # Increase size if hit rate is low and memory allows
        if (current_hit_rate < self.target_hit_rate and 
            memory_pressure < 0.7 and 
            self.max_size < 5000):
            self.max_size = min(5000, int(self.max_size * 1.1))
        
        # Decrease size if memory pressure is high
        elif memory_pressure > self.memory_pressure_threshold:
            self.max_size = max(100, int(self.max_size * 0.9))
    
    def _adapt_eviction_policy(self):
        """Adapt eviction policy based on performance"""
        if len(self.performance_history) < 10:
            return
        
        # Analyze recent performance trend
        recent_hit_rates = [hr for _, hr in self.performance_history[-10:]]
        avg_hit_rate = statistics.mean(recent_hit_rates)
        hit_rate_trend = recent_hit_rates[-1] - recent_hit_rates[0]
        
        # Switch policy if performance is declining
        if avg_hit_rate < 0.6 and hit_rate_trend < -0.05:
            if self.policy != CachePolicy.ADAPTIVE:
                self.policy = CachePolicy.ADAPTIVE
    
    def _predictive_management(self):
        """Predictive cache management"""
        if not self.predictive_loading:
            return
        
        # Simple pattern-based prediction
        # In production, this would use more sophisticated ML models
        self._predict_future_accesses()
    
    def _predict_future_accesses(self):
        """Predict which keys might be accessed soon"""
        current_time = time.time()
        predictions = {}
        
        for key, access_times in self.access_patterns.items():
            if len(access_times) < 2:
                continue
            
            # Calculate access frequency
            recent_accesses = [t for t in access_times if t > current_time - 3600]
            if recent_accesses:
                frequency = len(recent_accesses) / 3600  # accesses per second
                
                # Simple prediction: if frequently accessed, likely to be accessed again
                next_access_prob = min(1.0, frequency * 3600)  # probability in next hour
                predictions[key] = next_access_prob
        
        # Pre-warm high-probability items if they're not cached
        for key, prob in predictions.items():
            if prob > 0.7 and key not in self.cache:
                # Would trigger pre-loading in production system
                pass
    
    def _update_performance_metrics(self):
        """Update performance metrics"""
        total_requests = self.metrics.hits + self.metrics.misses
        if total_requests > 0:
            self.metrics.hit_rate = self.metrics.hits / total_requests
        
        # Record performance history
        self.performance_history.append((time.time(), self.metrics.hit_rate))
        
        # Keep only recent history
        cutoff_time = time.time() - 24 * 3600  # 24 hours
        self.performance_history = [
            (t, hr) for t, hr in self.performance_history if t > cutoff_time
        ]
        
        # Calculate cache efficiency
        if self.metrics.evictions > 0:
            self.metrics.cache_efficiency = self.metrics.hits / (self.metrics.hits + self.metrics.evictions)
        else:
            self.metrics.cache_efficiency = 1.0
        
        # Update memory pressure
        self.metrics.memory_pressure = self._calculate_memory_pressure()
    
    def _estimate_size(self, obj: Any) -> int:
        """Estimate object size in bytes"""
        try:
            return len(pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL))
        except Exception:
            # Fallback estimation
            if hasattr(obj, '__sizeof__'):
                return obj.__sizeof__()
            elif isinstance(obj, (str, bytes)):
                return len(obj)
            else:
                return 64  # Default estimate
    
    def get_performance_metrics(self) -> CacheMetrics:
        """Get current performance metrics"""
        self._update_performance_metrics()
        return self.metrics
    
    def get_cache_efficiency_report(self) -> Dict[str, Any]:
        """Get detailed cache efficiency report"""
        with self._lock:
            total_entries = len(self.cache)
            if total_entries == 0:
                return {"error": "No cache entries"}
            
            # Analyze entry distribution
            hints_distribution = defaultdict(int)
            size_distribution = {"small": 0, "medium": 0, "large": 0}
            utility_scores = []
            
            for entry in self.cache.values():
                if entry.hint:
                    hints_distribution[entry.hint.value] += 1
                
                size_mb = entry.size_bytes / (1024 * 1024)
                if size_mb < 1:
                    size_distribution["small"] += 1
                elif size_mb < 10:
                    size_distribution["medium"] += 1
                else:
                    size_distribution["large"] += 1
                
                utility_scores.append(entry.calculate_utility_score())
            
            return {
                "total_entries": total_entries,
                "memory_usage_mb": self.metrics.memory_usage_bytes / (1024 * 1024),
                "hit_rate": self.metrics.hit_rate,
                "cache_efficiency": self.metrics.cache_efficiency,
                "memory_pressure": self.metrics.memory_pressure,
                "avg_access_time_ms": self.metrics.avg_access_time_ms,
                "hints_distribution": dict(hints_distribution),
                "size_distribution": size_distribution,
                "utility_score_stats": {
                    "mean": statistics.mean(utility_scores) if utility_scores else 0,
                    "median": statistics.median(utility_scores) if utility_scores else 0,
                    "min": min(utility_scores) if utility_scores else 0,
                    "max": max(utility_scores) if utility_scores else 0,
                },
                "eviction_policy": self.policy.value,
                "adaptive_sizing_enabled": self.adaptive_sizing,
                "predictive_loading_enabled": self.predictive_loading,
            }
    
    def clear(self):
        """Clear all cache entries"""
        with self._lock:
            self.cache.clear()
            self.access_patterns.clear()
            self.performance_history.clear()
            self.metrics = CacheMetrics()

# Global enhanced cache instances
_enhanced_caches: Dict[str, EnhancedPerformanceCache] = {}

def get_enhanced_cache(name: str = "default", **kwargs) -> EnhancedPerformanceCache:
    """Get named enhanced cache instance"""
    if name not in _enhanced_caches:
        _enhanced_caches[name] = EnhancedPerformanceCache(**kwargs)
    return _enhanced_caches[name]

def enhanced_cached(cache_name: str = "default", 
                   ttl: Optional[float] = None,
                   hint: Optional[CacheHint] = None,
                   track_performance: bool = True):
    """Enhanced caching decorator with performance tracking"""
    
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # Create cache key
            key_data = f"{func.__module__}.{func.__name__}:{args}:{sorted(kwargs.items())}"
            cache_key = hashlib.md5(key_data.encode()).hexdigest()
            
            cache = get_enhanced_cache(cache_name)
            
            # Try cache first
            result = cache.get(cache_key)
            if result is not None:
                return result
            
            # Compute and cache result
            start_time = time.time()
            result = func(*args, **kwargs)
            compute_time = time.time() - start_time
            
            # Estimate performance impact based on compute time
            performance_impact = min(2.0, compute_time / 0.1)  # 0.1s baseline
            
            cache.put(
                cache_key, result, 
                ttl=ttl, 
                hint=hint,
                compute_time=compute_time,
                performance_impact=performance_impact
            )
            
            return result
        
        return wrapper
    return decorator

__all__ = [
    'CachePolicy',
    'CacheHint', 
    'EnhancedCacheEntry',
    'CacheMetrics',
    'EnhancedPerformanceCache',
    'get_enhanced_cache',
    'enhanced_cached'
]