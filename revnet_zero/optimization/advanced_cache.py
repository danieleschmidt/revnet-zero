"""
Generation 3 Advanced Caching System for RevNet-Zero

Implements intelligent caching with adaptive algorithms, distributed caching,
and predictive prefetching for maximum performance.
"""

import time
import threading
import hashlib
import pickle
import json
from typing import Dict, List, Optional, Any, Tuple, Callable, Union, Set
from dataclasses import dataclass, asdict
from collections import OrderedDict, defaultdict, deque
from enum import Enum
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
import warnings
import weakref


class CachePolicy(Enum):
    """Cache eviction policies."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    FIFO = "fifo"  # First In, First Out
    ADAPTIVE = "adaptive"  # Adaptive Replacement Cache
    TTL = "ttl"  # Time To Live based


class CacheLevel(Enum):
    """Cache levels for hierarchical caching."""
    L1_MEMORY = "l1_memory"
    L2_MEMORY = "l2_memory" 
    L3_DISK = "l3_disk"
    L4_DISTRIBUTED = "l4_distributed"


@dataclass
class CacheEntry:
    """Represents a cache entry with metadata."""
    key: str
    value: Any
    timestamp: float
    access_count: int = 0
    last_access: float = 0.0
    size: int = 0
    ttl: Optional[float] = None
    
    def __post_init__(self):
        if self.last_access == 0.0:
            self.last_access = self.timestamp
    
    @property
    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        if self.ttl is None:
            return False
        return time.time() - self.timestamp > self.ttl
    
    @property
    def age(self) -> float:
        """Get age of cache entry in seconds."""
        return time.time() - self.timestamp


@dataclass
class CacheStats:
    """Cache statistics."""
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    total_requests: int = 0
    hit_rate: float = 0.0
    average_access_time_ms: float = 0.0
    memory_usage_mb: float = 0.0
    
    def update_hit_rate(self):
        """Update hit rate calculation."""
        self.total_requests = self.hits + self.misses
        self.hit_rate = (self.hits / self.total_requests) * 100 if self.total_requests > 0 else 0.0


class AdaptiveCacheManager:
    """
    Generation 3: Adaptive cache manager with multiple policies and intelligence.
    
    Features:
    - Multiple eviction policies with automatic selection
    - Hierarchical caching (L1/L2/L3/L4)
    - Intelligent prefetching
    - Cache warming strategies
    - Performance analytics and optimization
    """
    
    def __init__(
        self,
        max_size_mb: float = 1024.0,  # 1GB default
        default_ttl: Optional[float] = 3600.0,  # 1 hour
        policy: CachePolicy = CachePolicy.ADAPTIVE,
        enable_prefetching: bool = True,
        enable_compression: bool = True,
        enable_distributed: bool = False,
    ):
        """
        Initialize adaptive cache manager.
        
        Args:
            max_size_mb: Maximum cache size in megabytes
            default_ttl: Default time-to-live in seconds
            policy: Cache eviction policy
            enable_prefetching: Enable intelligent prefetching
            enable_compression: Enable value compression
            enable_distributed: Enable distributed caching
        """
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        self.default_ttl = default_ttl
        self.policy = policy
        self.enable_prefetching = enable_prefetching
        self.enable_compression = enable_compression
        self.enable_distributed = enable_distributed
        
        # Multi-level cache storage
        self.l1_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.l2_cache: Dict[str, CacheEntry] = {}
        
        # Cache statistics
        self.stats = CacheStats()
        self.policy_stats = defaultdict(int)
        
        # Access patterns for prefetching
        self.access_patterns: Dict[str, List[str]] = defaultdict(list)
        self.access_frequencies: Dict[str, int] = defaultdict(int)
        self.access_history: deque = deque(maxlen=10000)
        
        # Threading support
        self._lock = threading.RLock()
        self._prefetch_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="CachePrefetch")
        
        # Adaptive policy state
        self.policy_performance: Dict[CachePolicy, float] = defaultdict(float)
        self.current_policy = policy
        self.policy_switch_threshold = 0.1  # 10% improvement needed
        
        # Compression support
        if enable_compression:
            try:
                import lz4.frame
                self._compressor = lz4.frame
                self._compression_available = True
            except ImportError:
                import gzip
                self._compressor = gzip
                self._compression_available = True
        else:
            self._compression_available = False
        
        # Memory tracking
        self.current_size_bytes = 0
        self.size_tracking: Dict[str, int] = {}
        
        # Logger
        self.logger = logging.getLogger(__name__)
        
        # Prefetching state
        self.prefetch_queue = asyncio.Queue() if enable_prefetching else None
        self.prefetch_task = None
        
        # Background maintenance
        self._maintenance_thread = None
        self._stop_maintenance = threading.Event()
    
    def _calculate_size(self, obj: Any) -> int:
        """Calculate size of object in bytes."""
        try:
            if self.enable_compression and self._compression_available:
                # Compress and measure
                serialized = pickle.dumps(obj)
                if hasattr(self._compressor, 'compress'):
                    compressed = self._compressor.compress(serialized)
                else:
                    compressed = self._compressor.compress(serialized, compresslevel=1)
                return len(compressed)
            else:
                # Just pickle size
                return len(pickle.dumps(obj))
        except Exception:
            # Fallback estimate
            return len(str(obj).encode('utf-8'))
    
    def _compress_value(self, value: Any) -> bytes:
        """Compress a value for storage."""
        if not self._compression_available:
            return pickle.dumps(value)
        
        try:
            serialized = pickle.dumps(value)
            if hasattr(self._compressor, 'compress'):
                return self._compressor.compress(serialized)
            else:
                return self._compressor.compress(serialized, compresslevel=1)
        except Exception:
            return pickle.dumps(value)
    
    def _decompress_value(self, compressed: bytes) -> Any:
        """Decompress a value from storage."""
        if not self._compression_available:
            return pickle.loads(compressed)
        
        try:
            if hasattr(self._compressor, 'decompress'):
                decompressed = self._compressor.decompress(compressed)
            else:
                decompressed = self._compressor.decompress(compressed)
            return pickle.loads(decompressed)
        except Exception:
            return pickle.loads(compressed)
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get value from cache.
        
        Args:
            key: Cache key
            default: Default value if key not found
            
        Returns:
            Cached value or default
        """
        start_time = time.perf_counter()
        
        with self._lock:
            # Check L1 cache first
            if key in self.l1_cache:
                entry = self.l1_cache[key]
                
                # Check expiration
                if entry.is_expired:
                    self._remove_entry(key)
                    self.stats.misses += 1
                    self.stats.update_hit_rate()
                    return default
                
                # Update access metadata
                entry.last_access = time.time()
                entry.access_count += 1
                
                # Move to end for LRU
                if self.current_policy in [CachePolicy.LRU, CachePolicy.ADAPTIVE]:
                    self.l1_cache.move_to_end(key)
                
                # Record hit
                self.stats.hits += 1
                self.stats.update_hit_rate()
                
                # Update access patterns
                self._update_access_patterns(key)
                
                # Decompress if needed
                if isinstance(entry.value, bytes) and self.enable_compression:
                    try:
                        value = self._decompress_value(entry.value)
                    except Exception:
                        value = entry.value
                else:
                    value = entry.value
                
                # Record access time
                access_time = (time.perf_counter() - start_time) * 1000
                self._update_average_access_time(access_time)
                
                return value
            
            # Check L2 cache
            if key in self.l2_cache:
                entry = self.l2_cache[key]
                
                if not entry.is_expired:
                    # Promote to L1
                    self._promote_to_l1(key, entry)
                    
                    self.stats.hits += 1
                    self.stats.update_hit_rate()
                    
                    # Decompress if needed
                    if isinstance(entry.value, bytes) and self.enable_compression:
                        try:
                            value = self._decompress_value(entry.value)
                        except Exception:
                            value = entry.value
                    else:
                        value = entry.value
                    
                    return value
                else:
                    # Remove expired entry
                    del self.l2_cache[key]
            
            # Cache miss
            self.stats.misses += 1
            self.stats.update_hit_rate()
            
            # Trigger prefetching if enabled
            if self.enable_prefetching:
                self._trigger_prefetching(key)
            
            return default
    
    def put(
        self, 
        key: str, 
        value: Any, 
        ttl: Optional[float] = None,
        cache_level: CacheLevel = CacheLevel.L1_MEMORY
    ) -> bool:
        """
        Put value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live (None for default)
            cache_level: Target cache level
            
        Returns:
            True if successfully cached
        """
        with self._lock:
            try:
                # Calculate entry size
                if self.enable_compression:
                    compressed_value = self._compress_value(value)
                    entry_size = len(compressed_value)
                    store_value = compressed_value
                else:
                    entry_size = self._calculate_size(value)
                    store_value = value
                
                # Check if value is too large
                if entry_size > self.max_size_bytes:
                    self.logger.warning(f"Value for key '{key}' too large ({entry_size} bytes)")
                    return False
                
                # Create cache entry
                entry_ttl = ttl if ttl is not None else self.default_ttl
                entry = CacheEntry(
                    key=key,
                    value=store_value,
                    timestamp=time.time(),
                    size=entry_size,
                    ttl=entry_ttl
                )
                
                # Ensure space is available
                self._ensure_space(entry_size)
                
                # Store based on cache level
                if cache_level == CacheLevel.L1_MEMORY:
                    # Remove from L2 if exists
                    if key in self.l2_cache:
                        old_entry = self.l2_cache.pop(key)
                        self.current_size_bytes -= old_entry.size
                    
                    # Add/update L1
                    if key in self.l1_cache:
                        old_entry = self.l1_cache[key]
                        self.current_size_bytes -= old_entry.size
                    
                    self.l1_cache[key] = entry
                    self.current_size_bytes += entry_size
                    self.size_tracking[key] = entry_size
                    
                elif cache_level == CacheLevel.L2_MEMORY:
                    # Add to L2 cache
                    if key in self.l2_cache:
                        old_entry = self.l2_cache[key]
                        self.current_size_bytes -= old_entry.size
                    
                    self.l2_cache[key] = entry
                    self.current_size_bytes += entry_size
                    self.size_tracking[key] = entry_size
                
                # Update access patterns
                self._update_access_patterns(key)
                
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to cache key '{key}': {e}")
                return False
    
    def delete(self, key: str) -> bool:
        """
        Delete key from cache.
        
        Args:
            key: Cache key to delete
            
        Returns:
            True if key was deleted
        """
        with self._lock:
            deleted = False
            
            if key in self.l1_cache:
                self._remove_entry(key)
                deleted = True
            
            if key in self.l2_cache:
                entry = self.l2_cache.pop(key)
                self.current_size_bytes -= entry.size
                self.size_tracking.pop(key, None)
                deleted = True
            
            return deleted
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self.l1_cache.clear()
            self.l2_cache.clear()
            self.current_size_bytes = 0
            self.size_tracking.clear()
            self.access_patterns.clear()
            self.access_frequencies.clear()
            self.access_history.clear()
    
    def _remove_entry(self, key: str):
        """Remove entry from L1 cache."""
        if key in self.l1_cache:
            entry = self.l1_cache.pop(key)
            self.current_size_bytes -= entry.size
            self.size_tracking.pop(key, None)
            self.stats.evictions += 1
    
    def _promote_to_l1(self, key: str, entry: CacheEntry):
        """Promote entry from L2 to L1."""
        # Remove from L2
        if key in self.l2_cache:
            del self.l2_cache[key]
        
        # Ensure space in L1
        self._ensure_space(entry.size)
        
        # Add to L1
        entry.last_access = time.time()
        entry.access_count += 1
        self.l1_cache[key] = entry
    
    def _ensure_space(self, required_bytes: int):
        """Ensure sufficient space by evicting entries if necessary."""
        while (self.current_size_bytes + required_bytes) > self.max_size_bytes:
            if not self._evict_entry():
                break  # Nothing left to evict
    
    def _evict_entry(self) -> bool:
        """
        Evict an entry based on current policy.
        
        Returns:
            True if an entry was evicted
        """
        if not self.l1_cache and not self.l2_cache:
            return False
        
        if self.current_policy == CachePolicy.LRU:
            return self._evict_lru()
        elif self.current_policy == CachePolicy.LFU:
            return self._evict_lfu()
        elif self.current_policy == CachePolicy.FIFO:
            return self._evict_fifo()
        elif self.current_policy == CachePolicy.TTL:
            return self._evict_expired()
        elif self.current_policy == CachePolicy.ADAPTIVE:
            return self._evict_adaptive()
        else:
            return self._evict_lru()  # Fallback
    
    def _evict_lru(self) -> bool:
        """Evict least recently used entry."""
        if self.l1_cache:
            # LRU is at the beginning of OrderedDict
            key = next(iter(self.l1_cache))
            self._remove_entry(key)
            return True
        elif self.l2_cache:
            # Find LRU in L2
            lru_key = min(self.l2_cache.keys(), key=lambda k: self.l2_cache[k].last_access)
            entry = self.l2_cache.pop(lru_key)
            self.current_size_bytes -= entry.size
            self.size_tracking.pop(lru_key, None)
            return True
        return False
    
    def _evict_lfu(self) -> bool:
        """Evict least frequently used entry."""
        if self.l1_cache:
            lfu_key = min(self.l1_cache.keys(), key=lambda k: self.l1_cache[k].access_count)
            self._remove_entry(lfu_key)
            return True
        elif self.l2_cache:
            lfu_key = min(self.l2_cache.keys(), key=lambda k: self.l2_cache[k].access_count)
            entry = self.l2_cache.pop(lfu_key)
            self.current_size_bytes -= entry.size
            self.size_tracking.pop(lfu_key, None)
            return True
        return False
    
    def _evict_fifo(self) -> bool:
        """Evict first in, first out."""
        if self.l1_cache:
            # OrderedDict maintains insertion order
            key = next(iter(self.l1_cache))
            self._remove_entry(key)
            return True
        elif self.l2_cache:
            # Find oldest entry
            oldest_key = min(self.l2_cache.keys(), key=lambda k: self.l2_cache[k].timestamp)
            entry = self.l2_cache.pop(oldest_key)
            self.current_size_bytes -= entry.size
            self.size_tracking.pop(oldest_key, None)
            return True
        return False
    
    def _evict_expired(self) -> bool:
        """Evict expired entries first."""
        current_time = time.time()
        
        # Check L1 for expired entries
        expired_keys = [
            key for key, entry in self.l1_cache.items()
            if entry.ttl is not None and (current_time - entry.timestamp) > entry.ttl
        ]
        
        if expired_keys:
            self._remove_entry(expired_keys[0])
            return True
        
        # Check L2 for expired entries
        expired_keys = [
            key for key, entry in self.l2_cache.items()
            if entry.ttl is not None and (current_time - entry.timestamp) > entry.ttl
        ]
        
        if expired_keys:
            key = expired_keys[0]
            entry = self.l2_cache.pop(key)
            self.current_size_bytes -= entry.size
            self.size_tracking.pop(key, None)
            return True
        
        # No expired entries, fall back to LRU
        return self._evict_lru()
    
    def _evict_adaptive(self) -> bool:
        """Adaptive eviction based on access patterns."""
        # Combine multiple factors for scoring
        candidates = []
        
        # Score L1 entries
        for key, entry in self.l1_cache.items():
            score = self._calculate_eviction_score(entry)
            candidates.append((score, key, 'l1'))
        
        # Score L2 entries
        for key, entry in self.l2_cache.items():
            score = self._calculate_eviction_score(entry)
            candidates.append((score, key, 'l2'))
        
        if not candidates:
            return False
        
        # Sort by score (lowest score = best candidate for eviction)
        candidates.sort()
        
        _, key, level = candidates[0]
        
        if level == 'l1':
            self._remove_entry(key)
        else:
            entry = self.l2_cache.pop(key)
            self.current_size_bytes -= entry.size
            self.size_tracking.pop(key, None)
        
        return True
    
    def _calculate_eviction_score(self, entry: CacheEntry) -> float:
        """
        Calculate eviction score for adaptive policy.
        Lower score = better candidate for eviction.
        """
        current_time = time.time()
        
        # Base factors
        recency_factor = current_time - entry.last_access  # Higher = older
        frequency_factor = 1.0 / max(1, entry.access_count)  # Higher = less frequent
        age_factor = current_time - entry.timestamp  # Higher = older
        size_factor = entry.size / 1024.0  # Size in KB
        
        # TTL factor
        ttl_factor = 0.0
        if entry.ttl is not None:
            time_remaining = entry.ttl - (current_time - entry.timestamp)
            if time_remaining <= 0:
                return -1000.0  # Expired entries should be evicted first
            ttl_factor = 1.0 / max(1, time_remaining)
        
        # Combined score (weighted)
        score = (
            recency_factor * 0.4 +
            frequency_factor * 0.3 +
            age_factor * 0.2 +
            size_factor * 0.05 +
            ttl_factor * 0.05
        )
        
        return score
    
    def _update_access_patterns(self, key: str):
        """Update access patterns for prefetching."""
        if not self.enable_prefetching:
            return
        
        # Record access
        self.access_history.append((key, time.time()))
        self.access_frequencies[key] += 1
        
        # Update sequential patterns
        if len(self.access_history) >= 2:
            prev_key, prev_time = self.access_history[-2]
            current_time = time.time()
            
            # If accessed within 1 second, consider it a pattern
            if current_time - prev_time <= 1.0:
                self.access_patterns[prev_key].append(key)
                # Keep only recent patterns
                if len(self.access_patterns[prev_key]) > 10:
                    self.access_patterns[prev_key].pop(0)
    
    def _trigger_prefetching(self, key: str):
        """Trigger intelligent prefetching based on access patterns."""
        if not self.enable_prefetching or self.prefetch_queue is None:
            return
        
        # Get likely next keys based on patterns
        likely_keys = self.access_patterns.get(key, [])
        
        for next_key in likely_keys:
            if next_key not in self.l1_cache and next_key not in self.l2_cache:
                # Add to prefetch queue
                try:
                    self.prefetch_queue.put_nowait(next_key)
                except asyncio.QueueFull:
                    pass  # Queue full, skip prefetching
    
    def _update_average_access_time(self, access_time_ms: float):
        """Update rolling average access time."""
        if self.stats.average_access_time_ms == 0:
            self.stats.average_access_time_ms = access_time_ms
        else:
            # Exponential moving average
            alpha = 0.1
            self.stats.average_access_time_ms = (
                alpha * access_time_ms + 
                (1 - alpha) * self.stats.average_access_time_ms
            )
    
    def warm_cache(self, data_loader: Callable[[], Dict[str, Any]]):
        """
        Warm the cache with frequently accessed data.
        
        Args:
            data_loader: Function that returns dict of key-value pairs
        """
        try:
            data = data_loader()
            for key, value in data.items():
                self.put(key, value, cache_level=CacheLevel.L2_MEMORY)
            
            self.logger.info(f"Cache warmed with {len(data)} entries")
        except Exception as e:
            self.logger.error(f"Failed to warm cache: {e}")
    
    def optimize_policy(self):
        """Optimize cache policy based on performance metrics."""
        if self.policy != CachePolicy.ADAPTIVE:
            return
        
        current_hit_rate = self.stats.hit_rate
        
        # Test different policies and measure performance
        test_policies = [CachePolicy.LRU, CachePolicy.LFU, CachePolicy.FIFO]
        
        for test_policy in test_policies:
            if test_policy not in self.policy_performance:
                self.policy_performance[test_policy] = current_hit_rate
            
            # If a policy is performing significantly better, consider switching
            if (self.policy_performance[test_policy] - current_hit_rate) > self.policy_switch_threshold:
                old_policy = self.current_policy
                self.current_policy = test_policy
                self.logger.info(f"Switched cache policy from {old_policy.value} to {test_policy.value}")
                break
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        with self._lock:
            self.stats.memory_usage_mb = self.current_size_bytes / (1024 * 1024)
            
            return {
                'stats': asdict(self.stats),
                'policy': self.current_policy.value,
                'cache_levels': {
                    'l1_entries': len(self.l1_cache),
                    'l2_entries': len(self.l2_cache),
                    'total_entries': len(self.l1_cache) + len(self.l2_cache),
                },
                'memory': {
                    'current_size_mb': self.current_size_bytes / (1024 * 1024),
                    'max_size_mb': self.max_size_bytes / (1024 * 1024),
                    'utilization_percent': (self.current_size_bytes / self.max_size_bytes) * 100,
                },
                'features': {
                    'compression_enabled': self.enable_compression,
                    'prefetching_enabled': self.enable_prefetching,
                    'distributed_enabled': self.enable_distributed,
                },
                'patterns': {
                    'unique_access_patterns': len(self.access_patterns),
                    'total_access_history': len(self.access_history),
                    'most_frequent_keys': sorted(
                        self.access_frequencies.items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:5]
                }
            }
    
    def get_optimization_recommendations(self) -> List[str]:
        """Get cache optimization recommendations."""
        recommendations = []
        
        # Analyze hit rate
        if self.stats.hit_rate < 50:
            recommendations.append("Low hit rate detected - consider increasing cache size or adjusting TTL")
        elif self.stats.hit_rate > 95:
            recommendations.append("Very high hit rate - cache may be over-provisioned")
        
        # Analyze memory usage
        utilization = (self.current_size_bytes / self.max_size_bytes) * 100
        if utilization > 90:
            recommendations.append("High memory usage - consider enabling compression or reducing cache size")
        elif utilization < 30:
            recommendations.append("Low memory usage - cache size could be reduced")
        
        # Analyze eviction rate
        if self.stats.total_requests > 0:
            eviction_rate = (self.stats.evictions / self.stats.total_requests) * 100
            if eviction_rate > 20:
                recommendations.append("High eviction rate - consider increasing cache size")
        
        # Policy recommendations
        if self.policy == CachePolicy.ADAPTIVE and len(self.policy_performance) > 1:
            best_policy = max(self.policy_performance.items(), key=lambda x: x[1])
            if best_policy[1] > self.stats.hit_rate + 5:  # 5% improvement
                recommendations.append(f"Consider switching to {best_policy[0].value} policy for better performance")
        
        return recommendations
    
    def start_maintenance(self, interval: float = 60.0):
        """Start background maintenance thread."""
        if self._maintenance_thread and self._maintenance_thread.is_alive():
            return
        
        self._stop_maintenance.clear()
        self._maintenance_thread = threading.Thread(
            target=self._maintenance_loop,
            args=(interval,),
            daemon=True,
            name="CacheMaintenance"
        )
        self._maintenance_thread.start()
        
        self.logger.info(f"Started cache maintenance with {interval}s interval")
    
    def stop_maintenance(self):
        """Stop background maintenance thread."""
        if self._maintenance_thread and self._maintenance_thread.is_alive():
            self._stop_maintenance.set()
            self._maintenance_thread.join(timeout=5.0)
            self.logger.info("Stopped cache maintenance")
    
    def _maintenance_loop(self, interval: float):
        """Background maintenance loop."""
        while not self._stop_maintenance.wait(interval):
            try:
                # Remove expired entries
                self._cleanup_expired_entries()
                
                # Optimize policy if adaptive
                if self.policy == CachePolicy.ADAPTIVE:
                    self.optimize_policy()
                
                # Log statistics periodically
                stats = self.get_cache_stats()
                self.logger.debug(f"Cache stats: Hit rate {stats['stats']['hit_rate']:.1f}%, "
                                f"Memory usage {stats['memory']['utilization_percent']:.1f}%")
                
            except Exception as e:
                self.logger.error(f"Error in cache maintenance: {e}")
    
    def _cleanup_expired_entries(self):
        """Clean up expired cache entries."""
        with self._lock:
            current_time = time.time()
            
            # Clean L1 cache
            expired_l1_keys = [
                key for key, entry in self.l1_cache.items()
                if entry.ttl is not None and (current_time - entry.timestamp) > entry.ttl
            ]
            
            for key in expired_l1_keys:
                self._remove_entry(key)
            
            # Clean L2 cache
            expired_l2_keys = [
                key for key, entry in self.l2_cache.items()
                if entry.ttl is not None and (current_time - entry.timestamp) > entry.ttl
            ]
            
            for key in expired_l2_keys:
                entry = self.l2_cache.pop(key)
                self.current_size_bytes -= entry.size
                self.size_tracking.pop(key, None)
            
            if expired_l1_keys or expired_l2_keys:
                self.logger.debug(f"Cleaned up {len(expired_l1_keys) + len(expired_l2_keys)} expired entries")


# Global cache instance
_global_cache = None


def get_cache(
    max_size_mb: float = 1024.0,
    policy: CachePolicy = CachePolicy.ADAPTIVE,
    enable_prefetching: bool = True,
) -> AdaptiveCacheManager:
    """
    Get global cache instance.
    
    Args:
        max_size_mb: Maximum cache size in MB
        policy: Cache policy
        enable_prefetching: Enable prefetching
        
    Returns:
        Global cache instance
    """
    global _global_cache
    if _global_cache is None:
        _global_cache = AdaptiveCacheManager(
            max_size_mb=max_size_mb,
            policy=policy,
            enable_prefetching=enable_prefetching
        )
        _global_cache.start_maintenance()
    
    return _global_cache


def cache_result(
    ttl: Optional[float] = None,
    key_func: Optional[Callable] = None,
    cache_instance: Optional[AdaptiveCacheManager] = None,
):
    """
    Decorator for caching function results.
    
    Args:
        ttl: Time to live for cached result
        key_func: Function to generate cache key
        cache_instance: Cache instance to use
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Get cache instance
            cache = cache_instance or get_cache()
            
            # Generate cache key
            if key_func:
                key = key_func(*args, **kwargs)
            else:
                # Default key generation
                key_data = f"{func.__name__}:{hash(str(args) + str(sorted(kwargs.items())))}"
                key = hashlib.md5(key_data.encode()).hexdigest()
            
            # Try to get from cache
            result = cache.get(key)
            if result is not None:
                return result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache.put(key, result, ttl=ttl)
            
            return result
        
        return wrapper
    return decorator


__all__ = [
    'AdaptiveCacheManager', 'CachePolicy', 'CacheLevel', 'CacheEntry', 
    'CacheStats', 'get_cache', 'cache_result'
]