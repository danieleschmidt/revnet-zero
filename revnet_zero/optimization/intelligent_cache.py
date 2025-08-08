"""
Intelligent multi-level caching system for RevNet-Zero.

This module implements adaptive caching strategies that automatically
optimize memory usage and computational efficiency based on access patterns,
model architecture, and available resources.
"""

import time
import threading
import weakref
import hashlib
import pickle
from typing import Any, Dict, Optional, Tuple, List, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import OrderedDict, defaultdict
import gc
import os


class CacheLevel(Enum):
    """Cache level priorities."""
    L1_MEMORY = "l1_memory"      # Fast in-memory cache
    L2_SHARED = "l2_shared"      # Shared memory across processes
    L3_DISK = "l3_disk"          # Persistent disk cache
    L4_DISTRIBUTED = "l4_distributed"  # Distributed cache cluster


class EvictionPolicy(Enum):
    """Cache eviction policies."""
    LRU = "lru"                  # Least Recently Used
    LFU = "lfu"                  # Least Frequently Used
    ARC = "arc"                  # Adaptive Replacement Cache
    INTELLIGENT = "intelligent"  # AI-driven eviction


@dataclass
class CacheEntry:
    """Individual cache entry with metadata."""
    key: str
    value: Any
    size_bytes: int
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    creation_time: float = field(default_factory=time.time)
    computation_cost: float = 1.0  # Relative cost to recompute
    priority_score: float = 0.0
    tags: Dict[str, str] = field(default_factory=dict)
    
    def update_access(self):
        """Update access statistics."""
        self.access_count += 1
        self.last_access = time.time()
    
    def age_seconds(self) -> float:
        """Get age in seconds."""
        return time.time() - self.creation_time
    
    def idle_seconds(self) -> float:
        """Get idle time in seconds."""
        return time.time() - self.last_access


class IntelligentCacheManager:
    """
    Advanced multi-level caching system with intelligent eviction.
    
    Features:
    - Multi-level caching (memory -> disk -> distributed)
    - Adaptive eviction based on access patterns
    - Automatic size management
    - Compression for large objects
    - Asynchronous background operations
    - Cache warming and prefetching
    """
    
    def __init__(
        self,
        max_memory_mb: int = 1024,
        max_disk_mb: int = 10240,
        eviction_policy: EvictionPolicy = EvictionPolicy.INTELLIGENT,
        cache_dir: Optional[str] = None,
        compression_enabled: bool = True,
        prefetch_enabled: bool = True
    ):
        """
        Initialize intelligent cache manager.
        
        Args:
            max_memory_mb: Maximum memory cache size in MB
            max_disk_mb: Maximum disk cache size in MB
            eviction_policy: Cache eviction strategy
            cache_dir: Directory for disk cache
            compression_enabled: Enable compression for large objects
            prefetch_enabled: Enable predictive prefetching
        """
        self.max_memory_bytes = max_memory_mb * 1024 * 1024
        self.max_disk_bytes = max_disk_mb * 1024 * 1024
        self.eviction_policy = eviction_policy
        self.compression_enabled = compression_enabled
        self.prefetch_enabled = prefetch_enabled
        
        # Cache levels
        self.l1_cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.l2_cache: Dict[str, CacheEntry] = {}
        
        # Cache statistics
        self.stats = {
            'hits': defaultdict(int),
            'misses': defaultdict(int),
            'evictions': defaultdict(int),
            'bytes_cached': defaultdict(int),
            'access_patterns': defaultdict(list)
        }
        
        # Thread safety
        self._lock = threading.RLock()
        self._background_thread = None
        self._shutdown_event = threading.Event()
        
        # Disk cache setup
        if cache_dir is None:
            cache_dir = os.path.expanduser("~/.revnet_zero/cache")
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Access pattern learning
        self.access_history: List[Tuple[str, float]] = []
        self.pattern_predictions: Dict[str, List[str]] = {}
        
        # Start background maintenance
        self._start_background_maintenance()
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get item from cache with intelligent prefetching.
        
        Args:
            key: Cache key
            default: Default value if not found
            
        Returns:
            Cached value or default
        """
        with self._lock:
            # Try L1 cache first (memory)
            if key in self.l1_cache:
                entry = self.l1_cache[key]
                entry.update_access()
                
                # Move to end (most recently used)
                self.l1_cache.move_to_end(key)
                
                # Update statistics
                self.stats['hits']['l1'] += 1
                self._record_access(key)
                
                # Trigger prefetch if enabled
                if self.prefetch_enabled:
                    self._trigger_prefetch(key)
                
                return entry.value
            
            # Try L2 cache (shared memory)
            if key in self.l2_cache:
                entry = self.l2_cache[key]
                entry.update_access()
                
                # Promote to L1 if there's space
                self._promote_to_l1(key, entry)
                
                self.stats['hits']['l2'] += 1
                self._record_access(key)
                return entry.value
            
            # Try L3 cache (disk)
            disk_value = self._get_from_disk(key)
            if disk_value is not None:
                # Create cache entry and promote
                entry = CacheEntry(
                    key=key,
                    value=disk_value,
                    size_bytes=self._calculate_size(disk_value),
                    computation_cost=1.0
                )
                
                self._promote_to_l1(key, entry)
                self.stats['hits']['l3'] += 1
                self._record_access(key)
                return disk_value
            
            # Cache miss
            self.stats['misses']['total'] += 1
            return default
    
    def put(
        self, 
        key: str, 
        value: Any, 
        ttl_seconds: Optional[int] = None,
        computation_cost: float = 1.0,
        tags: Optional[Dict[str, str]] = None
    ):
        """
        Store item in cache with intelligent placement.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl_seconds: Time to live in seconds
            computation_cost: Relative cost to recompute (higher = more important)
            tags: Optional metadata tags
        """
        with self._lock:
            size_bytes = self._calculate_size(value)
            
            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=value,
                size_bytes=size_bytes,
                computation_cost=computation_cost,
                tags=tags or {}
            )
            
            # Calculate priority score
            entry.priority_score = self._calculate_priority(entry)
            
            # Decide cache level based on size and importance
            if size_bytes < self.max_memory_bytes * 0.1 or computation_cost > 2.0:
                # Small or important items go to L1
                self._put_l1(key, entry)
            elif size_bytes < self.max_memory_bytes * 0.5:
                # Medium items can go to L2
                self._put_l2(key, entry)
            else:
                # Large items go directly to disk
                self._put_disk(key, entry)
            
            # Update access pattern
            self._record_access(key)
    
    def invalidate(self, key: str):
        """Remove item from all cache levels."""
        with self._lock:
            # Remove from all levels
            self.l1_cache.pop(key, None)
            self.l2_cache.pop(key, None)
            self._remove_from_disk(key)
    
    def invalidate_by_tags(self, tags: Dict[str, str]):
        """Remove items matching tag criteria."""
        with self._lock:
            keys_to_remove = []
            
            # Check L1 cache
            for cache_key, entry in self.l1_cache.items():
                if self._tags_match(entry.tags, tags):
                    keys_to_remove.append(cache_key)
            
            # Check L2 cache
            for cache_key, entry in self.l2_cache.items():
                if self._tags_match(entry.tags, tags):
                    keys_to_remove.append(cache_key)
            
            # Remove all matching keys
            for key in keys_to_remove:
                self.invalidate(key)
    
    def clear_all(self):
        """Clear all cache levels."""
        with self._lock:
            self.l1_cache.clear()
            self.l2_cache.clear()
            
            # Clear disk cache
            import shutil
            if os.path.exists(self.cache_dir):
                shutil.rmtree(self.cache_dir)
                os.makedirs(self.cache_dir, exist_ok=True)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        with self._lock:
            total_hits = sum(self.stats['hits'].values())
            total_misses = self.stats['misses']['total']
            hit_rate = total_hits / (total_hits + total_misses) if (total_hits + total_misses) > 0 else 0
            
            return {
                'hit_rate': hit_rate,
                'total_hits': total_hits,
                'total_misses': total_misses,
                'l1_entries': len(self.l1_cache),
                'l2_entries': len(self.l2_cache),
                'l1_size_mb': sum(e.size_bytes for e in self.l1_cache.values()) / (1024 * 1024),
                'l2_size_mb': sum(e.size_bytes for e in self.l2_cache.values()) / (1024 * 1024),
                'evictions': dict(self.stats['evictions']),
                'memory_efficiency': self._calculate_memory_efficiency(),
                'top_accessed_keys': self._get_top_accessed_keys(10)
            }
    
    def prefetch(self, keys: List[str]):
        """Prefetch specified keys if not already cached."""
        for key in keys:
            if key not in self.l1_cache and key not in self.l2_cache:
                # Trigger async prefetch
                threading.Thread(
                    target=self._background_prefetch,
                    args=(key,),
                    daemon=True
                ).start()
    
    def cache_warmer(self, warm_function: Callable[[], Dict[str, Any]]):
        """Run cache warming function to preload important data."""
        def _warm():
            try:
                warm_data = warm_function()
                for key, value in warm_data.items():
                    self.put(key, value, computation_cost=0.5)  # Medium priority for warmed data
            except Exception as e:
                print(f"Cache warming failed: {e}")
        
        threading.Thread(target=_warm, daemon=True).start()
    
    def _put_l1(self, key: str, entry: CacheEntry):
        """Store in L1 (memory) cache with eviction."""
        # Check if we need to evict
        while (self._get_l1_size() + entry.size_bytes > self.max_memory_bytes 
               and len(self.l1_cache) > 0):
            self._evict_l1()
        
        self.l1_cache[key] = entry
        self.stats['bytes_cached']['l1'] += entry.size_bytes
    
    def _put_l2(self, key: str, entry: CacheEntry):
        """Store in L2 (shared) cache."""
        self.l2_cache[key] = entry
        self.stats['bytes_cached']['l2'] += entry.size_bytes
    
    def _put_disk(self, key: str, entry: CacheEntry):
        """Store in L3 (disk) cache."""
        try:
            file_path = os.path.join(self.cache_dir, self._get_cache_filename(key))
            
            # Compress if enabled and beneficial
            data = entry.value
            if self.compression_enabled and entry.size_bytes > 1024:
                data = self._compress(data)
            
            # Store metadata and data
            cache_data = {
                'value': data,
                'metadata': {
                    'size_bytes': entry.size_bytes,
                    'creation_time': entry.creation_time,
                    'computation_cost': entry.computation_cost,
                    'tags': entry.tags
                }
            }
            
            with open(file_path, 'wb') as f:
                pickle.dump(cache_data, f)
                
        except Exception as e:
            print(f"Disk cache write failed for {key}: {e}")
    
    def _get_from_disk(self, key: str) -> Any:
        """Retrieve from disk cache."""
        try:
            file_path = os.path.join(self.cache_dir, self._get_cache_filename(key))
            if not os.path.exists(file_path):
                return None
            
            with open(file_path, 'rb') as f:
                cache_data = pickle.load(f)
            
            data = cache_data['value']
            
            # Decompress if needed
            if self.compression_enabled:
                try:
                    data = self._decompress(data)
                except:
                    pass  # Data might not be compressed
            
            return data
            
        except Exception as e:
            print(f"Disk cache read failed for {key}: {e}")
            return None
    
    def _evict_l1(self):
        """Evict items from L1 cache based on policy."""
        if not self.l1_cache:
            return
        
        if self.eviction_policy == EvictionPolicy.LRU:
            # Remove least recently used
            key, entry = self.l1_cache.popitem(last=False)
        elif self.eviction_policy == EvictionPolicy.LFU:
            # Remove least frequently used
            lfu_key = min(self.l1_cache.keys(), 
                         key=lambda k: self.l1_cache[k].access_count)
            entry = self.l1_cache.pop(lfu_key)
            key = lfu_key
        elif self.eviction_policy == EvictionPolicy.INTELLIGENT:
            # Use AI-based eviction
            key, entry = self._intelligent_eviction()
        else:  # ARC or fallback to LRU
            key, entry = self.l1_cache.popitem(last=False)
        
        # Try to promote to L2 or disk
        if entry.computation_cost > 1.5:  # Important data
            self._put_l2(key, entry)
        elif entry.size_bytes > 1024 * 1024:  # Large data
            self._put_disk(key, entry)
        
        self.stats['evictions']['l1'] += 1
    
    def _intelligent_eviction(self) -> Tuple[str, CacheEntry]:
        """AI-based eviction using multiple factors."""
        if not self.l1_cache:
            return None, None
        
        # Calculate eviction scores for all entries
        scores = {}
        for key, entry in self.l1_cache.items():
            score = self._calculate_eviction_score(entry)
            scores[key] = score
        
        # Select entry with lowest score (best candidate for eviction)
        evict_key = min(scores.keys(), key=lambda k: scores[k])
        return evict_key, self.l1_cache.pop(evict_key)
    
    def _calculate_eviction_score(self, entry: CacheEntry) -> float:
        """
        Calculate intelligent eviction score.
        
        Lower score = better candidate for eviction
        Higher score = keep in cache longer
        """
        age_factor = 1.0 / (entry.age_seconds() + 1)
        frequency_factor = entry.access_count / 100.0
        recency_factor = 1.0 / (entry.idle_seconds() + 1)
        cost_factor = entry.computation_cost
        size_penalty = entry.size_bytes / (1024 * 1024)  # MB
        
        # Weighted combination
        score = (
            frequency_factor * 0.3 +
            recency_factor * 0.3 +
            cost_factor * 0.25 +
            age_factor * 0.15 -
            size_penalty * 0.1
        )
        
        return score
    
    def _promote_to_l1(self, key: str, entry: CacheEntry):
        """Promote entry from lower level to L1."""
        if self._get_l1_size() + entry.size_bytes <= self.max_memory_bytes:
            self.l1_cache[key] = entry
            # Remove from L2 if it was there
            self.l2_cache.pop(key, None)
        else:
            # Not enough space, keep in L2
            self.l2_cache[key] = entry
    
    def _calculate_size(self, obj: Any) -> int:
        """Estimate object size in bytes."""
        try:
            return len(pickle.dumps(obj))
        except:
            # Fallback estimation
            import sys
            return sys.getsizeof(obj)
    
    def _calculate_priority(self, entry: CacheEntry) -> float:
        """Calculate cache priority score."""
        return entry.computation_cost * (entry.access_count + 1)
    
    def _get_l1_size(self) -> int:
        """Get total size of L1 cache in bytes."""
        return sum(entry.size_bytes for entry in self.l1_cache.values())
    
    def _record_access(self, key: str):
        """Record access for pattern learning."""
        self.access_history.append((key, time.time()))
        
        # Keep history bounded
        if len(self.access_history) > 1000:
            self.access_history = self.access_history[-500:]
        
        # Update pattern predictions
        if self.prefetch_enabled:
            self._update_access_patterns(key)
    
    def _update_access_patterns(self, key: str):
        """Update access pattern predictions."""
        recent_keys = [k for k, t in self.access_history[-10:]]
        
        if key not in self.pattern_predictions:
            self.pattern_predictions[key] = []
        
        # Find keys that commonly follow this key
        for i, (hist_key, _) in enumerate(self.access_history[:-1]):
            if hist_key == key:
                next_key = self.access_history[i + 1][0]
                if next_key not in self.pattern_predictions[key]:
                    self.pattern_predictions[key].append(next_key)
    
    def _trigger_prefetch(self, key: str):
        """Trigger prefetch based on access patterns."""
        if key in self.pattern_predictions:
            prefetch_keys = self.pattern_predictions[key][:3]  # Top 3 predictions
            for prefetch_key in prefetch_keys:
                if (prefetch_key not in self.l1_cache and 
                    prefetch_key not in self.l2_cache):
                    # Async prefetch
                    threading.Thread(
                        target=self._background_prefetch,
                        args=(prefetch_key,),
                        daemon=True
                    ).start()
    
    def _background_prefetch(self, key: str):
        """Background prefetch operation."""
        # This would integrate with the actual computation system
        # For now, just check disk cache
        value = self._get_from_disk(key)
        if value is not None:
            entry = CacheEntry(
                key=key,
                value=value,
                size_bytes=self._calculate_size(value),
                computation_cost=0.5  # Lower priority for prefetched
            )
            with self._lock:
                if len(self.l2_cache) < 100:  # Don't overfill
                    self.l2_cache[key] = entry
    
    def _start_background_maintenance(self):
        """Start background maintenance thread."""
        def maintenance_loop():
            while not self._shutdown_event.wait(300):  # 5 minute intervals
                try:
                    self._cleanup_expired_entries()
                    self._optimize_cache_distribution()
                    gc.collect()  # Force garbage collection
                except Exception as e:
                    print(f"Cache maintenance error: {e}")
        
        self._background_thread = threading.Thread(target=maintenance_loop, daemon=True)
        self._background_thread.start()
    
    def _cleanup_expired_entries(self):
        """Clean up old or unused entries."""
        with self._lock:
            current_time = time.time()
            keys_to_remove = []
            
            # Remove very old entries
            for key, entry in self.l1_cache.items():
                if (current_time - entry.last_access > 3600 and  # 1 hour idle
                    entry.computation_cost < 1.0):  # Not high priority
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                self.l1_cache.pop(key, None)
    
    def _optimize_cache_distribution(self):
        """Optimize cache distribution across levels."""
        # Move frequently accessed L2 items to L1
        with self._lock:
            candidates = sorted(
                self.l2_cache.items(),
                key=lambda x: x[1].access_count,
                reverse=True
            )[:5]  # Top 5 candidates
            
            for key, entry in candidates:
                if (entry.access_count > 10 and
                    self._get_l1_size() + entry.size_bytes <= self.max_memory_bytes * 0.8):
                    self.l1_cache[key] = self.l2_cache.pop(key)
    
    def _calculate_memory_efficiency(self) -> float:
        """Calculate cache memory efficiency score."""
        if not self.l1_cache:
            return 0.0
        
        total_accesses = sum(entry.access_count for entry in self.l1_cache.values())
        total_size = self._get_l1_size()
        
        if total_size == 0:
            return 0.0
        
        return total_accesses / (total_size / (1024 * 1024))  # Accesses per MB
    
    def _get_top_accessed_keys(self, n: int) -> List[str]:
        """Get top N most accessed keys."""
        all_entries = list(self.l1_cache.items()) + list(self.l2_cache.items())
        sorted_entries = sorted(all_entries, key=lambda x: x[1].access_count, reverse=True)
        return [key for key, _ in sorted_entries[:n]]
    
    def _get_cache_filename(self, key: str) -> str:
        """Generate safe filename for cache key."""
        # Use hash to create safe filename
        key_hash = hashlib.sha256(key.encode()).hexdigest()[:16]
        return f"cache_{key_hash}.pkl"
    
    def _compress(self, data: Any) -> bytes:
        """Compress data for storage."""
        try:
            import gzip
            serialized = pickle.dumps(data)
            return gzip.compress(serialized)
        except:
            return pickle.dumps(data)
    
    def _decompress(self, data: bytes) -> Any:
        """Decompress data from storage."""
        try:
            import gzip
            decompressed = gzip.decompress(data)
            return pickle.loads(decompressed)
        except:
            return pickle.loads(data)
    
    def _tags_match(self, entry_tags: Dict[str, str], filter_tags: Dict[str, str]) -> bool:
        """Check if entry tags match filter criteria."""
        for key, value in filter_tags.items():
            if key not in entry_tags or entry_tags[key] != value:
                return False
        return True
    
    def _remove_from_disk(self, key: str):
        """Remove item from disk cache."""
        try:
            file_path = os.path.join(self.cache_dir, self._get_cache_filename(key))
            if os.path.exists(file_path):
                os.remove(file_path)
        except Exception:
            pass  # Ignore removal errors
    
    def shutdown(self):
        """Gracefully shutdown cache manager."""
        self._shutdown_event.set()
        if self._background_thread:
            self._background_thread.join(timeout=5)


# Global cache instance
_global_cache: Optional[IntelligentCacheManager] = None


def get_cache() -> IntelligentCacheManager:
    """Get global cache instance."""
    global _global_cache
    if _global_cache is None:
        _global_cache = IntelligentCacheManager()
    return _global_cache


def cache_result(
    key_prefix: str = "",
    ttl_seconds: Optional[int] = None,
    computation_cost: float = 1.0,
    tags: Optional[Dict[str, str]] = None
):
    """
    Decorator to cache function results.
    
    Args:
        key_prefix: Prefix for cache key
        ttl_seconds: Time to live in seconds
        computation_cost: Relative computation cost (higher = more important to cache)
        tags: Optional tags for cache entry
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # Generate cache key
            key_data = f"{key_prefix}:{func.__name__}:{str(args)}:{str(sorted(kwargs.items()))}"
            cache_key = hashlib.sha256(key_data.encode()).hexdigest()[:32]
            
            # Try to get from cache
            cache = get_cache()
            result = cache.get(cache_key)
            
            if result is not None:
                return result
            
            # Compute and cache result
            result = func(*args, **kwargs)
            cache.put(
                cache_key, 
                result, 
                ttl_seconds=ttl_seconds,
                computation_cost=computation_cost,
                tags=tags
            )
            
            return result
        return wrapper
    return decorator


# Example usage and integration
def example_cache_usage():
    """Example of how to use the intelligent cache system."""
    cache = IntelligentCacheManager(
        max_memory_mb=512,
        max_disk_mb=2048,
        eviction_policy=EvictionPolicy.INTELLIGENT
    )
    
    # Cache some expensive computations
    @cache_result("model_inference", computation_cost=3.0)
    def expensive_model_inference(input_data):
        # Simulate expensive computation
        time.sleep(0.1)
        return f"result_for_{input_data}"
    
    # First call - cache miss
    result1 = expensive_model_inference("test_input_1")
    
    # Second call - cache hit
    result2 = expensive_model_inference("test_input_1")
    
    # Check cache statistics
    stats = cache.get_stats()
    print(f"Cache hit rate: {stats['hit_rate']:.2%}")
    
    # Cache warming
    def warm_cache():
        return {
            "precomputed_1": "value_1",
            "precomputed_2": "value_2"
        }
    
    cache.cache_warmer(warm_cache)
    
    return cache


__all__ = [
    'IntelligentCacheManager', 'CacheLevel', 'EvictionPolicy', 'CacheEntry',
    'get_cache', 'cache_result'
]