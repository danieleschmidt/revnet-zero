"""
Intelligent caching system for RevNet-Zero.

Provides multi-level caching for activations, attention weights,
and intermediate computations to reduce redundant calculations.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Tuple, List, Union
from dataclasses import dataclass
from collections import OrderedDict
import threading
import time
import hashlib
import pickle
from pathlib import Path


@dataclass
class CacheConfig:
    """Configuration for caching system."""
    max_memory_mb: int = 1024  # Maximum cache memory in MB
    max_entries: int = 1000    # Maximum number of cache entries
    ttl_seconds: int = 3600    # Time to live for cache entries
    enable_disk_cache: bool = False  # Enable persistent disk cache
    disk_cache_dir: Optional[str] = None
    compression_level: int = 6  # Compression level for disk cache
    cache_hit_threshold: float = 0.1  # Minimum cache hit rate to maintain cache


class CacheEntry:
    """Single cache entry with metadata."""
    
    def __init__(self, key: str, value: Any, size_bytes: int):
        self.key = key
        self.value = value
        self.size_bytes = size_bytes
        self.created_at = time.time()
        self.last_accessed = time.time()
        self.access_count = 0
    
    def access(self) -> Any:
        """Access cached value and update statistics."""
        self.last_accessed = time.time()
        self.access_count += 1
        return self.value
    
    def is_expired(self, ttl_seconds: int) -> bool:
        """Check if entry is expired."""
        return time.time() - self.created_at > ttl_seconds
    
    def get_priority_score(self) -> float:
        """Calculate priority score for eviction (higher = keep longer)."""
        age = time.time() - self.created_at
        recency = time.time() - self.last_accessed
        frequency = self.access_count
        
        # Weighted score: frequency is most important, then recency, then age
        return (frequency * 0.5) + (1.0 / (recency + 1.0) * 0.3) + (1.0 / (age + 1.0) * 0.2)


class MemoryCache:
    """In-memory cache with LRU eviction and intelligent sizing."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self.current_size_bytes = 0
        self.max_size_bytes = config.max_memory_mb * 1024 * 1024
        self.lock = threading.RLock()
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self.lock:
            if key in self.cache:
                entry = self.cache[key]
                
                # Check if expired
                if entry.is_expired(self.config.ttl_seconds):
                    self._remove_entry(key)
                    self.misses += 1
                    return None
                
                # Move to end (most recently used)
                self.cache.move_to_end(key)
                self.hits += 1
                return entry.access()
            else:
                self.misses += 1
                return None
    
    def put(self, key: str, value: Any, size_hint: Optional[int] = None) -> bool:
        """Put value in cache."""
        with self.lock:
            # Calculate size
            if size_hint is not None:
                size_bytes = size_hint
            else:
                size_bytes = self._estimate_size(value)
            
            # Check if value is too large for cache
            if size_bytes > self.max_size_bytes * 0.5:  # Don't cache items > 50% of cache
                return False
            
            # Remove existing entry if present
            if key in self.cache:
                self._remove_entry(key)
            
            # Make space if needed
            while (self.current_size_bytes + size_bytes > self.max_size_bytes or 
                   len(self.cache) >= self.config.max_entries):
                if not self._evict_lru():
                    break
            
            # Add new entry
            entry = CacheEntry(key, value, size_bytes)
            self.cache[key] = entry
            self.current_size_bytes += size_bytes
            
            return True
    
    def _remove_entry(self, key: str):
        """Remove entry from cache."""
        if key in self.cache:
            entry = self.cache[key]
            del self.cache[key]
            self.current_size_bytes -= entry.size_bytes
    
    def _evict_lru(self) -> bool:
        """Evict least recently used entry."""
        if not self.cache:
            return False
        
        # Find entry with lowest priority score
        min_score = float('inf')
        victim_key = None
        
        for key, entry in self.cache.items():
            score = entry.get_priority_score()
            if score < min_score:
                min_score = score
                victim_key = key
        
        if victim_key:
            self._remove_entry(victim_key)
            self.evictions += 1
            return True
        
        return False
    
    def _estimate_size(self, value: Any) -> int:
        """Estimate memory size of value."""
        if isinstance(value, torch.Tensor):
            return value.numel() * value.element_size()
        else:
            # Rough estimate using pickle
            try:
                return len(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
            except:
                return 1024  # Default size
    
    def clear(self):
        """Clear all cache entries."""
        with self.lock:
            self.cache.clear()
            self.current_size_bytes = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0.0
            
            return {
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
                "hit_rate": hit_rate,
                "entries": len(self.cache),
                "size_mb": self.current_size_bytes / (1024 * 1024),
                "utilization": self.current_size_bytes / self.max_size_bytes,
            }


class DiskCache:
    """Persistent disk cache for large objects."""
    
    def __init__(self, cache_dir: Path, compression_level: int = 6):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.compression_level = compression_level
        self.lock = threading.RLock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from disk cache."""
        cache_file = self._get_cache_file(key)
        
        if not cache_file.exists():
            return None
        
        try:
            with self.lock:
                import gzip
                with gzip.open(cache_file, 'rb', compresslevel=self.compression_level) as f:
                    return pickle.load(f)
        except Exception:
            # Remove corrupted cache file
            cache_file.unlink(missing_ok=True)
            return None
    
    def put(self, key: str, value: Any):
        """Put value in disk cache."""
        cache_file = self._get_cache_file(key)
        
        try:
            with self.lock:
                import gzip
                with gzip.open(cache_file, 'wb', compresslevel=self.compression_level) as f:
                    pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            # Remove failed cache file
            cache_file.unlink(missing_ok=True)
    
    def _get_cache_file(self, key: str) -> Path:
        """Get cache file path for key."""
        # Use hash to create safe filename
        key_hash = hashlib.md5(key.encode()).hexdigest()
        return self.cache_dir / f"{key_hash}.cache"
    
    def clear(self):
        """Clear disk cache."""
        with self.lock:
            for cache_file in self.cache_dir.glob("*.cache"):
                cache_file.unlink(missing_ok=True)


class CacheManager:
    """
    Multi-level cache manager for RevNet-Zero.
    
    Provides intelligent caching of computations, activations,
    and intermediate results to improve performance.
    """
    
    def __init__(self, config: CacheConfig = None):
        """
        Initialize cache manager.
        
        Args:
            config: Cache configuration
        """
        self.config = config or CacheConfig()
        
        # Initialize memory cache
        self.memory_cache = MemoryCache(self.config)
        
        # Initialize disk cache if enabled
        self.disk_cache = None
        if self.config.enable_disk_cache:
            cache_dir = Path(self.config.disk_cache_dir or "./cache")
            self.disk_cache = DiskCache(cache_dir, self.config.compression_level)
        
        # Cache statistics
        self.start_time = time.time()
    
    def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache (memory first, then disk).
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        # Try memory cache first
        value = self.memory_cache.get(key)
        if value is not None:
            return value
        
        # Try disk cache if enabled
        if self.disk_cache is not None:
            value = self.disk_cache.get(key)
            if value is not None:
                # Promote to memory cache
                self.memory_cache.put(key, value)
                return value
        
        return None
    
    def put(self, key: str, value: Any, cache_level: str = "memory") -> bool:
        """
        Put value in cache.
        
        Args:
            key: Cache key
            value: Value to cache
            cache_level: Cache level ("memory", "disk", or "both")
            
        Returns:
            True if successfully cached
        """
        success = False
        
        if cache_level in ["memory", "both"]:
            success = self.memory_cache.put(key, value) or success
        
        if cache_level in ["disk", "both"] and self.disk_cache is not None:
            self.disk_cache.put(key, value)
            success = True
        
        return success
    
    def cached_computation(self, key: str, computation_fn, *args, **kwargs):
        """
        Decorator for caching computation results.
        
        Args:
            key: Cache key
            computation_fn: Function to compute value if not cached
            *args, **kwargs: Arguments for computation function
            
        Returns:
            Cached or computed result
        """
        # Try to get from cache
        result = self.get(key)
        if result is not None:
            return result
        
        # Compute and cache
        result = computation_fn(*args, **kwargs)
        self.put(key, result)
        
        return result
    
    def cache_tensor_computation(
        self,
        inputs: Tuple[torch.Tensor, ...],
        computation_fn,
        operation_name: str,
    ) -> torch.Tensor:
        """
        Cache tensor computation based on input hashes.
        
        Args:
            inputs: Input tensors
            computation_fn: Computation function
            operation_name: Name of operation for key generation
            
        Returns:
            Cached or computed tensor
        """
        # Generate cache key from input tensor properties
        key_parts = [operation_name]
        for i, tensor in enumerate(inputs):
            if isinstance(tensor, torch.Tensor):
                # Use shape, dtype, and a hash of the data
                key_parts.append(f"t{i}_{tensor.shape}_{tensor.dtype}")
                # For small tensors, hash the actual data
                if tensor.numel() < 1000:
                    key_parts.append(str(hash(tensor.data.tobytes())))
                else:
                    # For large tensors, just use shape and a few sample values
                    sample_hash = hash(tensor.flatten()[:10].data.tobytes())
                    key_parts.append(str(sample_hash))
        
        cache_key = "_".join(key_parts)
        
        return self.cached_computation(cache_key, computation_fn, *inputs)
    
    def clear_cache(self, level: str = "both"):
        """
        Clear cache.
        
        Args:
            level: Cache level to clear ("memory", "disk", or "both")
        """
        if level in ["memory", "both"]:
            self.memory_cache.clear()
        
        if level in ["disk", "both"] and self.disk_cache is not None:
            self.disk_cache.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get comprehensive cache statistics."""
        memory_stats = self.memory_cache.get_stats()
        
        stats = {
            "memory_cache": memory_stats,
            "uptime_hours": (time.time() - self.start_time) / 3600,
            "config": {
                "max_memory_mb": self.config.max_memory_mb,
                "max_entries": self.config.max_entries,
                "ttl_seconds": self.config.ttl_seconds,
                "disk_cache_enabled": self.disk_cache is not None,
            }
        }
        
        return stats
    
    def optimize_cache(self):
        """Optimize cache performance based on usage patterns."""
        stats = self.memory_cache.get_stats()
        
        # If hit rate is too low, consider clearing cache
        if stats["hit_rate"] < self.config.cache_hit_threshold:
            self.memory_cache.clear()
        
        # If utilization is high but hit rate is good, consider increasing cache size
        if stats["utilization"] > 0.9 and stats["hit_rate"] > 0.5:
            # Could dynamically increase cache size here
            pass
    
    def warmup_cache(self, warmup_computations: List[Tuple[str, callable, tuple]]):
        """
        Warm up cache with common computations.
        
        Args:
            warmup_computations: List of (key, function, args) tuples
        """
        for key, func, args in warmup_computations:
            if self.get(key) is None:  # Only compute if not already cached
                result = func(*args)
                self.put(key, result)


# Global cache manager instance
_global_cache_manager = None


def get_cache_manager(config: CacheConfig = None) -> CacheManager:
    """Get or create global cache manager."""
    global _global_cache_manager
    if _global_cache_manager is None:
        _global_cache_manager = CacheManager(config)
    return _global_cache_manager


def cached_forward(cache_key_fn=None):
    """
    Decorator for caching forward pass results.
    
    Args:
        cache_key_fn: Function to generate cache key from inputs
        
    Returns:
        Decorated forward function
    """
    def decorator(forward_fn):
        def wrapper(self, *args, **kwargs):
            cache_manager = get_cache_manager()
            
            # Generate cache key
            if cache_key_fn:
                cache_key = cache_key_fn(self, *args, **kwargs)
            else:
                # Default key generation
                cache_key = f"{self.__class__.__name__}_{hash(str(args))}_{hash(str(kwargs))}"
            
            # Try cache first
            result = cache_manager.get(cache_key)
            if result is not None:
                return result
            
            # Compute and cache
            result = forward_fn(self, *args, **kwargs)
            cache_manager.put(cache_key, result)
            
            return result
        
        return wrapper
    return decorator