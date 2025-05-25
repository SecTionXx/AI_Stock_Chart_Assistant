"""
Smart Cache Manager for AI Stock Chart Assistant v2.0

Provides intelligent caching with:
- Local disk cache with TTL
- Memory cache for hot data
- Cache statistics and monitoring
- Automatic cleanup and optimization
"""

import asyncio
import json
import logging
import os
import pickle
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Union
import hashlib
import threading
from pathlib import Path

import diskcache as dc


@dataclass
class CacheStats:
    """Cache performance statistics"""
    hits: int = 0
    misses: int = 0
    sets: int = 0
    deletes: int = 0
    evictions: int = 0
    total_size: int = 0
    hit_rate: float = 0.0


class CacheManager:
    """
    Multi-level cache manager with disk and memory storage
    """
    
    def __init__(self, 
                 cache_dir: str = "data/cache",
                 max_memory_size: int = 100 * 1024 * 1024,  # 100MB
                 max_disk_size: int = 1024 * 1024 * 1024,   # 1GB
                 default_ttl: int = 3600):  # 1 hour
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_memory_size = max_memory_size
        self.max_disk_size = max_disk_size
        self.default_ttl = default_ttl
        
        # Initialize disk cache
        self.disk_cache = dc.Cache(
            str(self.cache_dir),
            size_limit=max_disk_size,
            eviction_policy='least-recently-used'
        )
        
        # Memory cache for hot data
        self.memory_cache: Dict[str, Dict[str, Any]] = {}
        self.memory_access_times: Dict[str, float] = {}
        self.memory_size = 0
        
        # Statistics
        self.stats = CacheStats()
        
        # Thread lock for memory cache
        self._lock = threading.RLock()
        
        # Logger
        self.logger = logging.getLogger(__name__)
        
        # Start cleanup task
        self._cleanup_task = None
        self._start_cleanup_task()
    
    async def get(self, key: str) -> Optional[Any]:
        """
        Get value from cache (memory first, then disk)
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found/expired
        """
        # Check memory cache first
        with self._lock:
            if key in self.memory_cache:
                entry = self.memory_cache[key]
                
                # Check if expired
                if entry['expires_at'] > time.time():
                    self.memory_access_times[key] = time.time()
                    self.stats.hits += 1
                    self.logger.debug(f"Memory cache hit for key: {key}")
                    return entry['value']
                else:
                    # Expired, remove from memory
                    self._remove_from_memory(key)
        
        # Check disk cache
        try:
            value = self.disk_cache.get(key)
            if value is not None:
                # Move to memory cache if it's hot data
                await self._promote_to_memory(key, value)
                self.stats.hits += 1
                self.logger.debug(f"Disk cache hit for key: {key}")
                return value
        except Exception as e:
            self.logger.error(f"Error reading from disk cache: {e}")
        
        # Cache miss
        self.stats.misses += 1
        self.logger.debug(f"Cache miss for key: {key}")
        return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set value in cache
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (default: use default_ttl)
            
        Returns:
            True if successful
        """
        if ttl is None:
            ttl = self.default_ttl
        
        expires_at = time.time() + ttl
        
        try:
            # Store in disk cache
            self.disk_cache.set(key, value, expire=ttl)
            
            # Store in memory cache if small enough
            value_size = self._estimate_size(value)
            if value_size < self.max_memory_size // 10:  # Max 10% of memory for single item
                await self._add_to_memory(key, value, expires_at, value_size)
            
            self.stats.sets += 1
            self.logger.debug(f"Cached value for key: {key}, TTL: {ttl}s")
            return True
            
        except Exception as e:
            self.logger.error(f"Error setting cache for key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        try:
            # Remove from memory
            with self._lock:
                if key in self.memory_cache:
                    self._remove_from_memory(key)
            
            # Remove from disk
            if key in self.disk_cache:
                del self.disk_cache[key]
            
            self.stats.deletes += 1
            return True
            
        except Exception as e:
            self.logger.error(f"Error deleting cache key {key}: {e}")
            return False
    
    async def clear(self) -> bool:
        """Clear all cache"""
        try:
            # Clear memory cache
            with self._lock:
                self.memory_cache.clear()
                self.memory_access_times.clear()
                self.memory_size = 0
            
            # Clear disk cache
            self.disk_cache.clear()
            
            self.logger.info("Cache cleared")
            return True
            
        except Exception as e:
            self.logger.error(f"Error clearing cache: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        total_requests = self.stats.hits + self.stats.misses
        hit_rate = (self.stats.hits / total_requests * 100) if total_requests > 0 else 0
        
        with self._lock:
            memory_items = len(self.memory_cache)
        
        disk_items = len(self.disk_cache)
        disk_size = self.disk_cache.volume()
        
        return {
            "hit_rate": hit_rate,
            "total_requests": total_requests,
            "hits": self.stats.hits,
            "misses": self.stats.misses,
            "sets": self.stats.sets,
            "deletes": self.stats.deletes,
            "evictions": self.stats.evictions,
            "memory_items": memory_items,
            "memory_size": self.memory_size,
            "disk_items": disk_items,
            "disk_size": disk_size,
            "max_memory_size": self.max_memory_size,
            "max_disk_size": self.max_disk_size
        }
    
    async def _promote_to_memory(self, key: str, value: Any):
        """Promote frequently accessed item to memory cache"""
        value_size = self._estimate_size(value)
        
        # Only promote if it fits in memory budget
        if value_size < self.max_memory_size // 10:
            expires_at = time.time() + self.default_ttl
            await self._add_to_memory(key, value, expires_at, value_size)
    
    async def _add_to_memory(self, key: str, value: Any, expires_at: float, size: int):
        """Add item to memory cache with size management"""
        with self._lock:
            # Make room if necessary
            while (self.memory_size + size > self.max_memory_size and 
                   len(self.memory_cache) > 0):
                await self._evict_lru_memory()
            
            # Add to memory cache
            self.memory_cache[key] = {
                'value': value,
                'expires_at': expires_at,
                'size': size
            }
            self.memory_access_times[key] = time.time()
            self.memory_size += size
    
    def _remove_from_memory(self, key: str):
        """Remove item from memory cache"""
        if key in self.memory_cache:
            size = self.memory_cache[key]['size']
            del self.memory_cache[key]
            del self.memory_access_times[key]
            self.memory_size -= size
    
    async def _evict_lru_memory(self):
        """Evict least recently used item from memory"""
        if not self.memory_access_times:
            return
        
        # Find LRU item
        lru_key = min(self.memory_access_times.items(), key=lambda x: x[1])[0]
        self._remove_from_memory(lru_key)
        self.stats.evictions += 1
        self.logger.debug(f"Evicted LRU item from memory: {lru_key}")
    
    def _estimate_size(self, obj: Any) -> int:
        """Estimate object size in bytes"""
        try:
            if isinstance(obj, (str, bytes)):
                return len(obj)
            elif isinstance(obj, (int, float)):
                return 8
            elif isinstance(obj, dict):
                return len(pickle.dumps(obj))
            else:
                return len(pickle.dumps(obj))
        except:
            return 1024  # Default estimate
    
    def _start_cleanup_task(self):
        """Start background cleanup task"""
        async def cleanup_loop():
            while True:
                try:
                    await asyncio.sleep(300)  # Run every 5 minutes
                    await self._cleanup_expired()
                except Exception as e:
                    self.logger.error(f"Error in cleanup task: {e}")
        
        # Start cleanup task if not already running
        if self._cleanup_task is None or self._cleanup_task.done():
            self._cleanup_task = asyncio.create_task(cleanup_loop())
    
    async def _cleanup_expired(self):
        """Clean up expired items from memory cache"""
        current_time = time.time()
        expired_keys = []
        
        with self._lock:
            for key, entry in self.memory_cache.items():
                if entry['expires_at'] <= current_time:
                    expired_keys.append(key)
        
        # Remove expired items
        for key in expired_keys:
            with self._lock:
                if key in self.memory_cache:
                    self._remove_from_memory(key)
        
        if expired_keys:
            self.logger.debug(f"Cleaned up {len(expired_keys)} expired items from memory")
    
    def close(self):
        """Close cache and cleanup resources"""
        try:
            if self._cleanup_task and not self._cleanup_task.done():
                self._cleanup_task.cancel()
            
            self.disk_cache.close()
            
            with self._lock:
                self.memory_cache.clear()
                self.memory_access_times.clear()
                self.memory_size = 0
            
            self.logger.info("Cache manager closed")
            
        except Exception as e:
            self.logger.error(f"Error closing cache manager: {e}")


class ImageCacheManager(CacheManager):
    """
    Specialized cache manager for image analysis results
    """
    
    def __init__(self, cache_dir: str = "data/cache/images", **kwargs):
        super().__init__(cache_dir=cache_dir, **kwargs)
    
    def generate_image_key(self, image_path: str, analysis_params: Dict[str, Any]) -> str:
        """Generate cache key for image analysis"""
        # Create hash from image content and parameters
        hasher = hashlib.md5()
        
        # Add image hash
        try:
            with open(image_path, 'rb') as f:
                hasher.update(f.read())
        except Exception:
            hasher.update(image_path.encode())
        
        # Add parameters hash
        params_str = json.dumps(analysis_params, sort_keys=True)
        hasher.update(params_str.encode())
        
        return f"img_{hasher.hexdigest()}"
    
    async def cache_analysis(self, 
                           image_path: str, 
                           analysis_params: Dict[str, Any],
                           analysis_result: Any,
                           ttl: int = 7200) -> str:  # 2 hours default
        """Cache image analysis result"""
        cache_key = self.generate_image_key(image_path, analysis_params)
        await self.set(cache_key, analysis_result, ttl)
        return cache_key
    
    async def get_cached_analysis(self, 
                                image_path: str, 
                                analysis_params: Dict[str, Any]) -> Optional[Any]:
        """Get cached image analysis result"""
        cache_key = self.generate_image_key(image_path, analysis_params)
        return await self.get(cache_key)


class ModelCacheManager(CacheManager):
    """
    Specialized cache manager for AI model responses
    """
    
    def __init__(self, cache_dir: str = "data/cache/models", **kwargs):
        super().__init__(cache_dir=cache_dir, **kwargs)
    
    def generate_model_key(self, 
                          model_name: str, 
                          prompt: str, 
                          image_hash: str) -> str:
        """Generate cache key for model response"""
        hasher = hashlib.md5()
        hasher.update(f"{model_name}:{prompt}:{image_hash}".encode())
        return f"model_{hasher.hexdigest()}"
    
    async def cache_model_response(self,
                                 model_name: str,
                                 prompt: str,
                                 image_hash: str,
                                 response: Any,
                                 ttl: int = 3600) -> str:  # 1 hour default
        """Cache model response"""
        cache_key = self.generate_model_key(model_name, prompt, image_hash)
        await self.set(cache_key, response, ttl)
        return cache_key
    
    async def get_cached_model_response(self,
                                      model_name: str,
                                      prompt: str,
                                      image_hash: str) -> Optional[Any]:
        """Get cached model response"""
        cache_key = self.generate_model_key(model_name, prompt, image_hash)
        return await self.get(cache_key) 