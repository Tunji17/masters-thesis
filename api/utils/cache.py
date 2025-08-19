"""
Caching utilities for the Medical Graph Extraction API
"""
from functools import lru_cache
from typing import Dict, Optional, Any
import hashlib


class EntityCache:
    """LRU Cache for entity linking results"""
    
    def __init__(self, maxsize: int = 10000):
        self.maxsize = maxsize
        self._cache = {}
        
    @lru_cache(maxsize=10000)
    def get_entity_lookup(self, entity_text: str) -> Optional[Dict[str, Any]]:
        """
        Cached entity lookup function.
        This will be implemented by the entity extraction service.
        """
        pass
    
    def clear_cache(self):
        """Clear the entity cache"""
        self.get_entity_lookup.cache_clear()
    
    def cache_info(self):
        """Get cache statistics"""
        return self.get_entity_lookup.cache_info()


def create_cache_key(*args, **kwargs) -> str:
    """Create a consistent cache key from arguments"""
    key_data = str(args) + str(sorted(kwargs.items()))
    return hashlib.md5(key_data.encode()).hexdigest()


# Global cache instance
entity_cache = EntityCache()