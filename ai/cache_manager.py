"""
Sliding Window Cache Manager
Memory-efficient caching for infinite movie stream

PRINCIPLE: Keep only 50-100 movies in memory at a time
When buffer gets low → fetch next batch
Discard old unused ones (LRU eviction)
"""

from collections import OrderedDict
from datetime import datetime
import time
import logging

from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SlidingWindowCache:
    """
    LRU Cache for movies with automatic eviction
    
    This is the "conveyor belt" - movies slide through memory
    """
    
    def __init__(self, max_size=Config.MOVIE_CACHE_SIZE):
        """
        Args:
            max_size: Maximum number of movies to keep in memory
        """
        self.max_size = max_size
        self.cache = OrderedDict()  # Maintains insertion order
        self.access_count = {}  # Track how many times each movie is accessed
        self.last_access = {}  # Track last access time
        # Hit/miss counters (cumulative, consumed by the cache monitor endpoint)
        self.hits = 0
        self.misses = 0
        
    def get(self, movie_id):
        """
        Get movie from cache
        
        Updates access time (moves to end of OrderedDict)
        This implements LRU behavior
        """
        if movie_id in self.cache:
            # Move to end (most recently used)
            self.cache.move_to_end(movie_id)
            self.access_count[movie_id] = self.access_count.get(movie_id, 0) + 1
            self.last_access[movie_id] = time.time()
            self.hits += 1
            return self.cache[movie_id]
        self.misses += 1
        return None
    
    def put(self, movie_id, movie_data):
        """
        Add movie to cache
        
        If cache is full, evict least recently used movie
        """
        if movie_id in self.cache:
            # Update existing entry
            self.cache.move_to_end(movie_id)
            self.cache[movie_id] = movie_data
        else:
            # Add new entry
            self.cache[movie_id] = movie_data
            self.access_count[movie_id] = 1
            self.last_access[movie_id] = time.time()
            
            # Evict if necessary
            if len(self.cache) > self.max_size:
                self._evict_lru()
    
    def _evict_lru(self):
        """Evict least recently used movie"""
        # Remove first item (least recently used)
        evicted_id, evicted_data = self.cache.popitem(last=False)
        
        # Clean up tracking data
        if evicted_id in self.access_count:
            del self.access_count[evicted_id]
        if evicted_id in self.last_access:
            del self.last_access[evicted_id]
        
        logger.debug(f"Evicted movie {evicted_id} from cache (LRU)")
    
    def bulk_put(self, movies):
        """
        Add multiple movies at once
        
        Args:
            movies: List of dicts with 'id' or 'movie_id' key
        """
        for movie in movies:
            movie_id = movie.get('id') or movie.get('movie_id')
            if movie_id:
                self.put(movie_id, movie)
    
    def contains(self, movie_id):
        """Check if movie is in cache"""
        return movie_id in self.cache
    
    def size(self):
        """Get current cache size"""
        return len(self.cache)
    
    def is_low(self, threshold=Config.CACHE_REFILL_THRESHOLD):
        """
        Check if cache is running low
        
        Args:
            threshold: Proportion (0-1) below which cache is considered "low"
        
        Returns:
            bool: True if cache should be refilled
        """
        return len(self.cache) < (self.max_size * threshold)
    
    def get_all_ids(self):
        """Get all movie IDs currently in cache"""
        return list(self.cache.keys())
    
    def get_all_movies(self):
        """Get all movies currently in cache"""
        return list(self.cache.values())
    
    def clear(self):
        """Clear entire cache"""
        self.cache.clear()
        self.access_count.clear()
        self.last_access.clear()
    
    def stats(self):
        """Get cache statistics"""
        total = self.hits + self.misses
        return {
            'size': len(self.cache),
            'max_size': self.max_size,
            'utilization': len(self.cache) / self.max_size if self.max_size > 0 else 0,
            'total_access_count': sum(self.access_count.values()),
            'avg_access_per_movie': sum(self.access_count.values()) / len(self.cache) if self.cache else 0,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': (self.hits / total * Config.PERCENT_SCALE) if total > 0 else 0.0,
        }


class VectorCache:
    """
    Cache for movie embeddings (vectors)
    
    Stores only:
    - movie_id
    - vector (small numeric array)
    
    NOT full movie objects!
    """
    
    def __init__(self, max_size=Config.VECTOR_CACHE_SIZE):
        """
        Args:
            max_size: Maximum number of vectors to cache
        """
        self.max_size = max_size
        self.vectors = OrderedDict()  # movie_id -> vector
        # Hit/miss counters (cumulative, consumed by the cache monitor endpoint)
        self.hits = 0
        self.misses = 0
    
    def get(self, movie_id):
        """Get vector for movie"""
        if movie_id in self.vectors:
            self.vectors.move_to_end(movie_id)
            self.hits += 1
            return self.vectors[movie_id]
        self.misses += 1
        return None
    
    def put(self, movie_id, vector):
        """Store (or overwrite) vector for movie"""
        if movie_id in self.vectors:
            # Overwrite the stale embedding and refresh LRU position
            self.vectors[movie_id] = vector
            self.vectors.move_to_end(movie_id)
        else:
            self.vectors[movie_id] = vector
            
            # Evict if necessary
            if len(self.vectors) > self.max_size:
                self.vectors.popitem(last=False)
    
    def bulk_put(self, vectors_dict):
        """
        Store multiple vectors
        
        Args:
            vectors_dict: {movie_id: vector, ...}
        """
        for movie_id, vector in vectors_dict.items():
            self.put(movie_id, vector)
    
    def contains(self, movie_id):
        """Check if vector exists"""
        return movie_id in self.vectors
    
    def size(self):
        """Get cache size"""
        return len(self.vectors)
    
    def stats(self):
        """Get vector cache statistics"""
        total = self.hits + self.misses
        return {
            'size': len(self.vectors),
            'max_size': self.max_size,
            'utilization': len(self.vectors) / self.max_size if self.max_size > 0 else 0,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': (self.hits / total * Config.PERCENT_SCALE) if total > 0 else 0.0,
        }
    
    def clear(self):
        """Clear all vectors"""
        self.vectors.clear()


class CacheManager:
    """
    Unified cache management
    
    Manages both:
    1. Movie data cache (sliding window)
    2. Vector cache (embeddings)
    """
    
    def __init__(self, movie_cache_size=Config.MOVIE_CACHE_SIZE, vector_cache_size=Config.VECTOR_CACHE_SIZE):
        self.movie_cache = SlidingWindowCache(max_size=movie_cache_size)
        self.vector_cache = VectorCache(max_size=vector_cache_size)
        
        logger.info(f"Cache initialized: {movie_cache_size} movies, {vector_cache_size} vectors")
    
    def get_movie(self, movie_id):
        """Get movie data"""
        return self.movie_cache.get(movie_id)
    
    def put_movie(self, movie_id, movie_data):
        """Store movie data"""
        self.movie_cache.put(movie_id, movie_data)
    
    def get_vector(self, movie_id):
        """Get movie vector"""
        return self.vector_cache.get(movie_id)
    
    def put_vector(self, movie_id, vector):
        """Store movie vector"""
        self.vector_cache.put(movie_id, vector)
    
    def is_movie_cached(self, movie_id):
        """Check if movie is in cache"""
        return self.movie_cache.contains(movie_id)
    
    def is_vector_cached(self, movie_id):
        """Check if vector is cached"""
        return self.vector_cache.contains(movie_id)
    
    def needs_refill(self, threshold=Config.CACHE_REFILL_THRESHOLD):
        """Check if movie cache needs refilling"""
        return self.movie_cache.is_low(threshold)
    
    def get_stats(self):
        """
        Get cache statistics.

        Returns the nested ``movie_cache``/``vector_cache`` blocks for internal
        callers *and* flat top-level counters (``movie_count``, ``movie_hits``,
        ``movie_misses``, ``vector_count``, ``vector_hits``, ``vector_misses``,
        ``timestamp``) that the API cache-monitor endpoint expects.
        """
        movie_stats = self.movie_cache.stats()
        vector_stats = self.vector_cache.stats()
        return {
            'movie_cache': movie_stats,
            'vector_cache': vector_stats,
            # Flat aliases consumed by /api/cache/monitor and the JS widget
            'movie_count': movie_stats['size'],
            'movie_hits': movie_stats.get('hits', 0),
            'movie_misses': movie_stats.get('misses', 0),
            'vector_count': vector_stats['size'],
            'vector_hits': vector_stats.get('hits', 0),
            'vector_misses': vector_stats.get('misses', 0),
            'timestamp': datetime.now().isoformat(),
        }
    
    def clear_all(self):
        """Clear all caches"""
        self.movie_cache.clear()
        self.vector_cache.clear()


# Global cache instance
cache_manager = CacheManager()


if __name__ == "__main__":
    # Test cache behavior
    print("Testing Sliding Window Cache...")
    
    cache = SlidingWindowCache(max_size=5)
    
    # Add movies
    for i in range(1, 8):
        movie = {'id': i, 'title': f'Movie {i}'}
        cache.put(i, movie)
        print(f"Added movie {i}, cache size: {cache.size()}")
    
    print(f"\nFinal cache IDs: {cache.get_all_ids()}")
    print("Movies 1-2 should be evicted (LRU)")
    
    # Access movie 3 (makes it most recently used)
    cache.get(3)
    print("\nAccessed movie 3")
    
    # Add one more movie
    cache.put(8, {'id': 8, 'title': 'Movie 8'})
    print(f"Added movie 8, cache size: {cache.size()}")
    print(f"Final cache IDs: {cache.get_all_ids()}")
    print("Movie 4 should be evicted, Movie 3 should remain")
    
    print(f"\nCache stats: {cache.stats()}")
