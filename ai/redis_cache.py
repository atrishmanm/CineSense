"""
Redis Caching System
Production-grade caching for recommendations, embeddings, and search results
"""

import redis
import pickle
import json
import hashlib
from typing import Optional, Any, List, Dict
import logging
from datetime import timedelta

logger = logging.getLogger(__name__)


class RedisCache:
    """
    Redis-based caching system for CineSense
    
    Features:
    - Recommendation caching
    - Embedding storage
    - Search result caching
    - User session data
    - Movie metadata caching
    """
    
    def __init__(
        self,
        host: str = 'localhost',
        port: int = 6379,
        db: int = 0,
        password: Optional[str] = None,
        default_ttl: int = 3600
    ):
        """
        Initialize Redis connection
        
        Args:
            host: Redis server host
            port: Redis server port
            db: Redis database number
            password: Redis password (if required)
            default_ttl: Default time-to-live in seconds (1 hour)
        """
        try:
            self.client = redis.Redis(
                host=host,
                port=port,
                db=db,
                password=password,
                decode_responses=False,  # For binary data (pickled objects)
                socket_connect_timeout=5,
                socket_timeout=5
            )
            
            # Test connection
            self.client.ping()
            
            self.default_ttl = default_ttl
            logger.info(f"✓ Redis connected: {host}:{port}")
        
        except redis.ConnectionError as e:
            logger.error(f"Failed to connect to Redis: {e}")
            self.client = None
        except Exception as e:
            logger.error(f"Redis initialization error: {e}")
            self.client = None
    
    def _make_key(self, prefix: str, *args) -> str:
        """
        Generate cache key
        
        Args:
            prefix: Key prefix (e.g., 'rec', 'emb', 'search')
            *args: Additional key components
            
        Returns:
            Cache key string
        """
        key_data = f"{prefix}:{'|'.join(map(str, args))}"
        
        # Hash long keys
        if len(key_data) > 200:
            hash_key = hashlib.md5(key_data.encode()).hexdigest()
            return f"{prefix}:{hash_key}"
        
        return key_data
    
    def is_available(self) -> bool:
        """Check if Redis is available"""
        if self.client is None:
            return False
        
        try:
            self.client.ping()
            return True
        except:
            return False
    
    # ========================================================================
    # RECOMMENDATIONS
    # ========================================================================
    
    def get_recommendations(self, user_id: int) -> Optional[List[Dict]]:
        """Get cached recommendations for user"""
        if not self.is_available():
            return None
        
        try:
            key = self._make_key('rec', user_id)
            data = self.client.get(key)
            
            if data:
                return pickle.loads(data)
            
            return None
        except Exception as e:
            logger.error(f"Error getting recommendations from cache: {e}")
            return None
    
    def set_recommendations(
        self,
        user_id: int,
        recommendations: List[Dict],
        ttl: Optional[int] = None
    ):
        """Cache recommendations for user"""
        if not self.is_available():
            return
        
        try:
            key = self._make_key('rec', user_id)
            ttl = ttl or self.default_ttl
            
            self.client.setex(
                key,
                ttl,
                pickle.dumps(recommendations)
            )
            
            logger.debug(f"Cached recommendations for user {user_id}")
        except Exception as e:
            logger.error(f"Error caching recommendations: {e}")
    
    # ========================================================================
    # EMBEDDINGS
    # ========================================================================
    
    def get_movie_embedding(self, movie_id: int) -> Optional[Any]:
        """Get cached movie embedding"""
        if not self.is_available():
            return None
        
        try:
            key = self._make_key('emb', 'movie', movie_id)
            data = self.client.get(key)
            
            if data:
                return pickle.loads(data)
            
            return None
        except Exception as e:
            logger.error(f"Error getting embedding from cache: {e}")
            return None
    
    def set_movie_embedding(
        self,
        movie_id: int,
        embedding: Any,
        ttl: Optional[int] = None
    ):
        """Cache movie embedding"""
        if not self.is_available():
            return
        
        try:
            key = self._make_key('emb', 'movie', movie_id)
            ttl = ttl or 86400  # 24 hours for embeddings
            
            self.client.setex(
                key,
                ttl,
                pickle.dumps(embedding)
            )
        except Exception as e:
            logger.error(f"Error caching embedding: {e}")
    
    def get_user_embedding(self, user_id: int) -> Optional[Any]:
        """Get cached user embedding"""
        if not self.is_available():
            return None
        
        try:
            key = self._make_key('emb', 'user', user_id)
            data = self.client.get(key)
            
            if data:
                return pickle.loads(data)
            
            return None
        except Exception as e:
            logger.error(f"Error getting user embedding: {e}")
            return None
    
    def set_user_embedding(
        self,
        user_id: int,
        embedding: Any,
        ttl: Optional[int] = None
    ):
        """Cache user embedding"""
        if not self.is_available():
            return
        
        try:
            key = self._make_key('emb', 'user', user_id)
            ttl = ttl or 7200  # 2 hours for user embeddings
            
            self.client.setex(
                key,
                ttl,
                pickle.dumps(embedding)
            )
        except Exception as e:
            logger.error(f"Error caching user embedding: {e}")
    
    # ========================================================================
    # SEARCH RESULTS
    # ========================================================================
    
    def get_search_results(self, query: str) -> Optional[List[Dict]]:
        """Get cached search results"""
        if not self.is_available():
            return None
        
        try:
            key = self._make_key('search', query)
            data = self.client.get(key)
            
            if data:
                return pickle.loads(data)
            
            return None
        except Exception as e:
            logger.error(f"Error getting search results: {e}")
            return None
    
    def set_search_results(
        self,
        query: str,
        results: List[Dict],
        ttl: Optional[int] = None
    ):
        """Cache search results"""
        if not self.is_available():
            return
        
        try:
            key = self._make_key('search', query)
            ttl = ttl or 1800  # 30 minutes for search results
            
            self.client.setex(
                key,
                ttl,
                pickle.dumps(results)
            )
        except Exception as e:
            logger.error(f"Error caching search results: {e}")
    
    # ========================================================================
    # MOVIE METADATA
    # ========================================================================
    
    def get_movie(self, movie_id: int) -> Optional[Dict]:
        """Get cached movie metadata"""
        if not self.is_available():
            return None
        
        try:
            key = self._make_key('movie', movie_id)
            data = self.client.get(key)
            
            if data:
                return json.loads(data)
            
            return None
        except Exception as e:
            logger.error(f"Error getting movie from cache: {e}")
            return None
    
    def set_movie(
        self,
        movie_id: int,
        movie_data: Dict,
        ttl: Optional[int] = None
    ):
        """Cache movie metadata"""
        if not self.is_available():
            return
        
        try:
            key = self._make_key('movie', movie_id)
            ttl = ttl or 86400  # 24 hours
            
            self.client.setex(
                key,
                ttl,
                json.dumps(movie_data)
            )
        except Exception as e:
            logger.error(f"Error caching movie: {e}")
    
    # ========================================================================
    # INVALIDATION
    # ========================================================================
    
    def invalidate_user(self, user_id: int):
        """Clear all cache for a user (after new interaction)"""
        if not self.is_available():
            return
        
        try:
            # Delete user's recommendations
            rec_key = self._make_key('rec', user_id)
            self.client.delete(rec_key)
            
            # Delete user's embedding
            emb_key = self._make_key('emb', 'user', user_id)
            self.client.delete(emb_key)
            
            logger.debug(f"Invalidated cache for user {user_id}")
        except Exception as e:
            logger.error(f"Error invalidating user cache: {e}")
    
    def invalidate_movie(self, movie_id: int):
        """Clear all cache for a movie (after metadata update)"""
        if not self.is_available():
            return
        
        try:
            movie_key = self._make_key('movie', movie_id)
            emb_key = self._make_key('emb', 'movie', movie_id)
            
            self.client.delete(movie_key)
            self.client.delete(emb_key)
            
            logger.debug(f"Invalidated cache for movie {movie_id}")
        except Exception as e:
            logger.error(f"Error invalidating movie cache: {e}")
    
    def clear_all(self):
        """Clear entire cache (use with caution!)"""
        if not self.is_available():
            return
        
        try:
            self.client.flushdb()
            logger.warning("Cleared entire Redis cache!")
        except Exception as e:
            logger.error(f"Error clearing cache: {e}")
    
    # ========================================================================
    # STATISTICS & MONITORING
    # ========================================================================
    
    def get_stats(self) -> Dict:
        """Get cache statistics"""
        if not self.is_available():
            return {'available': False}
        
        try:
            info = self.client.info()
            
            return {
                'available': True,
                'used_memory': info.get('used_memory_human', 'N/A'),
                'total_keys': self.client.dbsize(),
                'hits': info.get('keyspace_hits', 0),
                'misses': info.get('keyspace_misses', 0),
                'hit_rate': self._calculate_hit_rate(
                    info.get('keyspace_hits', 0),
                    info.get('keyspace_misses', 0)
                ),
                'connected_clients': info.get('connected_clients', 0)
            }
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return {'available': False, 'error': str(e)}
    
    def _calculate_hit_rate(self, hits: int, misses: int) -> float:
        """Calculate cache hit rate"""
        total = hits + misses
        if total == 0:
            return 0.0
        return (hits / total) * 100
    
    # ========================================================================
    # SESSION MANAGEMENT
    # ========================================================================
    
    def set_session(
        self,
        session_id: str,
        session_data: Dict,
        ttl: int = 86400
    ):
        """Store user session data"""
        if not self.is_available():
            return
        
        try:
            key = self._make_key('session', session_id)
            self.client.setex(
                key,
                ttl,
                json.dumps(session_data)
            )
        except Exception as e:
            logger.error(f"Error setting session: {e}")
    
    def get_session(self, session_id: str) -> Optional[Dict]:
        """Get user session data"""
        if not self.is_available():
            return None
        
        try:
            key = self._make_key('session', session_id)
            data = self.client.get(key)
            
            if data:
                return json.loads(data)
            
            return None
        except Exception as e:
            logger.error(f"Error getting session: {e}")
            return None


# Singleton instance
_redis_cache = None


def get_redis_cache(**kwargs) -> RedisCache:
    """Get global Redis cache instance.

    Accepts optional RedisCache constructor kwargs so callers can configure
    host/port/db/password without directly instantiating the class.
    """
    global _redis_cache
    
    if _redis_cache is None:
        _redis_cache = RedisCache(**kwargs)
    
    return _redis_cache


# Example usage
if __name__ == '__main__':
    cache = RedisCache()
    
    print("Redis Cache System")
    print("=" * 60)
    
    if cache.is_available():
        print("✓ Redis is available")
        
        # Test caching
        test_recommendations = [
            {'movie_id': 1, 'title': 'Inception', 'score': 4.5},
            {'movie_id': 2, 'title': 'The Matrix', 'score': 4.3}
        ]
        
        # Cache recommendations
        cache.set_recommendations(123, test_recommendations)
        
        # Retrieve recommendations
        cached = cache.get_recommendations(123)
        print(f"\nCached recommendations: {cached}")
        
        # Get stats
        stats = cache.get_stats()
        print(f"\nCache stats:")
        print(f"  Keys: {stats.get('total_keys')}")
        print(f"  Memory: {stats.get('used_memory')}")
        print(f"  Hit rate: {stats.get('hit_rate', 0):.2f}%")
    else:
        print("✗ Redis not available")
        print("Install: pip install redis")
        print("Start Redis: redis-server (or use Docker)")
