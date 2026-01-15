"""
Candidate Generation System
Real AI Trick: Don't score ALL movies, generate 200-500 candidates first

STRATEGY: Like a funnel
1. Infinite movies (TMDB API)
2. Generate 200-500 candidates (SQL + API filtering)
3. Rank only those candidates
4. Return top 20-30

This is how production systems work!
"""

import random
import logging
from tmdb.fetcher import TMDBFetcher
from ai.cache_manager import cache_manager
from database.db_manager import DatabaseManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CandidateGenerator:
    """
    Generates movie candidates for recommendation
    
    Uses multiple strategies:
    - Genre-based filtering
    - Director-based similarity
    - Popularity-based discovery
    - User history-based expansion
    """
    
    def __init__(self, tmdb_fetcher=None, db_manager=None):
        self.tmdb = tmdb_fetcher or TMDBFetcher()
        self.db = db_manager or DatabaseManager()
    
    def generate_candidates(self, user_id=None, target_count=300, strategy='mixed'):
        """
        Generate candidate movies for recommendation
        
        Args:
            user_id: User ID (optional, for personalized candidates)
            target_count: Number of candidates to generate (200-500)
            strategy: 'mixed', 'genre', 'popularity', 'exploration'
        
        Returns:
            list: Movie candidates (limited to target_count)
        """
        candidates = []
        candidate_ids = set()
        
        logger.info(f"Generating {target_count} candidates using '{strategy}' strategy...")
        
        if strategy == 'mixed':
            # 40% genre-based
            genre_candidates = self._generate_by_genre(user_id, int(target_count * 0.4))
            candidates.extend(genre_candidates)
            candidate_ids.update([m.get('id') for m in genre_candidates if m.get('id')])
            
            # 30% popularity-based
            popular_candidates = self._generate_by_popularity(int(target_count * 0.3))
            for movie in popular_candidates:
                if movie.get('id') not in candidate_ids:
                    candidates.append(movie)
                    candidate_ids.add(movie.get('id'))
            
            # 20% exploration (random discovery)
            explore_candidates = self._generate_exploration(int(target_count * 0.2))
            for movie in explore_candidates:
                if movie.get('id') not in candidate_ids:
                    candidates.append(movie)
                    candidate_ids.add(movie.get('id'))
            
            # 10% from cache (fast access)
            cache_candidates = self._generate_from_cache(int(target_count * 0.1))
            for movie in cache_candidates:
                if movie.get('id') not in candidate_ids:
                    candidates.append(movie)
                    candidate_ids.add(movie.get('id'))
        
        elif strategy == 'genre':
            candidates = self._generate_by_genre(user_id, target_count)
        
        elif strategy == 'popularity':
            candidates = self._generate_by_popularity(target_count)
        
        elif strategy == 'exploration':
            candidates = self._generate_exploration(target_count)
        
        # Trim to target count
        candidates = candidates[:target_count]
        
        logger.info(f"Generated {len(candidates)} unique candidates")
        
        # Cache the candidates for fast access
        cache_manager.movie_cache.bulk_put(candidates)
        
        return candidates
    
    def _generate_by_genre(self, user_id, count):
        """
        Generate candidates based on user's preferred genres
        
        If no user_id, use diverse genre selection
        """
        candidates = []
        
        if user_id:
            # Get user's favorite genres from interactions
            user_genres = self._get_user_favorite_genres(user_id)
        else:
            # Use diverse popular genres
            user_genres = self._get_popular_genres()
        
        if not user_genres:
            # Fallback to popular movies
            return self._generate_by_popularity(count)
        
        # Fetch movies for each genre
        movies_per_genre = max(1, count // len(user_genres))
        
        for genre_id in user_genres[:5]:  # Top 5 genres
            try:
                # Use TMDB discovery API
                data = self.tmdb.discover_movies(
                    with_genres=genre_id,
                    sort_by='vote_average.desc',
                    vote_count_gte=100,
                    page=1
                )
                
                if data and 'results' in data:
                    candidates.extend(data['results'][:movies_per_genre])
                
                if len(candidates) >= count:
                    break
                    
            except Exception as e:
                logger.error(f"Error fetching genre {genre_id}: {e}")
                continue
        
        return candidates[:count]
    
    def _generate_by_popularity(self, count):
        """
        Generate candidates based on popularity
        
        Uses TMDB popular and top-rated endpoints
        """
        candidates = []
        pages_needed = (count // 20) + 1  # 20 movies per page
        
        for page in range(1, min(pages_needed + 1, 6)):  # Max 5 pages
            try:
                # Alternate between popular and top-rated
                if page % 2 == 1:
                    data = self.tmdb.get_popular_movies(page=page)
                else:
                    data = self.tmdb.get_top_rated_movies(page=page)
                
                if data and 'results' in data:
                    candidates.extend(data['results'])
                
                if len(candidates) >= count:
                    break
                    
            except Exception as e:
                logger.error(f"Error fetching popular movies page {page}: {e}")
                continue
        
        return candidates[:count]
    
    def _generate_exploration(self, count):
        """
        Generate random exploration candidates
        
        Uses random genres, years, etc. for discovery
        """
        candidates = []
        
        # Get all genres
        genres_data = self.tmdb.get_genres()
        if not genres_data or 'genres' not in genres_data:
            return self._generate_by_popularity(count)
        
        all_genres = [g['id'] for g in genres_data['genres']]
        
        # Random sampling
        pages_needed = (count // 20) + 1
        
        for _ in range(pages_needed):
            # Random genre
            genre = random.choice(all_genres)
            
            # Random sort method
            sort_methods = [
                'popularity.desc',
                'vote_average.desc',
                'release_date.desc',
                'revenue.desc'
            ]
            sort_by = random.choice(sort_methods)
            
            # Random page (1-3 for variety)
            page = random.randint(1, 3)
            
            try:
                data = self.tmdb.discover_movies(
                    with_genres=genre,
                    sort_by=sort_by,
                    page=page,
                    vote_count_gte=50
                )
                
                if data and 'results' in data:
                    candidates.extend(data['results'])
                
                if len(candidates) >= count:
                    break
                    
            except Exception as e:
                logger.error(f"Error in exploration: {e}")
                continue
        
        # Shuffle for randomness
        random.shuffle(candidates)
        return candidates[:count]
    
    def _generate_from_cache(self, count):
        """Get random movies from cache (fast!)"""
        cached_movies = cache_manager.movie_cache.get_all_movies()
        
        if not cached_movies:
            return []
        
        # Random sample
        sample_size = min(count, len(cached_movies))
        return random.sample(cached_movies, sample_size)
    
    def _get_user_favorite_genres(self, user_id):
        """
        Get user's favorite genres from interaction history
        
        Returns:
            list: Genre IDs sorted by preference
        """
        try:
            # Query user's interacted movies
            query = """
            SELECT g.genre_id, COUNT(*) as interaction_count
            FROM user_interactions ui
            JOIN movie_genres mg ON ui.movie_id = mg.movie_id
            JOIN genres g ON mg.genre_id = g.genre_id
            WHERE ui.user_id = %s AND ui.chosen = TRUE
            GROUP BY g.genre_id
            ORDER BY interaction_count DESC
            LIMIT 5
            """
            
            results = self.db.execute_query(query, (user_id,))
            
            if results:
                return [row['genre_id'] for row in results]
            
        except Exception as e:
            logger.error(f"Error getting user genres: {e}")
        
        return []
    
    def _get_popular_genres(self):
        """
        Get most popular genres overall
        
        Returns:
            list: Popular genre IDs
        """
        # Common popular genres
        popular_genre_ids = [
            28,   # Action
            12,   # Adventure
            16,   # Animation
            35,   # Comedy
            18,   # Drama
            14,   # Fantasy
            27,   # Horror
            10749,  # Romance
            878,  # Science Fiction
            53    # Thriller
        ]
        
        # Shuffle for variety
        random.shuffle(popular_genre_ids)
        return popular_genre_ids
    
    def generate_pairwise_candidates(self, user_id=None, count=30):
        """
        Generate candidates specifically for pairwise comparison
        
        INFINITE PAIRWISE STRATEGY:
        - 1 known-preference movie (from user history or popular)
        - 1 unexplored movie (from API)
        
        Args:
            user_id: User ID
            count: Number of movies to generate
        
        Returns:
            dict: {'known': [...], 'explore': [...]}
        """
        known_movies = []
        explore_movies = []
        
        if user_id:
            # Get movies user has interacted with
            try:
                query = """
                SELECT m.movie_id, m.title, m.tmdb_id
                FROM movies m
                JOIN user_interactions ui ON m.movie_id = ui.movie_id
                WHERE ui.user_id = %s AND ui.chosen = TRUE
                ORDER BY ui.interaction_time DESC
                LIMIT %s
                """
                
                results = self.db.execute_query(query, (user_id, count // 2))
                
                if results:
                    known_movies = [dict(row) for row in results]
                
            except Exception as e:
                logger.error(f"Error getting user interactions: {e}")
        
        # If not enough known movies, use popular ones
        if len(known_movies) < count // 2:
            popular_data = self.tmdb.get_popular_movies(page=1)
            if popular_data and 'results' in popular_data:
                needed = (count // 2) - len(known_movies)
                known_movies.extend(popular_data['results'][:needed])
        
        # Generate exploration candidates (new movies)
        explore_movies = self._generate_exploration(count // 2)
        
        return {
            'known': known_movies[:count // 2],
            'explore': explore_movies[:count // 2]
        }


# Global instance
candidate_generator = CandidateGenerator()


if __name__ == "__main__":
    # Test candidate generation
    print("Testing Candidate Generator...")
    
    gen = CandidateGenerator()
    
    # Generate mixed candidates
    candidates = gen.generate_candidates(target_count=50, strategy='mixed')
    
    print(f"\nGenerated {len(candidates)} candidates")
    print(f"Sample titles:")
    for movie in candidates[:5]:
        print(f"  - {movie.get('title', 'Unknown')} (ID: {movie.get('id')})")
    
    # Test pairwise candidates
    print("\n\nTesting pairwise candidates...")
    pairwise = gen.generate_pairwise_candidates(count=20)
    
    print(f"Known movies: {len(pairwise['known'])}")
    print(f"Explore movies: {len(pairwise['explore'])}")
