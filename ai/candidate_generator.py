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
from config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _movie_key(movie):
    """Stable identifier for de-duplication across DB ('movie_id') and TMDB ('id') items.

    Entries that have neither id fall back to (tmdb_id, title) so they cannot
    all collapse onto the same None key (A186) — previously every such row
    deduplicated to a single survivor.
    """
    key = movie.get('id') or movie.get('movie_id')
    if key is not None:
        return key
    return ('no-id', movie.get('tmdb_id'), movie.get('title'))


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
    
    def generate_candidates(self, user_id=None, target_count=Config.CANDIDATE_COUNT, strategy=Config.CANDIDATE_STRATEGY):
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
        
        # Unknown strategies silently produced [] before; fall back explicitly.
        valid_strategies = ('mixed', 'genre', 'popularity', 'exploration')
        if strategy not in valid_strategies:
            logger.warning(f"Unknown strategy '{strategy}', falling back to '{Config.CANDIDATE_STRATEGY}'")
            strategy = Config.CANDIDATE_STRATEGY
        
        logger.info(f"Generating {target_count} candidates using '{strategy}' strategy...")
        
        if strategy == 'mixed':
            # 40% genre-based
            genre_candidates = self._generate_by_genre(user_id, int(target_count * Config.CANDIDATE_FUNNEL_CONTENT_RATIO))
            candidates.extend(genre_candidates)
            candidate_ids.update([_movie_key(m) for m in genre_candidates if _movie_key(m)])
            
            # 30% popularity-based
            popular_candidates = self._generate_by_popularity(int(target_count * Config.CANDIDATE_FUNNEL_POPULARITY_RATIO))
            for movie in popular_candidates:
                if _movie_key(movie) not in candidate_ids:
                    candidates.append(movie)
                    candidate_ids.add(_movie_key(movie))
            
            # 20% exploration (random discovery)
            explore_candidates = self._generate_exploration(int(target_count * Config.CANDIDATE_FUNNEL_EXPLORATION_RATIO))
            for movie in explore_candidates:
                if _movie_key(movie) not in candidate_ids:
                    candidates.append(movie)
                    candidate_ids.add(_movie_key(movie))
            
            # 10% from cache (fast access)
            cache_candidates = self._generate_from_cache(int(target_count * Config.CANDIDATE_FUNNEL_CACHE_RATIO))
            for movie in cache_candidates:
                if _movie_key(movie) not in candidate_ids:
                    candidates.append(movie)
                    candidate_ids.add(_movie_key(movie))
        
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
                    vote_count_gte=Config.TMDB_MIN_VOTES,
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
        pages_needed = (count // Config.MOVIES_PER_PAGE) + 1  # TMDB standard page size
        
        for page in range(1, min(pages_needed + 1, Config.MAX_PAGES_PER_FETCH + 1)):  # Capped at Config.MAX_PAGES_PER_FETCH pages
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
        pages_needed = (count // Config.MOVIES_PER_PAGE) + 1
        
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
                    vote_count_gte=Config.TMDB_EXPLORATION_MIN_VOTES
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
        Get user's favorite genres from interaction history.

        Returns TMDB genre ids (not the internal DB ``genre_id``) so the values
        are in the same namespace that ``_get_popular_genres`` and the TMDB
        discover API expect.

        Returns:
            list: TMDB genre IDs sorted by preference
        """
        try:
            # Query user's interacted movies (chosen_movie_id = pairwise winner).
            # Map the internal genre_id to the TMDB genre id via genres.tmdb_genre_id.
            query = """
            SELECT g.tmdb_genre_id, COUNT(*) as interaction_count
            FROM user_interactions ui
            JOIN movie_genres mg ON ui.chosen_movie_id = mg.movie_id
            JOIN genres g ON mg.genre_id = g.genre_id
            WHERE ui.user_id = %s
              AND ui.chosen_movie_id IS NOT NULL
              AND g.tmdb_genre_id IS NOT NULL
            GROUP BY g.tmdb_genre_id
            ORDER BY interaction_count DESC
            LIMIT 5
            """
            
            results = self.db.query(query, (user_id,))
            
            if results:
                return [row['tmdb_genre_id'] for row in results]
            
        except Exception as e:
            logger.error(f"Error getting user genres: {e}")
        
        return []
    
    def _get_popular_genres(self):
        """
        Get most popular genres overall
        
        Returns:
            list: Popular genre IDs
        """
        # Common popular genres (TMDB ids, single-sourced in config;
        # list() copies because this pool is shuffled in place).
        popular_genre_ids = list(Config.TMDB_POPULAR_GENRE_IDS)
        
        # Shuffle for variety
        random.shuffle(popular_genre_ids)
        return popular_genre_ids
    
    def generate_pairwise_candidates(self, user_id=None, count=Config.PAIRWISE_BATCH_SIZE):
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
            # Get movies user has interacted with, enriched with full display
            # metadata (poster/rating/year/genres) so the compare UI is complete.
            try:
                query = """
                SELECT md.movie_id, md.tmdb_id, md.title, md.original_title,
                       md.overview, md.release_year, md.runtime, md.poster_path,
                       md.backdrop_path, md.tmdb_rating, md.vote_count,
                       md.popularity, md.media_type, md.elo_score, md.genres
                FROM user_interactions ui
                JOIN movie_details md ON md.movie_id = ui.chosen_movie_id
                WHERE ui.user_id = %s AND ui.chosen_movie_id IS NOT NULL
                ORDER BY ui.timestamp DESC
                LIMIT %s
                """
                
                results = self.db.query(query, (user_id, count // 2))
                
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
        
        # Known movies come from the DB (tmdb_id) and explore movies from TMDB
        # (id); drop any explore candidate already present in the known half so
        # the pair can never surface the same movie twice.
        known_tmdb_ids = {
            m.get('tmdb_id') for m in known_movies
            if isinstance(m, dict) and m.get('tmdb_id')
        }
        explore_movies = [
            m for m in explore_movies
            if not (isinstance(m, dict) and m.get('id') in known_tmdb_ids)
        ]
        
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
    print("Sample titles:")
    for movie in candidates[:5]:
        print(f"  - {movie.get('title', 'Unknown')} (ID: {movie.get('id')})")
    
    # Test pairwise candidates
    print("\n\nTesting pairwise candidates...")
    pairwise = gen.generate_pairwise_candidates(count=20)
    
    print(f"Known movies: {len(pairwise['known'])}")
    print(f"Explore movies: {len(pairwise['explore'])}")
