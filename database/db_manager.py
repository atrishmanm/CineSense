"""
Database Manager for CineSense
Handles all MySQL database operations with connection pooling
"""

from mysql.connector import pooling, Error
from contextlib import contextmanager
from config import Config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatabaseManager:
    """Singleton database manager with connection pooling"""
    
    _instance = None
    _pool = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DatabaseManager, cls).__new__(cls)
            cls._instance._initialize_pool()
        return cls._instance
    
    def _initialize_pool(self):
        """Initialize connection pool"""
        try:
            self._pool = pooling.MySQLConnectionPool(
                pool_name="cinesense_pool",
                pool_size=10,
                pool_reset_session=True,
                **Config.DB_CONFIG
            )
            logger.info("Database connection pool initialized successfully")
        except Error as e:
            logger.error(f"Error creating connection pool: {e}")
            raise
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        connection = None
        try:
            connection = self._pool.get_connection()
            yield connection
        except Error as e:
            logger.error(f"Database connection error: {e}")
            if connection:
                connection.rollback()
            raise
        finally:
            if connection and connection.is_connected():
                connection.close()
    
    @contextmanager
    def get_cursor(self, dictionary=True, buffered=True, commit=True):
        """Context manager for database cursors.

        C291: the implicit commit is skipped when the session executed only
        SELECT statements (no point opening/committing a write transaction for
        pure reads). Pass commit=False to suppress the commit entirely;
        write paths behave exactly as before.
        """
        with self.get_connection() as connection:
            cursor = connection.cursor(dictionary=dictionary, buffered=buffered)

            # Track the statements executed through this cursor so the
            # post-yield commit decision can tell reads from writes without
            # changing any call site.
            executed = None
            try:
                original_execute = cursor.execute
                original_callproc = cursor.callproc

                def _tracked_execute(*args, **kwargs):
                    if args or kwargs:
                        executed.append(str(args[0]) if args else "<kwargs>")
                    return original_execute(*args, **kwargs)

                def _tracked_callproc(*args, **kwargs):
                    executed.append("CALL " + str(args[0] if args else "?"))
                    return original_callproc(*args, **kwargs)

                cursor.execute = _tracked_execute
                cursor.callproc = _tracked_callproc
                executed = []
            except (AttributeError, TypeError):
                # Cursor implementation disallows patching (C extension):
                # fall back to the legacy always-commit behavior.
                executed = None

            try:
                yield cursor
                if executed is None:
                    saw_write = True              # unpatchable cursor: legacy behavior
                elif executed:
                    saw_write = any(
                        not stmt.lstrip().upper().startswith("SELECT")
                        for stmt in executed
                    )
                else:
                    saw_write = False             # nothing executed, nothing to commit
                if commit and saw_write:
                    connection.commit()
            except Error as e:
                connection.rollback()
                logger.error(f"Database query error: {e}")
                raise
            finally:
                cursor.close()
    
    # ========================================================================
    # USER OPERATIONS
    # ========================================================================
    
    def create_user(self, username, email, password_hash):
        """Create a new user"""
        query = """
            INSERT INTO users (username, email, password_hash)
            VALUES (%s, %s, %s)
        """
        with self.get_cursor() as cursor:
            cursor.execute(query, (username, email, password_hash))
            return cursor.lastrowid
    
    def get_user_by_username(self, username):
        """Get user by username"""
        query = "SELECT * FROM users WHERE username = %s"
        with self.get_cursor() as cursor:
            cursor.execute(query, (username,))
            return cursor.fetchone()
    
    def get_user_by_id(self, user_id):
        """Get user by ID"""
        query = "SELECT * FROM users WHERE user_id = %s"
        with self.get_cursor() as cursor:
            cursor.execute(query, (user_id,))
            return cursor.fetchone()
    
    def update_user_interaction_count(self, user_id):
        """Increment user interaction count"""
        with self.get_cursor() as cursor:
            cursor.callproc('update_user_interaction_count', [user_id])
    
    # ========================================================================
    # MOVIE OPERATIONS
    # ========================================================================
    
    def insert_movie(self, movie_data):
        """Insert a movie with all metadata"""
        query = """
            INSERT INTO movies (
                movie_id, tmdb_id, title, original_title, overview,
                release_year, runtime, poster_path, backdrop_path,
                tmdb_rating, vote_count, popularity, watch_link, media_type
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                title = VALUES(title),
                overview = VALUES(overview),
                poster_path = VALUES(poster_path),
                tmdb_rating = VALUES(tmdb_rating),
                vote_count = VALUES(vote_count),
                popularity = VALUES(popularity),
                media_type = VALUES(media_type)
        """
        with self.get_cursor() as cursor:
            cursor.execute(query, (
                movie_data['movie_id'],
                movie_data['tmdb_id'],
                movie_data['title'],
                movie_data.get('original_title'),
                movie_data.get('overview'),
                movie_data.get('release_year'),
                movie_data.get('runtime'),
                movie_data.get('poster_path'),
                movie_data.get('backdrop_path'),
                movie_data.get('tmdb_rating'),
                movie_data.get('vote_count'),
                movie_data.get('popularity'),
                movie_data.get('watch_link'),
                movie_data.get('media_type', 'movie')
            ))
    
    def get_movie_by_id(self, movie_id):
        """Get movie details by ID"""
        query = "SELECT * FROM movie_details WHERE movie_id = %s"
        with self.get_cursor() as cursor:
            cursor.execute(query, (movie_id,))
            return cursor.fetchone()

    def get_movie_genres(self, movie_id):
        """Get genre names for a movie by ID"""
        query = """
            SELECT g.genre_name
            FROM movie_genres mg
            JOIN genres g ON mg.genre_id = g.genre_id
            WHERE mg.movie_id = %s
            ORDER BY g.genre_name
        """
        with self.get_cursor() as cursor:
            cursor.execute(query, (movie_id,))
            rows = cursor.fetchall()
            return [row['genre_name'] for row in rows] if rows else []
    
    def get_random_movies(self, limit=10, min_popularity=0):
        """Get random movies for comparison (avoids full-table ORDER BY RAND())"""
        # First get the count of eligible movies
        count_query = """
            SELECT COUNT(*) as cnt FROM movies
            WHERE popularity > %s AND tmdb_rating IS NOT NULL
        """
        select_query = """
            SELECT * FROM movies
            WHERE popularity > %s AND tmdb_rating IS NOT NULL
            LIMIT %s OFFSET %s
        """
        with self.get_cursor() as cursor:
            cursor.execute(count_query, (min_popularity,))
            total = cursor.fetchone()['cnt']
            if total == 0:
                return []
            # Collect distinct random offsets
            import random as _rand
            offsets = _rand.sample(range(total), min(limit, total))
            results = []
            for off in offsets:
                cursor.execute(select_query, (min_popularity, 1, off))
                row = cursor.fetchone()
                if row:
                    results.append(row)
            return results
    
    def get_top_movies(self, limit=20, order_by='elo_score', offset=0, media_type='all'):
        """Get top rated movies with pagination support and media type filter"""
        valid_orders = ['elo_score', 'tmdb_rating', 'popularity']
        if order_by not in valid_orders:
            order_by = 'elo_score'
        
        # Build media type filter
        media_filter = ""
        if media_type == 'movie':
            media_filter = "AND (media_type = 'movie' OR media_type IS NULL)"
        elif media_type == 'tv':
            media_filter = "AND media_type = 'tv'"
        
        query = """
            SELECT * FROM movie_details
            WHERE tmdb_rating IS NOT NULL {media_filter}
            ORDER BY {order_by} DESC
            LIMIT %s OFFSET %s
        """.format(media_filter=media_filter, order_by=order_by)
        with self.get_cursor() as cursor:
            cursor.execute(query, (limit, offset))
            return cursor.fetchall()
    
    def search_movies(self, search_term, limit=20):
        """
        Smart search with prioritization:
        1. Exact title matches (highest priority)
        2. Title starts with search term
        3. Title contains search term
        4. Overview/story contains search term
        """
        search_lower = search_term.lower()
        
        # Multi-tier search with relevance scoring
        # Query base tables directly for better control
        query = """
            SELECT 
                m.*,
                GROUP_CONCAT(DISTINCT g.genre_name ORDER BY g.genre_name SEPARATOR ', ') AS genres,
                GROUP_CONCAT(DISTINCT d.director_name ORDER BY d.director_name SEPARATOR ', ') AS directors,
                GROUP_CONCAT(DISTINCT a.actor_name ORDER BY ma.cast_order SEPARATOR ', ') AS cast,
                CASE
                    WHEN LOWER(m.title) = %s THEN 1000
                    WHEN LOWER(m.title) LIKE %s THEN 900
                    WHEN LOWER(m.title) LIKE %s THEN 800
                    WHEN LOWER(m.overview) LIKE %s THEN 100
                    ELSE 50
                END AS relevance_score
            FROM movies m
            LEFT JOIN movie_genres mg ON m.movie_id = mg.movie_id
            LEFT JOIN genres g ON mg.genre_id = g.genre_id
            LEFT JOIN movie_directors md ON m.movie_id = md.movie_id
            LEFT JOIN directors d ON md.director_id = d.director_id
            LEFT JOIN movie_actors ma ON m.movie_id = ma.movie_id
            LEFT JOIN actors a ON ma.actor_id = a.actor_id
            WHERE LOWER(m.title) LIKE %s 
               OR LOWER(m.overview) LIKE %s
            GROUP BY m.movie_id
            ORDER BY relevance_score DESC, m.popularity DESC, m.tmdb_rating DESC
            LIMIT %s
        """
        
        exact_match = search_lower
        starts_with = f"{search_lower}%"
        contains = f"%{search_lower}%"
        
        with self.get_cursor() as cursor:
            cursor.execute(query, (
                exact_match,      # Exact match
                starts_with,      # Starts with
                contains,         # Contains in title
                contains,         # Contains in overview
                contains,         # WHERE title LIKE
                contains,         # WHERE overview LIKE
                limit
            ))
            results = cursor.fetchall()
        
        return results
    
    def get_movies_by_genre_and_type(self, genre_name=None, media_type='all', limit=20, offset=0,
                                     year_from=None, year_to=None, sort_by='rating'):
        """Get movies/TV series by genre and media type, with optional
        year-range filtering and whitelisted sort keys."""
        conditions = ["tmdb_rating IS NOT NULL"]
        params = []

        if genre_name and genre_name != 'all':
            conditions.append("FIND_IN_SET(%s, genres) > 0")
            params.append(genre_name)

        if media_type == 'movie':
            conditions.append("(media_type = 'movie' OR media_type IS NULL)")
        elif media_type == 'tv':
            conditions.append("media_type = 'tv'")

        if year_from is not None:
            conditions.append("release_year >= %s")
            params.append(year_from)
        if year_to is not None:
            conditions.append("release_year <= %s")
            params.append(year_to)

        # Whitelisted sort keys (1:1 with the route's sort_by values)
        sort_columns = {
            'popularity':   'popularity DESC',
            'rating':       'tmdb_rating DESC',
            'release_date': 'release_year DESC, tmdb_rating DESC',
            'title':        'title ASC',
        }
        order_clause = sort_columns.get(sort_by, 'tmdb_rating DESC')

        where_clause = " AND ".join(conditions)
        query = """
            SELECT * FROM movie_details
            WHERE {where_clause}
            ORDER BY {order_clause}
            LIMIT %s OFFSET %s
        """.format(where_clause=where_clause, order_clause=order_clause)
        params.extend([limit, offset])

        with self.get_cursor() as cursor:
            cursor.execute(query, tuple(params))
            return cursor.fetchall()
    
    # ========================================================================
    # GENRE OPERATIONS
    # ========================================================================
    
    def insert_genre(self, genre_name, tmdb_genre_id=None):
        """Insert a genre"""
        query = """
            INSERT INTO genres (genre_name, tmdb_genre_id)
            VALUES (%s, %s)
            ON DUPLICATE KEY UPDATE genre_name = VALUES(genre_name)
        """
        with self.get_cursor() as cursor:
            cursor.execute(query, (genre_name, tmdb_genre_id))
            return cursor.lastrowid
    
    def get_genre_by_name(self, genre_name):
        """Get genre by name"""
        query = "SELECT * FROM genres WHERE genre_name = %s"
        with self.get_cursor() as cursor:
            cursor.execute(query, (genre_name,))
            return cursor.fetchone()
    
    def link_movie_genre(self, movie_id, genre_id):
        """Link movie to genre"""
        query = """
            INSERT IGNORE INTO movie_genres (movie_id, genre_id)
            VALUES (%s, %s)
        """
        with self.get_cursor() as cursor:
            cursor.execute(query, (movie_id, genre_id))
    
    def get_movies_by_genre(self, genre_name, limit=20):
        """Get movies by genre"""
        query = """
            SELECT md.* FROM movie_details md
            WHERE FIND_IN_SET(%s, md.genres) > 0
            ORDER BY md.tmdb_rating DESC
            LIMIT %s
        """
        with self.get_cursor() as cursor:
            cursor.execute(query, (genre_name, limit))
            return cursor.fetchall()
    
    # ========================================================================
    # DIRECTOR OPERATIONS
    # ========================================================================
    
    def insert_director(self, director_name, tmdb_person_id=None, popularity=None):
        """Insert a director"""
        query = """
            INSERT INTO directors (director_name, tmdb_person_id, popularity)
            VALUES (%s, %s, %s)
            ON DUPLICATE KEY UPDATE 
                director_name = VALUES(director_name),
                popularity = VALUES(popularity)
        """
        with self.get_cursor() as cursor:
            cursor.execute(query, (director_name, tmdb_person_id, popularity))
            return cursor.lastrowid
    
    def link_movie_director(self, movie_id, director_id):
        """Link movie to director"""
        query = """
            INSERT IGNORE INTO movie_directors (movie_id, director_id)
            VALUES (%s, %s)
        """
        with self.get_cursor() as cursor:
            cursor.execute(query, (movie_id, director_id))
    
    # ========================================================================
    # ACTOR OPERATIONS
    # ========================================================================
    
    def insert_actor(self, actor_name, tmdb_person_id=None, popularity=None):
        """Insert an actor"""
        query = """
            INSERT INTO actors (actor_name, tmdb_person_id, popularity)
            VALUES (%s, %s, %s)
            ON DUPLICATE KEY UPDATE 
                actor_name = VALUES(actor_name),
                popularity = VALUES(popularity)
        """
        with self.get_cursor() as cursor:
            cursor.execute(query, (actor_name, tmdb_person_id, popularity))
            return cursor.lastrowid
    
    def link_movie_actor(self, movie_id, actor_id, cast_order=0, character_name=None):
        """Link movie to actor"""
        query = """
            INSERT IGNORE INTO movie_actors (movie_id, actor_id, cast_order, character_name)
            VALUES (%s, %s, %s, %s)
        """
        with self.get_cursor() as cursor:
            cursor.execute(query, (movie_id, actor_id, cast_order, character_name))
    
    # ========================================================================
    # INTERACTION OPERATIONS
    # ========================================================================
    
    def record_interaction(self, user_id, movie_1_id, movie_2_id, chosen_movie_id, session_id=None):
        """Record a user's pairwise choice using stored procedure"""
        try:
            with self.get_connection() as connection:
                # Call the stored procedure that handles Elo updates automatically
                with connection.cursor() as cursor:
                    cursor.callproc('record_user_interaction',
                        [user_id, movie_1_id, movie_2_id, chosen_movie_id, session_id])
                connection.commit()
                return True
        except Exception as e:
            logger.error(f"Error recording interaction: {e}")
            return False
    
    def get_user_interactions(self, user_id, limit=100):
        """Get user's interaction history"""
        query = """
            SELECT * FROM user_interactions
            WHERE user_id = %s
            ORDER BY timestamp DESC
            LIMIT %s
        """
        with self.get_cursor() as cursor:
            cursor.execute(query, (user_id, limit))
            return cursor.fetchall()
    
    def update_movie_elo(self, winner_id, loser_id, k_factor=32):
        """Update ELO scores after comparison"""
        with self.get_cursor() as cursor:
            cursor.callproc('update_movie_elo', [winner_id, loser_id, k_factor])
    
    # ========================================================================
    # EMBEDDING OPERATIONS
    # ========================================================================
    
    def save_user_embedding(self, user_id, embedding_vector):
        """Save user's preference vector"""
        query = """
            INSERT INTO user_embeddings (user_id, feature_index, feature_value)
            VALUES (%s, %s, %s)
            ON DUPLICATE KEY UPDATE feature_value = VALUES(feature_value)
        """
        with self.get_cursor() as cursor:
            for idx, value in enumerate(embedding_vector):
                cursor.execute(query, (user_id, idx, float(value)))
    
    def get_user_embedding(self, user_id):
        """Get user's preference vector"""
        query = """
            SELECT feature_index, feature_value
            FROM user_embeddings
            WHERE user_id = %s
            ORDER BY feature_index
        """
        with self.get_cursor() as cursor:
            cursor.execute(query, (user_id,))
            results = cursor.fetchall()
            if not results:
                return None
            return [row['feature_value'] for row in results]
    
    def save_movie_embedding(self, movie_id, embedding_vector):
        """Save movie's feature vector"""
        query = """
            INSERT INTO movie_embeddings (movie_id, feature_index, feature_value)
            VALUES (%s, %s, %s)
            ON DUPLICATE KEY UPDATE feature_value = VALUES(feature_value)
        """
        with self.get_cursor() as cursor:
            for idx, value in enumerate(embedding_vector):
                cursor.execute(query, (movie_id, idx, float(value)))
    
    def get_movie_embedding(self, movie_id):
        """Get movie's feature vector"""
        query = """
            SELECT feature_index, feature_value
            FROM movie_embeddings
            WHERE movie_id = %s
            ORDER BY feature_index
        """
        with self.get_cursor() as cursor:
            cursor.execute(query, (movie_id,))
            results = cursor.fetchall()
            if not results:
                return None
            return [row['feature_value'] for row in results]
    
    def get_all_movie_embeddings(self):
        """Get all movie embeddings for batch processing"""
        query = """
            SELECT movie_id, feature_index, feature_value
            FROM movie_embeddings
            ORDER BY movie_id, feature_index
        """
        with self.get_cursor() as cursor:
            cursor.execute(query)
            return cursor.fetchall()
    
    # ========================================================================
    # STATISTICS & ANALYTICS
    # ========================================================================
    
    def get_user_stats(self, user_id):
        """Get user statistics"""
        query = "SELECT * FROM user_stats WHERE user_id = %s"
        with self.get_cursor() as cursor:
            cursor.execute(query, (user_id,))
            return cursor.fetchone()
    
    def get_movie_count(self):
        """Get total number of movies"""
        query = "SELECT COUNT(*) as count FROM movies"
        with self.get_cursor() as cursor:
            cursor.execute(query)
            return cursor.fetchone()['count']
    
    def get_user_count(self):
        """Get total number of users"""
        query = "SELECT COUNT(*) as count FROM users"
        with self.get_cursor() as cursor:
            cursor.execute(query)
            return cursor.fetchone()['count']
    
    # ========================================================================
    # ENHANCED STORED PROCEDURE OPERATIONS
    # ========================================================================
    
    def search_movies_advanced(self, search_query, search_type='title', user_id=None, limit=50):
        """Advanced search using stored procedure with storyline/semantic support"""
        try:
            with self.get_connection() as connection:
                with connection.cursor(dictionary=True) as cursor:
                    cursor.callproc('search_movies_advanced',
                        [search_query, search_type, user_id, limit])

                    # Fetch results from the procedure
                    results = []
                    for result in cursor.stored_results():
                        results = result.fetchall()

                connection.commit()
                return results
        except Exception as e:
            logger.error(f"Error in advanced search: {e}")
            # Fallback to regular search
            return self.search_movies(search_query, limit)
    
    def get_personalized_recommendations_sp(self, user_id, limit=20):
        """Get personalized recommendations using stored procedure with cursor"""
        try:
            with self.get_connection() as connection:
                with connection.cursor(dictionary=True) as cursor:
                    cursor.callproc('get_personalized_recommendations', [user_id, limit])

                    # Fetch results
                    results = []
                    for result in cursor.stored_results():
                        results = result.fetchall()

                connection.commit()
                return results
        except Exception as e:
            logger.error(f"Error getting personalized recommendations: {e}")
            return []
    
    def get_comparison_pair_sp(self, user_id):
        """Get comparison pair using stored procedure"""
        try:
            with self.get_connection() as connection:
                with connection.cursor(dictionary=True) as cursor:
                    cursor.callproc('get_comparison_pair', [user_id])

                    # Fetch results
                    results = []
                    for result in cursor.stored_results():
                        results = result.fetchall()

                connection.commit()
                
                # Return two movies if available
                if len(results) >= 2:
                    return results[0], results[1]
                return None, None
        except Exception as e:
            logger.error(f"Error getting comparison pair: {e}")
            return None, None
    
    def update_user_preferences_sp(self, user_id):
        """Update user preferences using stored procedure"""
        try:
            with self.get_connection() as connection:
                with connection.cursor() as cursor:
                    cursor.callproc('update_user_preferences', [user_id])
                connection.commit()
                return True
        except Exception as e:
            logger.error(f"Error updating user preferences: {e}")
            return False
    
    def update_cache_stats(self, cache_type, hits, misses, response_time_ms, memory_usage_kb):
        """Update cache statistics using stored procedure"""
        try:
            with self.get_connection() as connection:
                with connection.cursor() as cursor:
                    cursor.callproc('update_cache_stats',
                        [cache_type, hits, misses, response_time_ms, memory_usage_kb])
                connection.commit()
                return True
        except Exception as e:
            logger.error(f"Error updating cache stats: {e}")
            return False
    
    def get_cache_statistics(self):
        """Get cache statistics from database"""
        query = """
            SELECT 
                cache_type,
                cache_hits,
                cache_misses,
                ROUND((cache_hits / NULLIF(cache_hits + cache_misses, 0)) * 100, 2) as hit_rate,
                avg_response_time_ms,
                memory_usage_kb,
                recorded_at
            FROM cache_stats
            WHERE stat_date = CURRENT_DATE
        """
        with self.get_cursor() as cursor:
            cursor.execute(query)
            results = cursor.fetchall()
            return results if results else []
    
    def insert_movie_keyword(self, movie_id, keyword, relevance_score=1.0):
        """Insert movie keyword for semantic search"""
        query = """
            INSERT INTO movie_keywords (movie_id, keyword, relevance_score)
            VALUES (%s, %s, %s)
            ON DUPLICATE KEY UPDATE relevance_score = VALUES(relevance_score)
        """
        with self.get_cursor() as cursor:
            cursor.execute(query, (movie_id, keyword, relevance_score))
    
    def get_comprehensive_movie_view(self, movie_id):
        """Get comprehensive movie details from view"""
        query = "SELECT * FROM comprehensive_movie_view WHERE movie_id = %s"
        with self.get_cursor() as cursor:
            cursor.execute(query, (movie_id,))
            return cursor.fetchone()
    
    def get_advanced_user_stats(self, user_id):
        """Get advanced user statistics from view"""
        query = "SELECT * FROM advanced_user_stats WHERE user_id = %s"
        with self.get_cursor() as cursor:
            cursor.execute(query, (user_id,))
            return cursor.fetchone()
    
    def get_genre_popularity(self):
        """Get genre popularity statistics"""
        query = "SELECT * FROM genre_popularity_view ORDER BY popularity_rank"
        with self.get_cursor() as cursor:
            cursor.execute(query)
            return cursor.fetchall()
    
    def get_trending_movies(self, limit=20):
        """Get trending movies from view"""
        query = "SELECT * FROM trending_movies_view LIMIT %s"
        with self.get_cursor() as cursor:
            cursor.execute(query, (limit,))
            return cursor.fetchall()
    
    # ========================================================================
    # WATCHLIST OPERATIONS
    # ========================================================================
    
    def add_to_watchlist(self, user_id, movie_id, priority=5, status='planned', note=None):
        """Add movie to user's watchlist"""
        query = """
            INSERT INTO watchlist (user_id, movie_id, priority, status, personal_note)
            VALUES (%s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE priority = VALUES(priority), status = VALUES(status), personal_note = VALUES(personal_note)
        """
        with self.get_cursor() as cursor:
            cursor.execute(query, (user_id, movie_id, priority, status, note))
            return cursor.lastrowid
    
    def get_user_watchlist(self, user_id, status=None, limit=50):
        """Get user's watchlist from the watchlist_details view"""
        if status:
            query = "SELECT * FROM watchlist_details WHERE user_id = %s AND status = %s ORDER BY priority DESC, added_at DESC LIMIT %s"
            params = (user_id, status, limit)
        else:
            query = "SELECT * FROM watchlist_details WHERE user_id = %s ORDER BY priority DESC, added_at DESC LIMIT %s"
            params = (user_id, limit)
        with self.get_cursor() as cursor:
            cursor.execute(query, params)
            return cursor.fetchall()
    
    def update_watchlist_status(self, user_id, movie_id, status, rating=None):
        """Update watchlist entry status"""
        if status == 'completed':
            query = "UPDATE watchlist SET status = %s, user_rating = %s, watched_at = CURRENT_TIMESTAMP WHERE user_id = %s AND movie_id = %s"
            params = (status, rating, user_id, movie_id)
        else:
            query = "UPDATE watchlist SET status = %s WHERE user_id = %s AND movie_id = %s"
            params = (status, user_id, movie_id)
        with self.get_cursor() as cursor:
            cursor.execute(query, params)
            return cursor.rowcount
    
    def remove_from_watchlist(self, user_id, movie_id):
        """Remove movie from user's watchlist"""
        query = "DELETE FROM watchlist WHERE user_id = %s AND movie_id = %s"
        with self.get_cursor() as cursor:
            cursor.execute(query, (user_id, movie_id))
            return cursor.rowcount
    
    # ========================================================================
    # REVIEW OPERATIONS
    # ========================================================================
    
    def add_review(self, user_id, movie_id, rating, review_text=None, is_spoiler=False):
        """Add or update a movie review"""
        query = """
            INSERT INTO movie_reviews (user_id, movie_id, rating, review_text, is_spoiler)
            VALUES (%s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE rating = VALUES(rating), review_text = VALUES(review_text), is_spoiler = VALUES(is_spoiler)
        """
        with self.get_cursor() as cursor:
            cursor.execute(query, (user_id, movie_id, rating, review_text, is_spoiler))
            return cursor.lastrowid
    
    def get_movie_reviews(self, movie_id, limit=20):
        """Get reviews for a movie"""
        query = """
            SELECT r.*, u.username FROM movie_reviews r
            INNER JOIN users u ON r.user_id = u.user_id
            WHERE r.movie_id = %s ORDER BY r.created_at DESC LIMIT %s
        """
        with self.get_cursor() as cursor:
            cursor.execute(query, (movie_id, limit))
            return cursor.fetchall()
    
    def get_user_reviews(self, user_id, limit=50):
        """Get all reviews by a user"""
        query = """
            SELECT r.*, m.title, m.poster_path FROM movie_reviews r
            INNER JOIN movies m ON r.movie_id = m.movie_id
            WHERE r.user_id = %s ORDER BY r.created_at DESC LIMIT %s
        """
        with self.get_cursor() as cursor:
            cursor.execute(query, (user_id, limit))
            return cursor.fetchall()
    
    def get_user_review_summary(self, user_id):
        """Get user review statistics from the user_review_summary view"""
        query = "SELECT * FROM user_review_summary WHERE user_id = %s"
        with self.get_cursor() as cursor:
            cursor.execute(query, (user_id,))
            return cursor.fetchone()
    
    def vote_review_helpful(self, review_id):
        """Increment helpful vote count for a review"""
        query = "UPDATE movie_reviews SET helpful_votes = helpful_votes + 1 WHERE review_id = %s"
        with self.get_cursor() as cursor:
            cursor.execute(query, (review_id,))
            return cursor.rowcount
    
    # ========================================================================
    # ADVANCED ANALYTICS (using stored functions)  
    # ========================================================================
    
    def get_movie_popularity_tier(self, movie_id):
        """Get movie popularity tier using stored function"""
        query = "SELECT get_popularity_tier(popularity) AS tier FROM movies WHERE movie_id = %s"
        with self.get_cursor() as cursor:
            cursor.execute(query, (movie_id,))
            result = cursor.fetchone()
            return result['tier'] if result else 'Unknown'
    
    def get_genre_similarity(self, movie_a_id, movie_b_id):
        """Get Jaccard genre similarity between two movies"""
        query = "SELECT jaccard_genre_similarity(%s, %s) AS similarity"
        with self.get_cursor() as cursor:
            cursor.execute(query, (movie_a_id, movie_b_id))
            result = cursor.fetchone()
            return float(result['similarity']) if result else 0.0
    
    def get_user_taste_profile(self, user_id):
        """Get user taste profile label using stored function"""
        query = "SELECT get_user_taste_profile(%s) AS profile"
        with self.get_cursor() as cursor:
            cursor.execute(query, (user_id,))
            result = cursor.fetchone()
            return result['profile'] if result else 'New Explorer'
    
    def get_weighted_score(self, user_id, movie_id):
        """Get weighted recommendation score for a user-movie pair"""
        query = "SELECT weighted_recommendation_score(%s, %s) AS score"
        with self.get_cursor() as cursor:
            cursor.execute(query, (user_id, movie_id))
            result = cursor.fetchone()
            return float(result['score']) if result else 0.0
    
    def get_genre_comparison_stats(self):
        """Get genre win/loss statistics from view"""
        query = "SELECT * FROM genre_comparison_stats ORDER BY win_rate_pct DESC"
        with self.get_cursor() as cursor:
            cursor.execute(query)
            return cursor.fetchall()
    
    def run_genre_audit(self):
        """Run genre audit report using cursor-based stored procedure"""
        try:
            with self.get_connection() as connection:
                with connection.cursor(dictionary=True) as cursor:
                    cursor.callproc('generate_genre_audit')
                    results = []
                    for result in cursor.stored_results():
                        results = result.fetchall()
                connection.commit()
                return results
        except Exception as e:
            logger.error(f"Genre audit error: {e}")
            return []
    
    # ========================================================================
    # GENERIC QUERY/EXECUTE (used by social_routes and other modules)
    # ========================================================================
    
    @staticmethod
    def _reject_multi_statement(sql, allow_multi):
        """C278: reject multi-statement payloads unless explicitly allowed."""
        if sql is None:
            return
        body = sql.strip()
        if body.endswith(';'):
            body = body[:-1]
        if not allow_multi and ';' in body:
            raise ValueError(
                "query()/execute() rejected multi-statement SQL (';' found). "
                "Statements must be single and fully %s-parameterized; pass "
                "allow_multi=True only for trusted internal callers."
            )

    def query(self, sql, params=None, fetch_all=True, allow_multi=False):
        """Execute a SELECT query and return results.

        WARNING (C278): ``sql`` is passed to MySQL verbatim, so this sink is
        only safe for constant strings with %s placeholders -- never
        interpolate user input into ``sql``. As a lightweight guard,
        multi-statement payloads (a ';' besides a trailing one) raise
        ValueError unless allow_multi=True is passed explicitly.
        """
        self._reject_multi_statement(sql, allow_multi)
        with self.get_cursor() as cursor:
            cursor.execute(sql, params or ())
            if fetch_all:
                return cursor.fetchall()
            return cursor.fetchone()
    
    def execute(self, sql, params=None, return_lastrowid=False, allow_multi=False):
        """Execute an INSERT/UPDATE/DELETE query.

        Same C278 safety net as query(): verbatim SQL, single statement only
        unless allow_multi=True is passed explicitly.
        """
        self._reject_multi_statement(sql, allow_multi)
        with self.get_cursor() as cursor:
            cursor.execute(sql, params or ())
            if return_lastrowid:
                return cursor.lastrowid
            return cursor.rowcount
    
    def ensure_social_tables(self):
        """Create social feature tables if they don't exist.

        Authoritative DDL lives in database/schema.sql; this is an idempotent
        runtime safety net for databases created before the social tables were
        added to the schema. Keep the two in sync.
        """
        tables = [
            """
            CREATE TABLE IF NOT EXISTS friend_requests (
                request_id INT AUTO_INCREMENT PRIMARY KEY,
                from_user_id INT NOT NULL,
                to_user_id INT NOT NULL,
                status ENUM('pending', 'accepted', 'rejected') DEFAULT 'pending',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (from_user_id) REFERENCES users(user_id) ON DELETE CASCADE,
                FOREIGN KEY (to_user_id) REFERENCES users(user_id) ON DELETE CASCADE,
                UNIQUE KEY unique_request (from_user_id, to_user_id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """,
            """
            CREATE TABLE IF NOT EXISTS friendships (
                friendship_id INT AUTO_INCREMENT PRIMARY KEY,
                user_id_1 INT NOT NULL,
                user_id_2 INT NOT NULL,
                status ENUM('accepted', 'blocked') DEFAULT 'accepted',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id_1) REFERENCES users(user_id) ON DELETE CASCADE,
                FOREIGN KEY (user_id_2) REFERENCES users(user_id) ON DELETE CASCADE,
                UNIQUE KEY unique_friendship (user_id_1, user_id_2)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """,
            """
            CREATE TABLE IF NOT EXISTS watch_parties (
                party_id INT AUTO_INCREMENT PRIMARY KEY,
                host_id INT NOT NULL,
                movie_id INT NOT NULL,
                scheduled_time DATETIME NOT NULL,
                status ENUM('scheduled', 'pending', 'active', 'live', 'completed', 'cancelled') DEFAULT 'scheduled',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (host_id) REFERENCES users(user_id) ON DELETE CASCADE,
                FOREIGN KEY (movie_id) REFERENCES movies(movie_id) ON DELETE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """,
            """
            CREATE TABLE IF NOT EXISTS watch_party_invites (
                invite_id INT AUTO_INCREMENT PRIMARY KEY,
                party_id INT NOT NULL,
                user_id INT NOT NULL,
                status ENUM('invited', 'pending', 'accepted', 'declined') DEFAULT 'invited',
                invited_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (party_id) REFERENCES watch_parties(party_id) ON DELETE CASCADE,
                FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
                UNIQUE KEY unique_invite (party_id, user_id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """,
            """
            CREATE TABLE IF NOT EXISTS collaborative_lists (
                list_id INT AUTO_INCREMENT PRIMARY KEY,
                creator_id INT NOT NULL,
                name VARCHAR(255) NOT NULL,
                description TEXT,
                is_public BOOLEAN DEFAULT FALSE,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (creator_id) REFERENCES users(user_id) ON DELETE CASCADE
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """,
            """
            CREATE TABLE IF NOT EXISTS collaborative_list_items (
                item_id INT AUTO_INCREMENT PRIMARY KEY,
                list_id INT NOT NULL,
                movie_id INT NOT NULL,
                added_by INT NOT NULL,
                notes VARCHAR(500) NULL,
                position INT NULL,
                added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (list_id) REFERENCES collaborative_lists(list_id) ON DELETE CASCADE,
                FOREIGN KEY (movie_id) REFERENCES movies(movie_id) ON DELETE CASCADE,
                FOREIGN KEY (added_by) REFERENCES users(user_id) ON DELETE CASCADE,
                UNIQUE KEY unique_list_movie (list_id, movie_id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """,
            """
            CREATE TABLE IF NOT EXISTS collaborative_list_members (
                member_id INT AUTO_INCREMENT PRIMARY KEY,
                list_id INT NOT NULL,
                user_id INT NOT NULL,
                role ENUM('owner', 'editor', 'viewer') DEFAULT 'editor',
                joined_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (list_id) REFERENCES collaborative_lists(list_id) ON DELETE CASCADE,
                FOREIGN KEY (user_id) REFERENCES users(user_id) ON DELETE CASCADE,
                UNIQUE KEY unique_list_member (list_id, user_id)
            ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4
            """
        ]
        
        with self.get_cursor() as cursor:
            for table_sql in tables:
                try:
                    cursor.execute(table_sql)
                except Exception as e:
                    logger.warning(f"Table creation warning: {e}")


# Singleton instance
db = DatabaseManager()


if __name__ == "__main__":
    # Test database connection
    try:
        print("Testing database connection...")
        movie_count = db.get_movie_count()
        user_count = db.get_user_count()
        print("Connection successful!")
        print(f"  Movies in database: {movie_count}")
        print(f"  Users in database: {user_count}")
    except Exception as e:
        print(f"Connection failed: {e}")
