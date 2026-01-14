"""
Database Manager for CineSense
Handles all MySQL database operations with connection pooling
"""

import mysql.connector
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
    def get_cursor(self, dictionary=True, buffered=True):
        """Context manager for database cursors"""
        with self.get_connection() as connection:
            cursor = connection.cursor(dictionary=dictionary, buffered=buffered)
            try:
                yield cursor
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
                tmdb_rating, vote_count, popularity, watch_link
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                title = VALUES(title),
                overview = VALUES(overview),
                poster_path = VALUES(poster_path),
                tmdb_rating = VALUES(tmdb_rating),
                vote_count = VALUES(vote_count),
                popularity = VALUES(popularity)
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
                movie_data.get('watch_link')
            ))
    
    def get_movie_by_id(self, movie_id):
        """Get movie details by ID"""
        query = "SELECT * FROM movie_details WHERE movie_id = %s"
        with self.get_cursor() as cursor:
            cursor.execute(query, (movie_id,))
            return cursor.fetchone()
    
    def get_random_movies(self, limit=10, min_popularity=0):
        """Get random movies for comparison"""
        query = """
            SELECT * FROM movies 
            WHERE popularity > %s AND tmdb_rating IS NOT NULL
            ORDER BY RAND()
            LIMIT %s
        """
        with self.get_cursor() as cursor:
            cursor.execute(query, (min_popularity, limit))
            return cursor.fetchall()
    
    def get_top_movies(self, limit=20, order_by='elo_score'):
        """Get top rated movies"""
        valid_orders = ['elo_score', 'tmdb_rating', 'popularity']
        if order_by not in valid_orders:
            order_by = 'elo_score'
        
        query = f"""
            SELECT * FROM movie_details 
            WHERE tmdb_rating IS NOT NULL
            ORDER BY {order_by} DESC
            LIMIT %s
        """
        with self.get_cursor() as cursor:
            cursor.execute(query, (limit,))
            return cursor.fetchall()
    
    def search_movies(self, search_term, limit=20):
        """Search movies by title"""
        query = """
            SELECT * FROM movie_details
            WHERE MATCH(title, overview) AGAINST(%s IN NATURAL LANGUAGE MODE)
            LIMIT %s
        """
        with self.get_cursor() as cursor:
            cursor.execute(query, (search_term, limit))
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
        """Record a user's pairwise choice"""
        rejected_movie_id = movie_2_id if chosen_movie_id == movie_1_id else movie_1_id
        
        query = """
            INSERT INTO user_interactions 
            (user_id, movie_1_id, movie_2_id, chosen_movie_id, rejected_movie_id, session_id)
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        with self.get_cursor() as cursor:
            cursor.execute(query, (
                user_id, movie_1_id, movie_2_id, 
                chosen_movie_id, rejected_movie_id, session_id
            ))
            return cursor.lastrowid
    
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


# Singleton instance
db = DatabaseManager()


if __name__ == "__main__":
    # Test database connection
    try:
        print("Testing database connection...")
        movie_count = db.get_movie_count()
        user_count = db.get_user_count()
        print(f"✓ Connection successful!")
        print(f"  Movies in database: {movie_count}")
        print(f"  Users in database: {user_count}")
    except Exception as e:
        print(f"✗ Connection failed: {e}")
