import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Application configuration"""
    
    # Flask
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    PORT = int(os.getenv('PORT', 5000))
    DEBUG = os.getenv('FLASK_ENV') == 'development'
    
    # Database
    DB_CONFIG = {
        'host': os.getenv('DB_HOST', 'localhost'),
        'port': int(os.getenv('DB_PORT', 3306)),
        'user': os.getenv('DB_USER', 'root'),
        'password': os.getenv('DB_PASSWORD', ''),
        'database': os.getenv('DB_NAME', 'cinesense'),
        'charset': 'utf8mb4',
        'collation': 'utf8mb4_unicode_ci'
    }
    
    # TMDB API
    TMDB_API_KEY = os.getenv('TMDB_API_KEY', '')
    TMDB_BASE_URL = os.getenv('TMDB_BASE_URL', 'https://api.themoviedb.org/3')
    TMDB_IMAGE_BASE_URL = 'https://image.tmdb.org/t/p/'
    
    # AI Configuration
    LEARNING_RATE = float(os.getenv('LEARNING_RATE', 0.1))
    EXPLORATION_RATE = float(os.getenv('EXPLORATION_RATE', 0.2))
    INITIAL_ELO_SCORE = int(os.getenv('INITIAL_ELO_SCORE', 1500))
    
    # Feature Vector Dimensions
    GENRE_DIM = 20
    DIRECTOR_DIM = 10
    ACTOR_DIM = 20
    METADATA_DIM = 5  # rating, popularity, year, etc.
    TOTAL_VECTOR_DIM = GENRE_DIM + DIRECTOR_DIM + ACTOR_DIM + METADATA_DIM
    
    # Recommendation Settings
    RECOMMENDATION_BATCH_SIZE = 20
    COMPARISON_BATCH_SIZE = 2
    MIN_INTERACTIONS_FOR_PERSONALIZATION = 5
