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
    
    # Redis Configuration (for caching)
    REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
    REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
    REDIS_DB = int(os.getenv('REDIS_DB', 0))
    REDIS_PASSWORD = os.getenv('REDIS_PASSWORD', None)
    
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
    
    # Advanced AI: Latent Space Dimensionality Reduction
    LATENT_DIM = 32  # Compressed representation (from 55 -> 32)
    USE_DIMENSIONALITY_REDUCTION = True  # Use PCA/SVD for latent features
    
    # Memory & Temporal Decay
    TEMPORAL_DECAY_FACTOR = 0.7  # Weight for recent interactions (0.7 recent + 0.3 past)
    INTERACTION_MEMORY_WINDOW = 50  # Keep last N interactions with full weight
    LEARNING_TARGET_COMPARISONS = int(os.getenv('LEARNING_TARGET_COMPARISONS', INTERACTION_MEMORY_WINDOW))
    
    # Model Paths
    MODEL_DIR = os.getenv('MODEL_DIR', './model')
    CHECKPOINT_DIR = os.getenv('CHECKPOINT_DIR', './checkpoints')
    
    # Probabilistic Decision Making
    USE_SOFTMAX_SELECTION = True  # Use probability distributions instead of argmax
    SOFTMAX_TEMPERATURE = 0.8  # Controls exploration (lower = more deterministic)
    
    # Implicit Signal Weights
    IMPLICIT_SIGNALS = {
        'hover_time': 0.15,      # Time spent hovering on movie
        'skip_penalty': -0.2,    # Penalty for skipping
        'repeat_view': 0.3,      # Bonus for repeated views
        'session_abandon': -0.1  # Penalty for abandoned sessions
    }
    
    # Recommendation Settings
    RECOMMENDATION_BATCH_SIZE = 20
    COMPARISON_BATCH_SIZE = 2
    MIN_INTERACTIONS_FOR_PERSONALIZATION = 5
    
    # Natural Language Generation
    ENABLE_EXPLANATIONS = True  # Generate human-readable explanations
    EXPLANATION_DETAIL_LEVEL = 'medium'  # 'low', 'medium', 'high'
    
    # LAZY LOADING & INFINITE STREAM Configuration
    
    # Sliding Window Cache
    MOVIE_CACHE_SIZE = 100  # Keep only 100 movies in memory
    VECTOR_CACHE_SIZE = 500  # Cache 500 movie vectors
    CACHE_REFILL_THRESHOLD = 0.3  # Refill when 30% full
    
    # Candidate Generation
    CANDIDATE_COUNT = 300  # Generate 200-500 candidates before ranking
    CANDIDATE_STRATEGY = 'mixed'  # 'mixed', 'genre', 'popularity', 'exploration'
    
    # TMDB API Pagination
    MAX_PAGES_PER_FETCH = 10  # Fetch max 10 pages (200 movies) at a time
    MOVIES_PER_PAGE = 20  # TMDB standard
    
    # Pairwise Comparison Strategy
    PAIRWISE_KNOWN_RATIO = 0.5  # 50% known movies, 50% exploration
    PAIRWISE_BATCH_SIZE = 30  # Generate 30 movies for pairwise pool
    
    # Memory Optimization
    STORE_ONLY_INTERACTED = True  # Only save movies user interacted with
    LAZY_EMBEDDING = True  # Compute embeddings on-demand, not precomputed
    EVICTION_STRATEGY = 'lru'  # 'lru' (Least Recently Used)
    
    # Recommendation Pipeline
    USE_CANDIDATE_GENERATION = True  # Enable candidate generation (vs scoring all)
    FINAL_RECOMMENDATION_COUNT = 20  # Return top 20 after ranking candidates

    # Deep Learning Ensemble (NeuMF V2)
    # Disabled by default unless checkpoints are explicitly provisioned.
    USE_DL_SCORING = os.getenv('USE_DL_SCORING', '0').lower() in {'1', 'true', 'yes', 'on'}
    DL_SCORE_WEIGHT = 0.15  # Weight of DL genre affinity in final score
    DL_V2_CHECKPOINT = os.getenv('DL_V2_CHECKPOINT', 'cinesense_v2.pt')  # Phase 2 ensemble (8 models)
    DL_V1_CHECKPOINT = os.getenv('DL_V1_CHECKPOINT', 'cinesense_model_final.pt')  # Phase 1 ensemble (5 models)
    DL_ENSEMBLE_RMSE = 0.8932  # Best: Mega optimized
