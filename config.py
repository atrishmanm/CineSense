import os
import secrets
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Application configuration"""
    
    # Flask — generate a random key if SECRET_KEY env var is not set.
    # NOTE: restarting the server invalidates all active sessions when
    # this fallback is used.  In production always set SECRET_KEY.
    SECRET_KEY = os.getenv('SECRET_KEY') or secrets.token_hex(32)
    PORT = int(os.getenv('PORT', 5000))
    DEBUG = os.getenv('FLASK_ENV') == 'development'

    # Request/upload size limit. The app itself enforces this in
    # app_integrated.py via app.config['MAX_CONTENT_LENGTH'] = 16 MB.
    # Per-image size/content validation for upload routes (/api/visual-search,
    # /api/text-to-image-search) belongs in those route handlers (api/).
    
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
    
    # Redis configuration is exposed via the canonical REDIS_CONFIG aggregate
    # below (host/port/db/password); consumers pass **Config.REDIS_CONFIG to
    # ai.redis_cache.get_redis_cache().
    REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
    REDIS_PORT = int(os.getenv('REDIS_PORT', 6379))
    REDIS_DB = int(os.getenv('REDIS_DB', 0))
    REDIS_PASSWORD = os.getenv('REDIS_PASSWORD', None)

    # Canonical aggregate for Redis consumers. ai.redis_cache.RedisCache accepts
    # host/port/db/password, so callers can use Config.REDIS_CONFIG instead of
    # ad-hoc os.getenv reads.
    REDIS_CONFIG = {
        'host': REDIS_HOST,
        'port': REDIS_PORT,
        'db': REDIS_DB,
        'password': REDIS_PASSWORD,
    }
    
    # TMDB API
    TMDB_API_KEY = os.getenv('TMDB_API_KEY', '')
    TMDB_BASE_URL = os.getenv('TMDB_BASE_URL', 'https://api.themoviedb.org/3')
    TMDB_IMAGE_BASE_URL = 'https://image.tmdb.org/t/p/'
    
    # AI Configuration
    LEARNING_RATE = float(os.getenv('LEARNING_RATE', 0.1))
    EXPLORATION_RATE = float(os.getenv('EXPLORATION_RATE', 0.2))  # epsilon-greedy exploration rate (wired: ai/reinforcement.py)
    INITIAL_ELO_SCORE = int(os.getenv('INITIAL_ELO_SCORE', 1500))

    # ------------------------------------------------------------------
    # AI layer tunables - single source of truth for magic values that were
    # previously hardcoded across ai/ (de-duplication of audit A088-A162).
    # Defaults preserve current behavior exactly; tune here, not at call sites.
    # ------------------------------------------------------------------
    ELO_K_FACTOR = 32               # max ELO rating change per comparison
    BANDIT_UCB_C = 2.0              # UCB exploration bonus coefficient
    PREFERENCE_SIGMOID_SCALE = 5.0  # steepness of the preference -> probability sigmoid
    # Scoring normalization anchor for (elo - ELO_BASE) / ELO_BASE, clamped to
    # [0, 1]. Intentionally distinct from INITIAL_ELO_SCORE (1500): 1500 is the
    # START rating for a fresh movie/user; 1000 is the floor the community ELO
    # scale is normalized against (values drift below 1500 via comparisons).
    ELO_BASE = 1000.0
    # Weighted linear rec-score mix (content / preference / ELO; sums to 1.0).
    RECOMMEND_WEIGHT_CONTENT = 0.50
    RECOMMEND_WEIGHT_PREFERENCE = 0.30
    RECOMMEND_WEIGHT_ELO = 0.20
    # Candidate generation funnel (mixed-strategy share of each channel).
    CANDIDATE_FUNNEL_CONTENT_RATIO = 0.4
    CANDIDATE_FUNNEL_POPULARITY_RATIO = 0.3
    CANDIDATE_FUNNEL_EXPLORATION_RATIO = 0.2
    CANDIDATE_FUNNEL_CACHE_RATIO = 0.1
    TMDB_MIN_VOTES = 100             # vote_count_gte filter for curated discovery
    TMDB_EXPLORATION_MIN_VOTES = 50  # looser vote filter for random exploration
    # TMDB genre ids forming the diverse/popular genre fallback pool.
    TMDB_POPULAR_GENRE_IDS = [
        28,     # Action
        12,     # Adventure
        16,     # Animation
        35,     # Comedy
        18,     # Drama
        14,     # Fantasy
        27,     # Horror
        10749,  # Romance
        878,    # Science Fiction
        53,     # Thriller
    ]
    # Recommender fetching limits.
    RECENT_INTERACTIONS_LIMIT = 100  # interaction rows fetched for history/seen-set
    COMPARISON_POOL_SIZE = 50        # random movie pool behind get_comparison_pair
    RECOMMEND_CANDIDATE_POOL = 500   # candidate pool ranked in get_recommendations
    MODEL_WARM_START_LIMIT = 100     # stored comparisons replayed to warm-start a RAM preference model
    DIVERSITY_MMR_LAMBDA = 0.3       # MMR relevance/diversity tradeoff (0 = pure relevance)
    # Implicit-signal reward shaping.
    HOVER_TIME_SCALE_SECONDS = 5.0   # hover-time normalization ceiling (0-5s -> 0-1)
    IMPLICIT_SIGNAL_MIX = 0.3        # weight of implicit reward stacked on the explicit choice
    # Online feature store.
    TASTE_EMA_DECAY_RATE = 0.10     # taste-vector EMA step. Deliberately separate from
                                    # TEMPORAL_DECAY_FACTOR (recency weighting of interactions).
    FEATURE_STORE_BASE_YEAR = 2000  # neutral default release year for unregistered items
    # Multi-stage ranker.
    PLATT_CALIBRATION_A = -2.5       # Platt scaling slope (logistic calibration)
    PLATT_CALIBRATION_B = 0.5        # Platt scaling intercept
    COLD_ITEM_IMPRESSION_THRESHOLD = 20  # below this an item earns an exploration bonus
    EXPLORATION_BONUS_COEFFICIENT = 0.15
    EXPLORATION_BONUS_MAX = 0.20
    MIN_SYSTEM_IMPRESSIONS = 10      # floor inside the UCB-style log term
    # Inference pipeline (offline ML-100k research serving path).
    INFERENCE_POPULAR_K = 100        # popular channel size
    INFERENCE_RECENT_K = 50          # newest-by-id channel size
    INFERENCE_GENRE_SIMILAR_K = 200  # genre-overlap channel size
    INFERENCE_CANDIDATE_POOL = 500   # total candidates before scoring
    INFERENCE_TOP_K = 10             # final recommendation count
    INFERENCE_MAX_PER_GENRE = 5      # DiversityReranker.rerank default per-genre cap
    INFERENCE_RECOMMEND_MAX_PER_GENRE = 3  # cap used by ProductionRecommender.recommend
    # A/B testing (ai/ab_testing.py).
    AB_MIN_SAMPLE_SIZE = 1000        # minimum impressions before significance tests
    AB_BUCKET_SCALE = 10000          # hashing bucket count for variant assignment
    AB_SIGNIFICANCE_ALPHA = 0.05     # p cutoff for CTR / rating / watch-time tests
    AB_MIN_GROUP_N = 30              # min observations per variant for t-tests
    # A/B experimentation (ai/ab_experimentation.py). Different test semantics
    # from AB_SIGNIFICANCE_ALPHA: SRM flags engineering bugs / tracking loss,
    # so it needs a much stricter bar.
    SRM_ALPHA = 0.001
    # Trending detector.
    TRENDING_TIME_WINDOWS_HOURS = [1, 6, 24, 72, 168]  # velocity windows: 1h/6h/1d/3d/7d
    TRENDING_LOOKBACK_HOURS = 168    # default lookback / analysis window (7 days)
    TRENDING_WEIGHT_RECENT_VELOCITY = 0.5
    TRENDING_WEIGHT_AVG_VELOCITY = 0.2
    TRENDING_WEIGHT_ACCELERATION = 0.3
    TRENDING_SCORE_SCALE = 10.0      # heuristic raw-score multiplier to a 0-100 scale
    TRENDING_SCORE_MAX = 100
    TRENDING_DEFAULT_LIMIT = 20
    TRENDING_MIN_INTERACTIONS = 10   # min interactions to be considered trending
    TRENDING_VIRAL_PREDICTION_MIN_SCORE = 20  # score band: growing but not peaked
    TRENDING_VIRAL_PREDICTION_MAX_SCORE = 60
    TRENDING_VIRAL_GROWTH_THRESHOLD = 0.3     # 30% 24h-over-24h growth
    
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
    LEARNING_TARGET_COMPARISONS = int(os.getenv('LEARNING_TARGET_COMPARISONS', '500'))
    
    # Model Paths
    MODEL_DIR = os.getenv('MODEL_DIR', './model')
    CHECKPOINT_DIR = os.getenv('CHECKPOINT_DIR', './checkpoints')  # reserved / not yet wired
    
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
    RECOMMENDATION_BATCH_SIZE = 20  # reserved / not yet wired
    COMPARISON_BATCH_SIZE = 2  # arms/pairs selected per comparison step (wired: ai/reinforcement.py select_arm top_k)
    MIN_INTERACTIONS_FOR_PERSONALIZATION = 5
    
    # Natural Language Generation
    ENABLE_EXPLANATIONS = True  # Generate human-readable explanations
    EXPLANATION_DETAIL_LEVEL = 'medium'  # 'low', 'medium', 'high'
    
    # LAZY LOADING & INFINITE STREAM Configuration
    
    # Sliding Window Cache
    MOVIE_CACHE_SIZE = 100  # Keep only 100 movies in memory
    VECTOR_CACHE_SIZE = 500  # ai/cache_manager.py VectorCache capacity (wired)
    CACHE_REFILL_THRESHOLD = 0.3  # refill trigger as fraction of max_size (wired: ai/cache_manager.py + recommender)
    PERCENT_SCALE = 100.0  # hits/total -> percent, used for cache hit-rate stats
    
    # Candidate Generation
    CANDIDATE_COUNT = 300  # Generate 200-500 candidates before ranking
    CANDIDATE_STRATEGY = 'mixed'  # 'mixed', 'genre', 'popularity', 'exploration'
    
    # TMDB API Pagination
    MAX_PAGES_PER_FETCH = 10  # Fetch max 10 pages (200 movies) at a time
    MOVIES_PER_PAGE = 20  # TMDB page size (wired: ai/candidate_generator.py + ai/recommender.py._refill_cache)

    # Pairwise Comparison Strategy
    PAIRWISE_KNOWN_RATIO = 0.5  # reserved / not yet wired
    PAIRWISE_BATCH_SIZE = 30  # Generate 30 movies for pairwise pool

    # Memory Optimization
    STORE_ONLY_INTERACTED = True  # reserved / not yet wired
    LAZY_EMBEDDING = True  # reserved / not yet wired
    EVICTION_STRATEGY = 'lru'  # reserved / not yet wired

    # Recommendation Pipeline
    USE_CANDIDATE_GENERATION = True  # reserved / not yet wired
    FINAL_RECOMMENDATION_COUNT = 20  # reserved / not yet wired

    # Deep Learning Ensemble (NeuMF V2) — OFFLINE / RESEARCH ONLY.
    # Checkpoints are not bundled and DL scoring stays disabled unless
    # checkpoints are explicitly provisioned. Do not cite training-time
    # numbers; use benchmarks/run_all_benchmarks.py for held-out results.
    USE_DL_SCORING = os.getenv('USE_DL_SCORING', '0').lower() in {'1', 'true', 'yes', 'on'}
    DL_SCORE_WEIGHT = 0.15  # Weight of DL genre affinity in final score
    DL_V2_CHECKPOINT = os.getenv('DL_V2_CHECKPOINT', 'cinesense_v2.pt')  # Phase 2 ensemble (8 models)
    DL_V1_CHECKPOINT = os.getenv('DL_V1_CHECKPOINT', 'cinesense_model_final.pt')  # Phase 1 ensemble (5 models)
