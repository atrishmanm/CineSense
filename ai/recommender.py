"""
Main Recommendation Engine
Combines all AI layers + Advanced features + Lazy Loading
"""

import numpy as np
from ai.pairwise_learning import PairwiseLearner, UserPreferenceModel
from ai.embeddings import MovieEmbedding, UserEmbedding, FeatureEncoder, ContentBasedRecommender
from ai.reinforcement import UCBBandit
from ai.advanced_ai import (
    LatentSpaceEncoder, ImplicitSignalProcessor, ProbabilisticSelector,
    TemporalMemoryManager, NaturalLanguageExplainer
)
from ai.cache_manager import cache_manager
from ai.candidate_generator import candidate_generator
from ai.neumf_scorer import get_scorer as _get_dl_scorer
from tmdb.fetcher import TMDBFetcher
from database.db_manager import db
from config import Config
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _normalize_elo(raw_elo):
    """Normalize a raw ELO score into a consistent [0, 1] range.

    Both the production (``get_recommendations``) and advanced
    (``get_advanced_recommendations``) paths use this so the ``recommendation_score``
    key has one documented scale. Clamped to 0..1 so extreme ELO values cannot
    swamp the other weighted components.
    """
    try:
        # ELO_BASE floors raw ELO for ranking; INITIAL_ELO_SCORE (1500) is
        # the separate start value for fresh movies/users (see config.py).
        return max(0.0, min(1.0, (float(raw_elo) - Config.ELO_BASE) / Config.ELO_BASE))
    except (TypeError, ValueError):
        return 0.5


def _to_float(value, default):
    """None-safe float coercion (wave5 fix C1.7).

    Movie rows come straight out of SQL where a numeric column can be NULL
    even though the key exists (``movie.get('tmdb_rating', 5.0)`` returns
    None, not 5.0); float(None) used to kill the whole recommendation call.
    """
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _to_int(value, default):
    """None-safe int coercion (wave5 fix C1.7); see _to_float."""
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _movie_key(m):
    """Stable identity key for de-duplication across DB ('movie_id') and
    TMDB ('id') payloads (wave5 fix C1.8).

    Entries with neither id fall back to a (tmdb_id, title) tuple so they
    cannot all collide on the None key; mirrors
    ai/candidate_generator._movie_key (:25-35).
    """
    m = m or {}
    key = m.get('id') or m.get('movie_id')
    if key is not None:
        return key
    return ('no-id', m.get('tmdb_id'), m.get('title'))


class CineSenseRecommender:
    """
    Main recommendation engine
    Integrates: Pairwise learning, embeddings, RL + Advanced AI + Lazy Loading
    """
    
    def __init__(self):
        """Initialize the recommendation system"""
        
        # AI Layer 1: Pairwise Learning
        self.pairwise_learner = PairwiseLearner(
            k_factor=Config.ELO_K_FACTOR,
            initial_rating=Config.INITIAL_ELO_SCORE
        )
        
        # AI Layer 2: Content-based filtering (with lazy loading)
        self.movie_embedder = MovieEmbedding()
        self.content_recommender = ContentBasedRecommender()
        
        # Feature encoders (will be fitted on first use)
        self.genre_encoder = FeatureEncoder(max_features=Config.GENRE_DIM)
        self.director_encoder = FeatureEncoder(max_features=Config.DIRECTOR_DIM)
        self.actor_encoder = FeatureEncoder(max_features=Config.ACTOR_DIM)
        self.encoders_fitted = False
        
        # AI Layer 3: Reinforcement Learning
        self.bandit = UCBBandit(c=Config.BANDIT_UCB_C)
        
        # Advanced AI Components
        self.latent_encoder = LatentSpaceEncoder() if Config.USE_DIMENSIONALITY_REDUCTION else None
        self.implicit_processor = ImplicitSignalProcessor()
        self.prob_selector = ProbabilisticSelector() if Config.USE_SOFTMAX_SELECTION else None
        self.memory_manager = TemporalMemoryManager()
        self.nlg_explainer = NaturalLanguageExplainer() if Config.ENABLE_EXPLANATIONS else None
        
        # LAZY LOADING Components
        self.cache = cache_manager  # Sliding window cache
        self.candidate_gen = candidate_generator  # Candidate generation
        self.tmdb = TMDBFetcher()  # Infinite movie stream
        
        # Deep Learning ensemble scorer (NeuMF V2)
        self.dl_scorer = None
        if Config.USE_DL_SCORING:
            try:
                model_dir = Path(Config.MODEL_DIR)

                def _resolve_checkpoint_path(raw_path: str) -> Path:
                    candidate = Path(raw_path)
                    search_paths = [
                        candidate,
                        Path.cwd() / candidate,
                        model_dir / candidate.name,
                    ]
                    for path in search_paths:
                        if path.exists():
                            return path
                    return model_dir / candidate.name

                v2_checkpoint = _resolve_checkpoint_path(Config.DL_V2_CHECKPOINT)
                v1_checkpoint = _resolve_checkpoint_path(Config.DL_V1_CHECKPOINT)

                if not v2_checkpoint.exists() and not v1_checkpoint.exists():
                    logger.info(
                        "DL scorer disabled: checkpoints not found at %s or %s",
                        v2_checkpoint,
                        v1_checkpoint,
                    )
                    self.dl_scorer = None
                else:
                    self.dl_scorer = _get_dl_scorer(
                        v2_path=str(v2_checkpoint),
                        v1_path=str(v1_checkpoint)
                    )
                    if self.dl_scorer.is_loaded:
                        logger.info("DL ensemble scorer loaded (13-model NeuMF)")
                    else:
                        logger.warning("DL scorer checkpoint load incomplete - genre affinity disabled")
                        self.dl_scorer = None
            except Exception as e:
                logger.warning(f"DL scorer init failed: {e} - genre affinity disabled")
                self.dl_scorer = None
        
        # User-specific models
        # NOTE: UserPreferenceModel state is RAM-only; _load_user_model
        # warm-starts it from the user's stored pairwise history on first
        # touch (audit N18), so a process restart no longer regresses users
        # to cold start. The model's replayed comparison_count and the DB
        # cold-start threshold (Config.MIN_INTERACTIONS_FOR_PERSONALIZATION)
        # now agree by construction.
        self.user_models = {}  # user_id -> UserPreferenceModel
        self.user_embeddings = {}  # user_id -> UserEmbedding
        
        logger.info("CineSense Recommender initialized with advanced AI features")
    
    def _fit_encoders_if_needed(self):
        """Fit encoders on first use"""
        if self.encoders_fitted:
            return
        
        try:
            # Get sample of movies to fit encoders
            movies = db.get_top_movies(limit=1000)
            
            if not movies:
                logger.warning("No movies found to fit encoders")
                return
            
            # Extract features
            all_genres = []
            all_directors = []
            all_actors = []
            
            for movie in movies:
                if movie.get('genres'):
                    genres = [g.strip() for g in movie['genres'].split(',')]
                    all_genres.append(genres)
                
                if movie.get('directors'):
                    directors = [d.strip() for d in movie['directors'].split(',')]
                    all_directors.append(directors)
                
                if movie.get('cast'):
                    actors = [a.strip() for a in movie['cast'].split(',')[:10]]  # Top 10
                    all_actors.append(actors)
            
            # Fit encoders
            if all_genres:
                self.genre_encoder.fit(all_genres)
            if all_directors:
                self.director_encoder.fit(all_directors)
            if all_actors:
                self.actor_encoder.fit(all_actors)
            
            self.encoders_fitted = True
            logger.info("Encoders fitted successfully")
            
        except Exception as e:
            logger.error(f"Error fitting encoders: {e}")
    
    def _get_user_model(self, user_id):
        """Get or create user preference model"""
        if user_id not in self.user_models:
            self.user_models[user_id] = self._load_user_model(user_id)
        return self.user_models[user_id]

    def _load_user_model(self, user_id):
        """Create a UserPreferenceModel, warm-started from stored DB history.

        UserPreferenceModel state lives only in RAM, so previously every
        process restart pushed every user back to cold start until they redo
        Config.MIN_INTERACTIONS_FOR_PERSONALIZATION fresh comparisons
        (audit N18). Replaying the stored pairwise rows into a fresh model
        reconstructs the same in-memory state the user would have
        accumulated live. Read-only warm start: new comparisons keep being
        persisted through db.record_interaction as before (no write-through
        is added). Any DB failure degrades gracefully to a cold model.
        """
        model = UserPreferenceModel(user_id)
        try:
            interactions = db.get_user_interactions(
                user_id, limit=Config.MODEL_WARM_START_LIMIT
            )
        except Exception as e:
            logger.warning(f"Could not warm-start user model for {user_id}: {e}")
            return model

        # Rows come back most-recent-first; replay chronologically so the
        # preference accumulation order matches the live experience.
        replayed = 0
        for interaction in reversed(interactions):
            chosen = interaction.get('chosen_movie_id')
            movie_1 = interaction.get('movie_1_id')
            movie_2 = interaction.get('movie_2_id')
            # 'chosen' must be one of the presented pair (skip malformed rows).
            rejected = movie_2 if chosen == movie_1 else (
                movie_1 if chosen == movie_2 else None
            )
            if chosen is None or rejected is None:
                continue
            model.record_preference(chosen, rejected)
            replayed += 1

        if replayed:
            logger.info(
                f"Warm-started preference model for user {user_id} from "
                f"{replayed} stored comparison(s)"
            )
        return model
    
    def _get_user_embedding(self, user_id):
        """Get or create user embedding"""
        if user_id not in self.user_embeddings:
            # Try to load from database
            saved_embedding = db.get_user_embedding(user_id)
            if saved_embedding:
                user_emb = UserEmbedding(Config.TOTAL_VECTOR_DIM)
                user_emb.set_embedding(saved_embedding)
                self.user_embeddings[user_id] = user_emb
            else:
                self.user_embeddings[user_id] = UserEmbedding(Config.TOTAL_VECTOR_DIM)
        
        return self.user_embeddings[user_id]
    
    def _build_user_genre_vec(self, interactions):
        """
        Build a user genre preference vector (18-dim, ML100K genre space)
        from the user's interaction history.  Used by the DL ensemble scorer
        for genre affinity scoring.
        """
        if not self.dl_scorer or not interactions:
            return np.zeros(18, dtype=np.float32)
        
        genre_sum = np.zeros(18, dtype=np.float32)
        count = 0
        for inter in interactions[:50]:
            chosen_id = inter.get('chosen_movie_id')
            if chosen_id:
                movie = db.get_movie_by_id(chosen_id)
                if movie and movie.get('genres'):
                    genre_sum += self.dl_scorer.tmdb_genre_to_vec(movie['genres'])
                    count += 1
        
        if count > 0:
            genre_sum /= count
        return genre_sum
    
    def _movie_to_embedding(self, movie):
        """Convert movie dictionary to embedding"""
        self._fit_encoders_if_needed()
        
        # Parse genres, directors, actors
        genres = []
        if movie.get('genres'):
            genres = [g.strip() for g in movie['genres'].split(',')]
        
        directors = []
        if movie.get('directors'):
            directors = [d.strip() for d in movie['directors'].split(',')]
        
        actors = []
        if movie.get('cast'):
            actors = [a.strip() for a in movie['cast'].split(',')[:10]]
        
        # Create movie data dictionary (convert Decimal to float).
        # wave5 fix (C1.7): None/bad values fall back to defaults instead of
        # one SQL-NULL row killing the whole recommendation call.
        movie_data = {
            'genres': genres,
            'directors': directors,
            'actors': actors,
            'tmdb_rating': _to_float(movie.get('tmdb_rating'), 5.0),
            'popularity': _to_float(movie.get('popularity'), 1.0),
            'release_year': _to_int(movie.get('release_year'), 2000),
            'vote_count': _to_int(movie.get('vote_count'), 0),
            'runtime': _to_int(movie.get('runtime'), 90)
        }
        
        # Create embedding
        embedding = self.movie_embedder.create_embedding(
            movie_data,
            self.genre_encoder,
            self.director_encoder,
            self.actor_encoder
        )
        
        return embedding
    
    def process_user_choice(self, user_id, chosen_movie_id, rejected_movie_id, session_id=None):
        """
        Process a user's pairwise choice and update all AI layers
        
        Args:
            user_id: User identifier
            chosen_movie_id: ID of chosen movie
            rejected_movie_id: ID of rejected movie
            session_id: Optional session identifier
        
        Returns:
            Boolean indicating success
        """
        try:
            # Fetch movie data
            chosen_movie = db.get_movie_by_id(chosen_movie_id)
            rejected_movie = db.get_movie_by_id(rejected_movie_id)
            
            if not chosen_movie or not rejected_movie:
                logger.error("Movie not found")
                return False
            
            # ==============================================================
            # AI LAYER 1: Update ELO scores (pairwise learning)
            # NOTE: the ELO delta is applied atomically by the stored procedure
            # record_user_interaction (schema.sql) — the explicit
            # db.update_movie_elo() call that used to run here applied the SAME
            # K-factor delta a second time, doubling every rating swing and
            # inflating comparison_count 2x. Removed (fresh-eyes audit).
            # ==============================================================

            # Update user preference model
            user_model = self._get_user_model(user_id)
            user_model.record_preference(chosen_movie_id, rejected_movie_id)
            
            # ==============================================================
            # AI LAYER 2: Update user embedding (content-based)
            # ==============================================================
            chosen_embedding = self._movie_to_embedding(chosen_movie)
            rejected_embedding = self._movie_to_embedding(rejected_movie)
            
            user_embedding = self._get_user_embedding(user_id)
            user_embedding.update_from_choice(
                chosen_embedding,
                rejected_embedding,
                learning_rate=Config.LEARNING_RATE
            )
            
            # Save user embedding to database
            db.save_user_embedding(user_id, user_embedding.get_embedding())
            
            # ==============================================================
            # AI LAYER 3: Update bandit (reinforcement learning)
            # ==============================================================
            self.bandit.update(chosen_movie_id, reward=1.0)
            self.bandit.update(rejected_movie_id, reward=0.0)
            
            # ==============================================================
            # Record interaction in database
            # ==============================================================
            db.record_interaction(
                user_id,
                chosen_movie_id,
                rejected_movie_id,
                chosen_movie_id,
                session_id
            )

            # Note: the stored procedure record_user_interaction already
            # recomputes interaction_count internally (schema.sql) — a second
            # explicit call here was a redundant COUNT(*) round-trip.
            
            logger.info(f"Processed choice: User {user_id} chose {chosen_movie_id} over {rejected_movie_id}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error processing user choice: {e}")
            return False
    
    def get_comparison_pair(self, user_id=None, min_popularity=0):
        """
        Get two movies for pairwise comparison
        Uses bandit algorithm to balance exploration/exploitation
        
        Args:
            user_id: Optional user ID for personalization
            min_popularity: Minimum popularity threshold (default 0 for all movies)
        
        Returns:
            Tuple of two movie dictionaries
        """
        try:
            # Get candidate movies (lower threshold for small databases)
            candidates = db.get_random_movies(limit=Config.COMPARISON_POOL_SIZE, min_popularity=min_popularity)
            
            if len(candidates) < 2:
                logger.warning(f"Not enough movies for comparison. Found {len(candidates)} movies")
                # Try without popularity filter
                candidates = db.get_random_movies(limit=Config.COMPARISON_POOL_SIZE, min_popularity=0)
            
            if len(candidates) < 2:
                logger.error("Still not enough movies for comparison")
                return None, None
            
            # Extract movie IDs
            candidate_ids = [m['movie_id'] for m in candidates]
            
            # Use bandit to select two movies
            selected_ids = self.bandit.select_arm(
                candidate_ids, top_k=Config.COMPARISON_BATCH_SIZE)
            
            if len(selected_ids) < 2:
                # Fallback to random (guard against an empty/short candidate pool
                # so np.random.choice([]) cannot raise)
                if len(candidate_ids) < 2:
                    logger.error("Not enough distinct candidates for comparison")
                    return None, None
                selected_ids = list(np.random.choice(candidate_ids, size=2, replace=False))
            
            # Fetch full movie details
            movie1 = db.get_movie_by_id(selected_ids[0])
            movie2 = db.get_movie_by_id(selected_ids[1])
            
            if not movie1 or not movie2:
                logger.error("Could not load full details for comparison pair")
                return None, None
            
            return movie1, movie2
            
        except Exception as e:
            logger.error(f"Error getting comparison pair: {e}")
            return None, None
    
    def get_recommendations(self, user_id, n=20, min_interactions=Config.MIN_INTERACTIONS_FOR_PERSONALIZATION):
        """
        Get personalized recommendations for a user
        
        Args:
            user_id: User identifier
            n: Number of recommendations
            min_interactions: Minimum interactions needed for personalization
        
        Returns:
            List of movie dictionaries with scores
        """
        try:
            user_model = self._get_user_model(user_id)
            
            # Check if user has enough interaction history
            if not user_model.has_enough_data(min_interactions):
                # Cold start: return popular movies
                logger.info(f"User {user_id} has insufficient data, returning popular movies")
                return db.get_top_movies(limit=n, order_by='popularity')
            
            # Get movies user has already interacted with
            interactions = db.get_user_interactions(user_id, limit=Config.RECENT_INTERACTIONS_LIMIT)
            seen_movie_ids = set()
            for interaction in interactions:
                seen_movie_ids.add(interaction['movie_1_id'])
                seen_movie_ids.add(interaction['movie_2_id'])
                if interaction.get('chosen_movie_id'):
                    seen_movie_ids.add(interaction['chosen_movie_id'])

            # Build user genre profile for DL scoring (if available)
            user_genre_vec = self._build_user_genre_vec(interactions) if self.dl_scorer else None

            # Get user embedding
            user_embedding = self._get_user_embedding(user_id)
            user_vector = user_embedding.get_embedding()

            # Get candidate movies
            all_movies = db.get_top_movies(limit=Config.RECOMMEND_CANDIDATE_POOL, order_by='elo_score')

            # Score weights (sum to exactly 1.0)
            if self.dl_scorer and user_genre_vec is not None:
                dl_w = float(getattr(Config, 'DL_SCORE_WEIGHT', 0.15))
                w_content = (1.0 - dl_w) * Config.RECOMMEND_WEIGHT_CONTENT
                w_pref = (1.0 - dl_w) * Config.RECOMMEND_WEIGHT_PREFERENCE
                w_elo = (1.0 - dl_w) * Config.RECOMMEND_WEIGHT_ELO
                w_dl = dl_w
            else:
                w_content = Config.RECOMMEND_WEIGHT_CONTENT
                w_pref = Config.RECOMMEND_WEIGHT_PREFERENCE
                w_elo = Config.RECOMMEND_WEIGHT_ELO
                w_dl = 0.0

            # Score each movie
            movie_scores = []
            u_norm = float(np.linalg.norm(user_vector))

            for movie in all_movies:
                movie_id = movie['movie_id']
                if movie_id in seen_movie_ids:
                    continue

                # 1. Content-based score: normalized cosine similarity [0, 1]
                movie_embedding = self._movie_to_embedding(movie)
                m_norm = float(np.linalg.norm(movie_embedding))
                if u_norm > 0 and m_norm > 0:
                    cosine = float(np.dot(user_vector, movie_embedding) / (u_norm * m_norm))
                    norm_content_score = max(0.0, min(1.0, (cosine + 1.0) / 2.0))
                else:
                    norm_content_score = 0.5

                # 2. Preference score: sigmoid-normalized to [0, 1]
                raw_pref = user_model.get_preference_score(movie_id)
                norm_preference_score = float(1.0 / (1.0 + np.exp(-Config.PREFERENCE_SIGMOID_SCALE * raw_pref)))

                # 3. Global quality: ELO normalized to [0, 1] (shared helper)
                raw_elo = float(movie.get('elo_score', Config.INITIAL_ELO_SCORE))
                norm_elo_score = _normalize_elo(raw_elo)

                # 4. Optional DL genre affinity: [0, 1]
                norm_dl_score = 0.0
                if self.dl_scorer and user_genre_vec is not None:
                    try:
                        movie_genre_vec = self.dl_scorer.tmdb_genre_to_vec(movie.get('genres', ''))
                        norm_dl_score = max(0.0, min(1.0, float(self.dl_scorer.genre_affinity_score(user_genre_vec, movie_genre_vec))))
                    except Exception as dl_err:
                        logger.warning(f"DL scoring error for movie {movie_id}: {dl_err}")
                        norm_dl_score = 0.0

                # Strictly weighted linear combination (weights sum to 1.0)
                final_score = (
                    w_content * norm_content_score +
                    w_pref * norm_preference_score +
                    w_elo * norm_elo_score +
                    w_dl * norm_dl_score
                )

                # Traceable component explanation payload
                explanation_payload = {
                    "content_similarity_contribution": round(w_content * norm_content_score, 4),
                    "preference_elo_contribution": round(w_pref * norm_preference_score + w_elo * norm_elo_score, 4),
                    "dl_contribution": round(w_dl * norm_dl_score, 4) if w_dl > 0 else 0.0,
                    "content_score": round(norm_content_score, 4),
                    "preference_score": round(norm_preference_score, 4),
                    "elo_score": round(norm_elo_score, 4),
                    "matched_genres": movie.get("genres", ""),
                    "reason": f"Matches preferred profile ({norm_content_score:.0%}) with community ELO {raw_elo:.0f}"
                }

                movie_scores.append({
                    **movie,
                    'recommendation_score': final_score,
                    'content_score': norm_content_score,
                    'preference_score': norm_preference_score,
                    'elo_score': norm_elo_score,
                    'dl_genre_affinity': norm_dl_score,
                    'explanation_payload': explanation_payload,
                })

            # Sort by final score
            movie_scores.sort(key=lambda x: x['recommendation_score'], reverse=True)
            return movie_scores[:n]
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            # Fallback to popular movies
            return db.get_top_movies(limit=n, order_by='popularity')
    
    def apply_diversity_rerank(self, ranked_items, top_n=20, lambda_diversity=Config.DIVERSITY_MMR_LAMBDA):
        """
        Maximal Marginal Relevance (MMR) diversity reranking.
        Balances item relevance score with pairwise dissimilarity to already selected items.

        MMR(i) = (1 - lambda) * relevance_score(i) - lambda * max_{j in S} similarity(i, j)

        Args:
            ranked_items: List of movie dicts with 'recommendation_score'
            top_n: Number of diverse items to select
            lambda_diversity: Tradeoff parameter (0.0 = pure relevance, 1.0 = maximum diversity)

        Returns:
            List of top_n reranked movie dicts with updated 'diversity_penalty' and 'mmr_score'
        """
        if not ranked_items or top_n <= 0:
            return []

        if lambda_diversity <= 0.0 or len(ranked_items) <= 1:
            return ranked_items[:top_n]

        # Extract or precompute genre sets for similarity
        def _get_genre_set(m):
            raw = m.get('genres', '')
            if isinstance(raw, str):
                return {g.strip().lower() for g in raw.split(',') if g.strip()}
            elif isinstance(raw, (list, set)):
                return {str(g).strip().lower() for g in raw if str(g).strip()}
            return set()

        genre_cache = {idx: _get_genre_set(item) for idx, item in enumerate(ranked_items)}

        def _jaccard_sim(idx_a, idx_b):
            set_a, set_b = genre_cache[idx_a], genre_cache[idx_b]
            if not set_a and not set_b:
                return 0.5
            union_len = len(set_a | set_b)
            return len(set_a & set_b) / union_len if union_len > 0 else 0.0

        unselected = list(range(len(ranked_items)))
        selected_indices = []

        # Select first item (purely highest score)
        best_first = max(unselected, key=lambda idx: ranked_items[idx].get('recommendation_score', 0.0))
        selected_indices.append(best_first)
        unselected.remove(best_first)

        # Greedily select remaining items using MMR formula
        while unselected and len(selected_indices) < top_n:
            best_idx = None
            best_mmr = float('-inf')

            for cand_idx in unselected:
                rel = float(ranked_items[cand_idx].get('recommendation_score', 0.0))
                max_sim = max(_jaccard_sim(cand_idx, sel_idx) for sel_idx in selected_indices)
                mmr_val = (1.0 - lambda_diversity) * rel - lambda_diversity * max_sim

                if mmr_val > best_mmr:
                    best_mmr = mmr_val
                    best_idx = cand_idx

            if best_idx is not None:
                item = ranked_items[best_idx]
                item['mmr_score'] = round(best_mmr, 4)
                item['diversity_penalty'] = round(lambda_diversity * max(_jaccard_sim(best_idx, s) for s in selected_indices), 4)
                selected_indices.append(best_idx)
                unselected.remove(best_idx)
            else:
                break

        return [ranked_items[i] for i in selected_indices]

    def get_featured_movie(self, user_id=None):
        """
        Get a featured movie for the hero banner
        
        Args:
            user_id: Optional user ID for personalization
        
        Returns:
            Movie dictionary
        """
        if user_id:
            recommendations = self.get_recommendations(user_id, n=5)
            if recommendations:
                return recommendations[0]
        
        # Fallback to top-rated movie
        top_movies = db.get_top_movies(limit=1, order_by='elo_score')
        return top_movies[0] if top_movies else None
    
    def explain_recommendation(self, user_id, movie_id):
        """
        Generate explanation for why a movie was recommended
        
        Args:
            user_id: User identifier
            movie_id: Movie identifier
        
        Returns:
            String explanation
        """
        try:
            movie = db.get_movie_by_id(movie_id)
            if not movie:
                return "Movie not found"
            
            # Get user's past interactions
            interactions = db.get_user_interactions(user_id, limit=10)
            
            if not interactions:
                return "Recommended based on high ratings and popularity."
            
            # Find similar movies user liked
            liked_movies = []
            for interaction in interactions[:5]:
                chosen = db.get_movie_by_id(interaction['chosen_movie_id'])
                if chosen:
                    liked_movies.append(chosen['title'])
            
            # Extract key features
            genres = movie.get('genres', 'Unknown')
            directors = movie.get('directors', 'Unknown')
            
            explanation = f"Recommended because you enjoyed {', '.join(liked_movies[:3])}. "
            explanation += f"This {genres} film "
            
            if directors != 'Unknown':
                explanation += f"directed by {directors} "
            
            explanation += f"has a rating of {movie.get('tmdb_rating', 'N/A')}/10."
            
            return explanation
            
        except Exception as e:
            logger.error(f"Error generating explanation: {e}")
            return "Recommended based on your viewing preferences."
    
    def get_advanced_recommendations(self, user_id, n=20, include_explanations=True):
        """
        Advanced recommendations using all enhanced AI features
        - Latent space representations
        - Probabilistic selection
        - Temporal memory
        - Natural language explanations
        
        Args:
            user_id: User identifier
            n: Number of recommendations
            include_explanations: Whether to generate NLG explanations
        
        Returns:
            List of movies with scores and explanations
        """
        try:
            # Get user interaction history with temporal weights
            interactions = db.get_user_interactions(user_id, limit=Config.RECENT_INTERACTIONS_LIMIT)
            
            # Same cold-start threshold as the production path (DB interaction count).
            if not interactions or len(interactions) < Config.MIN_INTERACTIONS_FOR_PERSONALIZATION:
                logger.info(f"User {user_id} needs more interactions for advanced recommendations")
                return self.get_recommendations(user_id, n)
            
            # Apply temporal decay to interactions
            weighted_interactions = self.memory_manager.apply_temporal_weights(interactions)
            
            # Build user preference profile with temporal weighting
            user_vector = self._build_temporal_user_vector(weighted_interactions)
            
            # Get candidate movies
            all_movies = db.get_top_movies(limit=Config.RECOMMEND_CANDIDATE_POOL)
            
            # Filter out already seen
            seen_ids = set()
            for inter in interactions:
                seen_ids.add(inter.get('movie_1_id'))
                seen_ids.add(inter.get('movie_2_id'))
            
            candidates = [m for m in all_movies if m['movie_id'] not in seen_ids]
            
            # Score each movie
            movie_scores = []
            for movie in candidates:
                # Get movie embedding
                movie_emb = self._movie_to_embedding(movie)
                
                # Transform to latent space if enabled
                if self.latent_encoder and self.latent_encoder.is_fitted:
                    movie_latent = self.latent_encoder.transform(movie_emb)
                    user_latent = self.latent_encoder.transform(user_vector)
                    # Compute similarity in latent space
                    content_score = np.dot(user_latent, movie_latent) / (
                        np.linalg.norm(user_latent) * np.linalg.norm(movie_latent) + 1e-8
                    )
                else:
                    content_score = np.dot(user_vector, movie_emb) / (
                        np.linalg.norm(user_vector) * np.linalg.norm(movie_emb) + 1e-8
                    )
                
                # Preference score from pairwise learning
                user_model = self._get_user_model(user_id)
                preference_score = user_model.get_preference_score(movie['movie_id'])
                
                # Combined score. The ELO term is normalized to [0, 1] via the
                # shared helper so this path matches the production path's scale
                # for the `recommendation_score` key.
                final_score = (
                    Config.RECOMMEND_WEIGHT_CONTENT * content_score +
                    Config.RECOMMEND_WEIGHT_PREFERENCE * preference_score +
                    Config.RECOMMEND_WEIGHT_ELO * _normalize_elo(movie.get('elo_score', Config.INITIAL_ELO_SCORE))
                )
                
                movie_scores.append({
                    **movie,
                    'recommendation_score': final_score,
                    'content_score': content_score,
                    'preference_score': preference_score
                })
            
            # Use probabilistic selection if enabled
            if self.prob_selector and Config.USE_SOFTMAX_SELECTION:
                scores = [m['recommendation_score'] for m in movie_scores]
                selected_indices = self.prob_selector.select_with_probability(scores, top_k=n)
                recommendations = [movie_scores[i] for i in selected_indices]
            else:
                # Deterministic: sort by score
                movie_scores.sort(key=lambda x: x['recommendation_score'], reverse=True)
                recommendations = movie_scores[:n]
            
            # Generate natural language explanations
            if include_explanations and self.nlg_explainer:
                user_profile = self._build_user_taste_profile(weighted_interactions)
                for rec in recommendations:
                    preference_factors = {
                        'preferred_genres': user_profile.get('top_genres', []),
                        'favorite_directors': user_profile.get('top_directors', []),
                        'pacing_preference': user_profile.get('pacing', 'balanced')
                    }
                    rec['explanation'] = self.nlg_explainer.explain_recommendation(
                        rec, user_profile, rec['content_score'], preference_factors
                    )
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Error in advanced recommendations: {e}")
            return self.get_recommendations(user_id, n)
    
    def _build_temporal_user_vector(self, weighted_interactions):
        """Build user vector with temporal weighting"""
        if not weighted_interactions:
            return np.zeros(Config.TOTAL_VECTOR_DIM)
        
        # Weight recent interactions more heavily
        recent_vectors = []
        weights = []
        
        for inter in weighted_interactions[:20]:  # Top 20 recent
            movie_id = inter.get('chosen_movie_id') or inter.get('movie_1_id')
            if movie_id:
                movie = db.get_movie_by_id(movie_id)
                if movie:
                    emb = self._movie_to_embedding(movie)
                    recent_vectors.append(emb)
                    weights.append(inter.get('temporal_weight', 1.0))
        
        if not recent_vectors:
            return np.zeros(Config.TOTAL_VECTOR_DIM)
        
        # Weighted average
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        user_vector = np.average(recent_vectors, axis=0, weights=weights)
        return user_vector
    
    def _build_user_taste_profile(self, interactions):
        """Extract interpretable taste profile for NLG"""
        profile = {}
        
        # Collect genres, directors from chosen movies
        genres_count = {}
        directors_count = {}
        years = []
        ratings = []
        
        for inter in interactions[:30]:
            movie_id = inter.get('chosen_movie_id')
            if movie_id:
                movie = db.get_movie_by_id(movie_id)
                if movie:
                    # Count genres
                    if movie.get('genres'):
                        for genre in movie['genres'].split(','):
                            genre = genre.strip()
                            genres_count[genre] = genres_count.get(genre, 0) + 1
                    
                    # Count directors
                    if movie.get('directors'):
                        for director in movie['directors'].split(','):
                            director = director.strip()
                            directors_count[director] = directors_count.get(director, 0) + 1
                    
                    # Track years and ratings
                    if movie.get('release_year'):
                        years.append(movie['release_year'])
                    if movie.get('tmdb_rating'):
                        ratings.append(float(movie['tmdb_rating']))
        
        # Top genres
        profile['top_genres'] = sorted(genres_count.keys(), key=genres_count.get, reverse=True)[:3]
        
        # Top directors
        profile['top_directors'] = sorted(directors_count.keys(), key=directors_count.get, reverse=True)[:2]
        
        # Average year
        profile['avg_release_year'] = int(np.mean(years)) if years else 2000
        
        # Average rating preference
        profile['avg_rating_preference'] = np.mean(ratings) if ratings else 7.0
        
        # Infer pacing (heuristic based on genres)
        action_count = genres_count.get('Action', 0) + genres_count.get('Thriller', 0)
        drama_count = genres_count.get('Drama', 0) + genres_count.get('Mystery', 0)
        
        if action_count > drama_count * 1.5:
            profile['pacing'] = 'fast-paced'
        elif drama_count > action_count * 1.5:
            profile['pacing'] = 'slow-burn'
        else:
            profile['pacing'] = 'balanced'
        
        return profile
    
    def get_taste_summary(self, user_id):
        """
        Generate natural language taste personality summary
        """
        if not self.nlg_explainer:
            return "Explanations not enabled"
        
        interactions = db.get_user_interactions(user_id, limit=50)
        if not interactions:
            return "Not enough data to build your taste profile yet."
        
        weighted = self.memory_manager.apply_temporal_weights(interactions)
        profile = self._build_user_taste_profile(weighted)
        
        return self.nlg_explainer.generate_taste_summary(profile)
    
    def get_comparison_pair_lazy(self, user_id=None):
        """
        LAZY LOADING: Get pairwise comparison using infinite stream
        
        Infinite Pairwise Strategy:
        - 50% known-preference movies (from user history or popular)
        - 50% unexplored movies (from TMDB API)
        
        This gives: Learning + Exploration + Infinite content
        
        Returns:
            Tuple of two movie dictionaries
        """
        try:
            # Check cache status
            if self.cache.needs_refill(threshold=Config.CACHE_REFILL_THRESHOLD):
                logger.info("Cache low - refilling from TMDB API...")
                self._refill_cache(user_id)
            
            # Generate pairwise candidates
            candidates = self.candidate_gen.generate_pairwise_candidates(
                user_id=user_id,
                count=Config.PAIRWISE_BATCH_SIZE
            )
            
            # Pick 1 known + 1 explore
            known_movies = candidates.get('known', [])
            explore_movies = candidates.get('explore', [])
            
            if not known_movies or not explore_movies:
                # Fallback to cache
                cached_movies = self.cache.movie_cache.get_all_movies()
                if len(cached_movies) >= 2:
                    import random
                    selected = random.sample(cached_movies, 2)
                    return selected[0], selected[1]
                else:
                    logger.error("Not enough movies for comparison")
                    return None, None
            
            # Random selection
            import random
            movie1 = random.choice(known_movies)
            movie2 = random.choice(explore_movies)
            
            # Never return the same movie twice (known/explore may overlap).
            # wave5 fix (C1.8): shared _movie_key helper (tuple fallback) so
            # id-less movies cannot all collapse onto the None key.
            if _movie_key(movie1) is not None and _movie_key(movie1) == _movie_key(movie2):
                alternatives = [m for m in explore_movies if _movie_key(m) != _movie_key(movie1)]
                if not alternatives:
                    alternatives = [m for m in known_movies if _movie_key(m) != _movie_key(movie1)]
                if alternatives:
                    movie2 = random.choice(alternatives)
            
            if not movie1 or not movie2:
                logger.error("Could not build a valid lazy comparison pair")
                return None, None
            
            # Cache these movies
            self.cache.put_movie(_movie_key(movie1), movie1)
            self.cache.put_movie(_movie_key(movie2), movie2)
            
            return movie1, movie2
            
        except Exception as e:
            logger.error(f"Error in lazy comparison pair: {e}")
            return None, None
    
    def get_recommendations_lazy(self, user_id, n=20, strategy=None):
        """
        LAZY LOADING: Get recommendations using candidate generation
        
        Production-Ready Strategy:
        1. Generate 200-500 candidates (not all movies!)
        2. Rank only candidates
        3. Return top N
        
        This is how real systems work!
        
        Args:
            user_id: User identifier
            n: Number of recommendations (default 20)
            strategy: Candidate strategy ('mixed', 'genre', 'popularity',
                'exploration'). Advisory: when omitted or unknown, the configured
                ``Config.CANDIDATE_STRATEGY`` is used.
        
        Returns:
            List of movie dictionaries with scores
        """
        try:
            # Unknown/None strategies fall back to the configured default so an
            # unexpected API value cannot silently yield an empty candidate set.
            valid_strategies = ('mixed', 'genre', 'popularity', 'exploration')
            if strategy not in valid_strategies:
                if strategy is not None:
                    logger.warning(f"Unknown strategy '{strategy}', using '{Config.CANDIDATE_STRATEGY}'")
                strategy = Config.CANDIDATE_STRATEGY
            
            # Step 1: Generate candidates (200-500 movies, not all!)
            candidates = self.candidate_gen.generate_candidates(
                user_id=user_id,
                target_count=Config.CANDIDATE_COUNT,
                strategy=strategy
            )
            
            if not candidates:
                logger.warning("No candidates generated, falling back to cache")
                candidates = self.cache.movie_cache.get_all_movies()
            
            # Extract candidate IDs (wave5 fix C1.8: _movie_key handles
            # id-less rows via a tuple fallback instead of all-None keys)
            candidate_ids = [_movie_key(m) for m in candidates]
            
            # Step 2: Get user embedding (if exists)
            user_model = self._get_user_model(user_id)
            
            if not user_model.has_enough_data(Config.MIN_INTERACTIONS_FOR_PERSONALIZATION):
                # Cold start: return popular from candidates
                logger.info(f"User {user_id} cold start - returning popular candidates")
                # Sort by popularity
                candidates_sorted = sorted(
                    candidates,
                    key=lambda x: x.get('popularity', 0),
                    reverse=True
                )
                return candidates_sorted[:n]
            
            # Step 3: Rank candidates (not all movies!)
            user_embedding = self._get_user_embedding(user_id)
            user_vector = user_embedding.get_embedding()
            
            # Initialize encoders if needed
            if not self.content_recommender.genre_encoder:
                self._fit_encoders_from_candidates(candidates)
                self.content_recommender.initialize_encoders(
                    self.genre_encoder,
                    self.director_encoder,
                    self.actor_encoder
                )
            
            # Rank candidates using content-based filtering
            ranked = self.content_recommender.recommend_for_user(
                user_embedding=user_vector,
                candidate_ids=candidate_ids,
                n=n
            )
            
            # Convert IDs back to full movie data (wave5 fix C1.8: same
            # _movie_key identity used above when building candidate_ids)
            id_to_movie = {_movie_key(m): m for m in candidates}
            recommendations = []
            
            for movie_id, score in ranked:
                if movie_id in id_to_movie:
                    movie = id_to_movie[movie_id].copy()
                    movie['recommendation_score'] = float(score)
                    recommendations.append(movie)
            
            return recommendations[:n]
            
        except Exception as e:
            logger.error(f"Error in lazy recommendations: {e}")
            # Fallback to cached movies
            return self.cache.movie_cache.get_all_movies()[:n]
    
    def _refill_cache(self, user_id=None):
        """
        Refill sliding window cache from TMDB API
        
        Fetches next batch when cache runs low
        """
        try:
            # Determine how many movies to fetch
            current_size = self.cache.movie_cache.size()
            target_size = Config.MOVIE_CACHE_SIZE
            fetch_count = target_size - current_size
            
            if fetch_count <= 0:
                return
            
            logger.info(f"Refilling cache: need {fetch_count} movies")
            
            # Get user's favorite genres if available.
            # (Removed: the old placeholder try/pass fetched interactions it
            # never used — dead code; the genre analysis was never wired up.)
            filters = {}
            
            # Fetch movies from TMDB
            pages_needed = (fetch_count // Config.MOVIES_PER_PAGE) + 1
            movies_fetched = []
            
            for page in range(1, min(pages_needed + 1, Config.MAX_PAGES_PER_FETCH + 1)):
                data = self.tmdb.discover_movies(page=page, **filters)
                
                if data and 'results' in data:
                    movies_fetched.extend(data['results'])
                
                if len(movies_fetched) >= fetch_count:
                    break
            
            # Add to cache
            self.cache.movie_cache.bulk_put(movies_fetched[:fetch_count])
            
            logger.info(f"Cache refilled: added {min(fetch_count, len(movies_fetched))} movies")
            
        except Exception as e:
            logger.error(f"Error refilling cache: {e}")
    
    def _fit_encoders_from_candidates(self, candidates):
        """
        Fit feature encoders from candidate movies
        
        Lazy fitting - only when needed
        
        Only genre NAMES are used to fit the shared encoder. Numeric TMDB genre
        IDs (e.g. '28') are skipped so the encoder vocabulary is not contaminated
        with IDs for genres that are later matched by name.
        """
        if self.encoders_fitted:
            return
        
        try:
            # Extract genre names only (skip numeric IDs / name-less entries)
            all_genres = []
            all_directors = []
            all_actors = []
            
            for movie in candidates:
                raw_genres = movie.get('genre_ids', []) or movie.get('genres', [])
                # wave5 fix (C1.2): DB rows carry 'Action,Drama' as a string;
                # without splitting, the loop below iterated the string per
                # character and contaminated the encoder vocabulary.
                if isinstance(raw_genres, str):
                    raw_genres = [p.strip() for p in raw_genres.split(',') if p.strip()]
                names = []
                for g in raw_genres:
                    if isinstance(g, dict):
                        name = g.get('name')
                    elif isinstance(g, str):
                        # Ignore bare numeric IDs from TMDB genre_ids
                        name = None if g.strip().isdigit() else g
                    else:
                        name = None
                    if name and str(name).strip():
                        names.append(str(name).strip())
                all_genres.append(names)
                
                # Directors and actors would need additional API calls
                # For now, use empty lists
                all_directors.append([])
                all_actors.append([])
            
            if not any(all_genres):
                # No usable genre names in the candidate pool - leave the
                # encoders unfitted rather than training them on ID strings.
                logger.info("Skipping encoder fit: candidates expose no genre names")
                return
            
            # Fit encoders
            self.genre_encoder.fit(all_genres)
            self.director_encoder.fit(all_directors)
            self.actor_encoder.fit(all_actors)
            
            self.encoders_fitted = True
            logger.info("Encoders fitted from candidates")
            
        except Exception as e:
            logger.error(f"Error fitting encoders: {e}")
    
    def get_cache_stats(self):
        """Get cache statistics for monitoring"""
        return self.cache.get_stats()
    
    def semantic_search(self, query, n=20, include_tv_series=True):
        """
        AI-powered semantic search using embeddings and NLP
        
        Args:
            query: Natural language search query
            n: Number of results to return
            include_tv_series: Whether to include TV series in results
        
        Returns:
            List of matching movies/TV series with relevance scores
        """
        try:
            logger.info(f"Semantic search: '{query}', include_tv={include_tv_series}")
            
            # Get all movies for semantic matching (skip problematic keyword search)
            media_filter = 'all' if include_tv_series else 'movie'
            all_content = db.get_top_movies(limit=2000, media_type=media_filter)
            
            # Create query embedding
            # For now, use simple keyword matching with genre/theme extraction
            query_lower = query.lower()
            
            # Extract potential genre/theme keywords
            genre_keywords = {
                'action': ['action', 'fight', 'battle', 'war', 'combat'],
                'thriller': ['thriller', 'suspense', 'tension', 'mystery', 'twist'],
                'comedy': ['comedy', 'funny', 'hilarious', 'laugh', 'humor'],
                'drama': ['drama', 'emotional', 'touching', 'powerful'],
                'horror': ['horror', 'scary', 'frightening', 'terror', 'creepy'],
                'romance': ['romance', 'love', 'romantic', 'relationship'],
                'sci-fi': ['sci-fi', 'science fiction', 'futuristic', 'space', 'alien'],
                'fantasy': ['fantasy', 'magic', 'magical', 'wizard', 'mythical'],
                'animation': ['animation', 'animated', 'cartoon'],
                'documentary': ['documentary', 'real', 'true story']
            }
            
            # Mood/theme keywords
            mood_keywords = {
                'mind-bending': ['mind-bending', 'complex', 'twist', 'psychological'],
                'dark': ['dark', 'gritty', 'noir', 'bleak'],
                'uplifting': ['uplifting', 'inspiring', 'heartwarming', 'feel-good'],
                'intense': ['intense', 'gripping', 'edge-of-seat', 'thrilling']
            }
            
            # Score each movie based on query
            scored_results = []
            query_words = query_lower.split()
            
            for content in all_content:
                score = 0.0
                
                # Get content fields
                title = (content.get('title') or '').lower()
                overview = (content.get('overview') or '').lower()
                content_genres = (content.get('genres') or '').lower()
                directors = (content.get('directors') or '').lower()
                cast = (content.get('cast') or '').lower()
                
                # 1. Exact title match (very high score)
                if query_lower in title:
                    score += 50.0
                
                # 2. Title word matching (high score)
                for word in query_words:
                    if len(word) > 2 and word in title:
                        score += 15.0
                
                # 3. Overview matching (medium score)
                if query_lower in overview:
                    score += 20.0
                
                # 4. Individual word matches in overview
                for word in query_words:
                    if len(word) > 2:
                        if word in overview:
                            score += 8.0
                        elif any(word in w for w in overview.split()):
                            # wave5 fix (C1.4): the substring hit above already
                            # scored +8; making this branch elif stops every
                            # hit contributing a double-counted 11.
                            score += 3.0
                
                # 5. Genre matching
                for genre, keywords in genre_keywords.items():
                    if any(kw in query_lower for kw in keywords):
                        # wave5 fix (C1.5): DB genre strings use the canonical
                        # 'Science Fiction', so the literal 'sci-fi' key must
                        # also match that alias or the boost never fires.
                        if genre == 'sci-fi':
                            if 'sci-fi' in content_genres or 'science fiction' in content_genres:
                                score += 10.0
                        elif genre in content_genres:
                            score += 10.0
                
                # 6. Mood/theme matching
                for mood, keywords in mood_keywords.items():
                    if any(kw in query_lower for kw in keywords):
                        if any(kw in overview for kw in keywords):
                            score += 5.0
                
                # 7. Director/actor matching
                for word in query_words:
                    if len(word) > 2:
                        if word in directors:
                            score += 12.0
                        if word in cast:
                            score += 8.0
                
                # 8. Rating and popularity boost
                rating = float(content.get('tmdb_rating') or 0)
                popularity = float(content.get('popularity') or 0)
                if score > 0:  # Only boost if there's already a match
                    score += (rating / 10) * 2.0  # Up to 2.0 bonus
                    score += min(popularity / 100, 2.0)  # Up to 2.0 bonus
                
                if score > 0:
                    scored_results.append({
                        **content,
                        'search_score': score,
                        'search_query': query
                    })
            
            # Sort by score
            scored_results.sort(key=lambda x: x['search_score'], reverse=True)
            
            # Return top N results
            results = scored_results[:n]
            
            logger.info(f"Semantic search found {len(results)} results")
            
            return results
            
        except Exception as e:
            logger.error(f"Semantic search error: {e}")
            import traceback
            traceback.print_exc()
            # Return empty list on error
            return []


# Singleton instance
recommender = CineSenseRecommender()


if __name__ == "__main__":
    print("Testing CineSense Recommender")
    print("=" * 50)
    
    # Note: This requires database connection and movie data
    try:
        # Test getting comparison pair
        print("\nTesting comparison pair selection...")
        movie1, movie2 = recommender.get_comparison_pair()
        
        if movie1 and movie2:
            print(f"Movie 1: {movie1['title']}")
            print(f"Movie 2: {movie2['title']}")
            print("Comparison pair generated")
        else:
            print("Could not generate comparison pair")
            print("  (This is expected if database is empty)")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        print("  Make sure database is set up and populated")
