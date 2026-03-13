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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CineSenseRecommender:
    """
    Main recommendation engine
    Integrates: Pairwise learning, embeddings, RL + Advanced AI + Lazy Loading
    """
    
    def __init__(self):
        """Initialize the recommendation system"""
        
        # AI Layer 1: Pairwise Learning
        self.pairwise_learner = PairwiseLearner(
            k_factor=32,
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
        self.bandit = UCBBandit(c=2.0)
        
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
                        logger.info("DL ensemble scorer loaded (13-model NeuMF, RMSE=0.8932)")
                    else:
                        logger.warning("DL scorer checkpoint load incomplete - genre affinity disabled")
                        self.dl_scorer = None
            except Exception as e:
                logger.warning(f"DL scorer init failed: {e} - genre affinity disabled")
                self.dl_scorer = None
        
        # User-specific models
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
            self.user_models[user_id] = UserPreferenceModel(user_id)
        return self.user_models[user_id]
    
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
        
        # Create movie data dictionary (convert Decimal to float)
        movie_data = {
            'genres': genres,
            'directors': directors,
            'actors': actors,
            'tmdb_rating': float(movie.get('tmdb_rating', 5.0)),
            'popularity': float(movie.get('popularity', 1.0)),
            'release_year': int(movie.get('release_year', 2000)),
            'vote_count': int(movie.get('vote_count', 0)),
            'runtime': int(movie.get('runtime', 90))
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
            # ==============================================================
            db.update_movie_elo(chosen_movie_id, rejected_movie_id, k_factor=32)
            
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
            
            # Update user interaction count
            db.update_user_interaction_count(user_id)
            
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
            candidates = db.get_random_movies(limit=50, min_popularity=min_popularity)
            
            if len(candidates) < 2:
                logger.warning(f"Not enough movies for comparison. Found {len(candidates)} movies")
                # Try without popularity filter
                candidates = db.get_random_movies(limit=50, min_popularity=0)
            
            if len(candidates) < 2:
                logger.error("Still not enough movies for comparison")
                return None, None
            
            # Extract movie IDs
            candidate_ids = [m['movie_id'] for m in candidates]
            
            # Use bandit to select two movies
            selected_ids = self.bandit.select_arm(candidate_ids, top_k=2)
            
            if len(selected_ids) < 2:
                # Fallback to random
                selected_ids = np.random.choice(candidate_ids, size=2, replace=False)
            
            # Fetch full movie details
            movie1 = db.get_movie_by_id(selected_ids[0])
            movie2 = db.get_movie_by_id(selected_ids[1])
            
            return movie1, movie2
            
        except Exception as e:
            logger.error(f"Error getting comparison pair: {e}")
            return None, None
    
    def get_recommendations(self, user_id, n=20, min_interactions=5):
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
            
            # Get user embedding
            user_embedding = self._get_user_embedding(user_id)
            user_vector = user_embedding.get_embedding()
            
            # Build user genre profile for DL scoring (if available)
            user_genre_vec = self._build_user_genre_vec(interactions) if self.dl_scorer else None
            
            # Get all movies
            all_movies = db.get_top_movies(limit=500, order_by='elo_score')
            
            # Get movies user has already interacted with
            interactions = db.get_user_interactions(user_id, limit=100)
            seen_movie_ids = set()
            for interaction in interactions:
                seen_movie_ids.add(interaction['movie_1_id'])
                seen_movie_ids.add(interaction['movie_2_id'])
                # Also add the chosen movie specifically
                if interaction.get('chosen_movie_id'):
                    seen_movie_ids.add(interaction['chosen_movie_id'])
            
            # Score each movie
            movie_scores = []
            
            for movie in all_movies:
                movie_id = movie['movie_id']
                
                # Skip seen movies
                if movie_id in seen_movie_ids:
                    continue
                
                # Get movie embedding
                movie_embedding = self._movie_to_embedding(movie)
                
                # Calculate combined score
                # 1. Content-based score (embedding similarity)
                content_score = np.dot(user_vector, movie_embedding)
                
                # 2. Preference score from pairwise learning
                preference_score = user_model.get_preference_score(movie_id)
                
                # 3. Global quality (ELO score, normalized)
                elo_score = movie.get('elo_score', 1500) / 3000  # Normalize to ~0-1
                
                # 4. DL genre affinity (learned from NeuMF ensemble)
                dl_score = 0.0
                if self.dl_scorer and user_genre_vec is not None:
                    movie_genre_vec = self.dl_scorer.tmdb_genre_to_vec(
                        movie.get('genres', ''))
                    dl_score = self.dl_scorer.genre_affinity_score(
                        user_genre_vec, movie_genre_vec)
                
                # Combine scores (weighted average)
                dl_w = Config.DL_SCORE_WEIGHT if self.dl_scorer else 0.0
                remaining = 1.0 - dl_w
                final_score = (
                    remaining * 0.5 / 0.85 * content_score +   # ~50% of remaining
                    remaining * 0.3 / 0.85 * preference_score + # ~30% of remaining
                    remaining * 0.2 / 0.85 * elo_score +        # ~20% of remaining (sums to remaining * 1.0/0.85 ≈ remaining when ~0.85)
                    dl_w * dl_score
                ) if dl_w > 0 else (
                    0.5 * content_score +
                    0.3 * preference_score +
                    0.2 * elo_score
                )
                
                movie_scores.append({
                    **movie,
                    'recommendation_score': final_score,
                    'content_score': content_score,
                    'preference_score': preference_score,
                    'elo_score': elo_score,
                    'dl_genre_affinity': dl_score
                })
            
            # Sort by final score
            movie_scores.sort(key=lambda x: x['recommendation_score'], reverse=True)
            
            return movie_scores[:n]
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            # Fallback to popular movies
            return db.get_top_movies(limit=n, order_by='popularity')
    
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
                return f"Recommended based on high ratings and popularity."
            
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
            interactions = db.get_user_interactions(user_id, limit=100)
            
            if not interactions or len(interactions) < Config.MIN_INTERACTIONS_FOR_PERSONALIZATION:
                logger.info(f"User {user_id} needs more interactions for advanced recommendations")
                return self.get_recommendations(user_id, n)
            
            # Apply temporal decay to interactions
            weighted_interactions = self.memory_manager.apply_temporal_weights(interactions)
            
            # Build user preference profile with temporal weighting
            user_vector = self._build_temporal_user_vector(weighted_interactions)
            
            # Get candidate movies
            all_movies = db.get_top_movies(limit=500)
            
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
                
                # Combined score
                final_score = 0.5 * content_score + 0.3 * preference_score + 0.2 * (movie.get('elo_score', 1500) / 3000)
                
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
            if self.cache.needs_refill(threshold=0.3):
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
            
            # Cache these movies
            self.cache.put_movie(movie1.get('id') or movie1.get('movie_id'), movie1)
            self.cache.put_movie(movie2.get('id') or movie2.get('movie_id'), movie2)
            
            return movie1, movie2
            
        except Exception as e:
            logger.error(f"Error in lazy comparison pair: {e}")
            return None, None
    
    def get_recommendations_lazy(self, user_id, n=20):
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
        
        Returns:
            List of movie dictionaries with scores
        """
        try:
            # Step 1: Generate candidates (200-500 movies, not all!)
            candidates = self.candidate_gen.generate_candidates(
                user_id=user_id,
                target_count=Config.CANDIDATE_COUNT,
                strategy=Config.CANDIDATE_STRATEGY
            )
            
            if not candidates:
                logger.warning("No candidates generated, falling back to cache")
                candidates = self.cache.movie_cache.get_all_movies()
            
            # Extract candidate IDs
            candidate_ids = [m.get('id') or m.get('movie_id') for m in candidates]
            
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
            
            # Convert IDs back to full movie data
            id_to_movie = {(m.get('id') or m.get('movie_id')): m for m in candidates}
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
            
            # Get user's favorite genres if available
            filters = {}
            if user_id:
                try:
                    # Get user's favorite genres
                    interactions = db.get_user_interactions(user_id, limit=20)
                    if interactions:
                        # Placeholder: would analyze genres from interactions
                        pass
                except Exception as e:
                    logger.error(f"Error getting user preferences: {e}")
            
            # Fetch movies from TMDB
            pages_needed = (fetch_count // 20) + 1
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
        """
        if self.encoders_fitted:
            return
        
        try:
            # Extract features from candidates
            all_genres = []
            all_directors = []
            all_actors = []
            
            for movie in candidates:
                genres = movie.get('genre_ids', []) or movie.get('genres', [])
                all_genres.append([str(g) if isinstance(g, int) else g.get('name', g) for g in genres])
                
                # Directors and actors would need additional API calls
                # For now, use empty lists
                all_directors.append([])
                all_actors.append([])
            
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
                        # Partial word matches
                        if any(word in w for w in overview.split()):
                            score += 3.0
                
                # 5. Genre matching
                for genre, keywords in genre_keywords.items():
                    if any(kw in query_lower for kw in keywords):
                        if genre in content_genres:
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
