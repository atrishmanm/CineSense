"""
Main Recommendation Engine
Combines all three AI layers for intelligent recommendations
"""

import numpy as np
from ai.pairwise_learning import PairwiseLearner, UserPreferenceModel
from ai.embeddings import MovieEmbedding, UserEmbedding, FeatureEncoder, ContentBasedRecommender
from ai.reinforcement import UCBBandit
from database.db_manager import db
from config import Config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CineSenseRecommender:
    """
    Main recommendation engine
    Integrates pairwise learning, embeddings, and reinforcement learning
    """
    
    def __init__(self):
        """Initialize the recommendation system"""
        
        # AI Layer 1: Pairwise Learning
        self.pairwise_learner = PairwiseLearner(
            k_factor=32,
            initial_rating=Config.INITIAL_ELO_SCORE
        )
        
        # AI Layer 2: Content-based filtering
        self.movie_embedder = MovieEmbedding()
        self.content_recommender = ContentBasedRecommender()
        
        # Feature encoders (will be fitted on first use)
        self.genre_encoder = FeatureEncoder(max_features=Config.GENRE_DIM)
        self.director_encoder = FeatureEncoder(max_features=Config.DIRECTOR_DIM)
        self.actor_encoder = FeatureEncoder(max_features=Config.ACTOR_DIM)
        self.encoders_fitted = False
        
        # AI Layer 3: Reinforcement Learning
        self.bandit = UCBBandit(c=2.0)
        
        # User-specific models
        self.user_models = {}  # user_id -> UserPreferenceModel
        self.user_embeddings = {}  # user_id -> UserEmbedding
        
        logger.info("CineSense Recommender initialized")
    
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
        
        # Create movie data dictionary
        movie_data = {
            'genres': genres,
            'directors': directors,
            'actors': actors,
            'tmdb_rating': movie.get('tmdb_rating', 5.0),
            'popularity': movie.get('popularity', 1.0),
            'release_year': movie.get('release_year', 2000),
            'vote_count': movie.get('vote_count', 0),
            'runtime': movie.get('runtime', 90)
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
    
    def get_comparison_pair(self, user_id=None, min_popularity=10):
        """
        Get two movies for pairwise comparison
        Uses bandit algorithm to balance exploration/exploitation
        
        Args:
            user_id: Optional user ID for personalization
            min_popularity: Minimum popularity threshold
        
        Returns:
            Tuple of two movie dictionaries
        """
        try:
            # Get candidate movies
            candidates = db.get_random_movies(limit=50, min_popularity=min_popularity)
            
            if len(candidates) < 2:
                logger.error("Not enough movies for comparison")
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
            
            # Get all movies
            all_movies = db.get_top_movies(limit=500, order_by='elo_score')
            
            # Get movies user has already interacted with
            interactions = db.get_user_interactions(user_id, limit=100)
            seen_movie_ids = set()
            for interaction in interactions:
                seen_movie_ids.add(interaction['movie_1_id'])
                seen_movie_ids.add(interaction['movie_2_id'])
            
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
                
                # Combine scores (weighted average)
                final_score = (
                    0.5 * content_score +
                    0.3 * preference_score +
                    0.2 * elo_score
                )
                
                movie_scores.append({
                    **movie,
                    'recommendation_score': final_score,
                    'content_score': content_score,
                    'preference_score': preference_score,
                    'elo_score': elo_score
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
            print("✓ Comparison pair generated")
        else:
            print("✗ Could not generate comparison pair")
            print("  (This is expected if database is empty)")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        print("  Make sure database is set up and populated")
