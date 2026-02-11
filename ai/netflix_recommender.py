"""
Netflix-Level Deep Learning Recommender
Integrates Two-Tower NCF + Content-Aware Hybrid Model with Flask app
"""

import torch
import numpy as np
import logging
from pathlib import Path
from typing import List, Dict
from ai.inference_pipeline import load_production_model, ProductionRecommender
from database.db_manager import db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class NetflixRecommender:
    """
    Production recommender using Netflix-level deep learning
    
    This is the main interface for the Flask app
    """
    
    def __init__(self, model_path='ai/models/best_model.pt', cache_dir='ai/cache'):
        self.model_path = Path(model_path)
        self.cache_dir = Path(cache_dir)
        
        self.recommender = None
        self._model_loaded = False
        
        # Try to load model on initialization
        self._try_load_model()
    
    def _try_load_model(self):
        """Try to load model if available"""
        if not self.model_path.exists():
            logger.warning(f"Model not found at {self.model_path}")
            logger.warning("Run training first: python -m ai.training_pipeline")
            return False
        
        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.recommender = load_production_model(
                model_path=str(self.model_path),
                cache_dir=str(self.cache_dir),
                device=device
            )
            
            self._model_loaded = True
            logger.info("✓ Netflix-level deep learning model loaded successfully")
            return True
        
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def is_ready(self):
        """Check if model is loaded and ready"""
        return self._model_loaded
    
    def get_recommendations(self, user_id, k=20, use_diversity=True):
        """
        Get personalized recommendations for a user
        
        Args:
            user_id: int - user ID
            k: int - number of recommendations
            use_diversity: bool - apply diversity reranking
        
        Returns:
            recommendations: List[Dict] with movie details
        """
        if not self._model_loaded:
            logger.warning("Model not loaded, returning empty recommendations")
            return []
        
        try:
            # Get user's watch history from database
            user_history = self._get_user_history(user_id)
            
            # Get recommendations from model
            recommendations = self.recommender.recommend(
                user_id=user_id,
                user_history=user_history,
                k=k,
                use_diversity=use_diversity,
                num_candidates=500
            )
            
            # Enrich with database info (posters, etc.)
            enriched_recommendations = self._enrich_recommendations(recommendations)
            
            return enriched_recommendations
        
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return []
    
    def _get_user_history(self, user_id):
        """Get user's watch history from database"""
        try:
            # Get user's rated movies
            ratings = db.get_user_ratings(user_id)
            
            history = []
            for rating in ratings:
                history.append({
                    'movieId': rating['movie_id'],
                    'rating': rating['rating'],
                    'timestamp': rating.get('timestamp')
                })
            
            return history
        
        except Exception as e:
            logger.error(f"Error fetching user history: {e}")
            return []
    
    def _enrich_recommendations(self, recommendations):
        """Enrich recommendations with database info"""
        enriched = []
        
        for rec in recommendations:
            movie_id = rec['movieId']
            
            # Get movie details from database
            movie = db.get_movie_by_id(movie_id)
            
            if movie:
                enriched.append({
                    'id': movie.get('id'),
                    'movie_id': movie_id,
                    'title': movie.get('title', rec['title']),
                    'poster_url': movie.get('poster_url', ''),
                    'overview': movie.get('overview', ''),
                    'genres': rec.get('genres', ''),
                    'rating': movie.get('vote_average', 0),
                    'release_date': movie.get('release_date', ''),
                    'recommendation_score': round(rec['score'], 4),
                    'rank': rec['rank']
                })
            else:
                # Fallback to basic info
                enriched.append({
                    'id': movie_id,
                    'movie_id': movie_id,
                    'title': rec['title'],
                    'genres': rec.get('genres', ''),
                    'recommendation_score': round(rec['score'], 4),
                    'rank': rec['rank']
                })
        
        return enriched
    
    def get_similar_movies(self, movie_id, k=10):
        """
        Get movies similar to a given movie
        Uses movie embeddings from the model
        """
        if not self._model_loaded:
            logger.warning("Model not loaded, returning empty similar movies")
            return []
        
        try:
            # Encode movie ID
            movie_idx = self.recommender._encode_movie_ids([movie_id])
            
            if not movie_idx:
                return []
            
            movie_idx = movie_idx[0]
            
            # Get movie embedding
            with torch.no_grad():
                movie_tensor = torch.tensor([movie_idx], 
                                           dtype=torch.long,
                                           device=self.recommender.device)
                query_embedding = self.recommender.model.get_movie_embedding(movie_tensor)
            
            # Get all movie embeddings (this can be expensive - consider caching)
            # For now, use candidate generation approach
            all_movie_ids = self.recommender.movies_df['movieId'].tolist()
            candidate_indices = self.recommender._encode_movie_ids(all_movie_ids[:1000])
            
            if not candidate_indices:
                return []
            
            # Compute similarities
            with torch.no_grad():
                candidate_tensor = torch.tensor(candidate_indices,
                                              dtype=torch.long,
                                              device=self.recommender.device)
                candidate_embeddings = self.recommender.model.get_movie_embedding(candidate_tensor)
                
                # Cosine similarity
                similarities = torch.matmul(query_embedding, candidate_embeddings.T).squeeze()
                
                # Get top-k (excluding the query movie itself)
                top_k_values, top_k_indices = torch.topk(similarities, k+1)
            
            # Decode movie IDs
            top_k_indices = top_k_indices.cpu().numpy()
            candidate_movie_ids = self.recommender._decode_movie_ids(
                [candidate_indices[i] for i in top_k_indices]
            )
            
            # Filter out the query movie
            similar_movie_ids = [mid for mid in candidate_movie_ids if mid != movie_id][:k]
            
            # Get movie details
            similar_movies = []
            for mid in similar_movie_ids:
                movie = db.get_movie_by_id(mid)
                if movie:
                    similar_movies.append(movie)
            
            return similar_movies
        
        except Exception as e:
            logger.error(f"Error finding similar movies: {e}")
            return []
    
    def predict_rating(self, user_id, movie_id):
        """
        Predict user's rating for a movie
        
        Args:
            user_id: int
            movie_id: int
        
        Returns:
            predicted_rating: float (0-5 scale)
        """
        if not self._model_loaded:
            return 2.5  # Default rating
        
        try:
            # Encode IDs
            user_idx = self.recommender._encode_user_id(user_id)
            movie_indices = self.recommender._encode_movie_ids([movie_id])
            
            if not movie_indices:
                return 2.5
            
            movie_idx = movie_indices[0]
            
            # Get prediction
            with torch.no_grad():
                user_tensor = torch.tensor([user_idx], 
                                          dtype=torch.long,
                                          device=self.recommender.device)
                movie_tensor = torch.tensor([movie_idx],
                                           dtype=torch.long,
                                           device=self.recommender.device)
                
                score = self.recommender.model(user_tensor, movie_tensor)
                score = score.cpu().item()
            
            # Convert from [0, 1] to [0, 5] rating scale
            predicted_rating = score * 5.0
            
            return round(predicted_rating, 2)
        
        except Exception as e:
            logger.error(f"Error predicting rating: {e}")
            return 2.5
    
    def get_top_movies(self, k=50):
        """
        Get top-k movies by popularity
        Fallback when model is not available
        """
        if self._model_loaded and hasattr(self.recommender, 'candidate_generator'):
            popular_ids = self.recommender.candidate_generator.get_popular_movies(k=k)
            
            movies = []
            for movie_id in popular_ids:
                movie = db.get_movie_by_id(movie_id)
                if movie:
                    movies.append(movie)
            
            return movies
        else:
            # Fallback to database
            return db.get_top_movies(limit=k)


# Global instance
netflix_recommender = NetflixRecommender()


def get_netflix_recommendations(user_id, k=20):
    """
    Convenience function for getting recommendations
    """
    return netflix_recommender.get_recommendations(user_id, k=k)


def get_similar_movies_netflix(movie_id, k=10):
    """
    Convenience function for getting similar movies
    """
    return netflix_recommender.get_similar_movies(movie_id, k=k)


if __name__ == '__main__':
    # Test the recommender
    logger.info("Testing Netflix-level recommender...")
    
    if netflix_recommender.is_ready():
        # Test recommendations
        user_id = 1
        recommendations = netflix_recommender.get_recommendations(user_id, k=10)
        
        logger.info(f"\nTop 10 recommendations for user {user_id}:")
        for rec in recommendations:
            logger.info(f"  {rec.get('rank')}. {rec.get('title')} (score: {rec.get('recommendation_score')})")
        
        logger.info("\n✓ Recommender test passed!")
    else:
        logger.warning("Model not loaded. Run training first:")
        logger.warning("  python -m ai.training_pipeline")
