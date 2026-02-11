"""
Production Inference Pipeline
Netflix-style recommendation serving

When user opens app:
    1. Get user embedding
    2. Get candidate movies (popular + new + similar)
    3. Compute scores using DL model
    4. Rank
    5. Apply diversity constraint
    6. Show results
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import pickle
import logging
from typing import List, Dict, Tuple
from collections import defaultdict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CandidateGenerator:
    """
    Generate candidate movies for ranking
    This is critical for performance - can't score ALL movies for each user
    """
    
    def __init__(self, movies_df, ratings_df=None):
        self.movies_df = movies_df
        self.ratings_df = ratings_df
        
        # Precompute popularity scores
        if ratings_df is not None:
            self._compute_popularity()
    
    def _compute_popularity(self):
        """Compute movie popularity from ratings"""
        popularity = self.ratings_df.groupby('movieId').agg({
            'rating': ['count', 'mean']
        })
        
        popularity.columns = ['num_ratings', 'avg_rating']
        popularity['popularity_score'] = (
            np.log1p(popularity['num_ratings']) * popularity['avg_rating']
        )
        
        self.popularity_scores = popularity['popularity_score'].to_dict()
    
    def get_popular_movies(self, k=100):
        """Get top-k popular movies"""
        if not hasattr(self, 'popularity_scores'):
            # Fallback: use any movies
            return self.movies_df['movieId'].head(k).tolist()
        
        # Sort by popularity
        popular = sorted(
            self.popularity_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:k]
        
        return [movie_id for movie_id, score in popular]
    
    def get_recent_movies(self, k=50):
        """Get recent movies (by movieId as proxy)"""
        # Higher movieIds are typically newer
        return self.movies_df['movieId'].nlargest(k).tolist()
    
    def get_genre_similar_movies(self, user_history, k=100):
        """Get movies similar to user's watch history by genre"""
        if not user_history:
            return []
        
        # Get genres from user history
        user_movie_ids = [h['movieId'] for h in user_history]
        user_movies = self.movies_df[self.movies_df['movieId'].isin(user_movie_ids)]
        
        # Extract genres (assuming 'genres' column exists)
        if 'genres' not in user_movies.columns:
            return []
        
        user_genres = set()
        for genres_str in user_movies['genres'].dropna():
            for genre in str(genres_str).split('|'):
                user_genres.add(genre.strip())
        
        if not user_genres:
            return []
        
        # Find movies with matching genres
        similar_movies = []
        
        for _, movie in self.movies_df.iterrows():
            if 'genres' not in movie or pd.isna(movie['genres']):
                continue
            
            movie_genres = set(str(movie['genres']).split('|'))
            overlap = len(user_genres & movie_genres)
            
            if overlap > 0 and movie['movieId'] not in user_movie_ids:
                similar_movies.append((movie['movieId'], overlap))
        
        # Sort by overlap
        similar_movies.sort(key=lambda x: x[1], reverse=True)
        
        return [movie_id for movie_id, _ in similar_movies[:k]]
    
    def get_candidates(self, user_id=None, user_history=None, k=500):
        """
        Get candidate movies for ranking
        
        Args:
            user_id: int - user ID
            user_history: List[Dict] - user's watch history
            k: int - number of candidates
        
        Returns:
            candidate_movie_ids: List[int]
        """
        candidates = set()
        
        # Add popular movies (always good baseline)
        popular = self.get_popular_movies(k=100)
        candidates.update(popular)
        
        # Add recent movies
        recent = self.get_recent_movies(k=50)
        candidates.update(recent)
        
        # Add genre-similar movies if user history available
        if user_history:
            similar = self.get_genre_similar_movies(user_history, k=200)
            candidates.update(similar)
        
        # Convert to list and limit
        candidate_list = list(candidates)[:k]
        
        return candidate_list


class DiversityReranker:
    """
    Apply diversity constraint to recommendations
    Prevents showing too many similar movies
    """
    
    def __init__(self, movies_df):
        self.movies_df = movies_df
        
        # Create genre index
        self._build_genre_index()
    
    def _build_genre_index(self):
        """Build index of movie -> genres"""
        self.movie_genres = {}
        
        for _, movie in self.movies_df.iterrows():
            if 'genres' in movie and pd.notna(movie['genres']):
                genres = set(str(movie['genres']).split('|'))
                self.movie_genres[movie['movieId']] = genres
            else:
                self.movie_genres[movie['movieId']] = set()
    
    def rerank(self, movie_ids, scores, diversity_weight=0.3, max_per_genre=5):
        """
        Rerank recommendations for diversity
        
        Args:
            movie_ids: List[int] - ranked movie IDs
            scores: List[float] - recommendation scores
            diversity_weight: float - weight for diversity (0-1)
            max_per_genre: int - max movies per primary genre
        
        Returns:
            reranked_movie_ids: List[int]
            reranked_scores: List[float]
        """
        if not movie_ids:
            return [], []
        
        # Track genre counts
        genre_counts = defaultdict(int)
        
        # Rerank
        reranked = []
        reranked_scores = []
        
        # Convert to (movie_id, score) pairs
        movie_score_pairs = list(zip(movie_ids, scores))
        
        for movie_id, score in movie_score_pairs:
            # Get primary genre
            genres = self.movie_genres.get(movie_id, set())
            
            if not genres:
                # No genre info, add anyway
                reranked.append(movie_id)
                reranked_scores.append(score)
                continue
            
            primary_genre = sorted(genres)[0]  # Use first genre alphabetically
            
            # Check if genre limit reached
            if genre_counts[primary_genre] >= max_per_genre:
                # Apply diversity penalty
                diversity_penalty = diversity_weight * (genre_counts[primary_genre] / max_per_genre)
                adjusted_score = score * (1 - diversity_penalty)
            else:
                adjusted_score = score
            
            reranked.append(movie_id)
            reranked_scores.append(adjusted_score)
            genre_counts[primary_genre] += 1
        
        # Re-sort by adjusted scores
        combined = list(zip(reranked, reranked_scores))
        combined.sort(key=lambda x: x[1], reverse=True)
        
        reranked_movie_ids = [m for m, s in combined]
        reranked_scores = [s for m, s in combined]
        
        return reranked_movie_ids, reranked_scores


class ProductionRecommender:
    """
    Production-ready recommender system
    Handles the full inference pipeline
    """
    
    def __init__(self, model, preprocessor, movies_df, device='cpu'):
        self.model = model
        self.model.eval()
        
        self.preprocessor = preprocessor
        self.movies_df = movies_df
        self.device = device
        
        # Move model to device
        self.model = self.model.to(device)
        
        # Initialize components
        self.candidate_generator = CandidateGenerator(movies_df)
        self.diversity_reranker = DiversityReranker(movies_df)
        
        logger.info("✓ Production recommender initialized")
    
    def _encode_user_id(self, user_id):
        """Convert user ID to index"""
        try:
            return self.preprocessor.user_encoder.transform([user_id])[0]
        except:
            # Unknown user - return random
            return 0
    
    def _encode_movie_ids(self, movie_ids):
        """Convert movie IDs to indices"""
        encoded = []
        
        for movie_id in movie_ids:
            try:
                idx = self.preprocessor.movie_encoder.transform([movie_id])[0]
                encoded.append(idx)
            except:
                # Unknown movie - skip
                continue
        
        return encoded
    
    def _decode_movie_ids(self, movie_indices):
        """Convert movie indices back to IDs"""
        return self.preprocessor.movie_encoder.inverse_transform(movie_indices)
    
    def recommend(self, user_id, user_history=None, k=10, 
                  use_diversity=True, num_candidates=500):
        """
        Generate recommendations for a user
        
        Args:
            user_id: int - user ID
            user_history: List[Dict] - user's watch history
            k: int - number of recommendations
            use_diversity: bool - apply diversity reranking
            num_candidates: int - number of candidates to generate
        
        Returns:
            recommendations: List[Dict] with keys: movieId, title, score, rank
        """
        # Step 1: Generate candidates
        candidate_movie_ids = self.candidate_generator.get_candidates(
            user_id=user_id,
            user_history=user_history,
            k=num_candidates
        )
        
        if not candidate_movie_ids:
            logger.warning(f"No candidates found for user {user_id}")
            return []
        
        # Step 2: Encode IDs
        user_idx = self._encode_user_id(user_id)
        candidate_indices = self._encode_movie_ids(candidate_movie_ids)
        
        if not candidate_indices:
            logger.warning(f"No valid movie indices for user {user_id}")
            return []
        
        # Step 3: Score candidates using model
        with torch.no_grad():
            user_tensor = torch.tensor([user_idx] * len(candidate_indices), 
                                      dtype=torch.long, device=self.device)
            movie_tensor = torch.tensor(candidate_indices, 
                                       dtype=torch.long, device=self.device)
            
            scores = self.model(user_tensor, movie_tensor)
            scores = scores.cpu().numpy()
        
        # Step 4: Decode movie IDs
        movie_ids = self._decode_movie_ids(candidate_indices)
        
        # Step 5: Sort by score
        movie_score_pairs = sorted(
            zip(movie_ids, scores),
            key=lambda x: x[1],
            reverse=True
        )
        
        top_movie_ids = [m for m, s in movie_score_pairs[:k*2]]  # Get 2x for diversity
        top_scores = [s for m, s in movie_score_pairs[:k*2]]
        
        # Step 6: Apply diversity reranking
        if use_diversity:
            top_movie_ids, top_scores = self.diversity_reranker.rerank(
                top_movie_ids,
                top_scores,
                diversity_weight=0.3,
                max_per_genre=3
            )
        
        # Step 7: Get top-k final
        final_movie_ids = top_movie_ids[:k]
        final_scores = top_scores[:k]
        
        # Step 8: Get movie details
        recommendations = []
        
        for rank, (movie_id, score) in enumerate(zip(final_movie_ids, final_scores), 1):
            movie = self.movies_df[self.movies_df['movieId'] == movie_id]
            
            if movie.empty:
                continue
            
            movie = movie.iloc[0]
            
            recommendations.append({
                'movieId': int(movie_id),
                'title': movie.get('title', 'Unknown'),
                'genres': movie.get('genres', ''),
                'score': float(score),
                'rank': rank
            })
        
        return recommendations
    
    def batch_recommend(self, user_ids, k=10):
        """
        Generate recommendations for multiple users
        Useful for batch processing
        """
        all_recommendations = {}
        
        for user_id in user_ids:
            all_recommendations[user_id] = self.recommend(user_id, k=k)
        
        return all_recommendations


def load_production_model(model_path='ai/models/best_model.pt', 
                          cache_dir='ai/cache',
                          device='cpu'):
    """
    Load trained model for production serving
    
    Args:
        model_path: str - path to model checkpoint
        cache_dir: str - path to preprocessed data
        device: str - 'cpu' or 'cuda'
    
    Returns:
        recommender: ProductionRecommender
    """
    logger.info("Loading production model...")
    
    # Load preprocessor
    from ai.data_preprocessor import DataPreprocessor
    preprocessor = DataPreprocessor()
    preprocessor.load_processed_data(input_dir=cache_dir)
    
    # Load movies
    movies_df = pd.read_pickle(Path(cache_dir) / 'movies_merged.pkl')
    
    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Reconstruct model
    config = checkpoint.get('config', {})
    model_type = config.get('model_type', 'hybrid')
    
    if model_type == 'hybrid':
        from ai.hybrid_model import create_hybrid_model
        model = create_hybrid_model(
            num_users=preprocessor.num_users,
            num_movies=preprocessor.num_movies,
            movie_encoder=preprocessor.movie_encoder,
            use_content=True,
            cache_dir=cache_dir
        )
    else:
        from ai.two_tower_ncf import TwoTowerNCF
        model = TwoTowerNCF(
            num_users=preprocessor.num_users,
            num_movies=preprocessor.num_movies
        )
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    logger.info(f"✓ Model loaded from {model_path}")
    
    # Create recommender
    recommender = ProductionRecommender(
        model=model,
        preprocessor=preprocessor,
        movies_df=movies_df,
        device=device
    )
    
    return recommender


if __name__ == '__main__':
    # Test inference
    logger.info("Testing production recommender...")
    
    try:
        recommender = load_production_model()
        
        # Test recommendation
        user_id = 1
        recommendations = recommender.recommend(user_id, k=10)
        
        logger.info(f"\nTop 10 recommendations for user {user_id}:")
        for rec in recommendations:
            logger.info(f"  {rec['rank']}. {rec['title']} (score: {rec['score']:.4f})")
        
        logger.info("\n✓ Inference test passed!")
    
    except Exception as e:
        logger.error(f"Test failed: {e}")
        logger.info("Run training first: python -m ai.training_pipeline")
