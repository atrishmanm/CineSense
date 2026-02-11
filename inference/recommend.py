"""
STEP 8: Online Inference (Flask Integration)
Production-ready inference using trained models.

Supports two backends:
  - 'ensemble' (default) — NeuMF V2 mega-ensemble (RMSE 0.8932)
  - 'hybrid'  — Legacy HybridRecommender from training/models.py
"""

import torch
import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from training.models import get_model
from ai.neumf_scorer import EnsembleScorer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProductionRecommender:
    """
    Production inference engine
    
    Loads trained model and generates recommendations in real-time
    """
    
    def __init__(self, model_type='hybrid'):
        self.model_type = model_type
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Loading {model_type} recommender...")
        
        # Load mappings
        with open("model/mappings.pkl", "rb") as f:
            mappings = pickle.load(f)
            self.user_encoder = mappings["user_encoder"]
            self.movie_encoder = mappings["movie_encoder"]
            self.num_users = mappings["num_users"]
            self.num_movies = mappings["num_movies"]
        
        # Load plot embeddings if hybrid
        self.plot_embeddings = None
        if model_type == 'hybrid':
            self.plot_embeddings = np.load("model/plot_embeddings.npy")
            logger.info(f"Loaded plot embeddings: {self.plot_embeddings.shape}")
        
        # Load model
        self.model = get_model(model_type, self.num_users, self.num_movies, plot_emb_dim=768)
        
        checkpoint = torch.load(f"model/{model_type}_recommender.pt", map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Load movie metadata
        self.movie_metadata = pd.read_csv("model/movie_metadata.csv")
        
        logger.info(f"✓ {model_type.upper()} recommender loaded and ready")
    
    def recommend(self, user_id, top_k=10, candidate_movies=None):
        """
        Generate recommendations for a user
        
        Args:
            user_id: original user ID
            top_k: number of recommendations
            candidate_movies: list of candidate movieIds (if None, use all)
        
        Returns:
            recommendations: list of (movieId, title, score)
        """
        try:
            # Encode user
            user_idx = self.user_encoder.transform([user_id])[0]
        except:
            logger.warning(f"Unknown user {user_id}, using random recommendations")
            return self._get_popular_movies(top_k)
        
        # Get candidates
        if candidate_movies is None:
            # Score all movies (can be slow for large catalogs)
            candidate_indices = np.arange(self.num_movies)
        else:
            # Score only candidates
            candidate_indices = []
            for movie_id in candidate_movies:
                try:
                    idx = self.movie_encoder.transform([movie_id])[0]
                    candidate_indices.append(idx)
                except:
                    continue
            candidate_indices = np.array(candidate_indices)
        
        # Prepare tensors
        user_tensor = torch.full((len(candidate_indices),), user_idx, dtype=torch.long, device=self.device)
        movie_tensor = torch.from_numpy(candidate_indices).long().to(self.device)
        
        # Score candidates
        with torch.no_grad():
            if self.model_type == 'hybrid':
                plot_emb_tensor = torch.from_numpy(
                    self.plot_embeddings[candidate_indices]
                ).float().to(self.device)
                scores = self.model(user_tensor, movie_tensor, plot_emb_tensor)
            else:
                scores = self.model(user_tensor, movie_tensor)
        
        # Get top-k
        scores = scores.cpu().numpy()
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        # Get movie details
        recommendations = []
        for idx in top_indices:
            movie_idx = candidate_indices[idx]
            score = scores[idx]
            
            movie_info = self.movie_metadata[self.movie_metadata["movie_idx"] == movie_idx].iloc[0]
            
            recommendations.append({
                "movieId": int(movie_info["movieId"]),
                "title": movie_info["title"],
                "score": float(score),
                "overview": movie_info["overview"][:200] if pd.notna(movie_info["overview"]) else ""
            })
        
        return recommendations
    
    def predict_rating(self, user_id, movie_id):
        """
        Predict user's rating for a specific movie
        
        Args:
            user_id: user ID
            movie_id: movie ID
        
        Returns:
            predicted_rating: float
        """
        try:
            user_idx = self.user_encoder.transform([user_id])[0]
            movie_idx = self.movie_encoder.transform([movie_id])[0]
        except:
            return 3.0  # Default rating
        
        user_tensor = torch.tensor([user_idx], dtype=torch.long, device=self.device)
        movie_tensor = torch.tensor([movie_idx], dtype=torch.long, device=self.device)
        
        with torch.no_grad():
            if self.model_type == 'hybrid':
                plot_emb_tensor = torch.from_numpy(
                    self.plot_embeddings[movie_idx:movie_idx+1]
                ).float().to(self.device)
                score = self.model(user_tensor, movie_tensor, plot_emb_tensor)
            else:
                score = self.model(user_tensor, movie_tensor)
        
        predicted_rating = float(score.cpu().item())
        
        # Clip to valid rating range
        predicted_rating = np.clip(predicted_rating, 0.5, 5.0)
        
        return predicted_rating
    
    def _get_popular_movies(self, k=10):
        """Fallback: return popular movies"""
        popular = self.movie_metadata.head(k)
        
        return [{
            "movieId": int(row["movieId"]),
            "title": row["title"],
            "score": 5.0,
            "overview": row["overview"][:200] if pd.notna(row["overview"]) else ""
        } for _, row in popular.iterrows()]


# Global instance for Flask integration
_recommender = None


def get_recommender(model_type='ensemble'):
    """Get global recommender instance (singleton).
    
    Args:
        model_type: 'ensemble' (NeuMF V2) or 'hybrid' (legacy)
    """
    global _recommender
    
    if _recommender is None or _recommender.model_type != model_type:
        if model_type == 'ensemble':
            _recommender = EnsembleProductionRecommender()
        else:
            _recommender = ProductionRecommender(model_type)
    
    return _recommender


class EnsembleProductionRecommender:
    """
    Production recommender using the NeuMF V2 mega-ensemble.
    
    Loads cinesense_v2.pt + cinesense_model_final.pt and scores
    MovieLens-100K user/movie pairs using the 13-model ensemble
    with optimized weights (RMSE = 0.8932).
    """
    
    model_type = 'ensemble'
    
    def __init__(self, v2_path='cinesense_v2.pt', v1_path='cinesense_model_final.pt'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.scorer = EnsembleScorer(v2_path=v2_path, v1_path=v1_path, device=self.device)
        
        if not self.scorer.load():
            raise RuntimeError("Failed to load ensemble checkpoints")
        
        self.num_users = self.scorer.stats['n_users']
        self.num_movies = self.scorer.stats['n_movies']
        
        # Load movie metadata if available
        meta_path = Path('model/movie_metadata.csv')
        self.movie_metadata = pd.read_csv(meta_path) if meta_path.exists() else None
        
        logger.info(f"✅ EnsembleProductionRecommender ready | "
                     f"{self.num_users} users, {self.num_movies} movies")
    
    def recommend(self, user_id, top_k=10, candidate_movies=None):
        """Generate top-k recommendations using the full ensemble."""
        try:
            user_idx = int(user_id)
            if user_idx < 0 or user_idx >= self.num_users:
                logger.warning(f"User {user_id} out of range, fallback")
                return self._get_popular_movies(top_k)
        except (ValueError, TypeError):
            return self._get_popular_movies(top_k)
        
        if candidate_movies is not None:
            movie_indices = [int(m) for m in candidate_movies
                            if 0 <= int(m) < self.num_movies]
        else:
            movie_indices = list(range(self.num_movies))
        
        user_arr = [user_idx] * len(movie_indices)
        scores = self.scorer.score(user_arr, movie_indices)
        
        top_idx = np.argsort(scores)[::-1][:top_k]
        
        results = []
        for i in top_idx:
            mid = movie_indices[i]
            entry = {'movieId': mid, 'score': float(scores[i])}
            if self.movie_metadata is not None:
                row = self.movie_metadata[self.movie_metadata['movie_idx'] == mid]
                if len(row):
                    entry['title'] = row.iloc[0].get('title', f'Movie {mid}')
                    ov = row.iloc[0].get('overview', '')
                    entry['overview'] = (ov[:200] if pd.notna(ov) else '')
            results.append(entry)
        
        return results
    
    def predict_rating(self, user_id, movie_id):
        """Predict rating for a single (user, movie) pair."""
        try:
            return self.scorer.score(int(user_id), int(movie_id))
        except Exception:
            return 3.0
    
    def _get_popular_movies(self, k=10):
        if self.movie_metadata is not None:
            pop = self.movie_metadata.head(k)
            return [{'movieId': int(r['movieId'] if 'movieId' in r else r.get('movie_idx', i)),
                     'title': r.get('title', ''), 'score': 5.0,
                     'overview': (r['overview'][:200] if pd.notna(r.get('overview')) else '')}
                    for i, (_, r) in enumerate(pop.iterrows())]
        return []


def recommend_for_user(user_id, top_k=10, model_type='ensemble'):
    """
    Convenience function for Flask routes
    
    Usage in Flask:
        from inference.recommend import recommend_for_user
        
        @app.route('/recommendations/<int:user_id>')
        def get_recommendations(user_id):
            recs = recommend_for_user(user_id, top_k=20)
            return jsonify(recs)
    """
    recommender = get_recommender(model_type)
    return recommender.recommend(user_id, top_k)


if __name__ == "__main__":
    # Test inference
    logger.info("Testing inference...")
    
    # Test with hybrid model
    recommender = ProductionRecommender('hybrid')
    
    # Test recommendation
    user_id = 1
    recommendations = recommender.recommend(user_id, top_k=10)
    
    logger.info(f"\nTop 10 recommendations for user {user_id}:")
    for i, rec in enumerate(recommendations, 1):
        logger.info(f"{i}. {rec['title']} (score: {rec['score']:.2f})")
    
    # Test rating prediction
    movie_id = 1
    predicted_rating = recommender.predict_rating(user_id, movie_id)
    logger.info(f"\nPredicted rating for user {user_id}, movie {movie_id}: {predicted_rating:.2f}")
    
    logger.info("\n✓ Inference test passed!")
