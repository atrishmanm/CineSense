"""
STEP 8: Online Inference (Flask Integration)
Production-ready inference using trained models.

Supports two backends:
  - 'ensemble' (default) — NeuMF V2 mega-ensemble (offline checkpoints, not bundled)
  - 'hybrid'  — Legacy HybridRecommender from training/models.py
"""

import os
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# Directory holding the locally trained legacy checkpoints (gitignored *.pt).
MODEL_DIR = Path(os.getenv("MODEL_DIR", "model"))


def _require_file(path: Path, what: str) -> Path:
    """Return *path* or raise with an actionable message (D387)."""
    if not Path(path).exists():
        raise FileNotFoundError(
            f"{what} not found: {path}. Trained artifacts under "
            f"'{MODEL_DIR}/' are gitignored and absent on a fresh clone; "
            "download them or retrain (training/train.py), or set MODEL_DIR."
        )
    return Path(path)


def _resolve_checkpoint(
    explicit: str | None, env_var: str, default_name: str, what: str
) -> Path | None:
    """Locate a checkpoint by explicit path, env var, CWD, or repo root."""
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    env_val = os.getenv(env_var)
    if env_val:
        candidates.append(Path(env_val))
    candidates.append(Path(default_name))
    candidates.append(PROJECT_ROOT / default_name)
    candidates.append(PROJECT_ROOT / "models" / default_name)
    for cand in candidates:
        if cand.exists():
            return cand
    return None


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
        mappings_path = _require_file(MODEL_DIR / "mappings.pkl", "Mappings file")
        with open(mappings_path, "rb") as f:
            mappings = pickle.load(f)
            self.user_encoder = mappings["user_encoder"]
            self.movie_encoder = mappings["movie_encoder"]
            self.num_users = mappings["num_users"]
            self.num_movies = mappings["num_movies"]
        
        # Load plot embeddings if hybrid
        self.plot_embeddings = None
        if model_type == 'hybrid':
            plot_path = _require_file(MODEL_DIR / "plot_embeddings.npy", "Plot embeddings")
            self.plot_embeddings = np.load(plot_path)
            logger.info(f"Loaded plot embeddings: {self.plot_embeddings.shape}")
        
        # Load model
        self.model = get_model(model_type, self.num_users, self.num_movies, plot_emb_dim=768)
        
        checkpoint_path = _require_file(
            MODEL_DIR / f"{model_type}_recommender.pt", f"{model_type.title()} checkpoint"
        )
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Load movie metadata
        self.movie_metadata = pd.read_csv(_require_file(MODEL_DIR / "movie_metadata.csv", "Movie metadata"))
        
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
    with optimized weights (offline checkpoints, not bundled).
    """
    
    model_type = 'ensemble'
    
    def __init__(self, v2_path=None, v1_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Resolve the offline ensemble checkpoints (D381/D387). These are not
        # bundled with the repo; locate them via explicit path, DL_V2_CHECKPOINT,
        # CWD, the repo root, or models/.
        resolved_v2 = _resolve_checkpoint(
            v2_path, "DL_V2_CHECKPOINT", "cinesense_v2.pt", "Ensemble V2 checkpoint"
        )
        if resolved_v2 is None:
            raise FileNotFoundError(
                "Ensemble V2 checkpoint 'cinesense_v2.pt' not found. Set "
                "DL_V2_CHECKPOINT to its path or place it in the repo root/models. "
                "These offline checkpoints are not bundled in the repository."
            )
        resolved_v1 = _resolve_checkpoint(
            v1_path, "DL_V1_CHECKPOINT", "cinesense_model_final.pt", "Ensemble V1 checkpoint"
        )
        if resolved_v1 is None:
            # V1 is optional: EnsembleScorer falls back to V2-only when it is
            # missing. Pass the (nonexistent) default so the existing warning fires.
            resolved_v1 = Path("cinesense_model_final.pt")
            logger.warning(
                "Ensemble V1 checkpoint not found; scoring with V2 models only."
            )
        
        self.scorer = EnsembleScorer(
            v2_path=str(resolved_v2), v1_path=str(resolved_v1), device=self.device
        )
        
        if not self.scorer.load():
            raise RuntimeError(
                f"Failed to load ensemble checkpoints (V2={resolved_v2}, V1={resolved_v1})"
            )
        
        self.num_users = self.scorer.stats['n_users']
        self.num_movies = self.scorer.stats['n_movies']
        
        # Load movie metadata if available
        meta_path = MODEL_DIR / 'movie_metadata.csv'
        self.movie_metadata = pd.read_csv(meta_path) if meta_path.exists() else None
        
        logger.info("✅ EnsembleProductionRecommender ready | "
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
