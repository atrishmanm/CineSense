"""
Content-Aware Hybrid Deep Learning Model
Netflix NEVER relies on CF alone - this adds content understanding

Architecture:
    User Tower (Deep NN) → user_embedding
    Movie Tower (Deep NN) → movie_cf_embedding
    Content Encoder (Transformer) → movie_content_embedding
    Fusion → Final Prediction

This handles:
    - Cold start problem
    - Story/content similarity
    - Unseen movies
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer
import numpy as np
import logging
from pathlib import Path
import pickle
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MovieContentEncoder(nn.Module):
    """
    Transformer-based content encoder using BERT/MiniLM
    Encodes movie metadata: overview, genres, keywords
    """
    
    def __init__(self, model_name='all-MiniLM-L6-v2', cache_dir='ai/cache'):
        super().__init__()
        
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Load pre-trained transformer
        logger.info(f"Loading transformer model: {model_name}...")
        self.transformer = SentenceTransformer(model_name)
        self.embedding_dim = self.transformer.get_sentence_embedding_dimension()
        
        # Freeze transformer (optional - can fine-tune later)
        for param in self.transformer.parameters():
            param.requires_grad = False
        
        logger.info(f"✓ Content encoder loaded, embedding_dim={self.embedding_dim}")
    
    def encode_texts(self, texts, batch_size=32, show_progress=True):
        """
        Encode a list of text strings to embeddings
        
        Args:
            texts: List[str] - movie content texts
            batch_size: int
            show_progress: bool
        
        Returns:
            embeddings: [num_texts, embedding_dim] numpy array
        """
        embeddings = self.transformer.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True
        )
        
        return embeddings
    
    def encode_movie_content(self, content_df):
        """
        Encode all movie content from dataframe
        
        Args:
            content_df: DataFrame with columns ['movieId', 'content_text']
        
        Returns:
            movie_content_embeddings: dict {movieId: embedding}
        """
        logger.info(f"Encoding {len(content_df)} movie contents...")
        
        texts = content_df['content_text'].fillna('').tolist()
        embeddings = self.encode_texts(texts, show_progress=True)
        
        # Create mapping
        movie_embeddings = {}
        for idx, row in content_df.iterrows():
            movie_embeddings[row['movieId']] = embeddings[idx]
        
        logger.info(f"✓ Encoded {len(movie_embeddings)} movies")
        
        return movie_embeddings
    
    def save_embeddings(self, movie_embeddings, filename='movie_content_embeddings.pkl'):
        """Save precomputed embeddings"""
        path = self.cache_dir / filename
        
        with open(path, 'wb') as f:
            pickle.dump(movie_embeddings, f)
        
        logger.info(f"✓ Saved content embeddings to {path}")
    
    def load_embeddings(self, filename='movie_content_embeddings.pkl'):
        """Load precomputed embeddings"""
        path = self.cache_dir / filename
        
        with open(path, 'rb') as f:
            movie_embeddings = pickle.load(f)
        
        logger.info(f"✓ Loaded {len(movie_embeddings)} content embeddings from {path}")
        
        return movie_embeddings


class ContentEmbeddingLayer(nn.Module):
    """
    Embedding layer that returns pre-computed content embeddings
    """
    
    def __init__(self, movie_embeddings_dict, movie_encoder, embedding_dim=384):
        super().__init__()
        
        self.movie_encoder = movie_encoder  # LabelEncoder for movieId -> idx
        self.embedding_dim = embedding_dim
        
        # Convert dict to tensor matrix [num_movies, embedding_dim]
        num_movies = len(movie_encoder.classes_)
        self.embeddings = torch.zeros(num_movies, embedding_dim)
        
        # Fill in embeddings
        for movieId in movie_encoder.classes_:
            idx = movie_encoder.transform([movieId])[0]
            
            if movieId in movie_embeddings_dict:
                embedding = movie_embeddings_dict[movieId]
                self.embeddings[idx] = torch.from_numpy(embedding).float()
        
        # Make it a parameter (but frozen)
        self.embeddings = nn.Parameter(self.embeddings, requires_grad=False)
        
        logger.info(f"✓ Content embedding layer created: {self.embeddings.shape}")
    
    def forward(self, movie_ids):
        """
        Args:
            movie_ids: [batch_size] tensor of movie indices
        
        Returns:
            content_embeddings: [batch_size, embedding_dim]
        """
        return self.embeddings[movie_ids]


class HybridMovieTower(nn.Module):
    """
    Hybrid Movie Tower that combines:
        1. Collaborative filtering embeddings (learned)
        2. Content embeddings (from transformer)
    
    This is production-grade architecture
    """
    
    def __init__(self, num_movies, cf_embedding_dim=128, content_embedding_layer=None, 
                 hidden_dims=[256, 128, 64], fusion_method='concat'):
        super().__init__()
        
        self.num_movies = num_movies
        self.cf_embedding_dim = cf_embedding_dim
        self.fusion_method = fusion_method
        
        # CF embeddings (learned from interactions)
        self.cf_embedding = nn.Embedding(num_movies, cf_embedding_dim)
        
        # Content embeddings (pre-computed from transformer)
        self.content_embedding_layer = content_embedding_layer
        
        # Determine input dimension for fusion
        if content_embedding_layer is not None:
            content_dim = content_embedding_layer.embedding_dim
            
            if fusion_method == 'concat':
                input_dim = cf_embedding_dim + content_dim
            elif fusion_method == 'add':
                # Project content to same dim as CF
                self.content_projection = nn.Linear(content_dim, cf_embedding_dim)
                input_dim = cf_embedding_dim
            else:
                raise ValueError(f"Unknown fusion method: {fusion_method}")
        else:
            input_dim = cf_embedding_dim
        
        # Deep layers for movie tower
        layers = []
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            input_dim = hidden_dim
        
        self.movie_tower = nn.Sequential(*layers)
        self.output_dim = hidden_dims[-1]
        
        self._init_weights()
        
        logger.info(f"✓ Hybrid Movie Tower initialized (fusion={fusion_method})")
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.cf_embedding.weight)
        
        for layer in self.movie_tower:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, movie_ids):
        """
        Args:
            movie_ids: [batch_size] tensor
        
        Returns:
            movie_embeddings: [batch_size, output_dim]
        """
        # Get CF embeddings
        cf_embed = self.cf_embedding(movie_ids)  # [batch, cf_dim]
        
        # Fuse with content if available
        if self.content_embedding_layer is not None:
            content_embed = self.content_embedding_layer(movie_ids)  # [batch, content_dim]
            
            if self.fusion_method == 'concat':
                fused = torch.cat([cf_embed, content_embed], dim=1)
            elif self.fusion_method == 'add':
                content_projected = self.content_projection(content_embed)
                fused = cf_embed + content_projected
        else:
            fused = cf_embed
        
        # Pass through deep network
        movie_repr = self.movie_tower(fused)  # [batch, output_dim]
        
        # L2 normalization
        movie_repr = F.normalize(movie_repr, p=2, dim=1)
        
        return movie_repr


class HybridTwoTowerModel(nn.Module):
    """
    Complete Hybrid Two-Tower Model
    
    Netflix-style architecture combining:
        - Neural Collaborative Filtering (user-item interactions)
        - Content-based filtering (movie metadata)
    
    This is production-grade and handles cold start
    """
    
    def __init__(self, num_users, num_movies, 
                 cf_embedding_dim=128,
                 content_embedding_layer=None,
                 hidden_dims=[256, 128, 64],
                 fusion_method='concat'):
        super().__init__()
        
        logger.info("Initializing Hybrid Two-Tower Model:")
        logger.info(f"  Users: {num_users:,}")
        logger.info(f"  Movies: {num_movies:,}")
        logger.info(f"  CF embedding dim: {cf_embedding_dim}")
        logger.info(f"  Fusion method: {fusion_method}")
        
        # Import user tower from two_tower_ncf
        from ai.two_tower_ncf import UserTower
        
        # User Tower (same as before)
        self.user_tower = UserTower(num_users, cf_embedding_dim, hidden_dims)
        
        # Hybrid Movie Tower (CF + Content)
        self.movie_tower = HybridMovieTower(
            num_movies,
            cf_embedding_dim,
            content_embedding_layer,
            hidden_dims,
            fusion_method
        )
        
        # Output dimensions must match
        assert self.user_tower.output_dim == self.movie_tower.output_dim
        self.output_dim = self.user_tower.output_dim
        
        logger.info(f"  Final embedding dim: {self.output_dim}")
        logger.info("✓ Hybrid Two-Tower Model initialized")
    
    def forward(self, user_ids, movie_ids):
        """
        Forward pass
        
        Args:
            user_ids: [batch_size]
            movie_ids: [batch_size]
        
        Returns:
            scores: [batch_size]
        """
        # Get embeddings from both towers
        user_embeddings = self.user_tower(user_ids)  # [batch, output_dim]
        movie_embeddings = self.movie_tower(movie_ids)  # [batch, output_dim]
        
        # Dot product (cosine similarity)
        scores = (user_embeddings * movie_embeddings).sum(dim=1)
        
        # Sigmoid to [0, 1]
        scores = torch.sigmoid(scores)
        
        return scores
    
    def get_user_embedding(self, user_ids):
        """Get user embeddings"""
        return self.user_tower(user_ids)
    
    def get_movie_embedding(self, movie_ids):
        """Get movie embeddings"""
        return self.movie_tower(movie_ids)
    
    def recommend_for_user(self, user_id, candidate_movie_ids, top_k=10):
        """
        Recommend top-k movies for a user
        
        Args:
            user_id: int
            candidate_movie_ids: [num_candidates] tensor
            top_k: int
        
        Returns:
            top_movie_ids: [top_k] tensor
            top_scores: [top_k] tensor
        """
        self.eval()
        
        with torch.no_grad():
            user_ids_batch = torch.full((len(candidate_movie_ids),), user_id, dtype=torch.long)
            
            if next(self.parameters()).is_cuda:
                user_ids_batch = user_ids_batch.cuda()
                candidate_movie_ids = candidate_movie_ids.cuda()
            
            scores = self.forward(user_ids_batch, candidate_movie_ids)
            
            top_scores, top_indices = torch.topk(scores, min(top_k, len(scores)))
            top_movie_ids = candidate_movie_ids[top_indices]
        
        return top_movie_ids, top_scores


def precompute_content_embeddings(content_df, cache_dir='ai/cache'):
    """
    Precompute and save movie content embeddings
    Run this once before training
    """
    logger.info("="*60)
    logger.info("PRECOMPUTING MOVIE CONTENT EMBEDDINGS")
    logger.info("="*60)
    
    # Initialize content encoder
    encoder = MovieContentEncoder(cache_dir=cache_dir)
    
    # Encode all movies
    movie_embeddings = encoder.encode_movie_content(content_df)
    
    # Save embeddings
    encoder.save_embeddings(movie_embeddings)
    
    logger.info("="*60)
    logger.info("✓ CONTENT EMBEDDINGS PRECOMPUTED!")
    logger.info("="*60)
    
    return movie_embeddings


def create_hybrid_model(num_users, num_movies, movie_encoder, use_content=True, 
                        cache_dir='ai/cache'):
    """
    Factory function to create hybrid model
    
    Args:
        num_users: int
        num_movies: int
        movie_encoder: LabelEncoder for movieId -> idx
        use_content: bool - whether to use content embeddings
        cache_dir: str - path to cached embeddings
    
    Returns:
        model: HybridTwoTowerModel
    """
    content_embedding_layer = None
    
    if use_content:
        # Load precomputed content embeddings
        encoder = MovieContentEncoder(cache_dir=cache_dir)
        movie_embeddings = encoder.load_embeddings()
        
        # Create embedding layer
        content_embedding_layer = ContentEmbeddingLayer(
            movie_embeddings,
            movie_encoder,
            embedding_dim=encoder.embedding_dim
        )
    
    # Create hybrid model
    model = HybridTwoTowerModel(
        num_users=num_users,
        num_movies=num_movies,
        cf_embedding_dim=128,
        content_embedding_layer=content_embedding_layer,
        hidden_dims=[256, 128, 64],
        fusion_method='concat'
    )
    
    return model


if __name__ == '__main__':
    logger.info("Testing Hybrid Two-Tower Model...")
    
    # This would normally load from preprocessed data
    logger.info("✓ See training pipeline for full usage")
