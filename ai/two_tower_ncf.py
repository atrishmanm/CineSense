"""
Two-Tower Neural Collaborative Filtering (NCF)
Netflix-style deep learning recommendation model

Architecture:
    User Tower (Deep NN) → User Embedding
    Movie Tower (Deep NN) → Movie Embedding
    Dot Product / Cosine Similarity → Score
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UserTower(nn.Module):
    """
    User Tower - Deep neural network for learning user representations
    Learns latent taste patterns automatically
    """
    
    def __init__(self, num_users, embedding_dim=128, hidden_dims=[256, 128, 64]):
        super().__init__()
        
        self.num_users = num_users
        self.embedding_dim = embedding_dim
        
        # User embedding layer
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        
        # Deep layers for user tower
        layers = []
        input_dim = embedding_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            input_dim = hidden_dim
        
        self.user_tower = nn.Sequential(*layers)
        
        # Final embedding dimension
        self.output_dim = hidden_dims[-1]
        
        self._init_weights()
    
    def _init_weights(self):
        """Xavier initialization for better convergence"""
        nn.init.xavier_uniform_(self.user_embedding.weight)
        
        for layer in self.user_tower:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, user_ids):
        """
        Args:
            user_ids: [batch_size] tensor of user indices
        
        Returns:
            user_embeddings: [batch_size, output_dim] tensor
        """
        # Get user embeddings
        user_embed = self.user_embedding(user_ids)  # [batch, embedding_dim]
        
        # Pass through deep network
        user_repr = self.user_tower(user_embed)  # [batch, output_dim]
        
        # L2 normalization for cosine similarity
        user_repr = F.normalize(user_repr, p=2, dim=1)
        
        return user_repr


class MovieTower(nn.Module):
    """
    Movie Tower - Deep neural network for learning movie representations
    Pure collaborative filtering (no content yet)
    """
    
    def __init__(self, num_movies, embedding_dim=128, hidden_dims=[256, 128, 64]):
        super().__init__()
        
        self.num_movies = num_movies
        self.embedding_dim = embedding_dim
        
        # Movie embedding layer
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)
        
        # Deep layers for movie tower
        layers = []
        input_dim = embedding_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            input_dim = hidden_dim
        
        self.movie_tower = nn.Sequential(*layers)
        
        # Final embedding dimension
        self.output_dim = hidden_dims[-1]
        
        self._init_weights()
    
    def _init_weights(self):
        """Xavier initialization for better convergence"""
        nn.init.xavier_uniform_(self.movie_embedding.weight)
        
        for layer in self.movie_tower:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, movie_ids):
        """
        Args:
            movie_ids: [batch_size] tensor of movie indices
        
        Returns:
            movie_embeddings: [batch_size, output_dim] tensor
        """
        # Get movie embeddings
        movie_embed = self.movie_embedding(movie_ids)  # [batch, embedding_dim]
        
        # Pass through deep network
        movie_repr = self.movie_tower(movie_embed)  # [batch, output_dim]
        
        # L2 normalization for cosine similarity
        movie_repr = F.normalize(movie_repr, p=2, dim=1)
        
        return movie_repr


class TwoTowerNCF(nn.Module):
    """
    Two-Tower Neural Collaborative Filtering
    
    This is Netflix-style deep CF, not matrix factorization.
    Learns latent taste automatically through deep neural networks.
    
    Architecture:
        User Tower → user_embedding [batch, 64]
        Movie Tower → movie_embedding [batch, 64]
        Interaction → dot_product → score
    """
    
    def __init__(self, num_users, num_movies, embedding_dim=128, hidden_dims=[256, 128, 64]):
        super().__init__()
        
        logger.info(f"Initializing Two-Tower NCF model:")
        logger.info(f"  Users: {num_users:,}")
        logger.info(f"  Movies: {num_movies:,}")
        logger.info(f"  Embedding dim: {embedding_dim}")
        logger.info(f"  Hidden dims: {hidden_dims}")
        
        # User and Movie towers
        self.user_tower = UserTower(num_users, embedding_dim, hidden_dims)
        self.movie_tower = MovieTower(num_movies, embedding_dim, hidden_dims)
        
        # Output dimension should match
        assert self.user_tower.output_dim == self.movie_tower.output_dim, \
            "User and Movie tower output dimensions must match"
        
        self.output_dim = self.user_tower.output_dim
        
        logger.info(f"  Final embedding dim: {self.output_dim}")
        logger.info("✓ Two-Tower NCF initialized")
    
    def forward(self, user_ids, movie_ids):
        """
        Forward pass - compute recommendation scores
        
        Args:
            user_ids: [batch_size] tensor
            movie_ids: [batch_size] tensor
        
        Returns:
            scores: [batch_size] tensor of predicted ratings/scores
        """
        # Get user and movie embeddings
        user_embeddings = self.user_tower(user_ids)  # [batch, output_dim]
        movie_embeddings = self.movie_tower(movie_ids)  # [batch, output_dim]
        
        # Compute dot product (cosine similarity since normalized)
        # Element-wise multiply then sum
        scores = (user_embeddings * movie_embeddings).sum(dim=1)  # [batch]
        
        # Sigmoid to get [0, 1] range for BCELoss
        scores = torch.sigmoid(scores)
        
        return scores
    
    def get_user_embedding(self, user_ids):
        """Get user embeddings (for candidate retrieval)"""
        return self.user_tower(user_ids)
    
    def get_movie_embedding(self, movie_ids):
        """Get movie embeddings (for candidate retrieval)"""
        return self.movie_tower(movie_ids)
    
    def predict_batch(self, user_ids, movie_ids):
        """
        Batch prediction for multiple user-movie pairs
        """
        self.eval()
        with torch.no_grad():
            scores = self.forward(user_ids, movie_ids)
        return scores
    
    def recommend_for_user(self, user_id, candidate_movie_ids, top_k=10):
        """
        Recommend top-k movies for a user from candidates
        
        Args:
            user_id: int - user index
            candidate_movie_ids: [num_candidates] tensor
            top_k: int - number of recommendations
        
        Returns:
            top_movie_ids: [top_k] tensor - recommended movie indices
            top_scores: [top_k] tensor - recommendation scores
        """
        self.eval()
        
        with torch.no_grad():
            # Get user embedding
            user_ids_batch = torch.full((len(candidate_movie_ids),), user_id, dtype=torch.long)
            
            if self.user_tower.user_embedding.weight.is_cuda:
                user_ids_batch = user_ids_batch.cuda()
                candidate_movie_ids = candidate_movie_ids.cuda()
            
            # Compute scores for all candidates
            scores = self.forward(user_ids_batch, candidate_movie_ids)
            
            # Get top-k
            top_scores, top_indices = torch.topk(scores, min(top_k, len(scores)))
            top_movie_ids = candidate_movie_ids[top_indices]
        
        return top_movie_ids, top_scores
    
    def compute_all_movie_scores(self, user_id, device='cpu'):
        """
        Compute scores for ALL movies for a user
        Memory intensive but useful for full ranking
        """
        self.eval()
        
        num_movies = self.movie_tower.num_movies
        batch_size = 1024  # Process in batches to avoid OOM
        
        all_scores = []
        
        with torch.no_grad():
            for start_idx in range(0, num_movies, batch_size):
                end_idx = min(start_idx + batch_size, num_movies)
                
                # Create batch
                movie_ids = torch.arange(start_idx, end_idx, dtype=torch.long, device=device)
                user_ids = torch.full((len(movie_ids),), user_id, dtype=torch.long, device=device)
                
                # Compute scores
                batch_scores = self.forward(user_ids, movie_ids)
                all_scores.append(batch_scores.cpu())
        
        return torch.cat(all_scores)


class GMF(nn.Module):
    """
    Generalized Matrix Factorization (GMF)
    Simpler baseline model - like traditional MF but with neural network
    """
    
    def __init__(self, num_users, num_movies, embedding_dim=64):
        super().__init__()
        
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)
        
        # Output layer
        self.fc = nn.Linear(embedding_dim, 1)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.movie_embedding.weight, std=0.01)
        nn.init.xavier_uniform_(self.fc.weight)
    
    def forward(self, user_ids, movie_ids):
        user_embed = self.user_embedding(user_ids)
        movie_embed = self.movie_embedding(movie_ids)
        
        # Element-wise product
        interaction = user_embed * movie_embed
        
        # Linear layer + sigmoid
        output = torch.sigmoid(self.fc(interaction).squeeze())
        
        return output


class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) model
    Another baseline - concatenates embeddings and passes through MLP
    """
    
    def __init__(self, num_users, num_movies, embedding_dim=64, hidden_dims=[128, 64, 32]):
        super().__init__()
        
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)
        
        # MLP layers
        layers = []
        input_dim = embedding_dim * 2  # Concatenated user + movie
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            input_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(input_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.movie_embedding.weight, std=0.01)
        
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
    
    def forward(self, user_ids, movie_ids):
        user_embed = self.user_embedding(user_ids)
        movie_embed = self.movie_embedding(movie_ids)
        
        # Concatenate
        concat = torch.cat([user_embed, movie_embed], dim=1)
        
        # Pass through MLP
        output = torch.sigmoid(self.mlp(concat).squeeze())
        
        return output


def get_model(model_type='two_tower', num_users=None, num_movies=None, **kwargs):
    """
    Factory function to get model
    
    Args:
        model_type: 'two_tower', 'gmf', or 'mlp'
        num_users: number of users
        num_movies: number of movies
        **kwargs: model-specific parameters
    """
    if model_type == 'two_tower':
        return TwoTowerNCF(num_users, num_movies, **kwargs)
    elif model_type == 'gmf':
        return GMF(num_users, num_movies, **kwargs)
    elif model_type == 'mlp':
        return MLP(num_users, num_movies, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == '__main__':
    # Test model
    logger.info("Testing Two-Tower NCF model...")
    
    num_users = 1000
    num_movies = 5000
    batch_size = 32
    
    model = TwoTowerNCF(num_users, num_movies)
    
    # Create sample data
    user_ids = torch.randint(0, num_users, (batch_size,))
    movie_ids = torch.randint(0, num_movies, (batch_size,))
    
    # Forward pass
    scores = model(user_ids, movie_ids)
    
    logger.info(f"Input shapes: users={user_ids.shape}, movies={movie_ids.shape}")
    logger.info(f"Output shape: {scores.shape}")
    logger.info(f"Score range: [{scores.min():.3f}, {scores.max():.3f}]")
    logger.info("✓ Model test passed!")
