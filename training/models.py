"""
STEP 4-5: Neural Recommendation Models
- NCF (Neural Collaborative Filtering)
- Hybrid Model (NCF + Content Embeddings)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NCF(nn.Module):
    """
    STEP 4: Neural Collaborative Filtering (NCF)
    
    This learns user taste automatically from behavior
    Netflix-style architecture
    
    Architecture:
        User Embedding (64) + Movie Embedding (64)
        → Concatenate (128)
        → Dense Layers
        → Output Score
    """
    
    def __init__(self, num_users, num_movies, embedding_dim=64):
        super().__init__()
        
        # Embeddings
        self.user_emb = nn.Embedding(num_users, embedding_dim)
        self.movie_emb = nn.Embedding(num_movies, embedding_dim)
        
        # Deep layers
        self.fc = nn.Sequential(
            nn.Linear(embedding_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Xavier initialization"""
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.movie_emb.weight)
        
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, user, movie):
        """
        Args:
            user: [batch_size] user indices
            movie: [batch_size] movie indices
        
        Returns:
            scores: [batch_size] predicted ratings
        """
        u = self.user_emb(user)  # [batch, 64]
        m = self.movie_emb(movie)  # [batch, 64]
        
        x = torch.cat([u, m], dim=1)  # [batch, 128]
        
        return self.fc(x).squeeze()  # [batch]


class HybridRecommender(nn.Module):
    """
    STEP 5: HYBRID DEEP LEARNING MODEL
    
    This is what Netflix actually uses!
    
    Combines:
        - User embeddings (collaborative filtering)
        - Movie CF embeddings
        - Movie plot embeddings (from transformers, 768-dim)
    
    Architecture:
        [User Emb (64)] + [Movie Emb (64)] + [Plot Emb (768)]
        → Concatenate (896)
        → Dense Layers
        → Final Score
    
    🔥 This handles cold-start movies — huge win!
    """
    
    def __init__(self, num_users, num_movies, plot_emb_dim=768, user_emb_dim=64):
        super().__init__()
        
        # Collaborative filtering embeddings
        self.user_emb = nn.Embedding(num_users, user_emb_dim)
        self.movie_emb = nn.Embedding(num_movies, user_emb_dim)
        
        # Deep fusion network
        input_dim = user_emb_dim + user_emb_dim + plot_emb_dim  # 64 + 64 + 768 = 896
        
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(64, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Xavier initialization"""
        nn.init.xavier_uniform_(self.user_emb.weight)
        nn.init.xavier_uniform_(self.movie_emb.weight)
        
        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, user, movie, plot_emb):
        """
        Args:
            user: [batch_size] user indices
            movie: [batch_size] movie indices
            plot_emb: [batch_size, 768] plot embeddings
        
        Returns:
            scores: [batch_size] predicted ratings
        """
        u = self.user_emb(user)  # [batch, 64]
        m = self.movie_emb(movie)  # [batch, 64]
        
        # Concatenate all features
        x = torch.cat([u, m, plot_emb], dim=1)  # [batch, 896]
        
        return self.fc(x).squeeze()  # [batch]


def get_model(model_type, num_users, num_movies, plot_emb_dim=768):
    """
    Factory function to get the right model
    
    Args:
        model_type: 'ncf' or 'hybrid'
        num_users: number of users
        num_movies: number of movies
        plot_emb_dim: dimension of plot embeddings (768 for mpnet)
    
    Returns:
        model: NCF or HybridRecommender
    """
    if model_type == 'ncf':
        return NCF(num_users, num_movies)
    elif model_type == 'hybrid':
        return HybridRecommender(num_users, num_movies, plot_emb_dim)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


if __name__ == "__main__":
    # Test models
    print("Testing NCF model...")
    ncf = NCF(num_users=1000, num_movies=5000)
    
    user = torch.randint(0, 1000, (32,))
    movie = torch.randint(0, 5000, (32,))
    
    output = ncf(user, movie)
    print(f"NCF output shape: {output.shape}")
    
    print("\nTesting Hybrid model...")
    hybrid = HybridRecommender(num_users=1000, num_movies=5000)
    
    plot_emb = torch.randn(32, 768)
    output = hybrid(user, movie, plot_emb)
    print(f"Hybrid output shape: {output.shape}")
    
    print("\n✓ Models test passed!")
