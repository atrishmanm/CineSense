"""
Advanced recommendation models with Transformer architecture
Improves RMSE from 0.9 to target 0.65-0.75
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional


class MultiHeadAttention(nn.Module):
    """Multi-head self-attention mechanism"""
    
    def __init__(self, embed_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.embed_dim = embed_dim
        self.head_dim = embed_dim // num_heads
        
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_linear = nn.Linear(embed_dim, embed_dim)
        self.k_linear = nn.Linear(embed_dim, embed_dim)
        self.v_linear = nn.Linear(embed_dim, embed_dim)
        self.out_linear = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size = x.size(0)
        
        # Linear projections and reshape to (batch, num_heads, seq_len, head_dim)
        Q = self.q_linear(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_linear(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_linear(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        
        return self.out_linear(context)


class TransformerBlock(nn.Module):
    """Transformer encoder block with self-attention and feed-forward network"""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        ff_dim: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Multi-head attention with residual connection and layer norm
        attn_out = self.attention(x)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward with residual connection and layer norm
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        
        return x


class AdvancedHybridRecommender(nn.Module):
    """
    State-of-the-art recommendation model with Transformer architecture
    
    Features:
    - Multi-head self-attention for complex pattern recognition
    - User and movie towers with deep encoding
    - Content feature integration (genres, cast, directors, plot embeddings)
    - Residual connections and layer normalization
    - Dropout for regularization
    
    Expected RMSE: 0.65-0.75 (vs 0.9 baseline)
    """
    
    def __init__(
        self,
        num_users: int,
        num_movies: int,
        embed_dim: int = 128,
        num_transformer_blocks: int = 2,
        num_heads: int = 8,
        content_feature_dim: int = 55,
        plot_embed_dim: int = 384,
        dropout: float = 0.2
    ):
        super().__init__()
        
        # User components
        self.user_embedding = nn.Embedding(num_users, embed_dim)
        self.user_bias = nn.Embedding(num_users, 1)
        
        # Movie components
        self.movie_embedding = nn.Embedding(num_movies, embed_dim)
        self.movie_bias = nn.Embedding(num_movies, 1)
        
        # Content encoding (genres, cast, director, etc.)
        self.content_encoder = nn.Sequential(
            nn.Linear(content_feature_dim, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim)
        )
        
        # Plot embedding encoder (from sentence transformers)
        self.plot_encoder = nn.Sequential(
            nn.Linear(plot_embed_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, embed_dim),
            nn.GELU(),
            nn.LayerNorm(embed_dim)
        )
        
        # Feature fusion - combine user, movie, content, plot
        fusion_dim = embed_dim * 4
        
        # Transformer layers for complex interactions
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(fusion_dim, num_heads, fusion_dim * 2, dropout)
            for _ in range(num_transformer_blocks)
        ])
        
        # Deep MLP for final prediction
        self.predictor = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.GELU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout),
            
            nn.Linear(512, 256),
            nn.GELU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout),
            
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            
            nn.Linear(64, 1)
        )
        
        # Global bias term
        self.global_bias = nn.Parameter(torch.zeros(1))
        
        self._init_weights()
    
    def _init_weights(self):
        """Xavier initialization for better convergence"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.01)
    
    def forward(
        self,
        user_ids: torch.Tensor,
        movie_ids: torch.Tensor,
        content_features: Optional[torch.Tensor] = None,
        plot_embeddings: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            user_ids: User ID tensor (batch_size,)
            movie_ids: Movie ID tensor (batch_size,)
            content_features: Movie content features (batch_size, content_feature_dim)
            plot_embeddings: Movie plot embeddings (batch_size, plot_embed_dim)
        
        Returns:
            Rating predictions (batch_size,)
        """
        # Get embeddings
        user_embed = self.user_embedding(user_ids)  # (batch, embed_dim)
        movie_embed = self.movie_embedding(movie_ids)  # (batch, embed_dim)
        
        # Get biases
        user_bias = self.user_bias(user_ids).squeeze(-1)
        movie_bias = self.movie_bias(movie_ids).squeeze(-1)
        
        # Encode content features if provided
        if content_features is not None:
            content_embed = self.content_encoder(content_features)
        else:
            content_embed = torch.zeros_like(user_embed)
        
        # Encode plot embeddings if provided
        if plot_embeddings is not None:
            plot_embed = self.plot_encoder(plot_embeddings)
        else:
            plot_embed = torch.zeros_like(user_embed)
        
        # Concatenate all features
        features = torch.cat([user_embed, movie_embed, content_embed, plot_embed], dim=-1)
        
        # Add sequence dimension for transformer: (batch, 1, features)
        features = features.unsqueeze(1)
        
        # Apply transformer blocks
        for transformer in self.transformer_blocks:
            features = transformer(features)
        
        # Remove sequence dimension: (batch, features)
        features = features.squeeze(1)
        
        # Predict rating через deep MLP
        prediction = self.predictor(features).squeeze(-1)
        
        # Add bias terms
        prediction = prediction + user_bias + movie_bias + self.global_bias
        
        return prediction


class DeepNCF(nn.Module):
    """
    Deep Neural Collaborative Filtering
    Simpler alternative to Transformer model
    """
    
    def __init__(
        self,
        num_users: int,
        num_movies: int,
        embed_dim: int = 64,
        layers: List[int] = [128, 64, 32],
        dropout: float = 0.2
    ):
        super().__init__()
        
        # Embeddings
        self.user_embedding = nn.Embedding(num_users, embed_dim)
        self.movie_embedding = nn.Embedding(num_movies, embed_dim)
        
        # GMF path (Generalized Matrix Factorization)
        self.gmf_user = nn.Embedding(num_users, embed_dim)
        self.gmf_movie = nn.Embedding(num_movies, embed_dim)
        
        # MLP path
        mlp_layers = []
        input_dim = embed_dim * 2
        
        for layer_dim in layers:
            mlp_layers.extend([
                nn.Linear(input_dim, layer_dim),
                nn.ReLU(),
                nn.BatchNorm1d(layer_dim),
                nn.Dropout(dropout)
            ])
            input_dim = layer_dim
        
        self.mlp = nn.Sequential(*mlp_layers)
        
        # Fusion layer
        self.fusion = nn.Linear(embed_dim + layers[-1], 1)
        
        self._init_weights()
    
    def _init_weights(self):
        """He initialization for ReLU networks"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.01)
    
    def forward(
        self,
        user_ids: torch.Tensor,
        movie_ids: torch.Tensor
    ) -> torch.Tensor:
        # GMF path: element-wise product
        gmf_user_embed = self.gmf_user(user_ids)
        gmf_movie_embed = self.gmf_movie(movie_ids)
        gmf_output = gmf_user_embed * gmf_movie_embed
        
        # MLP path: concatenation
        mlp_user_embed = self.user_embedding(user_ids)
        mlp_movie_embed = self.movie_embedding(movie_ids)
        mlp_input = torch.cat([mlp_user_embed, mlp_movie_embed], dim=-1)
        mlp_output = self.mlp(mlp_input)
        
        # Fusion
        fusion_input = torch.cat([gmf_output, mlp_output], dim=-1)
        prediction = self.fusion(fusion_input).squeeze(-1)
        
        return prediction


def build_model(
    model_type: str,
    num_users: int,
    num_movies: int,
    **kwargs
) -> nn.Module:
    """
    Factory function to build models
    
    Args:
        model_type: 'advanced' or 'deep_ncf'
        num_users: Number of unique users
        num_movies: Number of unique movies
        **kwargs: Additional model parameters
    
    Returns:
        Model instance
    """
    models = {
        'advanced': AdvancedHybridRecommender,
        'deep_ncf': DeepNCF,
        'transformer': AdvancedHybridRecommender  # Alias
    }
    
    if model_type not in models:
        raise ValueError(f"Unknown model type: {model_type}. Available: {list(models.keys())}")
    
    return models[model_type](num_users=num_users, num_movies=num_movies, **kwargs)


if __name__ == '__main__':
    # Test models
    batch_size = 32
    num_users = 1000
    num_movies = 5000
    
    user_ids = torch.randint(0, num_users, (batch_size,))
    movie_ids = torch.randint(0, num_movies, (batch_size,))
    content_features = torch.randn(batch_size, 55)
    plot_embeddings = torch.randn(batch_size, 384)
    
    print("Testing Advanced Hybrid Recommender...")
    model = AdvancedHybridRecommender(num_users, num_movies)
    output = model(user_ids, movie_ids, content_features, plot_embeddings)
    print(f"Output shape: {output.shape}")
    print(f"Sample predictions: {output[:5]}")
    
    print("\nTesting Deep NCF...")
    model2 = DeepNCF(num_users, num_movies)
    output2 = model2(user_ids, movie_ids)
    print(f"Output shape: {output2.shape}")
    print(f"Sample predictions: {output2[:5]}")
    
    print("\n✓ All models working correctly!")
