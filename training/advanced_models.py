"""
ADVANCED MODEL ARCHITECTURES
Designed to achieve RMSE < 0.75 (Netflix-level performance)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    """Multi-head self-attention for feature interactions"""
    
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        
    def forward(self, x):
        # x shape: (batch, embed_dim)
        x_expanded = x.unsqueeze(1)  # (batch, 1, embed_dim)
        attn_out, _ = self.attention(x_expanded, x_expanded, x_expanded)
        return self.norm(x + attn_out.squeeze(1))


class DeepNCF(nn.Module):
    """
    Enhanced NCF with deeper architecture and attention
    Target RMSE: < 0.80
    """
    
    def __init__(self, num_users, num_movies, embedding_dim=128):
        super().__init__()
        
        # Larger embeddings for better representation
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)
        
        # User bias and movie bias (critical for Netflix-level performance)
        self.user_bias = nn.Embedding(num_users, 1)
        self.movie_bias = nn.Embedding(num_movies, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        
        # Attention mechanism
        self.attention = AttentionLayer(embedding_dim * 2, num_heads=4)
        
        # Deep fusion network
        self.fusion = nn.Sequential(
            nn.Linear(embedding_dim * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(64, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        # Better initialization
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.movie_embedding.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.movie_bias.weight)
        
        for layer in self.fusion:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                nn.init.zeros_(layer.bias)
    
    def forward(self, user_ids, movie_ids):
        # Embeddings
        user_emb = self.user_embedding(user_ids)
        movie_emb = self.movie_embedding(movie_ids)
        
        # Biases (crucial for good performance)
        user_b = self.user_bias(user_ids).squeeze()
        movie_b = self.movie_bias(movie_ids).squeeze()
        
        # Concatenate
        combined = torch.cat([user_emb, movie_emb], dim=1)
        
        # Attention
        combined = self.attention(combined)
        
        # Deep fusion
        score = self.fusion(combined).squeeze()
        
        # Add biases
        score = score + user_b + movie_b + self.global_bias
        
        return score


class AdvancedHybrid(nn.Module):
    """
    State-of-the-art Hybrid Model with multiple feature interactions
    Target RMSE: < 0.70
    """
    
    def __init__(self, num_users, num_movies, embedding_dim=128, plot_emb_dim=768):
        super().__init__()
        
        # Embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.movie_embedding = nn.Embedding(num_movies, embedding_dim)
        
        # Biases
        self.user_bias = nn.Embedding(num_users, 1)
        self.movie_bias = nn.Embedding(num_movies, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        
        # Content feature projection
        self.content_projection = nn.Sequential(
            nn.Linear(plot_emb_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, embedding_dim)
        )
        
        # Multi-modal fusion with attention
        fusion_dim = embedding_dim * 3  # user + movie + content
        
        self.attention = AttentionLayer(fusion_dim, num_heads=8)
        
        # Deep fusion tower
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            nn.Linear(64, 1)
        )
        
        # Feature interaction (like DeepFM)
        self.feature_interaction = nn.Sequential(
            nn.Linear(fusion_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.movie_embedding.weight, std=0.01)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.movie_bias.weight)
        
        for module in [self.content_projection, self.fusion, self.feature_interaction]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)
    
    def forward(self, user_ids, movie_ids, plot_embeddings):
        # CF embeddings
        user_emb = self.user_embedding(user_ids)
        movie_emb = self.movie_embedding(movie_ids)
        
        # Biases
        user_b = self.user_bias(user_ids).squeeze()
        movie_b = self.movie_bias(movie_ids).squeeze()
        
        # Project content features
        content_emb = self.content_projection(plot_embeddings)
        
        # Multi-modal fusion
        combined = torch.cat([user_emb, movie_emb, content_emb], dim=1)
        
        # Attention
        combined_attn = self.attention(combined)
        
        # Two paths: deep fusion + feature interaction
        deep_score = self.fusion(combined_attn).squeeze()
        interaction_score = self.feature_interaction(combined).squeeze()
        
        # Combine paths
        score = deep_score + interaction_score * 0.3  # Weighted combination
        
        # Add biases
        score = score + user_b + movie_b + self.global_bias
        
        return score


def get_advanced_model(model_type, num_users, num_movies, plot_emb_dim=768):
    """Factory function for advanced models"""
    
    if model_type == 'deep_ncf':
        return DeepNCF(num_users, num_movies, embedding_dim=128)
    elif model_type == 'advanced_hybrid':
        return AdvancedHybrid(num_users, num_movies, embedding_dim=128, plot_emb_dim=plot_emb_dim)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
