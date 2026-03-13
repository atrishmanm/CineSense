"""
Sequential Recommender using GRU/LSTM
Models user behavior as a temporal sequence for better predictions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class SequentialRecommender(nn.Module):
    """
    Model user behavior as a sequence using GRU
    Captures temporal dynamics and user preference evolution
    """
    
    def __init__(
        self,
        num_movies: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.movie_embedding = nn.Embedding(num_movies, embed_dim)
        
        # GRU for sequence modeling
        self.gru = nn.GRU(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Attention over sequence
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Softmax(dim=1)
        )
        
        # Prediction head
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim + embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Xavier initialization"""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) >= 2:
                    nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(
        self, 
        movie_sequence: torch.Tensor, 
        target_movie: torch.Tensor,
        sequence_lengths: Optional[torch.Tensor] = None
    ):
        """
        Args:
            movie_sequence: (batch_size, seq_len) - user's watch history
            target_movie: (batch_size,) - movie to predict rating for
            sequence_lengths: (batch_size,) - actual length of each sequence
        
        Returns:
            predicted_rating: (batch_size,) - predicted rating
        """
        # Embed sequence
        seq_embed = self.movie_embedding(movie_sequence)  # (B, seq_len, embed_dim)
        
        # Pass through GRU
        if sequence_lengths is not None:
            # Pack padded sequence for efficiency
            packed = nn.utils.rnn.pack_padded_sequence(
                seq_embed, 
                sequence_lengths.cpu(), 
                batch_first=True, 
                enforce_sorted=False
            )
            gru_out, _ = self.gru(packed)
            gru_out, _ = nn.utils.rnn.pad_packed_sequence(gru_out, batch_first=True)
        else:
            gru_out, _ = self.gru(seq_embed)  # (B, seq_len, hidden_dim)
        
        # Apply attention
        attn_weights = self.attention(gru_out)  # (B, seq_len, 1)
        context = (gru_out * attn_weights).sum(dim=1)  # (B, hidden_dim)
        
        # Embed target movie
        target_embed = self.movie_embedding(target_movie)  # (B, embed_dim)
        
        # Combine context and target
        combined = torch.cat([context, target_embed], dim=1)
        
        # Predict rating
        rating = self.predictor(combined).squeeze()
        
        return rating


class LSTMSequentialRecommender(nn.Module):
    """
    LSTM-based sequential recommender
    Alternative to GRU with explicit cell state
    """
    
    def __init__(
        self,
        num_movies: int,
        embed_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()
        
        self.movie_embedding = nn.Embedding(num_movies, embed_dim)
        
        # LSTM for sequence modeling
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        # Multi-head self-attention
        self.self_attention = nn.MultiheadAttention(
            hidden_dim, 
            num_heads=4, 
            dropout=dropout,
            batch_first=True
        )
        
        # Prediction layers
        self.predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Xavier initialization"""
        for name, param in self.named_parameters():
            if 'weight' in name and len(param.shape) >= 2:
                nn.init.xavier_uniform_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
    
    def forward(self, movie_sequence: torch.Tensor, target_movie: torch.Tensor):
        """
        Args:
            movie_sequence: (batch_size, seq_len)
            target_movie: (batch_size,)
        
        Returns:
            predicted_rating: (batch_size,)
        """
        # Embed sequence
        seq_embed = self.movie_embedding(movie_sequence)  # (B, seq_len, embed_dim)
        
        # LSTM encoding
        lstm_out, (h_n, c_n) = self.lstm(seq_embed)  # (B, seq_len, hidden_dim)
        
        # Self-attention over sequence
        attn_out, _ = self.self_attention(lstm_out, lstm_out, lstm_out)
        
        # Use last hidden state and attention output
        last_hidden = h_n[-1]  # (B, hidden_dim)
        last_attn = attn_out[:, -1, :]  # (B, hidden_dim)
        
        # Combine representations
        combined = torch.cat([last_hidden, last_attn], dim=1)  # (B, hidden_dim * 2)
        
        # Predict
        rating = self.predictor(combined).squeeze()
        
        return rating


class TransformerSequentialRecommender(nn.Module):
    """
    Transformer-based sequential recommender (BERT4Rec style)
    Best for long sequences with complex dependencies
    """
    
    def __init__(
        self,
        num_movies: int,
        embed_dim: int = 128,
        num_heads: int = 8,
        num_layers: int = 3,
        ff_dim: int = 512,
        dropout: float = 0.1,
        max_seq_length: int = 100
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.max_seq_length = max_seq_length
        
        # Movie and position embeddings
        self.movie_embedding = nn.Embedding(num_movies + 1, embed_dim, padding_idx=0)
        self.position_embedding = nn.Embedding(max_seq_length, embed_dim)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output head
        self.output_layer = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights"""
        nn.init.xavier_uniform_(self.movie_embedding.weight[1:])  # Skip padding
        nn.init.xavier_uniform_(self.position_embedding.weight)
    
    def forward(self, movie_sequence: torch.Tensor, target_positions: torch.Tensor):
        """
        Args:
            movie_sequence: (batch_size, seq_len) - including target movie
            target_positions: (batch_size,) - position to predict
        
        Returns:
            predicted_rating: (batch_size,)
        """
        batch_size, seq_len = movie_sequence.shape
        
        # Create position indices
        positions = torch.arange(seq_len, device=movie_sequence.device).unsqueeze(0)
        positions = positions.expand(batch_size, -1)
        
        # Embeddings
        movie_emb = self.movie_embedding(movie_sequence)
        pos_emb = self.position_embedding(positions)
        
        # Combined embedding
        x = movie_emb + pos_emb
        
        # Create attention mask (causal for autoregressive)
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len, device=x.device)
        
        # Transformer
        transformer_out = self.transformer(x, mask=mask)
        
        # Get predictions at target positions
        batch_indices = torch.arange(batch_size, device=x.device)
        target_hidden = transformer_out[batch_indices, target_positions]
        
        # Predict rating
        rating = self.output_layer(target_hidden).squeeze()
        
        return rating


# Example training function
def train_sequential_model(
    model: nn.Module,
    train_loader,
    val_loader,
    num_epochs: int = 50,
    lr: float = 0.001,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Train sequential recommender
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    criterion = nn.MSELoss()
    
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        
        for batch in train_loader:
            sequences, targets, ratings = batch
            sequences = sequences.to(device)
            targets = targets.to(device)
            ratings = ratings.to(device)
            
            optimizer.zero_grad()
            predictions = model(sequences, targets)
            loss = criterion(predictions, ratings)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                sequences, targets, ratings = batch
                sequences = sequences.to(device)
                targets = targets.to(device)
                ratings = ratings.to(device)
                
                predictions = model(sequences, targets)
                loss = criterion(predictions, ratings)
                val_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        logger.info(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'model/sequential_best.pth')
            logger.info(f"✓ New best model saved! Val Loss: {best_val_loss:.4f}")
    
    return model


if __name__ == '__main__':
    # Test models
    num_movies = 10000
    batch_size = 32
    seq_len = 20
    
    # Test GRU model
    model_gru = SequentialRecommender(num_movies)
    sequences = torch.randint(0, num_movies, (batch_size, seq_len))
    targets = torch.randint(0, num_movies, (batch_size,))
    output = model_gru(sequences, targets)
    print(f"GRU Output shape: {output.shape}")
    
    # Test LSTM model
    model_lstm = LSTMSequentialRecommender(num_movies)
    output = model_lstm(sequences, targets)
    print(f"LSTM Output shape: {output.shape}")
    
    # Test Transformer model
    model_transformer = TransformerSequentialRecommender(num_movies)
    target_positions = torch.randint(0, seq_len, (batch_size,))
    output = model_transformer(sequences, target_positions)
    print(f"Transformer Output shape: {output.shape}")
