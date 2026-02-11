"""
STEP 6-7: Training Protocol + Model Saving
Complete training pipeline with proper ML practices
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from tqdm import tqdm
import logging
from datetime import datetime

from training.models import get_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MovieLensDataset(Dataset):
    """PyTorch dataset for MovieLens ratings"""
    
    def __init__(self, ratings_df, plot_embeddings=None):
        self.user_idx = torch.tensor(ratings_df["user_idx"].values, dtype=torch.long)
        self.movie_idx = torch.tensor(ratings_df["movie_idx"].values, dtype=torch.long)
        self.ratings = torch.tensor(ratings_df["rating"].values, dtype=torch.float32)
        
        self.plot_embeddings = plot_embeddings
        self.use_content = plot_embeddings is not None
    
    def __len__(self):
        return len(self.user_idx)
    
    def __getitem__(self, idx):
        item = {
            "user": self.user_idx[idx],
            "movie": self.movie_idx[idx],
            "rating": self.ratings[idx]
        }
        
        if self.use_content:
            movie_idx = self.movie_idx[idx].item()
            item["plot_emb"] = torch.from_numpy(self.plot_embeddings[movie_idx]).float()
        
        return item


def calculate_metrics(predictions, targets):
    """
    Calculate evaluation metrics
    
    Metrics:
        - RMSE: Rating prediction accuracy
        - MAE: Mean Absolute Error
    """
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # Clip predictions to rating range [0.5, 5.0]
    predictions = np.clip(predictions, 0.5, 5.0)
    
    rmse = np.sqrt(np.mean((predictions - targets) ** 2))
    mae = np.mean(np.abs(predictions - targets))
    
    return rmse, mae


def train_epoch(model, train_loader, optimizer, criterion, device, use_content=False):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    
    progress = tqdm(train_loader, desc="Training")
    
    for batch in progress:
        user = batch["user"].to(device)
        movie = batch["movie"].to(device)
        rating = batch["rating"].to(device)
        
        optimizer.zero_grad()
        
        if use_content:
            plot_emb = batch["plot_emb"].to(device)
            pred = model(user, movie, plot_emb)
        else:
            pred = model(user, movie)
        
        loss = criterion(pred, rating)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        progress.set_postfix({"loss": f"{loss.item():.4f}"})
    
    return total_loss / len(train_loader)


def validate(model, val_loader, criterion, device, use_content=False):
    """Validate model"""
    model.eval()
    total_loss = 0
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            user = batch["user"].to(device)
            movie = batch["movie"].to(device)
            rating = batch["rating"].to(device)
            
            if use_content:
                plot_emb = batch["plot_emb"].to(device)
                pred = model(user, movie, plot_emb)
            else:
                pred = model(user, movie)
            
            loss = criterion(pred, rating)
            total_loss += loss.item()
            
            predictions.extend(pred.cpu().numpy())
            targets.extend(rating.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    rmse, mae = calculate_metrics(predictions, targets)
    
    return avg_loss, rmse, mae


def train_model(model_type='hybrid', epochs=10, batch_size=1024, learning_rate=0.001):
    """
    Main training function
    
    Args:
        model_type: 'ncf' or 'hybrid'
        epochs: number of training epochs
        batch_size: batch size
        learning_rate: learning rate
    """
    logger.info("="*80)
    logger.info(f"TRAINING {model_type.upper()} MODEL")
    logger.info("="*80)
    
    # Device detection with GPU info
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        logger.info(f"\n🚀 GPU DETECTED: {torch.cuda.get_device_name(0)}")
        logger.info(f"   CUDA Version: {torch.version.cuda}")
        logger.info(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        logger.info(f"   Device: {device}")
    else:
        logger.info(f"\nDevice: {device} (⚠️  GPU not available)")
    
    # Load data
    logger.info("\nLoading data...")
    df = pd.read_csv("data/encoded.csv")
    
    # Load mappings
    with open("model/mappings.pkl", "rb") as f:
        mappings = pickle.load(f)
        num_users = mappings["num_users"]
        num_movies = mappings["num_movies"]
    
    logger.info(f"Users: {num_users:,}")
    logger.info(f"Movies: {num_movies:,}")
    logger.info(f"Ratings: {len(df):,}")
    
    # Time-based split (CRITICAL: prevents data leakage)
    logger.info("\nCreating time-based train/validation split...")
    df = df.sort_values("timestamp")
    
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]
    
    logger.info(f"Train: {len(train_df):,} ratings")
    logger.info(f"Val: {len(val_df):,} ratings")
    
    # Load plot embeddings if using hybrid model
    plot_embeddings = None
    if model_type == 'hybrid':
        logger.info("\nLoading plot embeddings...")
        plot_embeddings = np.load("model/plot_embeddings.npy")
        logger.info(f"Plot embeddings shape: {plot_embeddings.shape}")
    
    # Create datasets
    logger.info("\nCreating datasets...")
    train_dataset = MovieLensDataset(train_df, plot_embeddings)
    val_dataset = MovieLensDataset(val_df, plot_embeddings)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size*2, shuffle=False, num_workers=0)
    
    # Create model
    logger.info("\nCreating model...")
    model = get_model(model_type, num_users, num_movies, plot_emb_dim=768)
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Optimizer and loss
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    # Try to load checkpoint
    start_epoch = 1
    checkpoint_path = f"model/{model_type}_checkpoint.pt"
    if os.path.exists(checkpoint_path):
        logger.info(f"\n📂 Found checkpoint: {checkpoint_path}")
        logger.info("   Loading checkpoint to resume training...")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        best_rmse_from_checkpoint = checkpoint.get("best_rmse", float('inf'))
        logger.info(f"   ✓ Resumed from epoch {checkpoint['epoch']} (Best RMSE: {best_rmse_from_checkpoint:.4f})")
    else:
        logger.info(f"\n🆕 No checkpoint found. Starting fresh training.")
        best_rmse_from_checkpoint = float('inf')
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )
    
    # Training loop
    logger.info("\n" + "="*80)
    logger.info("TRAINING START")
    logger.info("="*80)
    
    best_rmse = best_rmse_from_checkpoint
    patience = 5
    patience_counter = 0
    
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_rmse": [],
        "val_mae": []
    }
    
    use_content = (model_type == 'hybrid')
    
    for epoch in range(start_epoch, epochs + 1):
        logger.info(f"\nEpoch {epoch}/{epochs}")
        logger.info("-" * 80)
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, use_content)
        
        # Validate
        val_loss, val_rmse, val_mae = validate(model, val_loader, criterion, device, use_content)
        
        # Update scheduler
        scheduler.step(val_rmse)
        
        # Log metrics
        logger.info(f"\nTrain Loss: {train_loss:.4f}")
        logger.info(f"Val Loss:   {val_loss:.4f}")
        logger.info(f"Val RMSE:   {val_rmse:.4f}")
        logger.info(f"Val MAE:    {val_mae:.4f}")
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_rmse"].append(val_rmse)
        history["val_mae"].append(val_mae)
        
        # Save checkpoint after EVERY epoch (so you never lose progress)
        checkpoint_path = f"model/{model_type}_checkpoint.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "val_rmse": val_rmse,
            "val_mae": val_mae,
            "best_rmse": best_rmse,
            "num_users": num_users,
            "num_movies": num_movies,
            "model_type": model_type,
            "history": history
        }, checkpoint_path)
        logger.info(f"💾 Checkpoint saved: {checkpoint_path}")
        
        # Save best model
        if val_rmse < best_rmse:
            best_rmse = val_rmse
            patience_counter = 0
            
            model_path = f"model/{model_type}_recommender.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "val_rmse": val_rmse,
                "val_mae": val_mae,
                "num_users": num_users,
                "num_movies": num_movies,
                "model_type": model_type
            }, model_path)
            
            logger.info(f"✓ Best model saved (RMSE: {val_rmse:.4f})")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            logger.info(f"\nEarly stopping triggered (patience={patience})")
            break
    
    # Save training history
    import json
    with open(f"model/{model_type}_history.json", "w") as f:
        json.dump(history, f, indent=2)
    
    logger.info("\n" + "="*80)
    logger.info("✓ TRAINING COMPLETE!")
    logger.info("="*80)
    logger.info(f"\nBest Val RMSE: {best_rmse:.4f}")
    logger.info(f"Model saved to: model/{model_type}_recommender.pt")
    logger.info("="*80)
    
    return model, history


if __name__ == "__main__":
    # Train both models
    
    # 1. NCF (Pure Collaborative Filtering)
    logger.info("\n" + "="*80)
    logger.info("TRAINING NEURAL COLLABORATIVE FILTERING (NCF)")
    logger.info("="*80)
    ncf_model, ncf_history = train_model(
        model_type='ncf',
        epochs=10,
        batch_size=2048,
        learning_rate=0.001
    )
    
    # 2. Hybrid (CF + Content)
    logger.info("\n\n" + "="*80)
    logger.info("TRAINING HYBRID MODEL (CF + CONTENT)")
    logger.info("="*80)
    hybrid_model, hybrid_history = train_model(
        model_type='hybrid',
        epochs=10,
        batch_size=1024,
        learning_rate=0.001
    )
    
    logger.info("\n" + "="*80)
    logger.info("✓ ALL TRAINING COMPLETE!")
    logger.info("="*80)
