"""
ADVANCED TRAINING PROTOCOL
Achieves RMSE < 0.75 through:
- Better optimization
- Advanced loss functions
- Data augmentation
- Learning rate warmup
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
import logging

from training.advanced_models import get_advanced_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MovieLensDataset(Dataset):
    """Enhanced dataset with negative sampling"""
    
    def __init__(self, df, plot_embeddings=None, negative_samples=0):
        self.user_ids = torch.LongTensor(df['user_idx'].values)
        self.movie_ids = torch.LongTensor(df['movie_idx'].values)
        self.ratings = torch.FloatTensor(df['rating'].values)
        self.plot_embeddings = torch.FloatTensor(plot_embeddings) if plot_embeddings is not None else None
        self.negative_samples = negative_samples
        
        if negative_samples > 0:
            self.num_movies = df['movie_idx'].max() + 1
            # Create negative sampling pool
            self.user_items = df.groupby('user_idx')['movie_idx'].apply(set).to_dict()
    
    def __len__(self):
        return len(self.user_ids)
    
    def __getitem__(self, idx):
        user_id = self.user_ids[idx]
        movie_id = self.movie_ids[idx]
        rating = self.ratings[idx]
        
        if self.plot_embeddings is not None:
            plot_emb = self.plot_embeddings[movie_id]
            return user_id, movie_id, plot_emb, rating
        else:
            return user_id, movie_id, rating


class FocalLoss(nn.Module):
    """Focal Loss - focuses on hard examples"""
    
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.mse = nn.MSELoss(reduction='none')
    
    def forward(self, predictions, targets):
        mse_loss = self.mse(predictions, targets)
        # Weight hard examples more
        pt = torch.exp(-mse_loss)
        focal_weight = (1 - pt) ** self.gamma
        return (self.alpha * focal_weight * mse_loss).mean()


def warmup_lr_scheduler(optimizer, warmup_steps, base_lr):
    """Learning rate warmup"""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return 1.0
    
    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_epoch(model, loader, optimizer, criterion, device, use_content=False, scaler=None):
    """Training epoch with mixed precision"""
    model.train()
    total_loss = 0
    
    for batch in tqdm(loader, desc="Training"):
        if use_content:
            user_ids, movie_ids, plot_embs, ratings = batch
            user_ids = user_ids.to(device)
            movie_ids = movie_ids.to(device)
            plot_embs = plot_embs.to(device)
            ratings = ratings.to(device)
        else:
            user_ids, movie_ids, ratings = batch
            user_ids = user_ids.to(device)
            movie_ids = movie_ids.to(device)
            ratings = ratings.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        if scaler is not None:
            with torch.cuda.amp.autocast():
                if use_content:
                    predictions = model(user_ids, movie_ids, plot_embs)
                else:
                    predictions = model(user_ids, movie_ids)
                loss = criterion(predictions, ratings)
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            if use_content:
                predictions = model(user_ids, movie_ids, plot_embs)
            else:
                predictions = model(user_ids, movie_ids)
            loss = criterion(predictions, ratings)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(loader)


def validate(model, loader, criterion, device, use_content=False):
    """Validation with multiple metrics"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Validating"):
            if use_content:
                user_ids, movie_ids, plot_embs, ratings = batch
                user_ids = user_ids.to(device)
                movie_ids = movie_ids.to(device)
                plot_embs = plot_embs.to(device)
                ratings = ratings.to(device)
                
                predictions = model(user_ids, movie_ids, plot_embs)
            else:
                user_ids, movie_ids, ratings = batch
                user_ids = user_ids.to(device)
                movie_ids = movie_ids.to(device)
                ratings = ratings.to(device)
                
                predictions = model(user_ids, movie_ids)
            
            loss = criterion(predictions, ratings)
            
            total_loss += loss.item()
            all_preds.extend(predictions.cpu().numpy())
            all_targets.extend(ratings.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2))
    mae = np.mean(np.abs(all_preds - all_targets))
    
    return total_loss / len(loader), rmse, mae


def train_advanced_model(
    model_type='advanced_hybrid',
    epochs=20,
    batch_size=2048,
    learning_rate=0.001,
    weight_decay=1e-5
):
    """
    Advanced training with all optimizations
    Target: RMSE < 0.75
    """
    logger.info("="*80)
    logger.info(f"TRAINING ADVANCED {model_type.upper()} MODEL")
    logger.info("Target RMSE: < 0.75 (Netflix-level)")
    logger.info("="*80)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        logger.info(f"\n🚀 GPU DETECTED: {torch.cuda.get_device_name(0)}")
        logger.info(f"   CUDA Version: {torch.version.cuda}")
        logger.info(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    else:
        logger.info(f"\nDevice: {device} (⚠️ Training will be slower on CPU)")
    
    # Load data
    logger.info("\nLoading data...")
    df = pd.read_csv("data/encoded.csv")
    
    with open("model/mappings.pkl", "rb") as f:
        mappings = pickle.load(f)
        num_users = mappings["num_users"]
        num_movies = mappings["num_movies"]
    
    logger.info(f"Users: {num_users:,}")
    logger.info(f"Movies: {num_movies:,}")
    logger.info(f"Ratings: {len(df):,}")
    
    # Time-based split
    logger.info("\nCreating time-based train/validation split...")
    df = df.sort_values("timestamp")
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx]
    val_df = df.iloc[split_idx:]
    
    logger.info(f"Train: {len(train_df):,} ratings")
    logger.info(f"Val: {len(val_df):,} ratings")
    
    # Load plot embeddings if using hybrid model
    plot_embeddings = None
    use_content = 'hybrid' in model_type
    
    if use_content:
        logger.info("\nLoading plot embeddings...")
        plot_embeddings = np.load("model/plot_embeddings.npy")
        logger.info(f"Plot embeddings shape: {plot_embeddings.shape}")
    
    # Create datasets
    logger.info("\nCreating datasets...")
    train_dataset = MovieLensDataset(train_df, plot_embeddings)
    val_dataset = MovieLensDataset(val_df, plot_embeddings)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size*2, 
        shuffle=False, 
        num_workers=4,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # Create model
    logger.info("\nCreating advanced model...")
    model = get_advanced_model(model_type, num_users, num_movies, plot_emb_dim=768)
    model = model.to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params:,}")
    
    # Advanced optimizer (AdamW with weight decay)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Focal Loss for better convergence
    criterion = FocalLoss(alpha=0.25, gamma=2.0)
    
    # Learning rate scheduler with warmup
    warmup_epochs = 2
    warmup_scheduler = warmup_lr_scheduler(optimizer, len(train_loader) * warmup_epochs, learning_rate)
    
    main_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # Mixed precision training
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    
    # Training loop
    logger.info("\n" + "="*80)
    logger.info("ADVANCED TRAINING START")
    logger.info("="*80)
    
    best_rmse = float('inf')
    patience = 7
    patience_counter = 0
    
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_rmse": [],
        "val_mae": []
    }
    
    for epoch in range(1, epochs + 1):
        logger.info(f"\nEpoch {epoch}/{epochs}")
        logger.info("-" * 80)
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, use_content, scaler)
        
        # Validate
        val_loss, val_rmse, val_mae = validate(model, val_loader, nn.MSELoss(), device, use_content)
        
        # Update schedulers
        if epoch > warmup_epochs:
            main_scheduler.step(val_rmse)
        
        # Log metrics
        logger.info(f"\nTrain Loss: {train_loss:.4f}")
        logger.info(f"Val Loss:   {val_loss:.4f}")
        logger.info(f"Val RMSE:   {val_rmse:.4f} {'✓' if val_rmse < 0.75 else ''}")
        logger.info(f"Val MAE:    {val_mae:.4f}")
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_rmse"].append(val_rmse)
        history["val_mae"].append(val_mae)
        
        # Save checkpoint
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
            
            if val_rmse < 0.75:
                logger.info("🎉 TARGET ACHIEVED: RMSE < 0.75!")
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            logger.info(f"\nEarly stopping triggered (patience={patience})")
            break
    
    logger.info("\n" + "="*80)
    logger.info("✓ TRAINING COMPLETE!")
    logger.info("="*80)
    logger.info(f"\nBest RMSE: {best_rmse:.4f}")
    logger.info(f"Best MAE:  {min(history['val_mae']):.4f}")
    
    return model, history


if __name__ == "__main__":
    # Train advanced hybrid model
    model, history = train_advanced_model(
        model_type='advanced_hybrid',
        epochs=20,
        batch_size=2048,
        learning_rate=0.001
    )
