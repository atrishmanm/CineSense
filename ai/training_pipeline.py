"""
Netflix-Level Training Pipeline
Complete production-grade training with:
    - Adam optimizer
    - Learning rate scheduling
    - Early stopping
    - Gradient clipping
    - Checkpointing
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import logging
from pathlib import Path
from tqdm import tqdm
import json
from datetime import datetime

from ai.data_preprocessor import DataPreprocessor, RecommenderDataset
from ai.two_tower_ncf import TwoTowerNCF
from ai.hybrid_model import create_hybrid_model, precompute_content_embeddings
from ai.evaluation import RecommenderEvaluator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EarlyStopping:
    """
    Early stopping to prevent overfitting
    """
    
    def __init__(self, patience=5, min_delta=0.001, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
        self.best_epoch = 0
    
    def __call__(self, score, epoch):
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            return False
        
        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:  # max
            improved = score > (self.best_score + self.min_delta)
        
        if improved:
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
        else:
            self.counter += 1
            
            if self.counter >= self.patience:
                logger.info(f"Early stopping triggered! Best epoch: {self.best_epoch}")
                self.early_stop = True
        
        return self.early_stop


class TrainingPipeline:
    """
    Complete training pipeline for recommendation models
    """
    
    def __init__(self, config):
        self.config = config
        
        # Paths
        self.model_dir = Path(config.get('model_dir', 'ai/models'))
        self.model_dir.mkdir(exist_ok=True)
        
        # Device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Will be set during training
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_metrics': []
        }
    
    def prepare_data(self, use_preprocessed=True, sample_size=None):
        """
        Prepare data for training
        
        Args:
            use_preprocessed: bool - load preprocessed data if available
            sample_size: int - sample size for testing (None = full dataset)
        """
        logger.info("="*60)
        logger.info("DATA PREPARATION")
        logger.info("="*60)
        
        preprocessor = DataPreprocessor()
        
        if use_preprocessed:
            try:
                preprocessor.load_processed_data()
                logger.info("✓ Loaded preprocessed data")
            except:
                logger.info("No preprocessed data found, running preprocessing...")
                from ai.data_preprocessor import preprocess_pipeline
                preprocessor = preprocess_pipeline(sample_size=sample_size)
        else:
            from ai.data_preprocessor import preprocess_pipeline
            preprocessor = preprocess_pipeline(sample_size=sample_size)
        
        # Create train/val/test splits
        train_df, val_df, test_df = preprocessor.create_time_based_split(
            train_ratio=0.8,
            val_ratio=0.1
        )
        
        # Save splits
        train_df.to_pickle('ai/cache/train_split.pkl')
        val_df.to_pickle('ai/cache/val_split.pkl')
        test_df.to_pickle('ai/cache/test_split.pkl')
        
        # Create PyTorch datasets
        use_implicit = self.config.get('use_implicit_feedback', True)
        
        train_dataset = RecommenderDataset(train_df, use_implicit=use_implicit)
        val_dataset = RecommenderDataset(val_df, use_implicit=use_implicit)
        test_dataset = RecommenderDataset(test_df, use_implicit=use_implicit)
        
        # Create data loaders
        batch_size = self.config.get('batch_size', 1024)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # Windows compatibility
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size * 2,  # Larger batch for validation
            shuffle=False,
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size * 2,
            shuffle=False,
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        logger.info(f"✓ Data loaders created:")
        logger.info(f"  Train: {len(train_dataset):,} samples")
        logger.info(f"  Validation: {len(val_dataset):,} samples")
        logger.info(f"  Test: {len(test_dataset):,} samples")
        
        return train_loader, val_loader, test_loader, preprocessor
    
    def build_model(self, preprocessor, model_type='hybrid'):
        """
        Build model
        
        Args:
            preprocessor: DataPreprocessor with metadata
            model_type: 'two_tower' or 'hybrid'
        """
        logger.info("="*60)
        logger.info("MODEL BUILDING")
        logger.info("="*60)
        
        num_users = preprocessor.num_users
        num_movies = preprocessor.num_movies
        
        if model_type == 'hybrid':
            # Precompute content embeddings if not already done
            content_path = Path('ai/cache/movie_content_embeddings.pkl')
            
            if not content_path.exists():
                logger.info("Content embeddings not found, computing...")
                import pandas as pd
                content_df = pd.read_pickle('ai/cache/movie_content.pkl')
                precompute_content_embeddings(content_df)
            
            # Create hybrid model
            self.model = create_hybrid_model(
                num_users=num_users,
                num_movies=num_movies,
                movie_encoder=preprocessor.movie_encoder,
                use_content=True
            )
        else:
            # Create pure Two-Tower NCF
            self.model = TwoTowerNCF(
                num_users=num_users,
                num_movies=num_movies,
                embedding_dim=self.config.get('embedding_dim', 128),
                hidden_dims=self.config.get('hidden_dims', [256, 128, 64])
            )
        
        # Move to device
        self.model = self.model.to(self.device)
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        logger.info(f"✓ Model created:")
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Trainable parameters: {trainable_params:,}")
        
        return self.model
    
    def setup_training(self):
        """
        Setup optimizer, loss, scheduler
        """
        logger.info("="*60)
        logger.info("TRAINING SETUP")
        logger.info("="*60)
        
        # Loss function
        if self.config.get('use_implicit_feedback', True):
            self.criterion = nn.BCELoss()  # Binary Cross Entropy for implicit
            logger.info("Loss: Binary Cross Entropy (implicit feedback)")
        else:
            self.criterion = nn.MSELoss()  # MSE for rating prediction
            logger.info("Loss: Mean Squared Error (rating prediction)")
        
        # Optimizer
        lr = self.config.get('learning_rate', 0.001)
        weight_decay = self.config.get('weight_decay', 1e-5)
        
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        logger.info(f"Optimizer: Adam (lr={lr}, weight_decay={weight_decay})")
        
        # Learning rate scheduler
        scheduler_type = self.config.get('scheduler', 'reduce_on_plateau')
        
        if scheduler_type == 'reduce_on_plateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=3,
                verbose=True
            )
        elif scheduler_type == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.get('num_epochs', 20)
            )
        else:
            self.scheduler = None
        
        logger.info(f"Scheduler: {scheduler_type}")
        logger.info("✓ Training setup complete")
    
    def train_epoch(self, train_loader, epoch):
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        num_batches = len(train_loader)
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
        
        for batch in progress_bar:
            # Get data
            user_ids = batch['user_id'].to(self.device)
            movie_ids = batch['movie_id'].to(self.device)
            labels = batch['label'].to(self.device)
            
            # Forward pass
            predictions = self.model(user_ids, movie_ids)
            loss = self.criterion(predictions, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping (prevent exploding gradients)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self, val_loader):
        """Validate model"""
        self.model.eval()
        
        total_loss = 0
        num_batches = len(val_loader)
        
        with torch.no_grad():
            for batch in val_loader:
                user_ids = batch['user_id'].to(self.device)
                movie_ids = batch['movie_id'].to(self.device)
                labels = batch['label'].to(self.device)
                
                predictions = self.model(user_ids, movie_ids)
                loss = self.criterion(predictions, labels)
                
                total_loss += loss.item()
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def train(self, train_loader, val_loader, test_loader):
        """
        Full training loop
        """
        logger.info("="*60)
        logger.info("TRAINING START")
        logger.info("="*60)
        
        num_epochs = self.config.get('num_epochs', 20)
        early_stopping = EarlyStopping(
            patience=self.config.get('early_stopping_patience', 5),
            min_delta=0.0001,
            mode='min'
        )
        
        best_val_loss = float('inf')
        
        for epoch in range(1, num_epochs + 1):
            logger.info(f"\nEpoch {epoch}/{num_epochs}")
            logger.info("-" * 60)
            
            # Train
            train_loss = self.train_epoch(train_loader, epoch)
            
            # Validate
            val_loss = self.validate(val_loader)
            
            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Log
            logger.info(f"Train Loss: {train_loss:.4f}")
            logger.info(f"Val Loss:   {val_loss:.4f}")
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_checkpoint('best_model.pt', epoch, val_loss)
                logger.info(f"✓ Best model saved (val_loss: {val_loss:.4f})")
            
            # Early stopping check
            if early_stopping(val_loss, epoch):
                logger.info("Early stopping triggered!")
                break
        
        logger.info("="*60)
        logger.info("TRAINING COMPLETE")
        logger.info("="*60)
        
        # Load best model and evaluate on test set
        self.load_checkpoint('best_model.pt')
        
        logger.info("\nFinal evaluation on test set...")
        evaluator = RecommenderEvaluator(self.model, self.device)
        test_metrics = evaluator.evaluate_full(test_loader, k_values=[5, 10, 20])
        
        # Save metrics
        self.history['test_metrics'] = test_metrics
        self.save_history()
        
        return test_metrics
    
    def save_checkpoint(self, filename, epoch, val_loss):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_loss': val_loss,
            'config': self.config
        }
        
        path = self.model_dir / filename
        torch.save(checkpoint, path)
    
    def load_checkpoint(self, filename):
        """Load model checkpoint"""
        path = self.model_dir / filename
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        logger.info(f"✓ Loaded checkpoint from {filename}")
    
    def save_history(self):
        """Save training history"""
        path = self.model_dir / 'training_history.json'
        
        with open(path, 'w') as f:
            json.dump(self.history, f, indent=2)
        
        logger.info(f"✓ Training history saved to {path}")


def train_netflix_model(config=None):
    """
    Main training function
    
    Args:
        config: dict with training configuration
    """
    if config is None:
        config = {
            # Data
            'batch_size': 1024,
            'use_implicit_feedback': True,
            
            # Model
            'model_type': 'hybrid',  # 'two_tower' or 'hybrid'
            'embedding_dim': 128,
            'hidden_dims': [256, 128, 64],
            
            # Training
            'num_epochs': 20,
            'learning_rate': 0.001,
            'weight_decay': 1e-5,
            'scheduler': 'reduce_on_plateau',
            'early_stopping_patience': 5,
            
            # Paths
            'model_dir': 'ai/models'
        }
    
    # Initialize pipeline
    pipeline = TrainingPipeline(config)
    
    # Prepare data
    train_loader, val_loader, test_loader, preprocessor = pipeline.prepare_data(
        use_preprocessed=True,
        sample_size=None  # Use full dataset (set to 100000 for testing)
    )
    
    # Build model
    pipeline.build_model(preprocessor, model_type=config['model_type'])
    
    # Setup training
    pipeline.setup_training()
    
    # Train
    test_metrics = pipeline.train(train_loader, val_loader, test_loader)
    
    logger.info("="*60)
    logger.info("✓ TRAINING PIPELINE COMPLETE!")
    logger.info("="*60)
    
    return pipeline, test_metrics


if __name__ == '__main__':
    # Run training
    pipeline, metrics = train_netflix_model()
