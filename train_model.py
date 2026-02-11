"""
Training Script for Netflix-Level Recommender System

This script trains the complete Two-Tower + Content-Aware Hybrid model
on MovieLens + TMDB data.

Usage:
    python train_model.py [--sample] [--epochs 20] [--batch-size 1024]
"""

import argparse
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Train Netflix-level recommender model')
    
    # Data arguments
    parser.add_argument('--sample', action='store_true',
                       help='Use sample of data for testing (100K ratings)')
    parser.add_argument('--preprocess-only', action='store_true',
                       help='Only run data preprocessing, skip training')
    
    # Model arguments
    parser.add_argument('--model-type', type=str, default='hybrid',
                       choices=['two_tower', 'hybrid'],
                       help='Model type: two_tower (CF only) or hybrid (CF + Content)')
    parser.add_argument('--embedding-dim', type=int, default=128,
                       help='Embedding dimension')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=1024,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--implicit', action='store_true', default=True,
                       help='Use implicit feedback (binary labels)')
    
    # System arguments
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use for training')
    
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("NETFLIX-LEVEL RECOMMENDER TRAINING")
    logger.info("="*80)
    logger.info(f"Configuration:")
    logger.info(f"  Model type: {args.model_type}")
    logger.info(f"  Epochs: {args.epochs}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Learning rate: {args.lr}")
    logger.info(f"  Sample mode: {args.sample}")
    logger.info("="*80)
    
    # Step 1: Preprocess data
    logger.info("\n>>> STEP 1: DATA PREPROCESSING")
    
    from ai.data_preprocessor import preprocess_pipeline
    
    cache_dir = Path('ai/cache')
    
    if not (cache_dir / 'ratings_processed.pkl').exists() or args.sample:
        logger.info("Running preprocessing pipeline...")
        
        sample_size = 100000 if args.sample else None
        preprocessor = preprocess_pipeline(sample_size=sample_size)
    else:
        logger.info("Using existing preprocessed data")
    
    if args.preprocess_only:
        logger.info("\n✓ Preprocessing complete (--preprocess-only flag set)")
        return
    
    # Step 2: Precompute content embeddings (for hybrid model)
    if args.model_type == 'hybrid':
        logger.info("\n>>> STEP 2: CONTENT EMBEDDING PRECOMPUTATION")
        
        from ai.hybrid_model import precompute_content_embeddings
        import pandas as pd
        
        embeddings_path = cache_dir / 'movie_content_embeddings.pkl'
        
        if not embeddings_path.exists():
            logger.info("Precomputing movie content embeddings...")
            logger.info("This will take ~5-10 minutes (one-time only)...")
            
            content_df = pd.read_pickle(cache_dir / 'movie_content.pkl')
            precompute_content_embeddings(content_df, cache_dir=str(cache_dir))
        else:
            logger.info("Using existing content embeddings")
    
    # Step 3: Train model
    logger.info("\n>>> STEP 3: MODEL TRAINING")
    
    from ai.training_pipeline import train_netflix_model
    
    config = {
        # Data
        'batch_size': args.batch_size,
        'use_implicit_feedback': args.implicit,
        
        # Model
        'model_type': args.model_type,
        'embedding_dim': args.embedding_dim,
        'hidden_dims': [256, 128, 64],
        
        # Training
        'num_epochs': args.epochs,
        'learning_rate': args.lr,
        'weight_decay': 1e-5,
        'scheduler': 'reduce_on_plateau',
        'early_stopping_patience': 5,
        
        # Paths
        'model_dir': 'ai/models'
    }
    
    pipeline, metrics = train_netflix_model(config)
    
    logger.info("\n" + "="*80)
    logger.info("✓ TRAINING COMPLETE!")
    logger.info("="*80)
    logger.info("\nFinal Test Metrics:")
    logger.info(f"  RMSE: {metrics.get('RMSE', 'N/A'):.4f}")
    logger.info(f"  Hit@10: {metrics.get('Hit@10', 'N/A'):.4f}")
    logger.info(f"  NDCG@10: {metrics.get('NDCG@10', 'N/A'):.4f}")
    logger.info("\nModel saved to: ai/models/best_model.pt")
    logger.info("\nYou can now use the model in your Flask app!")
    logger.info("="*80)


if __name__ == '__main__':
    main()
