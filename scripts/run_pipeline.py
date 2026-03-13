"""
COMPLETE PIPELINE EXECUTION
Runs all steps: Data Merging → Feature Engineering → Training

This is the master script that executes everything
"""

import sys
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_data_exists():
    """Check if required data files exist"""
    logger.info("Checking data files...")
    
    required_files = [
        "data/movie-lens_ml-32m/ratings.csv",
        "data/movie-lens_ml-32m/movies.csv",
        "data/movie-lens_ml-32m/links.csv",
        "data/tmdb/TMDB_movie_dataset_v11.csv"
    ]
    
    missing = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing.append(file_path)
            logger.error(f"❌ Missing: {file_path}")
        else:
            logger.info(f"✓ Found: {file_path}")
    
    if missing:
        logger.error("\n❌ Missing required data files!")
        logger.error("Please ensure all MovieLens and TMDB data is in the data/ folder")
        return False
    
    logger.info("✓ All data files found!\n")
    return True


def run_step_1_merge():
    """STEP 1-2: Data Merging"""
    logger.info("\n" + "="*100)
    logger.info("STEP 1-2: DATA MERGING (MovieLens + TMDB)")
    logger.info("="*100 + "\n")
    
    from preprocessing.merge_datasets import merge_movielens_tmdb
    
    merged_df = merge_movielens_tmdb()
    
    logger.info("\n✓ Step 1-2 Complete: Data merged successfully")
    logger.info(f"  Output: data/merged.csv ({len(merged_df):,} ratings)")
    
    return merged_df


def run_step_3_features():
    """STEP 3: Feature Engineering"""
    logger.info("\n" + "="*100)
    logger.info("STEP 3: FEATURE ENGINEERING (Encoding + Plot Embeddings)")
    logger.info("="*100 + "\n")
    
    from preprocessing.feature_engineering import main as feature_main
    
    feature_main()
    
    logger.info("\n✓ Step 3 Complete: Features engineered successfully")
    logger.info("  Outputs:")
    logger.info("    - data/encoded.csv")
    logger.info("    - model/mappings.pkl")
    logger.info("    - model/plot_embeddings.npy")
    
    return True


def run_step_4_5_6_train():
    """STEP 4-6: Model Training"""
    logger.info("\n" + "="*100)
    logger.info("STEP 4-6: MODEL TRAINING (NCF + Hybrid)")
    logger.info("="*100 + "\n")
    
    from training.train import train_model
    
    # Train NCF model
    logger.info("\n>>> Training Model 1: NCF (Pure Collaborative Filtering)")
    ncf_model, ncf_history = train_model(
        model_type='ncf',
        epochs=10,
        batch_size=2048,
        learning_rate=0.001
    )
    
    # Train Hybrid model
    logger.info("\n\n>>> Training Model 2: Hybrid (CF + Content Embeddings)")
    hybrid_model, hybrid_history = train_model(
        model_type='hybrid',
        epochs=10,
        batch_size=1024,
        learning_rate=0.001
    )
    
    logger.info("\n✓ Step 4-6 Complete: Models trained successfully")
    logger.info("  Outputs:")
    logger.info("    - model/ncf_recommender.pt")
    logger.info("    - model/hybrid_recommender.pt")
    
    return ncf_history, hybrid_history


def main():
    """Execute complete pipeline"""
    logger.info("\n" + "="*100)
    logger.info(" " * 30 + "NETFLIX-LEVEL RECOMMENDER TRAINING")
    logger.info(" " * 30 + "COMPLETE PIPELINE EXECUTION")
    logger.info("="*100)
    
    # Check prerequisites
    if not check_data_exists():
        logger.error("\n❌ Pipeline aborted: Missing data files")
        return False
    
    try:
        # Step 1-2: Merge data
        merged_df = run_step_1_merge()
        
        # Step 3: Feature engineering
        run_step_3_features()
        
        # Step 4-6: Train models
        ncf_history, hybrid_history = run_step_4_5_6_train()
        
        # Final summary
        logger.info("\n" + "="*100)
        logger.info(" " * 35 + "✓ PIPELINE COMPLETE!")
        logger.info("="*100)
        
        logger.info("\n📊 FINAL RESULTS:")
        logger.info(f"\nNCF Model (Pure CF):")
        logger.info(f"  Best Val RMSE: {min(ncf_history['val_rmse']):.4f}")
        logger.info(f"  Best Val MAE:  {min(ncf_history['val_mae']):.4f}")
        
        logger.info(f"\nHybrid Model (CF + Content):")
        logger.info(f"  Best Val RMSE: {min(hybrid_history['val_rmse']):.4f}")
        logger.info(f"  Best Val MAE:  {min(hybrid_history['val_mae']):.4f}")
        
        logger.info("\n📁 Generated Files:")
        logger.info("  Data:")
        logger.info("    - data/merged.csv")
        logger.info("    - data/encoded.csv")
        logger.info("  Models:")
        logger.info("    - model/ncf_recommender.pt")
        logger.info("    - model/hybrid_recommender.pt")
        logger.info("  Artifacts:")
        logger.info("    - model/mappings.pkl")
        logger.info("    - model/plot_embeddings.npy")
        logger.info("    - model/movie_metadata.csv")
        
        logger.info("\n" + "="*100)
        logger.info("🎉 YOU NOW HAVE A NETFLIX-LEVEL RECOMMENDATION SYSTEM!")
        logger.info("="*100)
        
        logger.info("\n💡 What you can say:")
        logger.info('  "We trained a hybrid deep learning recommender using MovieLens behavioral')
        logger.info('   data and TMDB content embeddings, combining neural collaborative filtering')
        logger.info('   with transformer-based plot understanding, similar to Netflix\'s production')
        logger.info('   architecture."')
        
        logger.info("\n🚀 Next steps:")
        logger.info("  1. Use models for inference (see inference/ folder)")
        logger.info("  2. Integrate with Flask app")
        logger.info("  3. Deploy to production")
        
        logger.info("\n" + "="*100)
        
        return True
    
    except Exception as e:
        logger.error(f"\n❌ Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
