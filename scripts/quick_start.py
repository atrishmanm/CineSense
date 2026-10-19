"""
Quick Start Script for Netflix-Level Recommender

This script helps you get started with training the model.
"""

import sys
import os
import logging
from pathlib import Path

# Ensure working directory is project root
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
os.chdir(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

TRAIN_SCRIPT = os.path.join('scripts', 'train_model.py')

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def check_data():
    """Check if data files exist"""
    logger.info("Checking data files...")
    
    data_dir = Path('data')
    
    # Check MovieLens data
    ml_dir = data_dir / 'movie-lens_ml-32m'
    ml_files = ['ratings.csv', 'movies.csv', 'links.csv']
    
    for filename in ml_files:
        filepath = ml_dir / filename
        if not filepath.exists():
            logger.error(f"❌ Missing: {filepath}")
            logger.error("Please ensure MovieLens-32M data is in data/movie-lens_ml-32m/")
            return False
        else:
            size_mb = filepath.stat().st_size / (1024 * 1024)
            logger.info(f"✓ Found: {filename} ({size_mb:.1f} MB)")
    
    # Check TMDB data
    tmdb_file = data_dir / 'tmdb' / 'TMDB_movie_dataset_v11.csv'
    if not tmdb_file.exists():
        logger.error(f"❌ Missing: {tmdb_file}")
        logger.error("Please ensure TMDB data is in data/tmdb/")
        return False
    else:
        size_mb = tmdb_file.stat().st_size / (1024 * 1024)
        logger.info(f"✓ Found: TMDB_movie_dataset_v11.csv ({size_mb:.1f} MB)")
    
    logger.info("✓ All data files found!\n")
    return True


def check_dependencies():
    """Check if required packages are installed"""
    logger.info("Checking dependencies...")
    
    required = {
        'torch': 'PyTorch',
        'transformers': 'Transformers',
        'sentence_transformers': 'Sentence Transformers',
        'pandas': 'Pandas',
        'numpy': 'NumPy',
        'sklearn': 'Scikit-learn',
        'tqdm': 'tqdm'
    }
    
    missing = []
    
    for package, name in required.items():
        try:
            __import__(package)
            logger.info(f"✓ {name} installed")
        except ImportError:
            logger.error(f"❌ {name} NOT installed")
            missing.append(package)
    
    if missing:
        logger.error(f"\nMissing packages: {', '.join(missing)}")
        logger.error("Install with: pip install -r requirements.txt")
        return False
    
    logger.info("✓ All dependencies installed!\n")
    return True


def print_menu():
    """Print menu options"""
    print("\n" + "="*80)
    print("NETFLIX-LEVEL RECOMMENDER - QUICK START")
    print("="*80)
    print("\nWhat would you like to do?\n")
    print("1. Quick Test (100K samples, ~10 minutes)")
    print("2. Full Training (33M samples, ~4-6 hours CPU / ~1 hour GPU)")
    print("3. Preprocess Data Only")
    print("4. Check System Status")
    print("5. Exit")
    print("\n" + "="*80)


def quick_test():
    """Run quick test with sample data"""
    logger.info("\n>>> Starting Quick Test Mode")
    logger.info("This will train on 100K ratings (takes ~10 minutes)\n")
    
    import subprocess
    
    cmd = [
        sys.executable, TRAIN_SCRIPT,
        '--sample',
        '--epochs', '5',
        '--batch-size', '512'
    ]
    
    subprocess.run(cmd)


def full_training():
    """Run full training"""
    logger.info("\n>>> Starting Full Training")
    logger.info("This will train on 33M ratings (takes ~4-6 hours)\n")
    
    response = input("Continue? (y/n): ").strip().lower()
    
    if response != 'y':
        logger.info("Training cancelled")
        return
    
    import subprocess
    
    cmd = [
        sys.executable, TRAIN_SCRIPT,
        '--epochs', '20',
        '--batch-size', '1024'
    ]
    
    subprocess.run(cmd)


def preprocess_only():
    """Run preprocessing only"""
    logger.info("\n>>> Running Data Preprocessing Only\n")
    
    import subprocess
    
    cmd = [
        sys.executable, TRAIN_SCRIPT,
        '--preprocess-only'
    ]
    
    subprocess.run(cmd)


def check_status():
    """Check system status"""
    logger.info("\n>>> System Status\n")
    
    # Training cache produced by ai/data_preprocessor.py (cache_dir='ai/cache')
    cache_dir = Path('ai/cache')
    
    if cache_dir.exists():
        cache_files = list(cache_dir.glob('*.pkl'))
        if cache_files:
            logger.info(f"✓ Preprocessed data found: {len(cache_files)} files")
        else:
            logger.info("⚠ No preprocessed data found")
    else:
        logger.info("⚠ Cache directory doesn't exist")
    
    # Training checkpoints produced by ai/training_pipeline.py (model_dir='ai/models'),
    # consumed by ai/netflix_recommender.py + ai/inference_pipeline.py.
    train_model_file = Path('ai/models') / 'best_model.pt'
    if train_model_file.exists():
        size_mb = train_model_file.stat().st_size / (1024 * 1024)
        logger.info(f"✓ Trained model found: ai/models/best_model.pt ({size_mb:.1f} MB)")
    else:
        logger.info("⚠ No trained model found in ai/models (train to create best_model.pt)")
    
    # Serving artifacts actually loaded by the app: ai/recommender.py reads
    # Config.MODEL_DIR (default './model').
    try:
        from config import Config
        serve_dir = Path(Config.MODEL_DIR)
    except Exception:
        serve_dir = Path('model')
    serve_artifacts = ['hybrid_recommender.pt', 'ncf_recommender.pt', 'plot_embeddings.npy']
    present = [name for name in serve_artifacts if (serve_dir / name).exists()]
    if present:
        logger.info(f"✓ Serving artifacts found in {serve_dir}: {', '.join(present)}")
    else:
        logger.info(f"⚠ No serving artifacts found in {serve_dir} (app will use fallbacks)")
    
    # Check if model loads
    try:
        from ai.netflix_recommender import netflix_recommender
        
        if netflix_recommender.is_ready():
            logger.info("✓ Model loaded and ready to use!")
        else:
            logger.info("⚠ Model not loaded (needs training)")
    except Exception as e:
        logger.error(f"❌ Error loading model: {e}")
    
    print()


def main():
    """Main function"""
    # Initial checks
    if not check_data():
        logger.error("\n⚠ Data files not found. Please add data to data/ folder.")
        return
    
    if not check_dependencies():
        logger.error("\n⚠ Dependencies not installed. Run: pip install -r requirements.txt")
        return
    
    # Main loop
    while True:
        print_menu()
        
        choice = input("Enter choice (1-5): ").strip()
        
        if choice == '1':
            quick_test()
        elif choice == '2':
            full_training()
        elif choice == '3':
            preprocess_only()
        elif choice == '4':
            check_status()
        elif choice == '5':
            logger.info("Goodbye!")
            break
        else:
            logger.warning("Invalid choice. Please enter 1-5.")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\nInterrupted by user. Goodbye!")
        sys.exit(0)
