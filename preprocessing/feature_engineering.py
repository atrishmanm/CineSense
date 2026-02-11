"""
STEP 3: Feature Engineering
- Encode users and movies (MANDATORY for neural nets)
- Generate plot embeddings using transformers
"""

import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer
import pickle
from pathlib import Path
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def encode_users_and_movies():
    """
    STEP 3.1: Encode Users & Movies
    Neural nets need indices, not raw IDs
    """
    logger.info("="*80)
    logger.info("ENCODING USERS & MOVIES")
    logger.info("="*80)
    
    # Load merged data
    df = pd.read_csv("data/merged.csv")
    logger.info(f"Loaded {len(df):,} ratings")
    
    # Create encoders
    user_enc = LabelEncoder()
    movie_enc = LabelEncoder()
    
    # Fit and transform
    logger.info("\nEncoding user IDs...")
    df["user_idx"] = user_enc.fit_transform(df["userId"])
    
    logger.info("Encoding movie IDs...")
    df["movie_idx"] = movie_enc.fit_transform(df["movieId"])
    
    num_users = len(user_enc.classes_)
    num_movies = len(movie_enc.classes_)
    
    logger.info(f"\n✓ Encoded {num_users:,} users")
    logger.info(f"✓ Encoded {num_movies:,} movies")
    
    # Save encoders
    Path("model").mkdir(exist_ok=True)
    with open("model/mappings.pkl", "wb") as f:
        pickle.dump({
            "user_encoder": user_enc,
            "movie_encoder": movie_enc,
            "num_users": num_users,
            "num_movies": num_movies
        }, f)
    
    logger.info("\n✓ Saved encoders to model/mappings.pkl")
    
    # Save encoded data
    df.to_csv("data/encoded.csv", index=False)
    logger.info("✓ Saved encoded data to data/encoded.csv")
    
    return df, user_enc, movie_enc


def generate_plot_embeddings():
    """
    STEP 3.2: Movie Plot Embeddings (DEEP NLP)
    This is where your system becomes content-aware
    Uses sentence transformers for semantic understanding
    """
    logger.info("\n" + "="*80)
    logger.info("GENERATING PLOT EMBEDDINGS")
    logger.info("="*80)
    
    # Load data
    df = pd.read_csv("data/encoded.csv")
    
    # Get unique movies with their overviews
    logger.info("\nPreparing movie metadata...")
    movies_df = df[["movie_idx", "movieId", "title", "overview", "keywords"]].drop_duplicates("movie_idx")
    movies_df = movies_df.sort_values("movie_idx").reset_index(drop=True)
    
    logger.info(f"Processing {len(movies_df):,} unique movies")
    
    # Combine overview + keywords for richer embeddings
    logger.info("\nCombining overview and keywords...")
    movies_df["content"] = movies_df.apply(
        lambda row: f"{row['overview']} {row['keywords']}" if pd.notna(row['keywords']) else row['overview'],
        axis=1
    )
    
    # Fill empty content
    movies_df["content"] = movies_df["content"].fillna("")
    
    # Load sentence transformer model
    logger.info("\nLoading sentence transformer (all-mpnet-base-v2)...")
    logger.info("This is a 768-dimensional model trained on 1B+ sentence pairs")
    
    model = SentenceTransformer("all-mpnet-base-v2")
    
    # Generate embeddings
    logger.info("\nGenerating embeddings...")
    logger.info("This will take 5-10 minutes depending on your hardware...")
    
    plot_embeddings = model.encode(
        movies_df["content"].tolist(),
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    logger.info(f"\n✓ Generated embeddings shape: {plot_embeddings.shape}")
    logger.info(f"  - Number of movies: {plot_embeddings.shape[0]:,}")
    logger.info(f"  - Embedding dimension: {plot_embeddings.shape[1]}")
    
    # Save embeddings
    Path("model").mkdir(exist_ok=True)
    np.save("model/plot_embeddings.npy", plot_embeddings)
    logger.info("\n✓ Saved embeddings to model/plot_embeddings.npy")
    
    # Save movie metadata
    movies_df.to_csv("model/movie_metadata.csv", index=False)
    logger.info("✓ Saved movie metadata to model/movie_metadata.csv")
    
    logger.info("\n" + "="*80)
    logger.info("✓ FEATURE ENGINEERING COMPLETE!")
    logger.info("="*80)
    
    return plot_embeddings


def main():
    """Run full feature engineering pipeline"""
    logger.info("\n" + "="*80)
    logger.info("FEATURE ENGINEERING PIPELINE")
    logger.info("="*80)
    
    # Step 1: Encode users and movies
    df, user_enc, movie_enc = encode_users_and_movies()
    
    # Step 2: Generate plot embeddings
    plot_embeddings = generate_plot_embeddings()
    
    logger.info("\n" + "="*80)
    logger.info("✓ ALL FEATURE ENGINEERING COMPLETE!")
    logger.info("="*80)
    logger.info("\nGenerated files:")
    logger.info("  - data/encoded.csv")
    logger.info("  - model/mappings.pkl")
    logger.info("  - model/plot_embeddings.npy")
    logger.info("  - model/movie_metadata.csv")
    logger.info("\nReady for model training!")
    logger.info("="*80)


if __name__ == "__main__":
    main()
