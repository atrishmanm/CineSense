"""
STEP 1-2: Data Merging - MovieLens + TMDB
Critical step: Map MovieLens movieId → TMDB metadata
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def merge_movielens_tmdb():
    """
    Merge MovieLens behavior data with TMDB content data
    This is the foundation for hybrid recommendation
    """
    logger.info("="*80)
    logger.info("DATA MERGING: MovieLens + TMDB")
    logger.info("="*80)
    
    # Load MovieLens data
    logger.info("\nLoading MovieLens data...")
    ratings = pd.read_csv("data/movie-lens_ml-32m/ratings.csv")
    movies = pd.read_csv("data/movie-lens_ml-32m/movies.csv")
    links = pd.read_csv("data/movie-lens_ml-32m/links.csv")
    
    logger.info(f"✓ Ratings: {len(ratings):,} rows")
    logger.info(f"✓ Movies: {len(movies):,} rows")
    logger.info(f"✓ Links: {len(links):,} rows")
    
    # Load TMDB data
    logger.info("\nLoading TMDB data...")
    tmdb = pd.read_csv("data/tmdb/TMDB_movie_dataset_v11.csv", low_memory=False)
    logger.info(f"✓ TMDB: {len(tmdb):,} rows")
    
    # Merge step 1: ratings + movies
    logger.info("\nMerging ratings + movies...")
    merged = ratings.merge(movies, on="movieId", how="left")
    logger.info(f"✓ After ratings+movies: {len(merged):,} rows")
    
    # Merge step 2: + links (to get tmdbId)
    logger.info("Merging with links...")
    merged = merged.merge(links[["movieId", "tmdbId"]], on="movieId", how="left")
    logger.info(f"✓ After adding links: {len(merged):,} rows")
    
    # Merge step 3: + TMDB metadata (the critical step)
    logger.info("Merging with TMDB metadata...")
    # Convert IDs to string for matching
    merged["tmdbId"] = merged["tmdbId"].astype(str)
    tmdb["id"] = tmdb["id"].astype(str)
    
    # Select TMDB columns we need
    tmdb_cols = ["id", "overview", "genres", "keywords", "vote_average", "release_date"]
    tmdb_subset = tmdb[tmdb_cols].copy()
    
    # Merge
    merged = merged.merge(
        tmdb_subset,
        left_on="tmdbId",
        right_on="id",
        how="left",
        suffixes=("_ml", "_tmdb")
    )
    
    logger.info(f"✓ Final merged data: {len(merged):,} rows")
    
    # Select and rename columns for clarity
    logger.info("\nCleaning and selecting columns...")
    final_columns = {
        "userId": "userId",
        "movieId": "movieId", 
        "rating": "rating",
        "timestamp": "timestamp",
        "title": "title",
        "genres_ml": "genres",  # MovieLens genres
        "overview": "overview",  # TMDB overview
        "keywords": "keywords",  # TMDB keywords
        "vote_average": "vote_average"
    }
    
    merged_clean = merged[list(final_columns.keys())].copy()
    merged_clean.columns = list(final_columns.values())
    
    # Fill missing overviews
    merged_clean["overview"] = merged_clean["overview"].fillna("")
    merged_clean["keywords"] = merged_clean["keywords"].fillna("")
    
    # Statistics
    logger.info("\n" + "="*80)
    logger.info("MERGE STATISTICS")
    logger.info("="*80)
    logger.info(f"Total ratings: {len(merged_clean):,}")
    logger.info(f"Unique users: {merged_clean['userId'].nunique():,}")
    logger.info(f"Unique movies: {merged_clean['movieId'].nunique():,}")
    logger.info(f"Movies with TMDB overview: {merged_clean['overview'].ne('').sum():,}")
    logger.info(f"Coverage: {merged_clean['overview'].ne('').sum() / len(merged_clean) * 100:.1f}%")
    
    # Save merged data
    output_path = Path("data/merged.csv")
    logger.info(f"\nSaving to {output_path}...")
    merged_clean.to_csv(output_path, index=False)
    
    logger.info("="*80)
    logger.info("✓ DATA MERGING COMPLETE!")
    logger.info("="*80)
    
    return merged_clean


if __name__ == "__main__":
    merged_df = merge_movielens_tmdb()
    print("\nSample of merged data:")
    print(merged_df.head())
