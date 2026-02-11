"""
Data Preprocessing Pipeline for Netflix-Level Recommender
Merges MovieLens + TMDB and prepares data for deep learning
"""

import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pickle
import os
from pathlib import Path
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Industry-standard data preprocessing for recommendation systems
    Handles: MovieLens + TMDB merge, encoding, normalization, splitting
    """
    
    def __init__(self, data_dir='data', cache_dir='ai/cache'):
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Encoders
        self.user_encoder = LabelEncoder()
        self.movie_encoder = LabelEncoder()
        
        # Metadata
        self.num_users = 0
        self.num_movies = 0
        self.rating_mean = 0
        self.rating_std = 1
        
        # Data
        self.ratings_df = None
        self.movies_df = None
        self.tmdb_df = None
        self.merged_df = None
        
    def load_datasets(self):
        """Load MovieLens and TMDB datasets"""
        logger.info("Loading datasets...")
        
        # Load MovieLens ratings
        ratings_path = self.data_dir / 'movie-lens_ml-32m' / 'ratings.csv'
        logger.info(f"Loading ratings from {ratings_path}...")
        self.ratings_df = pd.read_csv(ratings_path)
        logger.info(f"Loaded {len(self.ratings_df):,} ratings")
        
        # Load MovieLens movies
        movies_path = self.data_dir / 'movie-lens_ml-32m' / 'movies.csv'
        logger.info(f"Loading movies from {movies_path}...")
        self.movies_df = pd.read_csv(movies_path)
        logger.info(f"Loaded {len(self.movies_df):,} movies")
        
        # Load TMDB metadata
        tmdb_path = self.data_dir / 'tmdb' / 'TMDB_movie_dataset_v11.csv'
        logger.info(f"Loading TMDB metadata from {tmdb_path}...")
        self.tmdb_df = pd.read_csv(tmdb_path, low_memory=False)
        logger.info(f"Loaded {len(self.tmdb_df):,} TMDB movies")
        
        # Load MovieLens links for joining
        links_path = self.data_dir / 'movie-lens_ml-32m' / 'links.csv'
        self.links_df = pd.read_csv(links_path)
        
        return self
    
    def merge_datasets(self):
        """
        Merge MovieLens + TMDB datasets
        This is exactly what IMDb-like systems do
        """
        logger.info("Merging MovieLens + TMDB datasets...")
        
        # Convert tmdb_id to match
        self.links_df['tmdbId'] = self.links_df['tmdbId'].astype(str)
        self.tmdb_df['id'] = self.tmdb_df['id'].astype(str)
        
        # Merge movies with links
        movies_with_tmdb = self.movies_df.merge(
            self.links_df[['movieId', 'tmdbId']], 
            on='movieId', 
            how='left'
        )
        
        # Merge with TMDB metadata
        self.merged_df = movies_with_tmdb.merge(
            self.tmdb_df,
            left_on='tmdbId',
            right_on='id',
            how='left',
            suffixes=('_ml', '_tmdb')
        )
        
        logger.info(f"Merged dataset size: {len(self.merged_df):,} movies")
        logger.info(f"Movies with TMDB data: {self.merged_df['overview'].notna().sum():,}")
        
        return self
    
    def encode_ids(self):
        """
        Encode userId and movieId to contiguous integers
        Required for embedding layers
        """
        logger.info("Encoding user and movie IDs...")
        
        # Fit encoders
        self.user_encoder.fit(self.ratings_df['userId'].unique())
        self.movie_encoder.fit(self.ratings_df['movieId'].unique())
        
        # Transform IDs
        self.ratings_df['user_idx'] = self.user_encoder.transform(self.ratings_df['userId'])
        self.ratings_df['movie_idx'] = self.movie_encoder.transform(self.ratings_df['movieId'])
        
        # Save metadata
        self.num_users = len(self.user_encoder.classes_)
        self.num_movies = len(self.movie_encoder.classes_)
        
        logger.info(f"Encoded {self.num_users:,} users and {self.num_movies:,} movies")
        
        return self
    
    def normalize_ratings(self):
        """Normalize ratings to 0-1 range for neural networks"""
        logger.info("Normalizing ratings...")
        
        # Store original stats for denormalization
        self.rating_mean = self.ratings_df['rating'].mean()
        self.rating_std = self.ratings_df['rating'].std()
        
        # Min-max normalization to [0, 1]
        min_rating = self.ratings_df['rating'].min()
        max_rating = self.ratings_df['rating'].max()
        self.ratings_df['rating_normalized'] = (
            (self.ratings_df['rating'] - min_rating) / (max_rating - min_rating)
        )
        
        logger.info(f"Rating stats: mean={self.rating_mean:.2f}, std={self.rating_std:.2f}")
        logger.info(f"Normalized to range [0, 1]")
        
        return self
    
    def create_time_based_split(self, train_ratio=0.8, val_ratio=0.1):
        """
        Create time-based train/val/test split
        This is production best practice - prevents data leakage
        """
        logger.info("Creating time-based train/validation/test split...")
        
        # Sort by timestamp
        self.ratings_df = self.ratings_df.sort_values('timestamp')
        
        # Calculate split indices
        n = len(self.ratings_df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))
        
        # Split
        train_df = self.ratings_df.iloc[:train_end]
        val_df = self.ratings_df.iloc[train_end:val_end]
        test_df = self.ratings_df.iloc[val_end:]
        
        logger.info(f"Train: {len(train_df):,} ({len(train_df)/n*100:.1f}%)")
        logger.info(f"Validation: {len(val_df):,} ({len(val_df)/n*100:.1f}%)")
        logger.info(f"Test: {len(test_df):,} ({len(test_df)/n*100:.1f}%)")
        
        return train_df, val_df, test_df
    
    def prepare_movie_content(self):
        """
        Prepare movie content features for Content-Aware model
        Combines: overview, genres, keywords (Netflix-style)
        """
        logger.info("Preparing movie content features...")
        
        content_features = []
        
        for _, row in tqdm(self.merged_df.iterrows(), total=len(self.merged_df), desc="Processing movies"):
            # Combine text features
            text_parts = []
            
            if pd.notna(row.get('overview')):
                text_parts.append(str(row['overview']))
            
            if pd.notna(row.get('genres')):
                text_parts.append(f"Genres: {row['genres']}")
            
            if pd.notna(row.get('keywords')):
                text_parts.append(f"Keywords: {row['keywords']}")
            
            if pd.notna(row.get('tagline')):
                text_parts.append(f"Tagline: {row['tagline']}")
            
            # Fallback to title if no content
            if not text_parts and pd.notna(row.get('title_ml')):
                text_parts.append(str(row['title_ml']))
            
            content = ' '.join(text_parts) if text_parts else ''
            content_features.append({
                'movieId': row['movieId'],
                'content_text': content,
                'title': row.get('title_ml', ''),
                'genres': row.get('genres', ''),
                'overview': row.get('overview', '')
            })
        
        content_df = pd.DataFrame(content_features)
        logger.info(f"Prepared content for {len(content_df):,} movies")
        
        return content_df
    
    def create_implicit_feedback(self, threshold=3.5):
        """
        Convert ratings to implicit feedback (binary)
        rating >= threshold -> 1 (positive), else -> 0 (negative)
        Used for Binary Cross Entropy loss
        """
        logger.info(f"Creating implicit feedback with threshold={threshold}...")
        
        self.ratings_df['implicit_label'] = (
            self.ratings_df['rating'] >= threshold
        ).astype(np.float32)
        
        positive_rate = self.ratings_df['implicit_label'].mean()
        logger.info(f"Positive rate: {positive_rate*100:.1f}%")
        
        return self
    
    def save_processed_data(self, output_dir='ai/cache'):
        """Save all preprocessed data"""
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        logger.info(f"Saving processed data to {output_path}...")
        
        # Save dataframes
        self.ratings_df.to_pickle(output_path / 'ratings_processed.pkl')
        self.merged_df.to_pickle(output_path / 'movies_merged.pkl')
        
        # Save encoders
        with open(output_path / 'user_encoder.pkl', 'wb') as f:
            pickle.dump(self.user_encoder, f)
        
        with open(output_path / 'movie_encoder.pkl', 'wb') as f:
            pickle.dump(self.movie_encoder, f)
        
        # Save metadata
        metadata = {
            'num_users': self.num_users,
            'num_movies': self.num_movies,
            'rating_mean': self.rating_mean,
            'rating_std': self.rating_std
        }
        
        with open(output_path / 'metadata.pkl', 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info("✓ All data saved successfully")
        
        return self
    
    def load_processed_data(self, input_dir='ai/cache'):
        """Load preprocessed data (for training/inference)"""
        input_path = Path(input_dir)
        
        logger.info(f"Loading processed data from {input_path}...")
        
        # Load dataframes
        self.ratings_df = pd.read_pickle(input_path / 'ratings_processed.pkl')
        self.merged_df = pd.read_pickle(input_path / 'movies_merged.pkl')
        
        # Load encoders
        with open(input_path / 'user_encoder.pkl', 'rb') as f:
            self.user_encoder = pickle.load(f)
        
        with open(input_path / 'movie_encoder.pkl', 'rb') as f:
            self.movie_encoder = pickle.load(f)
        
        # Load metadata
        with open(input_path / 'metadata.pkl', 'rb') as f:
            metadata = pickle.load(f)
            self.num_users = metadata['num_users']
            self.num_movies = metadata['num_movies']
            self.rating_mean = metadata['rating_mean']
            self.rating_std = metadata['rating_std']
        
        logger.info("✓ Data loaded successfully")
        
        return self


class RecommenderDataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for recommendation
    """
    
    def __init__(self, dataframe, use_implicit=False):
        self.user_ids = torch.tensor(dataframe['user_idx'].values, dtype=torch.long)
        self.movie_ids = torch.tensor(dataframe['movie_idx'].values, dtype=torch.long)
        
        if use_implicit:
            self.labels = torch.tensor(dataframe['implicit_label'].values, dtype=torch.float32)
        else:
            self.labels = torch.tensor(dataframe['rating_normalized'].values, dtype=torch.float32)
    
    def __len__(self):
        return len(self.user_ids)
    
    def __getitem__(self, idx):
        return {
            'user_id': self.user_ids[idx],
            'movie_id': self.movie_ids[idx],
            'label': self.labels[idx]
        }


def preprocess_pipeline(sample_size=None):
    """
    Full preprocessing pipeline
    Run this once to prepare data for training
    """
    logger.info("="*60)
    logger.info("NETFLIX-LEVEL DATA PREPROCESSING PIPELINE")
    logger.info("="*60)
    
    preprocessor = DataPreprocessor()
    
    # Load datasets
    preprocessor.load_datasets()
    
    # Sample data if requested (for faster testing)
    if sample_size:
        logger.info(f"Sampling {sample_size:,} ratings for testing...")
        preprocessor.ratings_df = preprocessor.ratings_df.sample(n=sample_size, random_state=42)
    
    # Merge MovieLens + TMDB
    preprocessor.merge_datasets()
    
    # Encode IDs
    preprocessor.encode_ids()
    
    # Normalize ratings
    preprocessor.normalize_ratings()
    
    # Create implicit feedback
    preprocessor.create_implicit_feedback(threshold=3.5)
    
    # Prepare movie content
    content_df = preprocessor.prepare_movie_content()
    content_df.to_pickle('ai/cache/movie_content.pkl')
    
    # Save everything
    preprocessor.save_processed_data()
    
    logger.info("="*60)
    logger.info("✓ PREPROCESSING COMPLETE!")
    logger.info("="*60)
    
    return preprocessor


if __name__ == '__main__':
    # Run full pipeline (use sample_size for testing)
    # preprocessor = preprocess_pipeline(sample_size=100000)  # 100K for testing
    preprocessor = preprocess_pipeline()  # Full dataset for production
