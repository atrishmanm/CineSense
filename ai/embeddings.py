"""
AI Layer 2: Vector Embeddings
Content-based filtering using movie and user feature vectors
"""

import numpy as np
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from config import Config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MovieEmbedding:
    """
    Represents a movie as a feature vector
    Features: genres, directors, actors, metadata (rating, popularity, year)
    """
    
    def __init__(self):
        self.vector_dim = Config.TOTAL_VECTOR_DIM
        self.genre_dim = Config.GENRE_DIM
        self.director_dim = Config.DIRECTOR_DIM
        self.actor_dim = Config.ACTOR_DIM
        self.metadata_dim = Config.METADATA_DIM
    
    def create_embedding(self, movie_data, genre_encoder, director_encoder, actor_encoder):
        """
        Create feature vector for a movie
        
        Args:
            movie_data: Dictionary with movie information
            genre_encoder: Encoder for genres
            director_encoder: Encoder for directors
            actor_encoder: Encoder for actors
        
        Returns:
            numpy array of shape (vector_dim,)
        """
        embedding = np.zeros(self.vector_dim)
        
        # Part 1: Genre features (one-hot encoded)
        genres = movie_data.get('genres', [])
        genre_vector = genre_encoder.encode(genres)
        embedding[:self.genre_dim] = genre_vector[:self.genre_dim]
        
        # Part 2: Director features
        directors = movie_data.get('directors', [])
        director_vector = director_encoder.encode(directors)
        embedding[self.genre_dim:self.genre_dim + self.director_dim] = director_vector[:self.director_dim]
        
        # Part 3: Actor features (top cast)
        actors = movie_data.get('actors', [])
        actor_vector = actor_encoder.encode(actors)
        start_idx = self.genre_dim + self.director_dim
        embedding[start_idx:start_idx + self.actor_dim] = actor_vector[:self.actor_dim]
        
        # Part 4: Metadata features (normalized)
        metadata_start = self.genre_dim + self.director_dim + self.actor_dim
        
        # Rating (0-10 scale, normalized to 0-1)
        rating = float(movie_data.get('tmdb_rating', 5.0))
        embedding[metadata_start] = rating / 10.0
        
        # Popularity (log-normalized)
        popularity = float(movie_data.get('popularity', 1.0))
        embedding[metadata_start + 1] = np.log1p(popularity) / 10.0
        
        # Year (normalized: 1900-2030)
        year = movie_data.get('release_year', 2000)
        embedding[metadata_start + 2] = (year - 1900) / 130.0
        
        # Vote count (log-normalized)
        vote_count = movie_data.get('vote_count', 0)
        embedding[metadata_start + 3] = np.log1p(vote_count) / 15.0
        
        # Runtime (normalized: 0-300 minutes)
        runtime = movie_data.get('runtime', 90)
        if runtime:
            embedding[metadata_start + 4] = min(runtime, 300) / 300.0
        
        return embedding
    
    def similarity(self, embedding1, embedding2):
        """
        Calculate cosine similarity between two embeddings
        
        Args:
            embedding1: First movie embedding
            embedding2: Second movie embedding
        
        Returns:
            Similarity score (0 to 1)
        """
        # Reshape for sklearn
        e1 = np.array(embedding1).reshape(1, -1)
        e2 = np.array(embedding2).reshape(1, -1)
        
        return cosine_similarity(e1, e2)[0][0]


class FeatureEncoder:
    """
    Encodes categorical features (genres, directors, actors) into vectors
    """
    
    def __init__(self, max_features=20):
        """
        Initialize encoder
        
        Args:
            max_features: Maximum number of features to encode
        """
        self.max_features = max_features
        self.feature_to_idx = {}
        self.idx_to_feature = {}
        self.feature_count = {}
        self.next_idx = 0
    
    def fit(self, feature_lists):
        """
        Learn feature vocabulary from data
        
        Args:
            feature_lists: List of feature lists (e.g., list of genre lists)
        """
        # Count feature occurrences
        for features in feature_lists:
            for feature in features:
                self.feature_count[feature] = self.feature_count.get(feature, 0) + 1
        
        # Select top features by frequency
        top_features = sorted(
            self.feature_count.items(),
            key=lambda x: x[1],
            reverse=True
        )[:self.max_features]
        
        # Create mappings
        for idx, (feature, count) in enumerate(top_features):
            self.feature_to_idx[feature] = idx
            self.idx_to_feature[idx] = feature
        
        self.next_idx = len(self.feature_to_idx)
        
        logger.info(f"Encoder fitted with {self.next_idx} features")
    
    def encode(self, features):
        """
        Encode a list of features into a vector
        
        Args:
            features: List of feature names
        
        Returns:
            numpy array with multi-hot encoding
        """
        vector = np.zeros(self.max_features)
        
        for feature in features:
            if feature in self.feature_to_idx:
                idx = self.feature_to_idx[feature]
                vector[idx] = 1.0
        
        # Normalize if any features present
        if np.sum(vector) > 0:
            vector = vector / np.sum(vector)
        
        return vector
    
    def decode(self, vector, threshold=0.1):
        """
        Decode a vector back to feature names
        
        Args:
            vector: Encoded vector
            threshold: Minimum value to include feature
        
        Returns:
            List of feature names
        """
        features = []
        for idx, value in enumerate(vector):
            if value >= threshold and idx in self.idx_to_feature:
                features.append(self.idx_to_feature[idx])
        return features


class UserEmbedding:
    """
    Represents a user's preferences as a vector
    Learned from their movie choices
    """
    
    def __init__(self, vector_dim):
        """
        Initialize user embedding
        
        Args:
            vector_dim: Dimension of feature space
        """
        self.vector_dim = vector_dim
        self.embedding = np.zeros(vector_dim)
        self.update_count = 0
    
    def update_from_choice(self, chosen_movie_embedding, rejected_movie_embedding, learning_rate=0.1):
        """
        Update user embedding based on a pairwise choice
        
        Args:
            chosen_movie_embedding: Embedding of chosen movie
            rejected_movie_embedding: Embedding of rejected movie
            learning_rate: Step size for update
        """
        chosen = np.array(chosen_movie_embedding)
        rejected = np.array(rejected_movie_embedding)
        
        # Move towards chosen, away from rejected
        self.embedding += learning_rate * chosen
        self.embedding -= learning_rate * rejected
        
        self.update_count += 1
        
        # Normalize periodically to prevent unbounded growth
        if self.update_count % 10 == 0:
            norm = np.linalg.norm(self.embedding)
            if norm > 0:
                self.embedding = self.embedding / norm
    
    def predict_preference(self, movie_embedding):
        """
        Predict user's preference for a movie
        
        Args:
            movie_embedding: Movie feature vector
        
        Returns:
            Preference score (higher = more preferred)
        """
        movie = np.array(movie_embedding)
        
        # Dot product (inner product similarity)
        score = np.dot(self.embedding, movie)
        
        return score
    
    def get_embedding(self):
        """Get current user embedding vector"""
        return self.embedding.copy()
    
    def set_embedding(self, embedding):
        """Set user embedding vector"""
        self.embedding = np.array(embedding)


class ContentBasedRecommender:
    """
    Content-based recommendation using embeddings
    WITH LAZY LOADING SUPPORT
    """
    
    def __init__(self):
        self.movie_embedder = MovieEmbedding()
        self.movie_embeddings = {}  # movie_id -> embedding
        
        # Lazy loading components
        from ai.cache_manager import cache_manager
        self.cache = cache_manager
        
        # Encoders (shared across all movies)
        self.genre_encoder = None
        self.director_encoder = None
        self.actor_encoder = None
    
    def initialize_encoders(self, genre_encoder, director_encoder, actor_encoder):
        """
        Initialize feature encoders
        
        These are shared across all movies for consistent encoding
        """
        self.genre_encoder = genre_encoder
        self.director_encoder = director_encoder
        self.actor_encoder = actor_encoder
    
    def get_or_create_embedding(self, movie_id, movie_data=None):
        """
        LAZY EMBEDDING: Get embedding from cache or compute on-demand
        
        Args:
            movie_id: Movie ID
            movie_data: Movie data dict (optional, will fetch if not provided)
        
        Returns:
            numpy array: Movie embedding
        """
        # Check vector cache first (FAST)
        cached_vector = self.cache.get_vector(movie_id)
        if cached_vector is not None:
            return cached_vector
        
        # Check in-memory embeddings
        if movie_id in self.movie_embeddings:
            return self.movie_embeddings[movie_id]
        
        # Need to compute - ensure we have movie data
        if movie_data is None:
            # Try cache
            movie_data = self.cache.get_movie(movie_id)
            
            if movie_data is None:
                logger.warning(f"Cannot create embedding for movie {movie_id} - no data available")
                return None
        
        # Compute embedding ON-DEMAND
        if not all([self.genre_encoder, self.director_encoder, self.actor_encoder]):
            logger.error("Encoders not initialized - call initialize_encoders() first")
            return None
        
        embedding = self.movie_embedder.create_embedding(
            movie_data,
            self.genre_encoder,
            self.director_encoder,
            self.actor_encoder
        )
        
        # Cache for future use
        self.cache.put_vector(movie_id, embedding)
        self.movie_embeddings[movie_id] = embedding
        
        logger.debug(f"Created embedding for movie {movie_id} on-demand")
        
        return embedding
    
    def add_movie(self, movie_id, embedding):
        """
        Add movie embedding to index
        
        DEPRECATED: Use get_or_create_embedding() instead for lazy loading
        """
        self.movie_embeddings[movie_id] = np.array(embedding)
        self.cache.put_vector(movie_id, embedding)
    
    def find_similar_movies(self, movie_id, n=10, candidate_ids=None):
        """
        Find movies similar to given movie
        WITH CANDIDATE FILTERING (don't search all movies!)
        
        Args:
            movie_id: Reference movie ID
            n: Number of similar movies to return
            candidate_ids: List of candidate movie IDs to search (IMPORTANT!)
        
        Returns:
            List of (movie_id, similarity_score) tuples
        """
        # Get target embedding (lazy)
        target_embedding = self.get_or_create_embedding(movie_id)
        if target_embedding is None:
            return []
        
        similarities = []
        
        # Search only candidates (not all movies!)
        if candidate_ids:
            search_ids = candidate_ids
        else:
            # Fallback to cached movies only
            search_ids = list(self.movie_embeddings.keys())
        
        for mid in search_ids:
            if mid != movie_id:
                # Get embedding (lazy)
                embedding = self.get_or_create_embedding(mid)
                if embedding is not None:
                    sim = self.movie_embedder.similarity(target_embedding, embedding)
                    similarities.append((mid, sim))
        
        # Sort by similarity descending
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:n]
    
    def recommend_for_user(self, user_embedding, n=10, exclude_ids=None, candidate_ids=None):
        """
        Recommend movies for a user based on their embedding
        WITH CANDIDATE FILTERING (production-ready!)
        
        Args:
            user_embedding: User's preference vector
            n: Number of recommendations
            exclude_ids: Set of movie IDs to exclude
            candidate_ids: List of candidate IDs to rank (IMPORTANT!)
        
        Returns:
            List of (movie_id, score) tuples
        """
        if exclude_ids is None:
            exclude_ids = set()
        
        user_emb = np.array(user_embedding)
        scores = []
        
        # Rank only candidates (not all movies!)
        if candidate_ids:
            search_ids = candidate_ids
        else:
            # Fallback to cached movies only
            search_ids = list(self.movie_embeddings.keys())
        
        for movie_id in search_ids:
            if movie_id not in exclude_ids:
                # Get embedding (lazy)
                movie_emb = self.get_or_create_embedding(movie_id)
                if movie_emb is not None:
                    # Cosine similarity
                    score = self.movie_embedder.similarity(user_emb, movie_emb)
                    scores.append((movie_id, score))
        
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return scores[:n]


if __name__ == "__main__":
    # Test embedding module
    print("Testing Vector Embeddings Module")
    print("=" * 50)
    
    # Test Feature Encoder
    print("\n1. Feature Encoder:")
    encoder = FeatureEncoder(max_features=5)
    
    # Sample data
    feature_lists = [
        ['Action', 'Sci-Fi'],
        ['Action', 'Adventure'],
        ['Drama', 'Romance'],
        ['Action', 'Thriller'],
        ['Comedy']
    ]
    
    encoder.fit(feature_lists)
    print(f"Learned features: {list(encoder.feature_to_idx.keys())}")
    
    # Encode a feature list
    encoded = encoder.encode(['Action', 'Sci-Fi'])
    print(f"Encoded ['Action', 'Sci-Fi']: {encoded}")
    
    # Test Movie Embedding
    print("\n2. Movie Embedding:")
    movie_embedder = MovieEmbedding()
    
    # Create sample encoders
    genre_enc = FeatureEncoder(max_features=Config.GENRE_DIM)
    director_enc = FeatureEncoder(max_features=Config.DIRECTOR_DIM)
    actor_enc = FeatureEncoder(max_features=Config.ACTOR_DIM)
    
    genre_enc.fit([['Action', 'Sci-Fi'], ['Drama']])
    director_enc.fit([['Nolan'], ['Spielberg']])
    actor_enc.fit([['DiCaprio'], ['Hanks']])
    
    # Sample movie
    movie1 = {
        'genres': ['Action', 'Sci-Fi'],
        'directors': ['Nolan'],
        'actors': ['DiCaprio'],
        'tmdb_rating': 8.5,
        'popularity': 150,
        'release_year': 2010,
        'vote_count': 20000,
        'runtime': 148
    }
    
    embedding1 = movie_embedder.create_embedding(movie1, genre_enc, director_enc, actor_enc)
    print(f"Movie embedding shape: {embedding1.shape}")
    print(f"First 10 values: {embedding1[:10]}")
    
    # Test similarity
    movie2 = {
        'genres': ['Action', 'Thriller'],
        'directors': ['Nolan'],
        'actors': ['Bale'],
        'tmdb_rating': 8.0,
        'popularity': 140,
        'release_year': 2008,
        'vote_count': 18000,
        'runtime': 152
    }
    
    embedding2 = movie_embedder.create_embedding(movie2, genre_enc, director_enc, actor_enc)
    similarity = movie_embedder.similarity(embedding1, embedding2)
    print(f"\nSimilarity between movies: {similarity:.3f}")
    
    # Test User Embedding
    print("\n3. User Embedding:")
    user_emb = UserEmbedding(vector_dim=Config.TOTAL_VECTOR_DIM)
    
    print(f"Initial embedding norm: {np.linalg.norm(user_emb.embedding):.3f}")
    
    # User chooses movie1 over movie2
    user_emb.update_from_choice(embedding1, embedding2)
    print(f"After 1 choice, embedding norm: {np.linalg.norm(user_emb.embedding):.3f}")
    
    # Predict preference
    pref1 = user_emb.predict_preference(embedding1)
    pref2 = user_emb.predict_preference(embedding2)
    print(f"Preference for movie1: {pref1:.3f}")
    print(f"Preference for movie2: {pref2:.3f}")
    
    print("\n✓ Vector embeddings module working correctly!")
