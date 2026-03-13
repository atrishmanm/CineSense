"""
FAISS Vector Store for fast similarity search
Replaces linear search with Approximate Nearest Neighbors (ANN)
"""

import faiss
import numpy as np
from typing import List, Dict, Tuple
import os
import pickle
import logging

logger = logging.getLogger(__name__)


class FAISSVectorStore:
    """
    Vector store using FAISS for efficient similarity search
    """
    
    def __init__(self, dimension: int = 768, index_type: str = 'flat'):
        """
        Initialize FAISS vector store
        
        Args:
            dimension: Dimensionality of embeddings
            index_type: Type of FAISS index ('flat', 'ivf', 'hnsw')
                - 'flat': Exact search (slower but accurate)
                - 'ivf': Inverted file index (faster, approximate)
                - 'hnsw': Hierarchical NSW (fast, good recall)
        """
        self.dimension = dimension
        self.index_type = index_type
        self.index = None
        self.movie_ids = []
        
        self._initialize_index()
    
    def _initialize_index(self):
        """Initialize FAISS index based on type"""
        if self.index_type == 'flat':
            # Exact search using inner product (for normalized vectors = cosine similarity)
            self.index = faiss.IndexFlatIP(self.dimension)
            logger.info(f"✓ Initialized FAISS Flat index (exact search, dim={self.dimension})")
            
        elif self.index_type == 'ivf':
            # Inverted file index for faster approximate search
            nlist = 100  # Number of Voronoi cells
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist)
            logger.info(f"✓ Initialized FAISS IVF index (approximate, nlist={nlist})")
            
        elif self.index_type == 'hnsw':
            # Hierarchical Navigable Small World - fast and accurate
            M = 32  # Number of connections per layer
            self.index = faiss.IndexHNSWFlat(self.dimension, M)
            logger.info(f"✓ Initialized FAISS HNSW index (M={M})")
            
        else:
            raise ValueError(f"Unknown index type: {self.index_type}")
    
    def add_embeddings(self, embeddings: np.ndarray, movie_ids: List[int]):
        """
        Add movie embeddings to the index
        
        Args:
            embeddings: Numpy array of shape (N, dimension)
            movie_ids: List of movie IDs corresponding to embeddings
        """
        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension {embeddings.shape[1]} != index dimension {self.dimension}")
        
        # Normalize vectors for cosine similarity
        faiss.normalize_L2(embeddings)
        
        # Train index if needed (for IVF)
        if self.index_type == 'ivf' and not self.index.is_trained:
            logger.info("Training IVF index...")
            self.index.train(embeddings)
            logger.info("✓ Index trained")
        
        # Add vectors to index
        self.index.add(embeddings)
        self.movie_ids.extend(movie_ids)
        
        logger.info(f"✓ Added {len(movie_ids)} embeddings to index (total: {len(self.movie_ids)})")
    
    def search(
        self, 
        query_embedding: np.ndarray, 
        k: int = 20,
        nprobe: int = 10
    ) -> Tuple[List[int], List[float]]:
        """
        Search for k nearest neighbors
        
        Args:
            query_embedding: Query vector of shape (dimension,) or (1, dimension)
            k: Number of nearest neighbors to return
            nprobe: Number of cells to visit (for IVF index)
            
        Returns:
            Tuple of (movie_ids, distances)
        """
        if self.index is None or self.index.ntotal == 0:
            logger.warning("Index is empty, returning no results")
            return [], []
        
        # Ensure query is 2D
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        # Normalize query
        faiss.normalize_L2(query_embedding)
        
        # Set nprobe for IVF index
        if self.index_type == 'ivf':
            self.index.nprobe = nprobe
        
        # Search
        k = min(k, len(self.movie_ids))
        distances, indices = self.index.search(query_embedding, k)
        
        # Convert indices to movie IDs
        movie_ids = [self.movie_ids[i] for i in indices[0] if i >= 0]
        scores = distances[0].tolist()
        
        return movie_ids, scores
    
    def batch_search(
        self,
        query_embeddings: np.ndarray,
        k: int = 20
    ) -> Tuple[List[List[int]], List[List[float]]]:
        """
        Batch search for multiple queries
        
        Args:
            query_embeddings: Array of shape (N, dimension)
            k: Number of results per query
            
        Returns:
            Lists of movie IDs and scores for each query
        """
        # Normalize queries
        faiss.normalize_L2(query_embeddings)
        
        # Search
        k = min(k, len(self.movie_ids))
        distances, indices = self.index.search(query_embeddings, k)
        
        # Convert to lists
        all_movie_ids = []
        all_scores = []
        
        for i in range(len(query_embeddings)):
            movie_ids = [self.movie_ids[idx] for idx in indices[i] if idx >= 0]
            scores = distances[i].tolist()
            all_movie_ids.append(movie_ids)
            all_scores.append(scores)
        
        return all_movie_ids, all_scores
    
    def save(self, filepath: str):
        """
        Save index to disk
        
        Args:
            filepath: Path to save the index
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save FAISS index
        faiss.write_index(self.index, f"{filepath}.faiss")
        
        # Save metadata
        metadata = {
            'movie_ids': self.movie_ids,
            'dimension': self.dimension,
            'index_type': self.index_type
        }
        with open(f"{filepath}.pkl", 'wb') as f:
            pickle.dump(metadata, f)
        
        logger.info(f"✓ Index saved to {filepath}")
    
    def load(self, filepath: str):
        """
        Load index from disk
        
        Args:
            filepath: Path to load the index from
        """
        # Load FAISS index
        self.index = faiss.read_index(f"{filepath}.faiss")
        
        # Load metadata
        with open(f"{filepath}.pkl", 'rb') as f:
            metadata = pickle.load(f)
        
        self.movie_ids = metadata['movie_ids']
        self.dimension = metadata['dimension']
        self.index_type = metadata['index_type']
        
        logger.info(f"✓ Index loaded from {filepath} ({len(self.movie_ids)} movies)")
    
    def get_stats(self) -> Dict:
        """Get statistics about the index"""
        return {
            'total_vectors': self.index.ntotal if self.index else 0,
            'dimension': self.dimension,
            'index_type': self.index_type,
            'num_movies': len(self.movie_ids)
        }


# Example usage
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    # Create sample embeddings
    dimension = 768
    num_movies = 1000
    
    embeddings = np.random.randn(num_movies, dimension).astype('float32')
    movie_ids = list(range(1, num_movies + 1))
    
    # Test different index types
    for index_type in ['flat', 'hnsw']:
        print(f"\n{'='*50}")
        print(f"Testing {index_type.upper()} index")
        print('='*50)
        
        # Create index
        store = FAISSVectorStore(dimension=dimension, index_type=index_type)
        
        # Add embeddings
        store.add_embeddings(embeddings, movie_ids)
        
        # Search
        query = np.random.randn(dimension).astype('float32')
        result_ids, scores = store.search(query, k=10)
        
        print(f"\nTop 10 results:")
        for i, (movie_id, score) in enumerate(zip(result_ids, scores), 1):
            print(f"  {i}. Movie {movie_id}: {score:.4f}")
        
        # Stats
        stats = store.get_stats()
        print(f"\nIndex stats: {stats}")
