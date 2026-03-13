"""
Extended Visual Movie Search
Image-based search, poster similarity, scene matching
"""

from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests
from io import BytesIO
import torch
import numpy as np
from typing import List, Dict, Union, Optional
import logging
import base64
import os

logger = logging.getLogger(__name__)


class VisualMovieSearch:
    """
    Advanced visual search for movies
    - Search by uploaded image
    - Find visually similar posters
    - Text-to-image search ("dark moody poster")
    - Color-based search
    """
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", db_manager=None):
        """
        Initialize CLIP model for visual search
        """
        logger.info(f"Initializing visual search with model: {model_name}")
        self.db = db_manager
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.local_only = os.getenv('VISUAL_SEARCH_LOCAL_ONLY', '0').lower() in {'1', 'true', 'yes', 'on'}
        self.poster_image_base = os.getenv('TMDB_POSTER_BASE_URL', 'https://image.tmdb.org/t/p/w500')
        self.max_index_movies = int(os.getenv('VISUAL_SEARCH_INDEX_LIMIT', '500'))
        self.model_load_attempts = 0
        self.max_model_load_attempts = int(os.getenv('VISUAL_SEARCH_MODEL_LOAD_RETRIES', '2'))
        self.index_build_attempts = 0
        self.max_index_build_attempts = int(os.getenv('VISUAL_SEARCH_INDEX_RETRIES', '3'))
        
        self.movie_poster_embeddings = None
        self.movie_metadata = None
    
    def _ensure_model_loaded(self) -> bool:
        """Load CLIP lazily to avoid unnecessary startup downloads and latency."""
        if self.model is not None and self.processor is not None:
            return True
        if self.model_load_attempts >= self.max_model_load_attempts:
            logger.warning("CLIP model load retry limit reached")
            return False

        self.model_load_attempts += 1
        
        try:
            logger.info("Loading CLIP model from local cache first...")
            self.model = CLIPModel.from_pretrained(self.model_name, local_files_only=True)
            self.processor = CLIPProcessor.from_pretrained(self.model_name, local_files_only=True)
            self.model.eval()
            logger.info("CLIP model loaded from local cache")
            return True
        except Exception as e:
            if self.local_only:
                logger.warning(f"CLIP local-only mode enabled and local model not found: {e}")
                self.model = None
                self.processor = None
                return False
        
        try:
            logger.info("CLIP local cache miss, downloading model from Hugging Face...")
            self.model = CLIPModel.from_pretrained(self.model_name)
            self.processor = CLIPProcessor.from_pretrained(self.model_name)
            self.model.eval()
            logger.info("CLIP model downloaded and ready")
            return True
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            self.model = None
            self.processor = None
            return False

    def _normalize_poster_url(self, poster_path: str) -> Optional[str]:
        """Normalize poster path to a valid absolute URL."""
        if not poster_path:
            return None
        if poster_path.startswith('http'):
            return poster_path
        if poster_path.startswith('/'):
            return f"{self.poster_image_base}{poster_path}"
        return f"{self.poster_image_base}/{poster_path}"

    def _ensure_index_ready(self) -> bool:
        """Build poster index on-demand from dataset-backed DB records."""
        if self.movie_poster_embeddings is not None:
            return True
        if self.index_build_attempts >= self.max_index_build_attempts:
            logger.warning("Poster index build retry limit reached")
            return False

        if not self.db:
            logger.warning("No DB manager available for poster indexing")
            return False

        self.index_build_attempts += 1

        try:
            movies = self.db.get_top_movies(limit=self.max_index_movies, order_by='popularity')
            if not movies:
                logger.warning("No movies available for poster indexing")
                return False
            self.index_movie_posters(movies)
            return self.movie_poster_embeddings is not None
        except Exception as e:
            logger.warning(f"Automatic poster indexing failed: {e}")
            return False
    
    def index_movie_posters(self, movies: List[Dict]):
        """
        Pre-compute embeddings for all movie posters
        
        Args:
            movies: List of movie dicts with 'poster_path' field
        """
        if not self._ensure_model_loaded():
            logger.error("CLIP model not loaded")
            return
        
        logger.info(f"Indexing {len(movies)} movie posters...")
        
        embeddings = []
        valid_movies = []
        
        for movie in movies:
            poster_url = self._normalize_poster_url(movie.get('poster_path'))
            if poster_url:
                try:
                    embedding = self._encode_image_url(poster_url)
                    embeddings.append(embedding)
                    normalized_movie = movie.copy()
                    normalized_movie['poster_path'] = poster_url
                    valid_movies.append(normalized_movie)
                except Exception as e:
                    logger.debug(f"Failed to encode poster for {movie.get('title')}: {e}")
                    continue
        
        if embeddings:
            self.movie_poster_embeddings = torch.stack(embeddings)
            self.movie_metadata = valid_movies
            logger.info(f"✓ Indexed {len(valid_movies)} movie posters")
        else:
            logger.warning("No movie posters could be indexed")
    
    def _encode_image_url(self, url: str, timeout: int = 5) -> torch.Tensor:
        """
        Encode image from URL to embedding
        """
        if not self._ensure_model_loaded():
            raise RuntimeError("CLIP model unavailable")

        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert('RGB')
        
        inputs = self.processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            features = self.model.get_image_features(**inputs)
        
        # Normalize for cosine similarity
        features = features / features.norm(dim=-1, keepdim=True)
        
        return features.squeeze()
    
    def _encode_image_base64(self, base64_str: str) -> torch.Tensor:
        """
        Encode base64 image to embedding
        """
        if not self._ensure_model_loaded():
            raise RuntimeError("CLIP model unavailable")

        image_data = base64.b64decode(base64_str)
        image = Image.open(BytesIO(image_data)).convert('RGB')
        
        inputs = self.processor(images=image, return_tensors="pt")
        
        with torch.no_grad():
            features = self.model.get_image_features(**inputs)
        
        features = features / features.norm(dim=-1, keepdim=True)
        
        return features.squeeze()
    
    def search_by_uploaded_image(
        self,
        image_path_or_url_or_base64: str,
        top_k: int = 10,
        is_base64: bool = False
    ) -> List[Dict]:
        """
        Find similar movies by uploaded image
        
        Args:
            image_path_or_url_or_base64: Image to search with
            top_k: Number of results
            is_base64: Whether input is base64 encoded
            
        Returns:
            List of similar movies
        """
        if not self._ensure_index_ready():
            logger.warning("Poster index not available; returning empty visual results")
            return []
        
        try:
            # Encode query image
            if is_base64:
                query_embedding = self._encode_image_base64(image_path_or_url_or_base64)
            elif image_path_or_url_or_base64.startswith('http'):
                query_embedding = self._encode_image_url(image_path_or_url_or_base64)
            else:
                # Local file
                image = Image.open(image_path_or_url_or_base64).convert('RGB')
                inputs = self.processor(images=image, return_tensors="pt")
                with torch.no_grad():
                    query_embedding = self.model.get_image_features(**inputs).squeeze()
                query_embedding = query_embedding / query_embedding.norm()
            
            # Compute similarities
            similarities = torch.cosine_similarity(
                query_embedding.unsqueeze(0),
                self.movie_poster_embeddings
            )
            
            # Get top matches
            top_indices = similarities.argsort(descending=True)[:top_k]
            
            results = []
            for idx in top_indices:
                movie = self.movie_metadata[idx].copy()
                movie['visual_similarity'] = float(similarities[idx])
                results.append(movie)
            
            return results
        
        except Exception as e:
            logger.error(f"Visual search failed: {e}")
            return []
    
    def search_by_text_description(
        self,
        text: str,
        top_k: int = 10
    ) -> List[Dict]:
        """
        Find movies matching visual description
        
        Examples:
        - "dark poster with blue tones"
        - "colorful animated movie"
        - "red and black horror poster"
        
        Args:
            text: Visual description
            top_k: Number of results
            
        Returns:
            List of matching movies
        """
        if not self._ensure_index_ready():
            logger.warning("Poster index not available; returning empty visual results")
            return []
        
        try:
            # Encode text query
            inputs = self.processor(text=[text], return_tensors="pt", padding=True)
            
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
            
            # Normalize
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            # Find matching posters
            similarities = torch.cosine_similarity(
                text_features,
                self.movie_poster_embeddings
            )
            
            top_indices = similarities.argsort(descending=True)[:top_k]
            
            results = []
            for idx in top_indices:
                movie = self.movie_metadata[idx].copy()
                movie['visual_match_score'] = float(similarities[idx])
                results.append(movie)
            
            return results
        
        except Exception as e:
            logger.error(f"Text-to-image search failed: {e}")
            return []

    def text_to_image_search(self, text_description: str = '', top_k: int = 10) -> List[Dict]:
        """Alias for search_by_text_description — used by API routes"""
        return self.search_by_text_description(text=text_description, top_k=top_k)

    def search_by_image(self, image_input: str = '', top_k: int = 10, is_base64: bool = True) -> List[Dict]:
        """Alias for search_by_uploaded_image — used by API routes"""
        return self.search_by_uploaded_image(
            image_path_or_url_or_base64=image_input,
            top_k=top_k,
            is_base64=is_base64
        )

    def find_similar_posters(
        self,
        movie_id: int,
        top_k: int = 10
    ) -> List[Dict]:
        """
        Find movies with visually similar posters
        
        Args:
            movie_id: Reference movie ID
            top_k: Number of similar movies to return
            
        Returns:
            List of visually similar movies
        """
        if not self._ensure_index_ready():
            return []
        
        try:
            # Find movie in metadata
            movie_idx = None
            for idx, movie in enumerate(self.movie_metadata):
                if movie.get('movie_id') == movie_id:
                    movie_idx = idx
                    break
            
            if movie_idx is None:
                logger.error(f"Movie {movie_id} not in index")
                return []
            
            # Get embedding
            query_embedding = self.movie_poster_embeddings[movie_idx]
            
            # Compute similarities (excluding self)
            similarities = torch.cosine_similarity(
                query_embedding.unsqueeze(0),
                self.movie_poster_embeddings
            )
            
            # Set self similarity to -1 to exclude
            similarities[movie_idx] = -1
            
            # Get top matches
            top_indices = similarities.argsort(descending=True)[:top_k]
            
            results = []
            for idx in top_indices:
                if idx != movie_idx:
                    movie = self.movie_metadata[idx].copy()
                    movie['similarity_score'] = float(similarities[idx])
                    results.append(movie)
            
            return results
        
        except Exception as e:
            logger.error(f"Poster similarity search failed: {e}")
            return []
    
    def extract_dominant_colors(self, image_url: str, n_colors: int = 5) -> List[str]:
        """
        Extract dominant colors from movie poster
        
        Returns:
            List of hex color codes
        """
        try:
            from sklearn.cluster import KMeans
            
            # Load image
            response = requests.get(image_url, timeout=5)
            image = Image.open(BytesIO(response.content)).convert('RGB')
            
            # Resize for speed
            image = image.resize((100, 100))
            
            # Convert to array
            pixels = np.array(image).reshape(-1, 3)
            
            # K-means clustering
            kmeans = KMeans(n_clusters=n_colors, random_state=42)
            kmeans.fit(pixels)
            
            # Get colors
            colors = kmeans.cluster_centers_.astype(int)
            
            # Convert to hex
            hex_colors = []
            for color in colors:
                hex_code = '#{:02x}{:02x}{:02x}'.format(color[0], color[1], color[2])
                hex_colors.append(hex_code)
            
            return hex_colors
        
        except Exception as e:
            logger.error(f"Color extraction failed: {e}")
            return []
    
    def search_by_color_palette(
        self,
        target_colors: List[str],
        top_k: int = 10
    ) -> List[Dict]:
        """
        Find movies with similar color palettes
        
        Args:
            target_colors: List of hex colors (e.g., ['#FF0000', '#0000FF'])
            top_k: Number of results
            
        Returns:
            List of movies with similar colors
        """
        # This would require pre-computing color palettes for all movies
        # For now, return placeholder
        logger.warning("Color palette search not yet fully implemented")
        return []


# Example usage
if __name__ == '__main__':
    visual_search = VisualMovieSearch()
    
    print("Visual Movie Search System")
    print("=" * 60)
    
    # Example: search by text description
    print("\nSearching for: 'dark moody poster with blue tones'")
    
    # Note: Would need actual movie data with poster URLs
    sample_movies = [
        {
            'movie_id': 1,
            'title': 'The Dark Knight',
            'poster_path': 'https://example.com/poster1.jpg'
        }
    ]
    
    # visual_search.index_movie_posters(sample_movies)
    # results = visual_search.search_by_text_description('dark moody poster')
    
    print("Visual search module loaded successfully!")
