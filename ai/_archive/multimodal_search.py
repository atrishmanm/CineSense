"""
Multi-modal search using CLIP
Enables searching by images, finding similar posters, and text-to-image search
"""

from transformers import CLIPProcessor, CLIPModel
from PIL import Image
import requests
from io import BytesIO
import torch
import numpy as np
from typing import List, Dict, Union
import logging

logger = logging.getLogger(__name__)


class MultiModalSearch:
    """
    CLIP-based multi-modal search for movies
    Supports text-to-image and image-to-image search
    """
    
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"):
        """
        Initialize CLIP model
        
        Args:
            model_name: HuggingFace model name for CLIP
        """
        logger.info(f"Loading CLIP model: {model_name}")
        try:
            self.model = CLIPModel.from_pretrained(model_name)
            self.processor = CLIPProcessor.from_pretrained(model_name)
            self.model.eval()
            logger.info("✓ CLIP model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            self.model = None
            self.processor = None
    
    def encode_text(self, text: str) -> np.ndarray:
        """
        Encode text query to embedding
        
        Args:
            text: Text description
            
        Returns:
            Embedding vector as numpy array
        """
        if self.model is None:
            raise RuntimeError("CLIP model not loaded")
        
        try:
            inputs = self.processor(text=[text], return_tensors="pt", padding=True)
            
            with torch.no_grad():
                text_features = self.model.get_text_features(**inputs)
            
            # Normalize for cosine similarity
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            return text_features.cpu().numpy()[0]
            
        except Exception as e:
            logger.error(f"Text encoding failed: {e}")
            raise
    
    def encode_image(self, image_input: Union[str, Image.Image]) -> np.ndarray:
        """
        Encode image to embedding
        
        Args:
            image_input: Either URL string or PIL Image object
            
        Returns:
            Embedding vector as numpy array
        """
        if self.model is None:
            raise RuntimeError("CLIP model not loaded")
        
        try:
            # Load image if URL provided
            if isinstance(image_input, str):
                if image_input.startswith('http'):
                    response = requests.get(image_input, timeout=5)
                    image = Image.open(BytesIO(response.content))
                else:
                    # Local file path
                    image = Image.open(image_input)
            else:
                image = image_input
            
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Process and encode
            inputs = self.processor(images=image, return_tensors="pt")
            
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
            
            # Normalize
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            return image_features.cpu().numpy()[0]
            
        except Exception as e:
            logger.error(f"Image encoding failed: {e}")
            raise
    
    def search_by_text(
        self, 
        text_query: str, 
        movie_posters: List[Dict],
        top_k: int = 10
    ) -> List[Dict]:
        """
        Search movies by text description (finds matching posters)
        
        Args:
            text_query: Natural language description
            movie_posters: List of dicts with 'movie_id', 'poster_url', etc.
            top_k: Number of results to return
            
        Returns:
            List of movies with similarity scores
        """
        if self.model is None:
            logger.warning("CLIP model not loaded")
            return []
        
        # Encode query text
        text_embedding = self.encode_text(text_query)
        
        # Encode all posters and compute similarities
        results = []
        for movie in movie_posters:
            try:
                poster_url = movie.get('poster_url') or movie.get('poster_path')
                if not poster_url:
                    continue
                
                # Encode poster
                image_embedding = self.encode_image(poster_url)
                
                # Compute similarity
                similarity = np.dot(text_embedding, image_embedding)
                
                movie_result = movie.copy()
                movie_result['clip_score'] = float(similarity)
                results.append(movie_result)
                
            except Exception as e:
                logger.debug(f"Failed to process poster for movie {movie.get('movie_id')}: {e}")
                continue
        
        # Sort by similarity
        results.sort(key=lambda x: x['clip_score'], reverse=True)
        
        return results[:top_k]
    
    def search_by_image(
        self,
        query_image: Union[str, Image.Image],
        movie_posters: List[Dict],
        top_k: int = 10
    ) -> List[Dict]:
        """
        Find similar movies by poster image
        
        Args:
            query_image: Query image (URL or PIL Image)
            movie_posters: List of movie poster info
            top_k: Number of results
            
        Returns:
            List of similar movies with scores
        """
        if self.model is None:
            logger.warning("CLIP model not loaded")
            return []
        
        # Encode query image
        query_embedding = self.encode_image(query_image)
        
        # Compare with all movie posters
        results = []
        for movie in movie_posters:
            try:
                poster_url = movie.get('poster_url') or movie.get('poster_path')
                if not poster_url:
                    continue
                
                # Encode poster
                poster_embedding = self.encode_image(poster_url)
                
                # Compute similarity
                similarity = np.dot(query_embedding, poster_embedding)
                
                movie_result = movie.copy()
                movie_result['clip_score'] = float(similarity)
                results.append(movie_result)
                
            except Exception as e:
                logger.debug(f"Failed to process poster: {e}")
                continue
        
        # Sort by similarity
        results.sort(key=lambda x: x['clip_score'], reverse=True)
        
        return results[:top_k]
    
    def batch_encode_images(self, image_urls: List[str]) -> np.ndarray:
        """
        Encode multiple images in batch for efficiency
        
        Args:
            image_urls: List of image URLs
            
        Returns:
            Array of embeddings (N, embedding_dim)
        """
        if self.model is None:
            raise RuntimeError("CLIP model not loaded")
        
        embeddings = []
        
        # Process in batches
        batch_size = 8
        for i in range(0, len(image_urls), batch_size):
            batch_urls = image_urls[i:i + batch_size]
            
            # Load images
            images = []
            for url in batch_urls:
                try:
                    if url.startswith('http'):
                        response = requests.get(url, timeout=5)
                        image = Image.open(BytesIO(response.content))
                    else:
                        image = Image.open(url)
                    
                    if image.mode != 'RGB':
                        image = image.convert('RGB')
                    
                    images.append(image)
                    
                except Exception as e:
                    logger.warning(f"Failed to load image {url}: {e}")
                    # Add placeholder for failed images
                    images.append(None)
            
            # Filter out None values
            valid_images = [img for img in images if img is not None]
            
            if valid_images:
                # Process batch
                inputs = self.processor(images=valid_images, return_tensors="pt")
                
                with torch.no_grad():
                    batch_features = self.model.get_image_features(**inputs)
                
                # Normalize
                batch_features = batch_features / batch_features.norm(dim=-1, keepdim=True)
                embeddings.append(batch_features.cpu().numpy())
        
        if embeddings:
            return np.vstack(embeddings)
        else:
            return np.array([])
    
    def compute_similarity_matrix(
        self,
        text_queries: List[str],
        image_urls: List[str]
    ) -> np.ndarray:
        """
        Compute similarity matrix between texts and images
        
        Args:
            text_queries: List of text descriptions
            image_urls: List of image URLs
            
        Returns:
            Similarity matrix (len(texts), len(images))
        """
        # Encode texts
        text_embeddings = []
        for text in text_queries:
            text_embeddings.append(self.encode_text(text))
        text_embeddings = np.array(text_embeddings)
        
        # Encode images
        image_embeddings = self.batch_encode_images(image_urls)
        
        # Compute similarity matrix
        similarity = np.dot(text_embeddings, image_embeddings.T)
        
        return similarity


# Example usage
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    # Initialize
    search = MultiModalSearch()
    
    # Sample movie posters
    sample_movies = [
        {
            'movie_id': 1,
            'title': 'Inception',
            'poster_url': 'https://image.tmdb.org/t/p/w500/9gk7adHYeDvHkCSEqAvQNLV5Uge.jpg'
        }
    ]
    
    # Test text-to-image search
    text_query = "sci-fi thriller with dark blue tones"
    print(f"\nSearching for: '{text_query}'")
    results = search.search_by_text(text_query, sample_movies, top_k=5)
    
    for i, movie in enumerate(results, 1):
        print(f"{i}. {movie['title']} (CLIP score: {movie['clip_score']:.3f})")
