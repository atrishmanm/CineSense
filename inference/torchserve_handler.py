"""
TorchServe Handler for CineSense Recommendation Models
Production-ready model serving with batching and preprocessing
"""

import torch
import logging
import json
import numpy as np
from ts.torch_handler.base_handler import BaseHandler
from typing import List, Dict, Any

logger = logging.getLogger(__name__)


class CineSenseRecommenderHandler(BaseHandler):
    """
    Custom TorchServe handler for CineSense recommendation models
    Supports batched inference with preprocessing and postprocessing
    """
    
    def __init__(self):
        super().__init__()
        self.model = None
        self.device = None
        self.initialized = False
        self.genre_encoder = None
        self.plot_embeddings = None
        
    def initialize(self, context):
        """
        Initialize model, load artifacts, and prepare for inference
        
        Args:
            context: TorchServe context containing model artifacts
        """
        logger.info("Initializing CineSense Recommender Handler...")
        
        # Get model properties
        properties = context.system_properties
        self.device = torch.device(
            "cuda:" + str(properties.get("gpu_id")) 
            if torch.cuda.is_available() 
            else "cpu"
        )
        
        # Load model architecture
        model_dir = properties.get("model_dir")
        
        # Load model checkpoint
        model_path = f"{model_dir}/model.pth"
        try:
            self.model = torch.load(model_path, map_location=self.device)
            self.model.eval()
            logger.info(f"✓ Model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
        
        # Load auxiliary artifacts
        try:
            # Genre encoder
            with open(f"{model_dir}/genre_encoder.json", 'r') as f:
                self.genre_encoder = json.load(f)
            
            # Plot embeddings
            self.plot_embeddings = np.load(f"{model_dir}/plot_embeddings.npy")
            
            logger.info("✓ Auxiliary artifacts loaded")
        except Exception as e:
            logger.warning(f"Could not load auxiliary artifacts: {e}")
        
        self.initialized = True
        logger.info("✓ Handler initialized successfully")
    
    def preprocess(self, requests: List[Dict]) -> torch.Tensor:
        """
        Preprocess incoming requests into model inputs
        
        Args:
            requests: List of inference requests with user/movie data
            
        Returns:
            Preprocessed tensors ready for model
        """
        batch_data = []
        
        for request in requests:
            data = request.get("body") or request.get("data")
            
            if isinstance(data, (bytes, bytearray)):
                data = json.loads(data)
            
            # Extract features
            user_id = data.get("user_id", 0)
            movie_id = data.get("movie_id", 0)
            
            # Content features (genres, duration, etc.)
            content_features = self._encode_content_features(data)
            
            # Plot embedding
            plot_embedding = self._get_plot_embedding(movie_id)
            
            batch_data.append({
                "user_id": user_id,
                "movie_id": movie_id,
                "content_features": content_features,
                "plot_embedding": plot_embedding
            })
        
        # Convert to tensors
        user_ids = torch.LongTensor([d["user_id"] for d in batch_data])
        movie_ids = torch.LongTensor([d["movie_id"] for d in batch_data])
        content_features = torch.FloatTensor([d["content_features"] for d in batch_data])
        plot_embeddings = torch.FloatTensor([d["plot_embedding"] for d in batch_data])
        
        return {
            "user_ids": user_ids.to(self.device),
            "movie_ids": movie_ids.to(self.device),
            "content_features": content_features.to(self.device),
            "plot_embeddings": plot_embeddings.to(self.device)
        }
    
    def inference(self, model_input: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Run model inference
        
        Args:
            model_input: Preprocessed tensors
            
        Returns:
            Model predictions
        """
        with torch.no_grad():
            predictions = self.model(
                model_input["user_ids"],
                model_input["movie_ids"],
                model_input["content_features"],
                model_input["plot_embeddings"]
            )
        
        return predictions
    
    def postprocess(self, inference_output: torch.Tensor) -> List[Dict]:
        """
        Postprocess model outputs into response format
        
        Args:
            inference_output: Raw model predictions
            
        Returns:
            List of formatted responses
        """
        predictions = inference_output.cpu().numpy()
        
        responses = []
        for pred in predictions:
            # Convert prediction to rating (0-5 scale)
            rating = float(pred)
            rating = max(0.5, min(5.0, rating))  # Clip to valid range
            
            response = {
                "predicted_rating": round(rating, 2),
                "confidence": self._calculate_confidence(rating),
                "recommendation": "highly_recommended" if rating >= 4.0 else (
                    "recommended" if rating >= 3.0 else "not_recommended"
                )
            }
            
            responses.append(response)
        
        return responses
    
    def _encode_content_features(self, data: Dict) -> List[float]:
        """Encode movie content features"""
        features = []
        
        # Encode genres (multi-hot encoding)
        genres = data.get("genres", [])
        if self.genre_encoder:
            genre_vector = [0] * len(self.genre_encoder)
            for genre in genres:
                if genre in self.genre_encoder:
                    genre_vector[self.genre_encoder[genre]] = 1
            features.extend(genre_vector)
        
        # Add other features
        features.append(data.get("duration", 0) / 300.0)  # Normalize
        features.append(data.get("year", 2000) / 2025.0)  # Normalize
        features.append(data.get("vote_average", 0) / 10.0)  # Normalize
        
        return features
    
    def _get_plot_embedding(self, movie_id: int) -> List[float]:
        """Get plot embedding for movie"""
        if self.plot_embeddings is not None and movie_id < len(self.plot_embeddings):
            return self.plot_embeddings[movie_id].tolist()
        else:
            # Return zero vector if not available
            return [0.0] * 384
    
    def _calculate_confidence(self, rating: float) -> float:
        """Calculate confidence score based on rating"""
        # Higher confidence for extreme ratings (low or high)
        distance_from_middle = abs(rating - 2.75)
        confidence = 0.5 + (distance_from_middle / 2.25) * 0.5
        return round(confidence, 2)


def export_model_for_torchserve(
    model_path: str,
    output_dir: str = "torchserve_models"
):
    """
    Export CineSense model for TorchServe deployment
    
    Args:
        model_path: Path to trained model checkpoint
        output_dir: Directory to save TorchServe artifacts
    """
    import os
    import shutil
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Copy model file
    shutil.copy(model_path, f"{output_dir}/model.pth")
    
    print(f"✓ Model exported to {output_dir}")
    print("\nTo create TorchServe model archive:")
    print("""
    torch-model-archiver \\
        --model-name cinesense_recommender \\
        --version 1.0 \\
        --serialized-file torchserve_models/model.pth \\
        --handler inference/torchserve_handler.py \\
        --export-path model_store/
    """)
    
    print("\nTo start TorchServe:")
    print("""
    torchserve --start \\
        --model-store model_store \\
        --models cinesense=cinesense_recommender.mar
    """)
    
    print("\nTo make inference requests:")
    print("""
    curl -X POST http://localhost:8080/predictions/cinesense \\
        -H "Content-Type: application/json" \\
        -d '{"user_id": 123, "movie_id": 550, "genres": ["Action", "Thriller"]}'
    """)


if __name__ == "__main__":
    # Example: Export model for TorchServe
    model_checkpoint = "model/hybrid_recommender.pt"
    if os.path.exists(model_checkpoint):
        export_model_for_torchserve(model_checkpoint)
    else:
        print(f"Model checkpoint not found: {model_checkpoint}")
        print("Train a model first using train_model.py")
