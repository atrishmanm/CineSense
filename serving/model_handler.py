"""
TorchServe Model Handler
Production model serving with TorchServe
"""

import torch
import logging
import json
import numpy as np
from typing import List, Dict, Any
import io

logger = logging.getLogger(__name__)


class RecommenderHandler:
    """
    TorchServe handler for production model serving
    
    Handles:
    - Model loading
    - Input preprocessing
    - Inference
    - Output postprocessing
    """
    
    def __init__(self):
        self.model = None
        self.device = None
        self.initialized = False
        self.manifest = None
        self.map_location = None
    
    def initialize(self, context):
        """
        Load model on server startup
        
        Args:
            context: TorchServe context with model artifacts
        """
        try:
            self.manifest = context.manifest
            properties = context.system_properties
            model_dir = properties.get("model_dir")
            
            # Set device
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            logger.info(f"Using device: {self.device}")
            
            # Load model architecture
            # Import your model class here
            from training.advanced_models_v2 import AdvancedHybridRecommender
            
            # Load model checkpoint
            model_path = f"{model_dir}/best_model.pth"
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Get model parameters from checkpoint or config
            model_config = checkpoint.get('config', {
                'num_users': 100000,
                'num_movies': 100000,
                'embed_dim': 128,
                'num_transformer_blocks': 3
            })
            
            # Initialize model
            self.model = AdvancedHybridRecommender(**model_config)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.to(self.device)
            self.model.eval()
            
            self.initialized = True
            logger.info("✓ Model initialized successfully")
        
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise
    
    def preprocess(self, data: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Convert request to model input
        
        Args:
            data: List of request dictionaries
            
        Returns:
            Dictionary with tensors for model input
        """
        try:
            # Extract data from first request (batch of 1 for simplicity)
            request = data[0]
            
            # Parse JSON body
            if isinstance(request, dict):
                body = request.get('body', request)
            else:
                body = json.loads(request.get('body', '{}'))
            
            # Extract features
            user_id = body.get('user_id')
            movie_id = body.get('movie_id')
            content_features = body.get('content_features')
            plot_embedding = body.get('plot_embedding')
            
            # Convert to tensors
            inputs = {
                'user_ids': torch.LongTensor([user_id]).to(self.device),
                'movie_ids': torch.LongTensor([movie_id]).to(self.device)
            }
            
            if content_features:
                inputs['content_features'] = torch.FloatTensor(
                    [content_features]
                ).to(self.device)
            
            if plot_embedding:
                inputs['plot_embeddings'] = torch.FloatTensor(
                    [plot_embedding]
                ).to(self.device)
            
            return inputs
        
        except Exception as e:
            logger.error(f"Preprocessing error: {e}")
            raise
    
    def inference(self, model_input: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Run inference
        
        Args:
            model_input: Preprocessed input tensors
            
        Returns:
            Model predictions
        """
        try:
            with torch.no_grad():
                # Forward pass
                if 'content_features' in model_input and 'plot_embeddings' in model_input:
                    prediction = self.model(
                        model_input['user_ids'],
                        model_input['movie_ids'],
                        model_input['content_features'],
                        model_input['plot_embeddings']
                    )
                elif 'content_features' in model_input:
                    prediction = self.model(
                        model_input['user_ids'],
                        model_input['movie_ids'],
                        model_input['content_features']
                    )
                else:
                    prediction = self.model(
                        model_input['user_ids'],
                        model_input['movie_ids']
                    )
            
            return prediction
        
        except Exception as e:
            logger.error(f"Inference error: {e}")
            raise
    
    def postprocess(self, inference_output: torch.Tensor) -> List[Dict]:
        """
        Format response
        
        Args:
            inference_output: Model output tensor
            
        Returns:
            List of response dictionaries
        """
        try:
            # Convert to Python types
            prediction = float(inference_output.cpu().item())
            
            # Clip to valid rating range
            prediction = max(0.5, min(5.0, prediction))
            
            response = [{
                'predicted_rating': round(prediction, 2),
                'confidence': self._calculate_confidence(prediction),
                'rating_category': self._get_rating_category(prediction)
            }]
            
            return response
        
        except Exception as e:
            logger.error(f"Postprocessing error: {e}")
            raise
    
    def handle(self, data: List[Dict], context) -> List[Dict]:
        """
        Main handler function called by TorchServe
        
        Args:
            data: Input data
            context: TorchServe context
            
        Returns:
            Predictions
        """
        try:
            # Preprocess
            model_input = self.preprocess(data)
            
            # Inference
            model_output = self.inference(model_input)
            
            # Postprocess
            response = self.postprocess(model_output)
            
            return response
        
        except Exception as e:
            logger.error(f"Handler error: {e}")
            return [{"error": str(e)}]
    
    def _calculate_confidence(self, prediction: float) -> str:
        """Calculate confidence level"""
        if prediction >= 4.5 or prediction <= 1.5:
            return "high"
        elif 3.5 <= prediction <= 4.5 or 1.5 <= prediction <= 2.5:
            return "medium"
        else:
            return "low"
    
    def _get_rating_category(self, prediction: float) -> str:
        """Get human-readable rating category"""
        if prediction >= 4.5:
            return "excellent"
        elif prediction >= 4.0:
            return "very_good"
        elif prediction >= 3.5:
            return "good"
        elif prediction >= 3.0:
            return "average"
        elif prediction >= 2.0:
            return "below_average"
        else:
            return "poor"


# Handler instance for TorchServe
_service = RecommenderHandler()


def handle(data, context):
    """
    Entry point for TorchServe
    """
    if not _service.initialized:
        _service.initialize(context)
    
    return _service.handle(data, context)


"""
DEPLOYMENT INSTRUCTIONS:

1. Package the model:
   
   $ torch-model-archiver \\
       --model-name cinesense_recommender \\
       --version 1.0 \\
       --model-file training/advanced_models_v2.py \\
       --serialized-file model/best_model.pth \\
       --handler serving/model_handler.py \\
       --extra-files "ai/embeddings.py,config.py" \\
       --export-path model_store


2. Start TorchServe:
   
   $ torchserve --start \\
       --model-store model_store \\
       --models cinesense=cinesense_recommender.mar \\
       --ncs


3. Make predictions:
   
   $ curl -X POST http://localhost:8080/predictions/cinesense \\
       -H "Content-Type: application/json" \\
       -d '{
           "user_id": 123,
           "movie_id": 456,
           "content_features": [0.1, 0.2, ...],
           "plot_embedding": [0.3, 0.4, ...]
       }'


4. Check status:
   
   $ curl http://localhost:8081/models/cinesense


5. Scale workers:
   
   $ curl -X PUT "http://localhost:8081/models/cinesense?min_worker=2&max_worker=4"


6. Stop TorchServe:
   
   $ torchserve --stop


CONFIGURATION (config.properties):

inference_address=http://0.0.0.0:8080
management_address=http://0.0.0.0:8081
metrics_address=http://0.0.0.0:8082
number_of_netty_threads=32
job_queue_size=1000
model_store=/path/to/model_store
load_models=all
"""

if __name__ == '__main__':
    print("TorchServe Model Handler")
    print("=" * 60)
    print("\n✓ Handler ready for packaging")
    print("\nFeatures:")
    print("  • Production-grade serving")
    print("  • Auto-scaling workers")
    print("  • Metrics & monitoring")
    print("  • RESTful & gRPC APIs")
    print("  • Model versioning")
    print("\nSee deployment instructions above.")
