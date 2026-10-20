"""
TorchServe Handler for CineSense Recommendation Models
Production-ready model serving with batching and preprocessing

STATUS: EXPERIMENTAL / UNUSED (D404). Not referenced by the Flask app;
deploy with TorchServe only after verifying the checkpoint contract.
"""

import os
import torch
import logging
import json
import numpy as np
from typing import List, Dict

try:
    from ts.torch_handler.base_handler import BaseHandler
except ImportError:  # torchserve (`ts`) is optional and only present in a worker
    class BaseHandler:  # minimal stand-in so this module stays importable
        """Fallback used when TorchServe's ``ts`` package is not installed."""

        def __init__(self):  # pragma: no cover - trivial
            pass

logger = logging.getLogger(__name__)

# Production plot embeddings are 768-d (all-mpnet-base-v2); keep the fallback
# zero-vector in the same dimension as the trained model.
PLOT_EMBED_DIM = 768


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
        # Set in initialize() from inspect.signature: True for architectures
        # whose forward takes (user, movie, content_features, plot), False for
        # the plain (user, movie, plot) HybridRecommender signer.
        self._forward_uses_content_features = True
        
    def initialize(self, context):
        """
        Initialize model, load artifacts, and prepare for inference
        
        Args:
            context: TorchServe context containing model artifacts
        """
        logger.info("Initializing CineSense Recommender Handler...")
        
        # Get model properties
        properties = context.system_properties
        # gpu_id may be absent/unset ("None") even when CUDA is visible, and
        # torch.device("cuda:None") raises ValueError; fall back to CPU
        # (mirrors serving/model_handler.py) (audit F1.3).
        gpu_id = properties.get("gpu_id")
        if torch.cuda.is_available() and gpu_id not in (None, -1):
            self.device = torch.device(f"cuda:{int(gpu_id)}")
        else:
            self.device = torch.device("cpu")
        
        # Load model architecture
        model_dir = properties.get("model_dir")
        
        # Load model checkpoint
        model_path = f"{model_dir}/model.pth"
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"TorchServe model artifact not found: {model_path}. "
                "Build the model archive with export_model_for_torchserve(...) "
                "or pass --serialized-file <checkpoint> when running "
                "torch-model-archiver."
            )
        try:
            payload = torch.load(model_path, map_location="cpu", weights_only=False)
            if isinstance(payload, torch.nn.Module):
                # Legacy contract: a full pickled nn.Module.
                self.model = payload
                logger.warning(
                    "%s contains a full pickled nn.Module; this contract is "
                    "deprecated — save state dicts (checkpoint dicts) instead.",
                    model_path,
                )
            elif isinstance(payload, dict):
                state = payload.get("model_state_dict", payload)
                if any(k.startswith(("user_emb.", "user_embedding.", "fusion.")) for k in state):
                    # HybridRecommender family: reuse the compatibility loader
                    # from training.models so both the current naming
                    # (user_emb.* / fc.*) and the legacy pre-BatchNorm naming
                    # (user_embedding.* / movie_embedding.* / fusion.*) load
                    # (audit F1.2). BatchNorm running stats are absent from
                    # legacy files and kept at init (eval-mode BN ≈ identity
                    # + eps); strict=False there is intentional.
                    try:
                        from training.models import load_hybrid_checkpoint
                    except ImportError as imp_err:
                        raise RuntimeError(
                            f"{model_path} is a HybridRecommender state dict; "
                            "loading it requires training.models "
                            "(training/) to be importable."
                        ) from imp_err
                    self.model = load_hybrid_checkpoint(
                        model_path, map_location="cpu"
                    )
                    # load_hybrid_checkpoint validates the remap and logs
                    # defaulted BatchNorm params itself (audit F1.1).
                else:
                    raise RuntimeError(
                        f"Unsupported state-dict keys in {model_path}: "
                        f"{sorted(state)[:5]}... The handler only supports the "
                        "HybridRecommender family (user_emb.* / fc.* or legacy "
                        "user_embedding.* / fusion.*)."
                    )
            else:
                raise TypeError(
                    f"Unsupported checkpoint payload in {model_path}: "
                    f"{type(payload)!r}. Expected a state dict or checkpoint dict."
                )
            self.model = self.model.to(self.device)
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

        # Detect the checkpoint's forward arity ONCE: HybridRecommender's
        # forward is (user, movie, plot_embeddings) while
        # AdvancedHybridRecommender also takes content_features. Calling a
        # 3-arg model with 4 tensors raised TypeError at first inference.
        import inspect
        try:
            forward_params = inspect.signature(self.model.forward).parameters
            self._forward_uses_content_features = 'content_features' in forward_params
        except (TypeError, ValueError):
            self._forward_uses_content_features = True

        # Validate the handler -> model contract at init by running one dummy
        # forward. This surfaces arity/shape mismatches (e.g. a genre-encoder
        # vocabulary whose width != the checkpoint's content_feature_dim)
        # with a clear error instead of failing on the first user request.
        if self.genre_encoder is not None:
            content_width = len(self.genre_encoder) + 3  # genres + duration + year + rating
            probe_args = [
                torch.LongTensor([0]).to(self.device),
                torch.LongTensor([0]).to(self.device),
            ]
            if self._forward_uses_content_features:
                probe_args.insert(2, torch.FloatTensor([[0.0] * content_width]).to(self.device))
            probe_args.append(torch.FloatTensor([[0.0] * PLOT_EMBED_DIM]).to(self.device))
            try:
                with torch.no_grad():
                    self.model(*probe_args)
            except Exception as e:
                raise RuntimeError(
                    "Model/feature contract check failed (content vector width "
                    f"= {content_width}, plot dim = {PLOT_EMBED_DIM}): {e}. "
                    "The genre_encoder.json vocabulary size must match the "
                    "checkpoint's content_feature_dim (training config); "
                    "regenerate the encoder or export the matching checkpoint."
                )

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
            args = [model_input["user_ids"], model_input["movie_ids"]]
            if self._forward_uses_content_features:
                args.append(model_input["content_features"])
            args.append(model_input["plot_embeddings"])
            predictions = self.model(*args)
        
        return predictions
    
    def postprocess(self, inference_output: torch.Tensor) -> List[Dict]:
        """
        Postprocess model outputs into response format

        Note on rating scale (audit F1.4): this handler clips predictions to
        the 0.5-5.0 MovieLens-100K star scale, matching training/recommend.py.
        The live app's TMDB-derived rating surfaces use 0-10 — would silently
        blend if ever wired through `DL_SCORE_WEIGHT`; keep the scales apart.

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
            # Return zero vector (same dim as the trained model) if not available
            return [0.0] * PLOT_EMBED_DIM
    
    def _calculate_confidence(self, rating: float) -> float:
        """Calculate confidence score based on rating"""
        # Higher confidence for extreme ratings (low or high)
        distance_from_middle = abs(rating - 2.75)
        confidence = 0.5 + (distance_from_middle / 2.25) * 0.5
        return round(confidence, 2)


def export_model_for_torchserve(
    model_path: str,
    output_dir: str = "torchserve_models",
    genre_encoder_path: str | None = None,
    plot_embeddings_path: str | None = None,
):
    """
    Export CineSense model for TorchServe deployment

    Args:
        model_path: Path to trained model checkpoint (must exist)
        output_dir: Directory to save TorchServe artifacts
        genre_encoder_path: Optional genre encoder JSON. Defaults to
            ``<model dir>/genre_encoder.json`` when present.
        plot_embeddings_path: Optional plot embeddings .npy. Defaults to
            ``<model dir>/plot_embeddings.npy`` when present.
    """
    import shutil

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path}. "
            "Train a model first and/or point at the real checkpoint "
            "(see models/ for the bundled Kaggle checkpoints)."
        )

    os.makedirs(output_dir, exist_ok=True)

    # Copy model file
    shutil.copy(model_path, f"{output_dir}/model.pth")

    # Copy the auxiliary artifacts the handler expects. The handler loads
    # genre_encoder.json for content-feature encoding; without it content
    # features silently degrade, so flag it loudly.
    model_dir = os.path.dirname(os.path.abspath(model_path))
    src_genre = genre_encoder_path or os.path.join(model_dir, "genre_encoder.json")
    src_plot = plot_embeddings_path or os.path.join(model_dir, "plot_embeddings.npy")

    if os.path.exists(src_genre):
        shutil.copy(src_genre, f"{output_dir}/genre_encoder.json")
        print(f"✓ Copied genre encoder: {src_genre}")
    else:
        # D386: the handler loads genre_encoder.json from the archive/model
        # dir on initialize; without it content-feature encoding degrades
        # silently. Write the canonical MovieLens-32M 20-genre vocabulary
        # (the order used by the training pipelines) so the export is
        # self-contained. Replace with the production encoder if it differs.
        try:
            from benchmarks.data_loader import GENRE_NAMES_32M as _genre_vocab
        except ImportError:
            _genre_vocab = []
        if _genre_vocab:
            with open(f"{output_dir}/genre_encoder.json", "w", encoding="utf-8") as fh:
                json.dump({g: i for i, g in enumerate(_genre_vocab)}, fh, indent=2)
            print(
                f"✓ Wrote default genre encoder (MovieLens-32M 20-genre order) to {output_dir}/genre_encoder.json\n"
                f"  NOTE: this yields content vectors {len(_genre_vocab) + 3} wide — the exported "
                "checkpoint's content_feature_dim MUST match (training/advanced_models_v2.py "
                "defaults to 55). Verify before packaging, or the handler's init-time contract "
                "check will fail."
            )
        else:
            print(
                "! WARNING: genre_encoder.json not found "
                f"(looked in {src_genre}) and the default genre-vocabulary "
                "import failed; the handler will run without genre encoding. "
                "Copy/produce one before packaging."
            )

    if os.path.exists(src_plot):
        shutil.copy(src_plot, f"{output_dir}/plot_embeddings.npy")
        print(f"✓ Copied plot embeddings: {src_plot}")
    else:
        print(
            "! NOTE: plot_embeddings.npy not found "
            f"(looked in {src_plot}); the handler will use zero vectors."
        )

    print(f"✓ Model exported to {output_dir}")
    print("\nTo create TorchServe model archive:")
    # --model-file is required: without it the packaged MAR lacks the module
    # class definition (the handler alone is not the model's arch) and
    # initialization fails (audit F1.4; mirrors serving/model_handler.py).
    print("""
    torch-model-archiver \\
        --model-name cinesense_recommender \\
        --version 1.0 \\
        --model-file training/models.py \\
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
