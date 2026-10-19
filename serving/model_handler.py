"""
TorchServe Model Handler
Production model serving with TorchServe

STATUS: EXPERIMENTAL / UNUSED (D404). Not referenced by the Flask app.

The handler loads the CineSense Deep V3 checkpoint architecture
(``training.deep_v3.CineSenseDeepV3Full32M``) — the only architecture present
in ``models/``.  Model shapes are inferred from the checkpoint state dict
(the previous hardcoded 100000/100000/128/3 defaults matched no checkpoint).

Checkpoints live under ``models/`` and are *not* all tracked by git; a fresh
clone may not contain them. Set ``MODEL_PATH`` or run the Kaggle training
pipeline, otherwise ``initialize`` fails loudly with a clear message.
"""

import os
import logging
from pathlib import Path
from typing import List, Dict

import torch

from training.deep_v3 import load_v3_checkpoint

logger = logging.getLogger(__name__)

# Real bundled checkpoints (Deep V3). Searched in order if no explicit path
# is supplied via the MODEL_PATH env var or the TorchServe model_dir.
DEFAULT_CHECKPOINT_CANDIDATES = [
    "models/kaggle_t4x2_full32m/cinesense_deep_v3_full32m_best.pt",
    "models/kaggle_t4x2_3m_subset/cinesense_deep_v3_3m_subset_best.pt",
]


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
        self.checkpoint_path = None

    def _resolve_checkpoint(self, model_dir: str | None) -> str:
        """Find a real checkpoint, or raise with an actionable message."""
        candidates: List[str] = []
        env_path = os.environ.get("MODEL_PATH")
        if env_path:
            candidates.append(env_path)
        if model_dir:
            candidates.append(os.path.join(model_dir, "cinesense_deep_v3_full32m_best.pt"))
            candidates.append(os.path.join(model_dir, "model.pth"))
        candidates.extend(self._checkpoint_dir_candidates())
        candidates.extend(DEFAULT_CHECKPOINT_CANDIDATES)

        for cand in candidates:
            if cand and os.path.exists(cand):
                return cand

        raise FileNotFoundError(
            "No CineSense checkpoint found. Searched: "
            + ", ".join(c for c in candidates if c)
            + ". Bundled checkpoints live under models/ but may be absent on a "
            "fresh clone; set MODEL_PATH=<path to *.pt> or run the Kaggle "
            "training pipeline (kaggle_runner/train_kaggle_t4x2.py) to produce one."
        )

    @staticmethod
    def _checkpoint_dir_candidates() -> List[str]:
        """Glob ``config.CHECKPOINT_DIR`` for *.pt checkpoints, newest first
        (D385).  Honors the CHECKPOINT_DIR env var first; config.py is an
        optional import, so TorchServe workers without the app config still
        work on the default 'checkpoints' dir."""
        try:
            from config import Config  # type: ignore  # optional in TorchServe workers
            checkpoint_dir = Path(os.environ.get("CHECKPOINT_DIR") or Config.CHECKPOINT_DIR)
        except Exception:
            checkpoint_dir = Path(os.environ.get("CHECKPOINT_DIR", "checkpoints"))
        try:
            found = sorted(
                (p for p in checkpoint_dir.glob("*.pt") if p.is_file()),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
        except OSError:
            return []
        return [str(p) for p in found]

    def initialize(self, context):
        """
        Load model on server startup

        Args:
            context: TorchServe context with model artifacts
        """
        try:
            self.manifest = getattr(context, "manifest", None)
            properties = context.system_properties
            model_dir = properties.get("model_dir")

            # Set device
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            logger.info(f"Using device: {self.device}")

            # Resolve + load the Deep V3 checkpoint (shapes inferred from it).
            self.checkpoint_path = self._resolve_checkpoint(model_dir)
            model, missing, unexpected = load_v3_checkpoint(
                self.checkpoint_path, device=str(self.device)
            )
            self.model = model.to(self.device)
            self.model.eval()

            if missing or unexpected:
                # Dense lookup buffers are intentionally zero-filled on load;
                # report anything else that did not match.
                logger.warning(
                    "Checkpoint load had missing=%s unexpected=%s keys "
                    "(dense lookup buffers are zero-filled by design).",
                    missing, unexpected,
                )

            logger.info(
                "✓ Model initialized from %s (users=%d, items=%d)",
                self.checkpoint_path,
                int(model.user_embed.num_embeddings),
                int(model.item_embed.num_embeddings),
            )
            self.initialized = True

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
                body = request.get('body', '{}')

            if isinstance(body, (bytes, bytearray)):
                import json
                body = json.loads(body)

            user_id = body.get('user_id')
            movie_id = body.get('movie_id')
            if user_id is None or movie_id is None:
                raise ValueError("Request must include 'user_id' and 'movie_id'")

            return {
                'user_ids': torch.LongTensor([int(user_id)]).to(self.device),
                'movie_ids': torch.LongTensor([int(movie_id)]).to(self.device),
            }

        except Exception as e:
            logger.error(f"Preprocessing error: {e}")
            raise

    def inference(self, model_input: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Run inference

        Args:
            model_input: Preprocessed input tensors

        Returns:
            Model predictions (raw 0.5-5.0 rating scale)
        """
        try:
            with torch.no_grad():
                # CineSense Deep V3 returns (normalized_pred, raw_pred).
                out = self.model(model_input['user_ids'], model_input['movie_ids'])
                if isinstance(out, tuple):
                    out = out[1]  # raw 0.5-5.0 rating
            return out

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
            prediction = float(inference_output.detach().cpu().reshape(-1)[0].item())

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


# Handler instance for TorchServe (created lazily on first handle() call to
# avoid import-time side effects — D384).
_service = None


def handle(data, context):
    """
    Entry point for TorchServe
    """
    global _service
    if _service is None:
        _service = RecommenderHandler()
    if not _service.initialized:
        _service.initialize(context)

    return _service.handle(data, context)


"""
DEPLOYMENT INSTRUCTIONS:

1. Package the model (point --serialized-file at a real checkpoint):

   $ torch-model-archiver \\
       --model-name cinesense_recommender \\
       --version 1.0 \\
       --model-file training/deep_v3.py \\
       --serialized-file models/kaggle_t4x2_full32m/cinesense_deep_v3_full32m_best.pt \\
       --handler serving/model_handler.py \\
       --export-path model_store


2. Start TorchServe:

   $ torchserve --start \\
       --model-store model_store \\
       --models cinesense=cinesense_recommender.mar \\
       --ncs


3. Make predictions:

   $ curl -X POST http://localhost:8080/predictions/cinesense \\
       -H "Content-Type: application/json" \\
       -d '{"user_id": 123, "movie_id": 456}'


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
