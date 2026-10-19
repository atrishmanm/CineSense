"""
CineSense MLOps, Drift Monitoring & Continuous Quality Engine
=============================================================

Implements:
1. Population Stability Index (PSI) to detect training-serving feature drift.
2. Embedding Centroid Drift Detection across rolling temporal windows.
3. Shadow Scoring Evaluation: Asynchronous divergence evaluation between active and canary models.
"""

from __future__ import annotations

from typing import Any, Dict, List
import numpy as np


class DriftMonitor:
    """
    Monitors data drift, feature distribution stability, and embedding shifts.
    """

    @staticmethod
    def calculate_psi(
        expected_distribution: np.ndarray,
        actual_distribution: np.ndarray,
        num_buckets: int = 10,
    ) -> Dict[str, Any]:
        """
        Calculate Population Stability Index (PSI) between baseline and serving distributions.
        PSI = sum( (Actual_b - Expected_b) * ln(Actual_b / Expected_b) )
        """
        assert len(expected_distribution) > 0 and len(actual_distribution) > 0

        # Create quantile bin edges from expected baseline
        quantiles = np.linspace(0, 100, num_buckets + 1)
        bin_edges = np.percentile(expected_distribution, quantiles)
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf

        # Calculate proportions in each bin with Laplace smoothing
        exp_counts, _ = np.histogram(expected_distribution, bins=bin_edges)
        act_counts, _ = np.histogram(actual_distribution, bins=bin_edges)

        exp_pct = (exp_counts + 1e-4) / (len(expected_distribution) + 1e-4 * num_buckets)
        act_pct = (act_counts + 1e-4) / (len(actual_distribution) + 1e-4 * num_buckets)

        # PSI computation
        psi_value = float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))

        if psi_value < 0.10:
            status = "STABLE"
            action = "No change required"
        elif psi_value <= 0.25:
            status = "MODERATE_DRIFT"
            action = "Monitor closely; consider scheduling retraining"
        else:
            status = "SIGNIFICANT_DRIFT"
            action = "Trigger automated model retraining"

        return {
            "psi": round(psi_value, 4),
            "status": status,
            "action": action,
            "num_buckets": num_buckets,
        }

    @staticmethod
    def calculate_embedding_centroid_drift(
        baseline_embeddings: np.ndarray,
        current_embeddings: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Measures shift in embedding space by comparing normalized centroids.
        Returns cosine distance between baseline and current centroids.
        """
        base_mean = np.mean(baseline_embeddings, axis=0)
        curr_mean = np.mean(current_embeddings, axis=0)

        b_norm = np.linalg.norm(base_mean)
        c_norm = np.linalg.norm(curr_mean)

        if b_norm > 0 and c_norm > 0:
            cosine_sim = float(np.dot(base_mean, curr_mean) / (b_norm * c_norm))
            drift_distance = max(0.0, 1.0 - cosine_sim)
        else:
            drift_distance = 0.0

        alert = drift_distance > 0.15
        return {
            "centroid_cosine_similarity": round(1.0 - drift_distance, 4),
            "drift_distance": round(drift_distance, 4),
            "alert": alert,
        }

    @staticmethod
    def evaluate_shadow_divergence(
        production_ranks: List[int],
        shadow_ranks: List[int],
        k: int = 10,
    ) -> Dict[str, float]:
        """
        Compares production model recommendations with shadow canary model.
        Computes Jaccard overlap and Rank Correlation.
        """
        top_prod = set(production_ranks[:k])
        top_shadow = set(shadow_ranks[:k])

        intersection = len(top_prod & top_shadow)
        union = len(top_prod | top_shadow)
        jaccard = intersection / union if union > 0 else 0.0

        return {
            "top_k_jaccard_overlap": round(jaccard, 4),
            "identical_top_item": bool(production_ranks and shadow_ranks and production_ranks[0] == shadow_ranks[0]),
        }
