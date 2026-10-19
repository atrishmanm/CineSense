"""
CineSense Diversity, Calibration & Explainability Engine
========================================================

Implements:
1. Determinantal Point Processes (DPP) fast greedy MAP inference for optimal diversity.
2. Calibrated Recommendations via Kullback-Leibler (KL) divergence minimization.
3. Traceable counterfactual explanation attribution.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List
import numpy as np


class DPPDiversifier:
    """
    Determinantal Point Processes (DPP) for fast diversity reranking.
    Kernel matrix: L_ij = q_i * S_ij * q_j
    where q_i = item relevance quality, S_ij = pairwise similarity.
    """

    @staticmethod
    def dpp_greedy(
        item_ids: List[int],
        quality_scores: np.ndarray,
        feature_vectors: np.ndarray,
        k: int = 10,
    ) -> List[int]:
        """
        Fast greedy MAP inference for DPP (Chen et al., NeurIPS 2018).
        """
        n = len(item_ids)
        if n <= k:
            return item_ids

        # Normalize feature vectors
        norms = np.linalg.norm(feature_vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        feats = feature_vectors / norms

        # Item qualities (positive)
        q = np.maximum(quality_scores, 1e-4)

        # Initialize L kernel columns
        selected: List[int] = []
        c = np.zeros((k, n), dtype=np.float64)
        d = np.copy(q ** 2)

        for i in range(k):
            # Select item with highest marginal gain d_j
            best_idx = int(np.argmax(d))
            if d[best_idx] < 1e-8:
                break
            selected.append(item_ids[best_idx])

            # Update Cholesky factor
            c[i, best_idx] = math.sqrt(d[best_idx])
            feat_best = feats[best_idx]
            sims = np.dot(feats, feat_best)  # Pairwise cosine similarities

            for j in range(n):
                if j in [item_ids.index(it) for it in selected]:
                    d[j] = 0.0
                    continue
                # Kernel cross-term: L_ij = q_i * q_j * S_ij
                l_val = q[best_idx] * q[j] * sims[j]
                proj = sum(c[m, best_idx] * c[m, j] for m in range(i))
                c[i, j] = (l_val - proj) / c[i, best_idx]
                d[j] = max(0.0, d[j] - c[i, j] ** 2)

        return selected


class CalibratedReranker:
    """
    Genre calibration minimizing KL divergence between user history and recommendation list.
    """

    @staticmethod
    def compute_kl_divergence(p_user: np.ndarray, p_rec: np.ndarray) -> float:
        """Kullback-Leibler Divergence: D_KL(P_user || P_rec)."""
        eps = 1e-6
        p = np.clip(p_user, eps, 1.0)
        p = p / np.sum(p)
        q = np.clip(p_rec, eps, 1.0)
        q = q / np.sum(q)
        return float(np.sum(p * np.log(p / q)))

    @classmethod
    def calibrate_recommendations(
        cls,
        candidate_items: List[Dict[str, Any]],
        user_genre_distribution: np.ndarray,
        k: int = 10,
        tradeoff_lambda: float = 0.3,
    ) -> List[Dict[str, Any]]:
        """
        Greedy selection maximizing: (1 - lambda) * score - lambda * D_KL.
        """
        if not candidate_items:
            return []

        selected: List[Dict[str, Any]] = []
        remaining = list(candidate_items)
        genre_accum = np.zeros_like(user_genre_distribution, dtype=np.float64)

        while len(selected) < k and remaining:
            best_item = None
            best_obj = float("-inf")

            for item in remaining:
                item_genre = item.get("genres_vector", np.zeros_like(user_genre_distribution))
                temp_accum = genre_accum + item_genre
                total_g = np.sum(temp_accum)
                p_rec = temp_accum / max(total_g, 1e-6)

                kl = cls.compute_kl_divergence(user_genre_distribution, p_rec)
                score = float(item.get("recommendation_score", 0.5))

                # Objective: balance high recommendation score with low KL divergence
                obj = (1.0 - tradeoff_lambda) * score - tradeoff_lambda * kl
                if obj > best_obj:
                    best_obj = obj
                    best_item = item

            if best_item is not None:
                selected.append(best_item)
                remaining.remove(best_item)
                genre_accum += best_item.get("genres_vector", np.zeros_like(user_genre_distribution))
            else:
                break

        return selected
