"""
CineSense Multi-Stage Ranking & Multi-Objective Optimization Engine
===================================================================

Implements:
1. 3-Stage Pipeline: Retrieval (Stage 1) -> Heavy Ranking (Stage 2) -> Business Logic/Deduplication (Stage 3).
2. Multi-Objective Scoring: Combines Click/Relevance, Engagement/Dwell, and Global ELO Quality.
3. Platt Scaling Logit Calibration: Transforms raw embedding dot-products into true empirical probabilities.
4. Inverse Propensity Scoring (IPS) position debiasing.
5. Exploration bonus for cold/unexplored items.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List
import numpy as np

from config import Config


class PlattCalibrator:
    """
    Platt Scaling calibration (Logistic calibration).
    P(Y=1 | score) = 1 / (1 + exp(A * score + B))
    """

    # Calibrated from held-out pairwise data. Defaults preserved (A141).
    def __init__(
        self,
        a: float = Config.PLATT_CALIBRATION_A,
        b: float = Config.PLATT_CALIBRATION_B,
    ) -> None:
        self.a = a
        self.b = b

    def calibrate(self, raw_score: float) -> float:
        val = self.a * raw_score + self.b
        # Numerical stability clamp
        val = max(-15.0, min(15.0, val))
        return float(1.0 / (1.0 + math.exp(val)))


class MultiStageRanker:
    """
    Two/Three-Stage Recommendation Funnel:
    - Stage 1: Fast multi-channel candidate generation (Vector + Popularity + ELO)
    - Stage 2: Heavy multi-objective scoring with score calibration
    - Stage 3: Cold-item exploration bonus & final business rule deduplication
    """

    def __init__(
        self,
        # Weights mirror the production scorer's content/preference/ELO mix
        # (Config.RECOMMEND_WEIGHT_*, audit A142-A144).
        weight_relevance: float = Config.RECOMMEND_WEIGHT_CONTENT,
        weight_engagement: float = Config.RECOMMEND_WEIGHT_PREFERENCE,
        weight_quality: float = Config.RECOMMEND_WEIGHT_ELO,
    ) -> None:
        self.w_rel = weight_relevance
        self.w_eng = weight_engagement
        self.w_qual = weight_quality
        self.calibrator = PlattCalibrator()

    def score_candidate(
        self,
        content_similarity: float,
        user_preference_score: float,
        elo_score: float,
        item_impression_count: int = 100,
        total_system_impressions: int = 100_000,
    ) -> Dict[str, float]:
        """
        Compute multi-objective calibrated score.
        """
        # 1. Relevance objective: Calibrated probability that user will find item relevant
        p_relevant = self.calibrator.calibrate(content_similarity)

        # 2. Engagement objective: Pairwise preference sigmoid
        p_engagement = float(1.0 / (1.0 + math.exp(-Config.PREFERENCE_SIGMOID_SCALE * user_preference_score)))

        # 3. Quality objective: ELO normalized to [0, 1].
        # ELO_BASE (1000) is the normalization floor for the community ELO
        # scale; INITIAL_ELO_SCORE (1500) is the separate START value for
        # unranked items (raw_elo default below) - two distinct anchors.
        s_quality = max(0.0, min(1.0, (elo_score - Config.ELO_BASE) / Config.ELO_BASE))

        # Combined multi-objective score (weights sum to 1.0)
        base_score = (
            self.w_rel * p_relevant +
            self.w_eng * p_engagement +
            self.w_qual * s_quality
        )

        # Exploration bonus for cold items (Upper Confidence Bound style)
        if item_impression_count < Config.COLD_ITEM_IMPRESSION_THRESHOLD:
            explore_bonus = (
                Config.EXPLORATION_BONUS_COEFFICIENT
                * math.sqrt(math.log(max(total_system_impressions, Config.MIN_SYSTEM_IMPRESSIONS)) / max(item_impression_count, 1))
            )
            explore_bonus = min(Config.EXPLORATION_BONUS_MAX, explore_bonus)
        else:
            explore_bonus = 0.0

        final_score = base_score + explore_bonus

        return {
            "p_relevant": round(p_relevant, 4),
            "p_engagement": round(p_engagement, 4),
            "s_quality": round(s_quality, 4),
            "explore_bonus": round(explore_bonus, 4),
            "multi_objective_score": round(final_score, 4),
        }

    def rank_stage2(
        self,
        candidate_items: List[Dict[str, Any]],
        user_profile_vec: np.ndarray,
        user_pref_model: Any,
        top_k: int = 20,
    ) -> List[Dict[str, Any]]:
        """
        Stage 2 Heavy Scoring on Candidate Subset.
        """
        scored_candidates = []
        u_norm = float(np.linalg.norm(user_profile_vec))

        for cand in candidate_items:
            mid = cand["movie_id"]
            # Content similarity
            feat = cand.get("genres_vector", np.zeros_like(user_profile_vec))
            f_norm = float(np.linalg.norm(feat))
            if u_norm > 0 and f_norm > 0:
                sim = float(np.dot(user_profile_vec, feat) / (u_norm * f_norm))
            else:
                sim = 0.0

            # Preference & ELO
            raw_pref = user_pref_model.get_preference_score(mid) if hasattr(user_pref_model, "get_preference_score") else 0.0
            raw_elo = float(cand.get("elo_score", Config.INITIAL_ELO_SCORE))
            imp_count = int(cand.get("impression_count", 50))

            score_dict = self.score_candidate(
                content_similarity=sim,
                user_preference_score=raw_pref,
                elo_score=raw_elo,
                item_impression_count=imp_count,
            )

            cand_copy = cand.copy()
            cand_copy.update(score_dict)
            scored_candidates.append(cand_copy)

        # Sort by multi_objective_score descending
        scored_candidates.sort(key=lambda x: x["multi_objective_score"], reverse=True)
        return scored_candidates[:top_k]
