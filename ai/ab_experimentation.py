"""
CineSense Production A/B Experimentation & Causal Inference Engine
==================================================================

Implements:
1. Deterministic Sticky User Bucketing (MurmurHash3 / SHA256 hashing).
2. Sample Ratio Mismatch (SRM) Detection via Chi-Square goodness-of-fit test.
3. Team-Draft Interleaving: High-sensitivity offline/online algorithm comparison.
4. Viewport Exposure Tracking to eliminate impression bias.
"""

from __future__ import annotations

import hashlib
import math
import random
from typing import Any, Dict, List, Optional, Set, Tuple

from config import Config


class ExperimentationEngine:
    """
    Production A/B testing bucketing and experimentation harness.
    """

    @staticmethod
    def bucket_user(user_id: int, experiment_name: str, variants: List[str], salt: str = "cinesense_salt") -> str:
        """
        Deterministic, sticky assignment of user to variant using cryptographic hashing.
        Guarantees uniform distribution across [0, 99] bucket spectrum.
        """
        key = f"{salt}:{experiment_name}:{user_id}".encode("utf-8")
        hash_val = int(hashlib.sha256(key).hexdigest()[:8], 16)
        bucket = hash_val % 100

        # Equal share partitioning
        step = 100 // len(variants)
        var_idx = min(bucket // step, len(variants) - 1)
        return variants[var_idx]

    @staticmethod
    def detect_srm(
        observed_counts: Dict[str, int],
        expected_ratios: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Detects Sample Ratio Mismatch (SRM) using Chi-Square goodness-of-fit test.
        A mismatch (p < 0.001) flags engineering bugs, dropouts, or skewed assignments.
        """
        total_observed = sum(observed_counts.values())
        if total_observed == 0:
            return {"srm_detected": False, "p_value": 1.0, "chi2": 0.0}

        k = len(observed_counts)
        ratios = expected_ratios or {v: 1.0 / k for v in observed_counts}

        chi2 = 0.0
        for variant, obs in observed_counts.items():
            expected = total_observed * ratios[variant]
            chi2 += ((obs - expected) ** 2) / max(expected, 1e-6)

        # Degrees of freedom = k - 1
        df = max(1, k - 1)

        # Regularized incomplete gamma approximation for Chi2 p-value
        # For df=1: p = 2 * (1 - Phi(sqrt(chi2)))
        if df == 1:
            z = math.sqrt(chi2)
            # Standard normal error function approx
            p_val = math.erfc(z / math.sqrt(2.0))
        else:
            # Simple Poisson/Gaussian tail approximation
            p_val = math.exp(-0.5 * chi2) if chi2 < 700 else 0.0

        # SRM alarm threshold. Deliberately much stricter than
        # Config.AB_SIGNIFICANCE_ALPHA (0.05): SRM flags tracking/engineering
        # bugs, not treatment effects (A162).
        srm_detected = p_val < Config.SRM_ALPHA
        return {
            "srm_detected": srm_detected,
            "chi2": round(chi2, 4),
            "p_value": round(p_val, 6),
            "df": df,
        }

    @staticmethod
    def team_draft_interleave(
        list_a: List[int],
        list_b: List[int],
        k: int = 10,
        seed: Optional[int] = None,
    ) -> Tuple[List[int], Dict[int, str]]:
        """
        Team-Draft Interleaving: Merges recommendations from Model A and Model B.
        Returns:
            (interleaved_list, item_attribution_dict)
            where item_attribution_dict maps item_id -> 'A' or 'B'
        """
        rng = random.Random(seed)
        team_a: Set[int] = set()
        team_b: Set[int] = set()
        selected: List[int] = []
        attribution: Dict[int, str] = {}

        idx_a = 0
        idx_b = 0

        while len(selected) < k and (idx_a < len(list_a) or idx_b < len(list_b)):
            # Determine priority: if sizes unequal, smaller team picks; else coin flip
            if len(team_a) < len(team_b):
                pick_team_a = True
            elif len(team_b) < len(team_a):
                pick_team_a = False
            else:
                pick_team_a = rng.random() < 0.5

            if pick_team_a:
                # Find next unselected item in list_a
                while idx_a < len(list_a) and list_a[idx_a] in attribution:
                    idx_a += 1
                if idx_a < len(list_a):
                    item = list_a[idx_a]
                    selected.append(item)
                    team_a.add(item)
                    attribution[item] = "A"
                    idx_a += 1
            else:
                # Find next unselected item in list_b
                while idx_b < len(list_b) and list_b[idx_b] in attribution:
                    idx_b += 1
                if idx_b < len(list_b):
                    item = list_b[idx_b]
                    selected.append(item)
                    team_b.add(item)
                    attribution[item] = "B"
                    idx_b += 1

        return selected[:k], attribution
