"""
Standard RecSys evaluation metrics.

Every function operates on *one user* unless explicitly stated.  The
``evaluate_all`` helper aggregates across a dict of per-user
recommendation lists.

Metrics implemented
-------------------
* Precision@K
* Recall@K
* NDCG@K  (binary relevance)
* Hit Rate@K
* Catalog Coverage  (across all users)
* Mean Self-Information  (novelty -- inverse popularity)
* Intra-List Diversity  (1 - mean pairwise cosine similarity)
"""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, List, Set

import numpy as np


# -- Per-user metrics ------------------------------------------------

def precision_at_k(recommended: List[int], relevant: Set[int], k: int = 10) -> float:
    """Fraction of the top-*k* items that are relevant."""
    rec_k = recommended[:k]
    if not rec_k:
        return 0.0
    return sum(1 for item in rec_k if item in relevant) / k


def recall_at_k(recommended: List[int], relevant: Set[int], k: int = 10) -> float:
    """Fraction of relevant items that appear in the top-*k*."""
    if not relevant:
        return 0.0
    return sum(1 for item in recommended[:k] if item in relevant) / len(relevant)


def ndcg_at_k(recommended: List[int], relevant: Set[int], k: int = 10) -> float:
    """Normalised Discounted Cumulative Gain at *k* (binary relevance)."""
    if not relevant:
        return 0.0

    dcg = sum(
        1.0 / math.log2(i + 2)          # position is 1-indexed -> log(i+2)
        for i, item in enumerate(recommended[:k])
        if item in relevant
    )
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))

    return dcg / idcg if idcg > 0 else 0.0


def hit_rate_at_k(recommended: List[int], relevant: Set[int], k: int = 10) -> float:
    """1 if *any* relevant item appears in the top-*k*, else 0."""
    return 1.0 if any(item in relevant for item in recommended[:k]) else 0.0


# -- Aggregate metrics -----------------------------------------------

def catalog_coverage(
    all_recommendations: Dict[int, List[int]],
    catalog_size: int,
) -> float:
    """Fraction of the catalog recommended to at least one user."""
    if catalog_size <= 0:
        return 0.0
    seen: Set[int] = set()
    for recs in all_recommendations.values():
        seen.update(recs)
    return len(seen) / catalog_size


def mean_self_information(
    recommendations: Dict[int, List[int]],
    item_popularity: Dict[int, float],
    k: int = 10,
) -> float:
    """Average self-information (novelty).

    ``-log(pop)`` per recommended item, averaged.  Higher -> more novel.
    """
    total, count = 0.0, 0
    for recs in recommendations.values():
        for item in recs[:k]:
            pop = item_popularity.get(item, 1e-10)
            total += -math.log2(max(pop, 1e-10))
            count += 1
    return total / max(count, 1)


def intra_list_diversity(
    recommendations: Dict[int, List[int]],
    item_features: Dict[int, np.ndarray],
    k: int = 10,
) -> float:
    """Mean pairwise cosine *distance* within recommendation lists.

    ILD = 1 - cosine_sim  averaged over all user-lists.
    """
    diversities: list[float] = []
    for recs in recommendations.values():
        vecs = [item_features[it] for it in recs[:k] if it in item_features]
        if len(vecs) < 2:
            continue
        dists: list[float] = []
        for i in range(len(vecs)):
            ni = np.linalg.norm(vecs[i])
            if ni == 0:
                continue
            for j in range(i + 1, len(vecs)):
                nj = np.linalg.norm(vecs[j])
                if nj == 0:
                    dists.append(1.0)
                else:
                    dists.append(1.0 - float(np.dot(vecs[i], vecs[j]) / (ni * nj)))
        if dists:
            diversities.append(float(np.mean(dists)))
    return float(np.mean(diversities)) if diversities else 0.0


def paired_bootstrap_ci(
    vals_a: List[float],
    vals_b: List[float],
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Compute paired bootstrap 95% confidence intervals for the difference (A - B).
    Samples users with replacement to estimate whether delta is statistically significant.
    """
    assert len(vals_a) == len(vals_b), "Arrays must be paired across the same users"
    n = len(vals_a)
    if n == 0:
        return {"mean_delta": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "significant": False}

    diffs = np.array(vals_a, dtype=np.float64) - np.array(vals_b, dtype=np.float64)
    observed_mean = float(np.mean(diffs))

    rng = np.random.default_rng(seed)
    boot_means = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_means[b] = np.mean(diffs[idx])

    ci_lower = float(np.percentile(boot_means, 100 * (alpha / 2)))
    ci_upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    # Significant if 0 is not in [ci_lower, ci_upper]
    significant = bool(ci_lower > 0.0 or ci_upper < 0.0)

    return {
        "mean_delta": observed_mean,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "significant": significant,
    }


# -- Convenience wrapper ---------------------------------------------

def evaluate_all(
    recommendations: Dict[int, List[int]],
    test_relevant: Dict[int, Set[int]],
    catalog_size: int,
    item_popularity: Dict[int, float],
    item_features: Dict[int, np.ndarray],
    k: int = 10,
    return_per_user: bool = False,
) -> Dict[str, Any]:
    """Compute every metric in one pass and return a flat dict."""

    uids, precs, recs, ndcgs, hits = [], [], [], [], []
    for uid in sorted(recommendations.keys()):
        rec_list = recommendations[uid]
        rel = test_relevant.get(uid, set())
        if not rel:
            continue
        uids.append(uid)
        precs.append(precision_at_k(rec_list, rel, k))
        recs.append(recall_at_k(rec_list, rel, k))
        ndcgs.append(ndcg_at_k(rec_list, rel, k))
        hits.append(hit_rate_at_k(rec_list, rel, k))

    n = len(precs)
    res = {
        "n_eval_users": n,
        f"precision@{k}": float(np.mean(precs)) if precs else 0.0,
        f"recall@{k}": float(np.mean(recs)) if recs else 0.0,
        f"ndcg@{k}": float(np.mean(ndcgs)) if ndcgs else 0.0,
        f"hit_rate@{k}": float(np.mean(hits)) if hits else 0.0,
        "catalog_coverage": catalog_coverage(recommendations, catalog_size),
        "novelty": mean_self_information(recommendations, item_popularity, k),
        "diversity": intra_list_diversity(recommendations, item_features, k),
    }
    if return_per_user:
        res["per_user"] = {
            "uids": uids,
            "precision": precs,
            "recall": recs,
            "ndcg": ndcgs,
            "hit_rate": hits,
        }
    return res
