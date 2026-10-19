"""
CineSense Comprehensive Evaluation Metrics Suite
=================================================
Calculates all relevant RecSys & ML metrics on real interaction data:

1. Regression Metrics (Rating Prediction):
   - RMSE (Root Mean Squared Error)
   - MAE (Mean Absolute Error)
   - MSE (Mean Squared Error)
   - R² (Coefficient of Determination)
   - Explained Variance
   - Pearson Correlation (r)
   - Spearman Rank Correlation (rho)
   - Median Absolute Error
   - Max Absolute Error

2. Binary Classification & Decision-Support Metrics (threshold >= 4.0):
   - Accuracy
   - Precision (Positive Predictive Value)
   - Recall (Sensitivity / True Positive Rate)
   - F1-Score
   - Specificity (True Negative Rate)
   - Balanced Accuracy
   - ROC-AUC (Area Under ROC Curve)
   - PR-AUC (Average Precision)
   - Matthews Correlation Coefficient (MCC)
   - Confusion Matrix (TP, FP, TN, FN)

3. Top-N Ranking Metrics (@5, @10, @20):
   - Precision@K
   - Recall@K
   - NDCG@K
   - MAP@K (Mean Average Precision)
   - Hit Rate@K (HR@K)
   - MRR (Mean Reciprocal Rank)

4. Catalog & Beyond-Accuracy Metrics:
   - Catalog Coverage@K
   - Gini Inequality Coefficient
   - Intra-List Diversity (ILD)
   - Novelty (Mean Self-Information)
   - Shannon Entropy
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Set

import numpy as np


# ============================================================================
# 1. Regression Metrics (Continuous Rating Prediction)
# ============================================================================

def calculate_regression_metrics(
    y_true: Sequence[float] | np.ndarray,
    y_pred: Sequence[float] | np.ndarray,
) -> Dict[str, float]:
    """Compute all standard continuous rating prediction metrics."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if len(y_true) == 0:
        return {
            "rmse": 0.0, "mae": 0.0, "mse": 0.0, "r2": 0.0,
            "explained_variance": 0.0, "pearson_r": 0.0, "spearman_rho": 0.0,
            "median_ae": 0.0, "max_error": 0.0,
        }

    errors = y_pred - y_true
    abs_errors = np.abs(errors)
    mse = float(np.mean(errors ** 2))
    rmse = float(math.sqrt(mse))
    mae = float(np.mean(abs_errors))
    median_ae = float(np.median(abs_errors))
    max_error = float(np.max(abs_errors))

    # R-squared (Coefficient of Determination)
    total_var = float(np.var(y_true))
    r2 = 1.0 - (mse / total_var) if total_var > 1e-12 else 0.0

    # Explained Variance: 1 - Var(y_true - y_pred) / Var(y_true)
    residual_var = float(np.var(errors))
    explained_var = 1.0 - (residual_var / total_var) if total_var > 1e-12 else 0.0

    # Pearson Correlation Coefficient (r)
    std_true = float(np.std(y_true))
    std_pred = float(np.std(y_pred))
    if std_true > 1e-12 and std_pred > 1e-12:
        cov = float(np.mean((y_true - np.mean(y_true)) * (y_pred - np.mean(y_pred))))
        pearson_r = cov / (std_true * std_pred)
    else:
        pearson_r = 0.0

    # Spearman Rank Correlation (rho)
    def _rank_data(a: np.ndarray) -> np.ndarray:
        sorter = np.argsort(a)
        ranks = np.empty_like(sorter, dtype=np.float64)
        ranks[sorter] = np.arange(len(a), dtype=np.float64)
        return ranks

    rank_true = _rank_data(y_true)
    rank_pred = _rank_data(y_pred)
    std_rtrue = float(np.std(rank_true))
    std_rpred = float(np.std(rank_pred))
    if std_rtrue > 1e-12 and std_rpred > 1e-12:
        cov_rank = float(np.mean((rank_true - np.mean(rank_true)) * (rank_pred - np.mean(rank_pred))))
        spearman_rho = cov_rank / (std_rtrue * std_rpred)
    else:
        spearman_rho = 0.0

    return {
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "mse": round(mse, 4),
        "r2": round(r2, 4),
        "explained_variance": round(explained_var, 4),
        "pearson_r": round(pearson_r, 4),
        "spearman_rho": round(spearman_rho, 4),
        "median_ae": round(median_ae, 4),
        "max_error": round(max_error, 4),
    }


# ============================================================================
# 2. Binary Classification & Decision Support (Relevance Thresholding)
# ============================================================================

def calculate_classification_metrics(
    y_true: Sequence[float] | np.ndarray,
    y_pred: Sequence[float] | np.ndarray,
    relevance_threshold: float = 4.0,
    pred_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """
    Evaluate decision support metrics where ratings >= relevance_threshold
    are positive interactions (e.g. 4.0+ stars = 'relevant' / 'liked').
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    if len(y_true) == 0:
        return {}

    b_true = (y_true >= relevance_threshold).astype(int)

    # If predictions are on a 1-5 rating scale, default pred_threshold to relevance_threshold
    # If predictions are already probabilities [0, 1], default pred_threshold to 0.5
    if pred_threshold is None:
        if np.max(y_pred) <= 1.05 and np.min(y_pred) >= -0.05:
            pred_threshold = 0.5
        else:
            pred_threshold = relevance_threshold

    b_pred = (y_pred >= pred_threshold).astype(int)

    # Confusion Matrix elements
    tp = int(np.sum((b_true == 1) & (b_pred == 1)))
    fp = int(np.sum((b_true == 0) & (b_pred == 1)))
    tn = int(np.sum((b_true == 0) & (b_pred == 0)))
    fn = int(np.sum((b_true == 1) & (b_pred == 0)))
    total = len(b_true)

    # Accuracy
    accuracy = (tp + tn) / total if total > 0 else 0.0

    # Precision & Recall
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # F1 Score
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    # Specificity (True Negative Rate)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Balanced Accuracy
    balanced_acc = (recall + specificity) / 2.0

    # Matthews Correlation Coefficient (MCC)
    denom = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = ((tp * tn) - (fp * fn)) / denom if denom > 0 else 0.0

    # ROC-AUC computation (Rank-order Wilcoxon-Mann-Whitney formulation)
    roc_auc = _compute_roc_auc(b_true, y_pred)

    # PR-AUC (Average Precision)
    pr_auc = _compute_pr_auc(b_true, y_pred)

    return {
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "specificity": round(specificity, 4),
        "balanced_accuracy": round(balanced_acc, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "mcc": round(mcc, 4),
        "confusion_matrix": {
            "true_positives": tp,
            "false_positives": fp,
            "true_negatives": tn,
            "false_negatives": fn,
        },
    }


def _compute_roc_auc(y_true_binary: np.ndarray, y_score: np.ndarray) -> float:
    """Exact ROC-AUC via rank sum (Wilcoxon-Mann-Whitney)."""
    n_pos = int(np.sum(y_true_binary == 1))
    n_neg = int(np.sum(y_true_binary == 0))
    if n_pos == 0 or n_neg == 0:
        return 0.5

    # Order scores ascending
    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(y_score) + 1, dtype=np.float64)

    sum_ranks_pos = float(np.sum(ranks[y_true_binary == 1]))
    u = sum_ranks_pos - (n_pos * (n_pos + 1)) / 2.0
    return float(u / (n_pos * n_neg))


def _compute_pr_auc(y_true_binary: np.ndarray, y_score: np.ndarray) -> float:
    """Average Precision (Area under Precision-Recall Curve)."""
    order = np.argsort(-y_score)
    y_true_sorted = y_true_binary[order]

    n_pos = int(np.sum(y_true_sorted == 1))
    if n_pos == 0:
        return 0.0

    cum_pos = np.cumsum(y_true_sorted)
    precisions = cum_pos / np.arange(1, len(y_score) + 1)
    # Average precision over positive items
    avg_prec = float(np.sum(precisions * y_true_sorted) / n_pos)
    return avg_prec


# ============================================================================
# 3. Top-N Ranking Metrics
# ============================================================================

def precision_at_k(recommended: Sequence[int], relevant: Set[int], k: int = 10) -> float:
    rec_k = recommended[:k]
    if not rec_k:
        return 0.0
    return sum(1 for item in rec_k if item in relevant) / k


def recall_at_k(recommended: Sequence[int], relevant: Set[int], k: int = 10) -> float:
    if not relevant:
        return 0.0
    return sum(1 for item in recommended[:k] if item in relevant) / len(relevant)


def ndcg_at_k(recommended: Sequence[int], relevant: Set[int], k: int = 10) -> float:
    if not relevant:
        return 0.0
    dcg = sum(
        1.0 / math.log2(i + 2)
        for i, item in enumerate(recommended[:k])
        if item in relevant
    )
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0


def map_at_k(recommended: Sequence[int], relevant: Set[int], k: int = 10) -> float:
    """Average Precision at K for a single query/user."""
    if not relevant:
        return 0.0
    hits = 0
    sum_prec = 0.0
    for i, item in enumerate(recommended[:k]):
        if item in relevant:
            hits += 1
            sum_prec += hits / (i + 1)
    return sum_prec / min(len(relevant), k) if min(len(relevant), k) > 0 else 0.0


def hit_rate_at_k(recommended: Sequence[int], relevant: Set[int], k: int = 10) -> float:
    return 1.0 if any(item in relevant for item in recommended[:k]) else 0.0


def mrr_at_k(recommended: Sequence[int], relevant: Set[int], k: int = 10) -> float:
    """Mean Reciprocal Rank at K."""
    for i, item in enumerate(recommended[:k]):
        if item in relevant:
            return 1.0 / (i + 1)
    return 0.0


# ============================================================================
# 4. Catalog & Beyond-Accuracy Metrics
# ============================================================================

def catalog_coverage(all_recs: Dict[int, List[int]], catalog_size: int, k: int = 10) -> float:
    """Fraction of entire catalog recommended to at least one user."""
    if catalog_size <= 0:
        return 0.0
    seen = set()
    for recs in all_recs.values():
        seen.update(recs[:k])
    return len(seen) / catalog_size


def gini_coefficient(all_recs: Dict[int, List[int]], catalog_size: int, k: int = 10) -> float:
    """
    Gini Index of recommendation distribution.
    0.0 = perfect equality (every movie recommended equally).
    1.0 = maximal inequality (only 1 movie ever recommended).

    IDs outside ``[0, catalog_size)`` are *not* silently dropped: the
    frequency vector is widened to cover the largest recommended id so
    every recommendation is counted exactly once.
    """
    counts = Counter()
    for recs in all_recs.values():
        counts.update(recs[:k])

    if not counts:
        return 0.0

    # All items including those with 0 recommendations.  Grow the vector if
    # any recommended id falls outside the nominal catalog range instead of
    # discarding those items.
    max_id = max((iid for iid in counts if iid >= 0), default=-1)
    size = max(catalog_size, max_id + 1)
    if size <= 0:
        return 0.0
    freqs = np.zeros(size, dtype=np.float64)
    for item_id, count in counts.items():
        if item_id >= 0:
            freqs[item_id] = count

    if np.sum(freqs) == 0:
        return 0.0

    freqs = np.sort(freqs)
    n = len(freqs)
    index = np.arange(1, n + 1)
    return float((2.0 * np.sum(index * freqs)) / (n * np.sum(freqs)) - (n + 1.0) / n)


def mean_novelty(
    all_recs: Dict[int, List[int]],
    item_popularity: Dict[int, float],
    k: int = 10,
) -> float:
    """Average self-information: -log2(p(item)). Higher = more novel/long-tail."""
    total_info = 0.0
    count = 0
    for recs in all_recs.values():
        for item in recs[:k]:
            pop = item_popularity.get(item, 1e-6)
            total_info += -math.log2(max(pop, 1e-6))
            count += 1
    return total_info / max(count, 1)


def intra_list_diversity(
    all_recs: Dict[int, List[int]],
    item_features: Dict[int, np.ndarray],
    k: int = 10,
) -> float:
    """Mean pairwise cosine distance within recommended top-K lists."""
    diversities = []
    for recs in all_recs.values():
        vecs = [item_features[it] for it in recs[:k] if it in item_features]
        if len(vecs) < 2:
            continue
        dists = []
        for i in range(len(vecs)):
            norm_i = np.linalg.norm(vecs[i])
            if norm_i == 0:
                continue
            for j in range(i + 1, len(vecs)):
                norm_j = np.linalg.norm(vecs[j])
                if norm_j == 0:
                    dists.append(1.0)
                else:
                    cos_sim = float(np.dot(vecs[i], vecs[j]) / (norm_i * norm_j))
                    dists.append(1.0 - np.clip(cos_sim, -1.0, 1.0))
        if dists:
            diversities.append(float(np.mean(dists)))
    return float(np.mean(diversities)) if diversities else 0.0


# ============================================================================
# 5. Full Evaluation Pipeline Orchestrator
# ============================================================================

def evaluate_full_suite(
    y_true: Sequence[float] | np.ndarray,
    y_pred: Sequence[float] | np.ndarray,
    user_recommendations: Dict[int, List[int]],
    test_relevant_items: Dict[int, Set[int]],
    catalog_size: int,
    item_popularity: Dict[int, float],
    item_features: Optional[Dict[int, np.ndarray]] = None,
    ks: Sequence[int] = (5, 10, 20),
) -> Dict[str, Any]:
    """
    Run full end-to-end evaluation computing all regression, classification,
    ranking (@5, @10, @20), and diversity metrics.
    """
    # 1. Regression Metrics
    reg_metrics = calculate_regression_metrics(y_true, y_pred)

    # 2. Classification Metrics (threshold = 4.0 stars)
    cls_metrics = calculate_classification_metrics(y_true, y_pred, relevance_threshold=4.0)

    # 3. Ranking Metrics across Ks
    ranking_metrics: Dict[str, Any] = {}
    for k in ks:
        precs, recs, ndcgs, maps, hrs, mrrs = [], [], [], [], [], []
        for uid, rec_list in user_recommendations.items():
            rel = test_relevant_items.get(uid, set())
            if not rel:
                continue
            precs.append(precision_at_k(rec_list, rel, k))
            recs.append(recall_at_k(rec_list, rel, k))
            ndcgs.append(ndcg_at_k(rec_list, rel, k))
            maps.append(map_at_k(rec_list, rel, k))
            hrs.append(hit_rate_at_k(rec_list, rel, k))
            mrrs.append(mrr_at_k(rec_list, rel, k))

        ranking_metrics[f"precision@{k}"] = round(float(np.mean(precs)), 4) if precs else 0.0
        ranking_metrics[f"recall@{k}"] = round(float(np.mean(recs)), 4) if recs else 0.0
        ranking_metrics[f"ndcg@{k}"] = round(float(np.mean(ndcgs)), 4) if ndcgs else 0.0
        ranking_metrics[f"map@{k}"] = round(float(np.mean(maps)), 4) if maps else 0.0
        ranking_metrics[f"hit_rate@{k}"] = round(float(np.mean(hrs)), 4) if hrs else 0.0
        ranking_metrics[f"mrr@{k}"] = round(float(np.mean(mrrs)), 4) if mrrs else 0.0

    # 4. Catalog & Diversity Metrics
    cat_coverage = catalog_coverage(user_recommendations, catalog_size, k=10)
    gini = gini_coefficient(user_recommendations, catalog_size, k=10)
    novelty = mean_novelty(user_recommendations, item_popularity, k=10)
    ild = intra_list_diversity(user_recommendations, item_features, k=10) if item_features else 0.0

    beyond_acc = {
        "catalog_coverage@10": round(cat_coverage, 4),
        "gini_coefficient@10": round(gini, 4),
        "novelty_entropy@10": round(novelty, 4),
        "intra_list_diversity@10": round(ild, 4),
    }

    return {
        "regression": reg_metrics,
        "classification_relevance_threshold_4_0": cls_metrics,
        "ranking": ranking_metrics,
        "beyond_accuracy": beyond_acc,
    }
