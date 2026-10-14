"""
Unit tests for CineSense Comprehensive Metrics Suite
"""

import math
import numpy as np
import pytest

from benchmarks.comprehensive_metrics import (
    calculate_regression_metrics,
    calculate_classification_metrics,
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
    map_at_k,
    hit_rate_at_k,
    mrr_at_k,
    catalog_coverage,
    gini_coefficient,
    mean_novelty,
    intra_list_diversity,
    evaluate_full_suite,
)


def test_regression_metrics_perfect():
    y_true = [1.0, 2.0, 3.0, 4.0, 5.0]
    y_pred = [1.0, 2.0, 3.0, 4.0, 5.0]
    m = calculate_regression_metrics(y_true, y_pred)
    assert m["rmse"] == 0.0
    assert m["mae"] == 0.0
    assert m["r2"] == 1.0
    assert m["pearson_r"] == 1.0
    assert m["spearman_rho"] == 1.0


def test_regression_metrics_noisy():
    y_true = [1.0, 2.0, 3.0, 4.0, 5.0]
    y_pred = [1.5, 2.5, 2.8, 3.8, 4.9]
    m = calculate_regression_metrics(y_true, y_pred)
    assert m["rmse"] > 0.0
    assert m["r2"] > 0.8
    assert m["pearson_r"] > 0.95


def test_classification_metrics():
    y_true = [5.0, 4.0, 3.0, 2.0, 1.0, 4.5]
    y_pred = [4.8, 4.2, 2.5, 1.8, 3.5, 4.6]
    # Relevance threshold 4.0: true positives are item 0, 1, 5
    m = calculate_classification_metrics(y_true, y_pred, relevance_threshold=4.0)
    assert m["accuracy"] == 1.0
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1_score"] == 1.0
    assert m["roc_auc"] == 1.0
    assert m["confusion_matrix"]["true_positives"] == 3
    assert m["confusion_matrix"]["false_positives"] == 0


def test_ranking_metrics():
    rec = [10, 20, 30, 40, 50]
    rel = {10, 30, 99}

    assert precision_at_k(rec, rel, k=5) == 2 / 5
    assert recall_at_k(rec, rel, k=5) == 2 / 3
    assert hit_rate_at_k(rec, rel, k=5) == 1.0
    assert mrr_at_k(rec, rel, k=5) == 1.0  # First item (pos 0) is relevant -> 1/(0+1)=1.0
    assert ndcg_at_k(rec, rel, k=5) > 0.5
    assert map_at_k(rec, rel, k=5) > 0.5


def test_catalog_and_diversity():
    all_recs = {
        1: [101, 102, 103],
        2: [103, 104, 105],
    }
    cat_size = 10
    cov = catalog_coverage(all_recs, cat_size, k=3)
    assert cov == 5 / 10  # items 101, 102, 103, 104, 105

    gini = gini_coefficient(all_recs, catalog_size=5, k=3)
    assert 0.0 <= gini <= 1.0

    features = {
        101: np.array([1.0, 0.0]),
        102: np.array([0.0, 1.0]),
        103: np.array([1.0, 1.0]),
    }
    ild = intra_list_diversity({1: [101, 102]}, features, k=2)
    assert ild == pytest.approx(1.0, 0.01)  # orthogonal vectors have cosine distance 1.0
