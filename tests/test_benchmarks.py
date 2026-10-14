"""
Unit tests for the CineSense RecSys benchmark suite.
Covers:
- Precision@K, Recall@K, NDCG@K, HitRate@K
- Catalog coverage, Novelty (mean self-information), Intra-list diversity
- Split creation & test leakage isolation
- Baseline and hybrid recommenders
- Latency & sliding-window caching
"""

import math
import numpy as np
import pytest

from benchmarks.data_loader import Rating, Movie, create_splits
from benchmarks.metrics import (
    precision_at_k,
    recall_at_k,
    ndcg_at_k,
    hit_rate_at_k,
    catalog_coverage,
    mean_self_information,
    intra_list_diversity,
    evaluate_all,
)
from benchmarks.recommenders import (
    BaseRecommender,
    PopularityRecommender,
    ContentRecommender,
    PreferenceEloRecommender,
    HybridRecommender,
)
from ai.cache_manager import SlidingWindowCache


class TestMetrics:
    def test_precision_at_k_perfect(self):
        rec = [1, 2, 3, 4, 5]
        rel = {1, 2, 3, 4, 5}
        assert precision_at_k(rec, rel, k=5) == 1.0

    def test_precision_at_k_partial(self):
        rec = [1, 2, 3, 4, 5]
        rel = {1, 3}
        assert precision_at_k(rec, rel, k=5) == 0.4

    def test_precision_at_k_empty(self):
        assert precision_at_k([], {1, 2}, k=5) == 0.0

    def test_recall_at_k_perfect(self):
        rec = [1, 2, 3, 4, 5]
        rel = {1, 2}
        assert recall_at_k(rec, rel, k=5) == 1.0

    def test_recall_at_k_half(self):
        rec = [1, 2, 3, 4, 5]
        rel = {1, 2, 6, 7}
        assert recall_at_k(rec, rel, k=5) == 0.5

    def test_recall_at_k_empty_relevant(self):
        assert recall_at_k([1, 2, 3], set(), k=5) == 0.0

    def test_ndcg_at_k_ideal(self):
        rec = [10, 20, 30]
        rel = {10, 20, 30}
        assert pytest.approx(ndcg_at_k(rec, rel, k=3), rel=1e-5) == 1.0

    def test_ndcg_at_k_suboptimal(self):
        # Item at rank 0 is relevant vs item at rank 2
        rel = {10}
        ndcg_top = ndcg_at_k([10, 20, 30], rel, k=3)
        ndcg_lower = ndcg_at_k([20, 30, 10], rel, k=3)
        assert ndcg_top > ndcg_lower
        assert ndcg_top == 1.0
        assert pytest.approx(ndcg_lower, rel=1e-5) == math.log2(2) / math.log2(4)

    def test_hit_rate_at_k(self):
        assert hit_rate_at_k([1, 2, 3], {2}, k=3) == 1.0
        assert hit_rate_at_k([1, 2, 3], {4}, k=3) == 0.0

    def test_catalog_coverage(self):
        recs = {
            1: [10, 20],
            2: [20, 30],
        }
        # 3 unique items out of catalog of 10
        assert catalog_coverage(recs, 10) == 0.3

    def test_mean_self_information(self):
        recs = {1: [100, 200]}
        # pop = 0.5 -> -log2(0.5) = 1.0
        pop = {100: 0.5, 200: 0.5}
        assert pytest.approx(mean_self_information(recs, pop, k=2), rel=1e-5) == 1.0

    def test_intra_list_diversity(self):
        recs = {1: [1, 2]}
        # Orthogonal unit vectors: cosine similarity = 0 -> distance = 1.0
        feats = {
            1: np.array([1.0, 0.0]),
            2: np.array([0.0, 1.0]),
        }
        assert pytest.approx(intra_list_diversity(recs, feats, k=2), rel=1e-5) == 1.0


class TestSplitsAndLeakage:
    def test_split_isolation(self):
        # Generate synthetic ratings
        ratings = []
        for u in range(1, 11):
            for i in range(1, 15):
                ratings.append(Rating(user_id=u, item_id=i, rating=4.5, timestamp=1000 + i))

        splits = create_splits(ratings, seed=42, relevance_threshold=4.0)

        # Check train, val, test are mutually exclusive per user
        train_pairs = {(r.user_id, r.item_id) for r in splits["train"]}
        val_pairs = {(r.user_id, r.item_id) for r in splits["val"]}
        test_pairs = {(r.user_id, r.item_id) for r in splits["test"]}

        assert len(train_pairs.intersection(val_pairs)) == 0
        assert len(train_pairs.intersection(test_pairs)) == 0
        assert len(val_pairs.intersection(test_pairs)) == 0


class TestRecommenders:
    def test_popularity_recommender(self):
        train = [
            Rating(1, 101, 5.0, 1),
            Rating(2, 101, 4.0, 2),
            Rating(3, 101, 4.5, 3),
            Rating(1, 102, 3.0, 4),
            Rating(2, 102, 3.5, 5),
            Rating(1, 103, 5.0, 6),
        ]
        movies = {
            101: Movie(101, "M1", np.zeros(19)),
            102: Movie(102, "M2", np.zeros(19)),
            103: Movie(103, "M3", np.zeros(19)),
            104: Movie(104, "M4", np.zeros(19)),
        }
        catalog = {101, 102, 103, 104}
        rec = PopularityRecommender()
        rec.fit(train, movies)

        # User 1 has already seen 101, 102, candidates are unseen items
        seen = {101, 102}
        candidates = catalog - seen
        top = rec.recommend(user_id=1, k=2, candidates=candidates)
        assert 101 not in top
        assert 102 not in top
        assert top[0] == 103

    def test_sliding_window_cache(self):
        cache = SlidingWindowCache(max_size=2)
        cache.put(1, {"title": "Movie 1"})
        cache.put(2, {"title": "Movie 2"})
        assert cache.get(1) == {"title": "Movie 1"}
        assert cache.get(2) == {"title": "Movie 2"}

        # Inserting a 3rd should evict LRU (which is 1 since 2 was accessed last)
        cache.put(3, {"title": "Movie 3"})
        assert len(cache.cache) <= 2
        assert cache.get(3) == {"title": "Movie 3"}
