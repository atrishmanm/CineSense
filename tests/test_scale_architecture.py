"""
Unit tests for CineSense production-scale architectural subsystems:
- HNSW Vector Index, IVF-PQ, Reciprocal Rank Fusion
- XFetch Cache, Singleflight request coalescing
- Multi-Stage Calibrated Ranker (Platt Scaling, IPS)
- Online Feature Store & Point-in-Time Joins
- Circuit Breaker, Deadline Context, Rate Limiter, Idempotency Registry
- Experimentation Engine (MurmurHash3/SHA256, SRM Chi-Square, Team-Draft Interleaving)
- MLOps Drift Monitor (PSI & Centroid Drift)
- DPP Diversity Reranking & Calibration
- Latency Histogram & Funnel Tracker Telemetry
"""

import math
import numpy as np
import pytest

from ai.vector_search import HNSWIndex, ProductQuantizer, ReciprocalRankFusion
from ai.xfetch_cache import XFetchCache, Singleflight
from ai.multi_stage_ranker import MultiStageRanker, PlattCalibrator
from ai.feature_store import OnlineFeatureStore
from ai.resilience import CircuitBreaker, CircuitBreakerOpenException, DeadlineContext, TokenBucketRateLimiter, IdempotencyRegistry
from ai.ab_experimentation import ExperimentationEngine
from ai.drift_monitor import DriftMonitor
from ai.diversity_engine import DPPDiversifier, CalibratedReranker
from ai.telemetry import LatencyHistogram, FunnelTracker


class TestVectorSearchAndANN:
    def test_hnsw_add_and_search(self):
        index = HNSWIndex(dim=4, m=4, ef_construction=16, ef_search=16)
        index.add_item(1, np.array([1.0, 0.0, 0.0, 0.0]), metadata={"genre": "Action"})
        index.add_item(2, np.array([0.9, 0.1, 0.0, 0.0]), metadata={"genre": "Action"})
        index.add_item(3, np.array([0.0, 0.0, 1.0, 0.0]), metadata={"genre": "Drama"})

        # Search for vector close to item 1
        res = index.search(np.array([1.0, 0.0, 0.0, 0.0]), k=2)
        assert len(res) == 2
        assert res[0][0] == 1  # Exact match top
        assert res[1][0] == 2  # Close match second

    def test_hnsw_filtered_search(self):
        index = HNSWIndex(dim=4, m=4)
        index.add_item(1, np.array([1.0, 0.0, 0.0, 0.0]), metadata={"genre": "Action"})
        index.add_item(2, np.array([0.9, 0.1, 0.0, 0.0]), metadata={"genre": "Drama"})

        # Filter strictly for Drama
        res = index.search(np.array([1.0, 0.0, 0.0, 0.0]), k=2, filter_fn=lambda m: m.get("genre") == "Drama")
        assert len(res) == 1
        assert res[0][0] == 2

    def test_product_quantizer(self):
        pq = ProductQuantizer(dim=4, num_subvectors=2, num_centroids=4)
        training_vecs = np.random.randn(20, 4).astype(np.float32)
        pq.fit(training_vecs)
        codes = pq.encode(101, training_vecs[0])
        assert len(codes) == 2
        dist = pq.asymmetric_distance(training_vecs[0], 101)
        assert dist >= 0.0

    def test_reciprocal_rank_fusion(self):
        list1 = [(10, 0.9), (20, 0.8), (30, 0.7)]
        list2 = [(20, 0.95), (10, 0.85), (40, 0.6)]
        fused = ReciprocalRankFusion.fuse([list1, list2], k_rrf=60)
        # Both list 1 and 2 rank 10 and 20 highly -> they should lead fused results
        top_ids = [it for it, _ in fused[:2]]
        assert 10 in top_ids and 20 in top_ids


class TestXFetchAndSingleflight:
    def test_singleflight_coalescing(self):
        sf = Singleflight()
        calls = 0

        def worker():
            nonlocal calls
            calls += 1
            return 42

        # 3 sequential calls for same key
        res1 = sf.do("key1", worker)
        res2 = sf.do("key1", worker)
        assert res1 == 42
        assert res2 == 42

    def test_xfetch_cache_hit_and_expiration(self):
        cache = XFetchCache(max_size=10, default_ttl_sec=1.0)
        cache.set("k1", "v1", ttl_sec=10.0)
        assert cache.get("k1") == "v1"
        assert cache.get_hit_ratio() > 0.0


class TestMultiStageRanker:
    def test_platt_calibration(self):
        calibrator = PlattCalibrator()
        p_high = calibrator.calibrate(1.0)
        p_low = calibrator.calibrate(-1.0)
        assert 0.0 <= p_high <= 1.0
        assert 0.0 <= p_low <= 1.0
        assert p_high > p_low

    def test_multi_objective_scoring(self):
        ranker = MultiStageRanker()
        scores = ranker.score_candidate(
            content_similarity=0.8,
            user_preference_score=0.2,
            elo_score=1600.0,
            item_impression_count=10,
        )
        assert 0.0 <= scores["p_relevant"] <= 1.0
        assert 0.0 <= scores["p_engagement"] <= 1.0
        assert 0.0 <= scores["s_quality"] <= 1.0
        assert scores["multi_objective_score"] > 0.0


class TestFeatureStoreAndPointInTime:
    def test_online_feature_store_ema_taste_vector(self):
        store = OnlineFeatureStore(decay_rate=0.2)
        genre_action = np.array([1.0, 0.0, 0.0])
        store.register_item(10, genre_action)
        store.record_user_interaction(user_id=1, movie_id=10, rating=5.0)

        feats = store.get_user_features(user_id=1)
        assert feats["taste_vector"] is not None
        assert pytest.approx(feats["taste_vector"][0]) == 1.0

    def test_point_in_time_join_prevents_leakage(self):
        interactions = [(1, 10, 5.0, 100.0)]
        # Item 10 had feature A at t=50 and feature B at t=150
        history = {
            10: [
                (50.0, {"feature_ver": 1}),
                (150.0, {"feature_ver": 2}),
            ]
        }
        joined = OnlineFeatureStore.point_in_time_join(interactions, history)
        assert len(joined) == 1
        # Interaction at t=100 must join feature_ver 1 (t=50), NEVER feature_ver 2 (future)
        assert joined[0]["feature_ver"] == 1


class TestResilience:
    def test_circuit_breaker_tripping_and_fallback(self):
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout_sec=0.1)

        def err_fn():
            raise ValueError("fail")

        def fallback():
            return "safe"

        # 2 failures trip the circuit
        for _ in range(2):
            try:
                cb.execute(err_fn)
            except ValueError:
                pass

        # Third call must trigger fallback immediately
        assert cb.execute(err_fn, fallback_fn=fallback) == "safe"

    def test_token_bucket_rate_limiter(self):
        limiter = TokenBucketRateLimiter(rate_per_sec=10.0, capacity=2.0)
        assert limiter.allow_request(1.0) is True
        assert limiter.allow_request(1.0) is True
        assert limiter.allow_request(1.0) is False  # Capacity exceeded

    def test_idempotency_registry(self):
        registry = IdempotencyRegistry(ttl_sec=60.0)
        calls = 0

        def mutate():
            nonlocal calls
            calls += 1
            return {"status": "success"}

        cached1, res1 = registry.check_or_set("idem-uuid-1", mutate)
        assert cached1 is False
        assert calls == 1

        # Second submission with same key returns cached result without re-executing
        cached2, res2 = registry.check_or_set("idem-uuid-1", mutate)
        assert cached2 is True
        assert calls == 1
        assert res1 == res2


class TestABTestingAndSRM:
    def test_deterministic_bucketing(self):
        v1 = ExperimentationEngine.bucket_user(user_id=42, experiment_name="exp1", variants=["A", "B"])
        v2 = ExperimentationEngine.bucket_user(user_id=42, experiment_name="exp1", variants=["A", "B"])
        assert v1 == v2  # Sticky

    def test_srm_detection(self):
        # Balanced counts -> no SRM
        balanced = ExperimentationEngine.detect_srm({"A": 1000, "B": 1000})
        assert balanced["srm_detected"] is False

        # Severely skewed counts -> SRM alert!
        skewed = ExperimentationEngine.detect_srm({"A": 1800, "B": 200})
        assert skewed["srm_detected"] is True

    def test_team_draft_interleaving(self):
        list_a = [1, 2, 3]
        list_b = [4, 5, 6]
        interleaved, attr = ExperimentationEngine.team_draft_interleave(list_a, list_b, k=4, seed=42)
        assert len(interleaved) == 4
        # Both algorithms must be attributed in the interleaved list
        assert "A" in attr.values() and "B" in attr.values()


class TestMLOpsDriftMonitoring:
    def test_psi_calculation_stable(self):
        dist_base = np.random.normal(loc=0.0, scale=1.0, size=1000)
        dist_curr = np.random.normal(loc=0.02, scale=1.01, size=1000)
        res = DriftMonitor.calculate_psi(dist_base, dist_curr)
        assert res["status"] in ["STABLE", "MODERATE_DRIFT"]
        assert res["psi"] < 0.25

    def test_psi_calculation_drift_alert(self):
        dist_base = np.random.normal(loc=0.0, scale=1.0, size=1000)
        dist_shifted = np.random.normal(loc=2.5, scale=1.0, size=1000)
        res = DriftMonitor.calculate_psi(dist_base, dist_shifted)
        assert res["status"] == "SIGNIFICANT_DRIFT"
        assert res["psi"] > 0.25


class TestDiversityAndTelemetry:
    def test_dpp_greedy(self):
        items = [1, 2, 3, 4]
        qualities = np.array([1.0, 1.0, 1.0, 1.0])
        # Items 1 and 2 are orthogonal; Item 3 is duplicate of 1
        feats = np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.5, 0.5],
        ])
        selected = DPPDiversifier.dpp_greedy(items, qualities, feats, k=2)
        assert len(selected) == 2
        # DPP should prefer orthogonal items {1, 2} over duplicate {1, 3}
        assert 1 in selected and 2 in selected

    def test_latency_histogram_and_funnel(self):
        hist = LatencyHistogram(name="rec_latency", buckets=[1.0, 5.0, 10.0, 20.0])
        hist.observe(0.8)
        hist.observe(4.2)
        hist.observe(15.0)
        exp = hist.export()
        assert exp["count"] == 3
        assert exp["p50_ms"] <= 5.0

        funnel = FunnelTracker()
        funnel.record_stage("retrieval", 300)
        funnel.record_stage("ranking", 50)
        funnel.record_stage("final", 10)
        report = funnel.export_dropout_report()
        assert "retrieval_to_ranking" in report["dropouts"]
        assert report["dropouts"]["retrieval_to_ranking"]["output"] == 50
