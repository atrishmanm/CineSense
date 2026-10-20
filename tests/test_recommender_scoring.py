"""
Unit tests for CineSense recommender scoring pipeline:
- Score normalization across content, preference, ELO, and DL
- Component weights summing to 1.0
- Deterministic behavior with DL on / off
- Traceable explanation payload
- MMR diversity-aware reranking
"""

import numpy as np
import pytest
from unittest.mock import MagicMock, patch

from config import Config
from ai.recommender import CineSenseRecommender


class TestRecommenderScoring:
    @patch("ai.recommender.db")
    def test_score_normalization_and_weights_dl_off(self, mock_db):
        mock_db.get_top_movies.return_value = [
            {"movie_id": 10, "title": "Inception", "elo_score": 1600, "genres": "Action, Sci-Fi"},
            {"movie_id": 20, "title": "Interstellar", "elo_score": 1700, "genres": "Sci-Fi, Drama"},
        ]
        mock_db.get_user_interactions.return_value = [
            {"movie_1_id": 1, "movie_2_id": 2, "chosen_movie_id": 1}
        ]

        rec = CineSenseRecommender()
        rec.dl_scorer = None  # DL disabled

        # Mock user model and embedding
        mock_user_model = MagicMock()
        mock_user_model.has_enough_data.return_value = True
        mock_user_model.get_preference_score.return_value = 0.5  # raw logit
        rec._get_user_model = MagicMock(return_value=mock_user_model)

        mock_user_emb = MagicMock()
        mock_user_emb.get_embedding.return_value = np.ones(10)
        rec._get_user_embedding = MagicMock(return_value=mock_user_emb)
        rec._movie_to_embedding = MagicMock(return_value=np.ones(10))

        recs = rec.get_recommendations(user_id=1, n=2)

        assert len(recs) == 2
        for item in recs:
            # Scores must be bounded in [0, 1]
            assert 0.0 <= item["content_score"] <= 1.0
            assert 0.0 <= item["preference_score"] <= 1.0
            assert 0.0 <= item["elo_score"] <= 1.0
            assert 0.0 <= item["recommendation_score"] <= 1.0

            # Explanation payload must be present and traceable
            exp = item["explanation_payload"]
            assert "content_similarity_contribution" in exp
            assert "preference_elo_contribution" in exp
            assert "reason" in exp

            # Weights sum to 1.0: 0.50 * content + 0.30 * pref + 0.20 * elo
            expected = 0.50 * item["content_score"] + 0.30 * item["preference_score"] + 0.20 * item["elo_score"]
            assert pytest.approx(item["recommendation_score"], rel=1e-4) == expected

    @patch("ai.recommender.db")
    def test_score_normalization_and_weights_dl_on(self, mock_db):
        mock_db.get_top_movies.return_value = [
            {"movie_id": 10, "title": "Inception", "elo_score": 1600, "genres": "Action, Sci-Fi"},
        ]
        mock_db.get_user_interactions.return_value = [
            {"movie_1_id": 1, "movie_2_id": 2, "chosen_movie_id": 1}
        ]

        rec = CineSenseRecommender()

        # Mock DL scorer
        mock_dl = MagicMock()
        mock_dl.tmdb_genre_to_vec.return_value = np.zeros(18)
        mock_dl.genre_affinity_score.return_value = 0.85
        rec.dl_scorer = mock_dl

        mock_user_model = MagicMock()
        mock_user_model.has_enough_data.return_value = True
        mock_user_model.get_preference_score.return_value = 0.0
        rec._get_user_model = MagicMock(return_value=mock_user_model)

        mock_user_emb = MagicMock()
        mock_user_emb.get_embedding.return_value = np.ones(10)
        rec._get_user_embedding = MagicMock(return_value=mock_user_emb)
        rec._movie_to_embedding = MagicMock(return_value=np.ones(10))

        recs = rec.get_recommendations(user_id=1, n=1)
        assert len(recs) == 1
        item = recs[0]
        assert item["dl_genre_affinity"] == 0.85

        # De-tautologized (audit I1.2): previously these weights were hard
        # literals on BOTH sides of the comparison (0.85 * ... / 0.15), so
        # the assertion was pure arithmetic that could never fail and tested
        # nothing about the code. Now the weights are read from the LIVE
        # production config that ai/recommender.py:525-533 actually applies
        # (DL_SCORE_WEIGHT split across RECOMMEND_WEIGHT_*), so this fails
        # if the shipped weight mix is ever edited to not sum to 1.0.
        dl_w = float(getattr(Config, "DL_SCORE_WEIGHT", 0.15))
        w_content = (1.0 - dl_w) * Config.RECOMMEND_WEIGHT_CONTENT
        w_pref = (1.0 - dl_w) * Config.RECOMMEND_WEIGHT_PREFERENCE
        w_elo = (1.0 - dl_w) * Config.RECOMMEND_WEIGHT_ELO
        total_w = w_content + w_pref + w_elo + dl_w
        assert pytest.approx(total_w, abs=1e-9) == 1.0


class TestMMRDiversityReranking:
    def test_mmr_zero_lambda_preserves_order(self):
        rec = CineSenseRecommender()
        items = [
            {"movie_id": 1, "title": "A", "recommendation_score": 0.9, "genres": "Action"},
            {"movie_id": 2, "title": "B", "recommendation_score": 0.8, "genres": "Action"},
            {"movie_id": 3, "title": "C", "recommendation_score": 0.7, "genres": "Drama"},
        ]
        reranked = rec.apply_diversity_rerank(items, top_n=3, lambda_diversity=0.0)
        assert [m["movie_id"] for m in reranked] == [1, 2, 3]

    def test_mmr_diversity_penalizes_redundancy(self):
        rec = CineSenseRecommender()
        # Item 1 and Item 2 share identical genres (Action).
        # Item 3 is Drama. With high lambda_diversity, Item 3 should be preferred over Item 2
        items = [
            {"movie_id": 1, "title": "Action 1", "recommendation_score": 0.90, "genres": "Action"},
            {"movie_id": 2, "title": "Action 2", "recommendation_score": 0.88, "genres": "Action"},
            {"movie_id": 3, "title": "Drama 1", "recommendation_score": 0.82, "genres": "Drama"},
        ]
        reranked = rec.apply_diversity_rerank(items, top_n=2, lambda_diversity=0.7)
        # First item is Action 1 (highest score). Second item should be Drama 1 (dissimilar to Action 1)
        assert reranked[0]["movie_id"] == 1
        assert reranked[1]["movie_id"] == 3
