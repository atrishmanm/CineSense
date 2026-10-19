"""
Unit tests for AI Layer 1: Pairwise Preference Learning
PairwiseLearner, BradleyTerryModel, and UserPreferenceModel
"""

from ai.pairwise_learning import PairwiseLearner, BradleyTerryModel, UserPreferenceModel


class TestPairwiseLearner:
    def test_expected_score_equal(self):
        learner = PairwiseLearner(k_factor=32, initial_rating=1500)
        prob = learner.expected_score(1500, 1500)
        assert prob == 0.5

    def test_expected_score_higher_rated(self):
        learner = PairwiseLearner()
        # Item with 1700 ELO vs 1500 ELO should have expected score > 0.5
        prob = learner.expected_score(1700, 1500)
        assert prob > 0.7

    def test_update_ratings(self):
        learner = PairwiseLearner(k_factor=32, initial_rating=1500)
        new_winner, new_loser = learner.update_ratings(1500, 1500)
        # With equal ratings, expected = 0.5, change = 32 * (1 - 0.5) = 16
        assert new_winner == 1516
        assert new_loser == 1484

    def test_rank_items(self):
        learner = PairwiseLearner()
        items = [(1, 1500), (2, 1600), (3, 1400)]
        ranked = learner.rank_items(items)
        assert [it[0] for it in ranked] == [2, 1, 3]


class TestBradleyTerryModel:
    def test_bt_probability_and_update(self):
        bt = BradleyTerryModel(n_items=3, learning_rate=0.2)
        # Initially all items have equal strength = 1.0 -> P(0 beats 1) = 0.5
        assert bt.probability_i_beats_j(0, 1) == 0.5

        # Item 0 beats item 1 multiple times
        for _ in range(5):
            bt.update_from_comparison(0, 1)

        # Strength of 0 should increase, strength of 1 should decrease
        assert bt.strengths[0] > bt.strengths[1]
        assert bt.probability_i_beats_j(0, 1) > 0.5
        rankings = bt.get_rankings()
        assert rankings[0] == 0


class TestUserPreferenceModel:
    def test_preference_accumulation(self):
        tracker = UserPreferenceModel(user_id=1)
        tracker.record_preference(winner_id=10, loser_id=20, strength=1.0)
        assert tracker.get_preference_score(10) > 0
        assert tracker.get_preference_score(20) < 0
        assert tracker.comparison_count == 1
        assert not tracker.has_enough_data(min_comparisons=5)

        for _ in range(5):
            tracker.record_preference(winner_id=10, loser_id=30)
        assert tracker.has_enough_data(min_comparisons=5)
