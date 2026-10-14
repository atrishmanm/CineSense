"""
Unit tests for AI Layer 3: Reinforcement Learning
Multi-Armed Bandit, UCB Bandit, Thompson Sampling, Contextual Bandit
"""

import numpy as np
import pytest
from ai.reinforcement import MultiArmedBandit, UCBBandit, ThompsonSampling, ContextualBandit


class TestMultiArmedBandit:
    def test_initialization(self):
        mab = MultiArmedBandit(exploration_rate=0.2)
        assert mab.exploration_rate == 0.2
        assert len(mab.arm_rewards) == 0

    def test_update_and_stats(self):
        mab = MultiArmedBandit(exploration_rate=0.0)
        mab.update(1, 1.0)
        mab.update(1, 0.5)
        stats = mab.get_arm_stats(1)
        assert stats["count"] == 2
        assert stats["total_reward"] == 1.5
        assert pytest.approx(stats["avg_reward"]) == 0.75

    def test_select_arm_exploitation(self):
        mab = MultiArmedBandit(exploration_rate=0.0)
        mab.update(1, 0.9)
        mab.update(2, 0.1)
        mab.update(3, 0.5)
        selected = mab.select_arm([1, 2, 3], top_k=2)
        assert selected == [1, 3]


class TestUCBBandit:
    def test_unexplored_arms_priority(self):
        ucb = UCBBandit(c=2.0)
        ucb.update(1, 1.0)
        # Arm 2 is unexplored -> should have inf UCB and be selected
        selected = ucb.select_arm([1, 2], top_k=1)
        assert selected == [2]

    def test_ucb_value_calculation(self):
        ucb = UCBBandit(c=1.0)
        ucb.update(1, 1.0)
        ucb.update(1, 1.0)
        val = ucb.ucb_value(1)
        # avg = 1.0, confidence = 1.0 * sqrt(ln(2) / 2)
        expected = 1.0 + np.sqrt(np.log(2) / 2)
        assert pytest.approx(val, rel=1e-4) == expected


class TestThompsonSampling:
    def test_thompson_sampling(self):
        ts = ThompsonSampling()
        # Arm 1 has high success
        for _ in range(20):
            ts.update(1, 1.0)
        # Arm 2 has high failure
        for _ in range(20):
            ts.update(2, 0.0)

        selected = ts.select_arm([1, 2], top_k=1)
        # Arm 1 should overwhelmingly be chosen
        assert selected == [1]


class TestContextualBandit:
    def test_linear_reward_prediction_and_update(self):
        cb = ContextualBandit(context_dim=3, learning_rate=0.1)
        context = np.array([1.0, 0.5, 0.0])
        initial_pred = cb.predict_reward(1, context)
        assert initial_pred == 0.0

        # Update with reward 1.0
        cb.update(1, context, reward=1.0)
        updated_pred = cb.predict_reward(1, context)
        assert updated_pred > 0.0
