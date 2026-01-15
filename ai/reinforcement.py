"""
AI Layer 3: Reinforcement Learning
Multi-Armed Bandit for exploration-exploitation balance
"""

import numpy as np
from config import Config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MultiArmedBandit:
    """
    Multi-Armed Bandit algorithm for recommendation
    Balances exploration (trying new movies) with exploitation (showing good movies)
    """
    
    def __init__(self, exploration_rate=0.2):
        """
        Initialize bandit
        
        Args:
            exploration_rate: Probability of exploring (0 to 1)
        """
        self.exploration_rate = exploration_rate
        self.arm_rewards = {}  # arm_id -> list of rewards
        self.arm_counts = {}   # arm_id -> selection count
    
    def select_arm(self, available_arms, top_k=2):
        """
        Select k arms (movies) to show user
        Uses epsilon-greedy strategy
        
        Args:
            available_arms: List of arm IDs (movie IDs)
            top_k: Number of arms to select
        
        Returns:
            List of selected arm IDs
        """
        if not available_arms:
            return []
        
        # Ensure we don't select more than available
        top_k = min(top_k, len(available_arms))
        
        # Decide: explore or exploit
        if np.random.random() < self.exploration_rate:
            # EXPLORE: Random selection
            selected = np.random.choice(available_arms, size=top_k, replace=False)
            return list(selected)
        else:
            # EXPLOIT: Select best performing arms
            arm_values = []
            for arm in available_arms:
                if arm in self.arm_rewards and len(self.arm_rewards[arm]) > 0:
                    # Use average reward
                    avg_reward = np.mean(self.arm_rewards[arm])
                else:
                    # Optimistic initial value for unexplored arms
                    avg_reward = 1.0
                
                arm_values.append((arm, avg_reward))
            
            # Sort by value and select top k
            arm_values.sort(key=lambda x: x[1], reverse=True)
            selected = [arm for arm, _ in arm_values[:top_k]]
            
            return selected
    
    def update(self, arm_id, reward):
        """
        Update arm statistics after receiving reward
        
        Args:
            arm_id: ID of selected arm
            reward: Reward received (1 for chosen, 0 for rejected)
        """
        if arm_id not in self.arm_rewards:
            self.arm_rewards[arm_id] = []
            self.arm_counts[arm_id] = 0
        
        self.arm_rewards[arm_id].append(reward)
        self.arm_counts[arm_id] += 1
    
    def get_arm_stats(self, arm_id):
        """
        Get statistics for an arm
        
        Args:
            arm_id: Arm identifier
        
        Returns:
            Dictionary with count, total_reward, avg_reward
        """
        if arm_id not in self.arm_rewards:
            return {'count': 0, 'total_reward': 0, 'avg_reward': 0}
        
        rewards = self.arm_rewards[arm_id]
        return {
            'count': self.arm_counts[arm_id],
            'total_reward': sum(rewards),
            'avg_reward': np.mean(rewards) if rewards else 0
        }


class UCBBandit:
    """
    Upper Confidence Bound (UCB) bandit
    More sophisticated than epsilon-greedy
    Automatically balances exploration and exploitation
    """
    
    def __init__(self, c=2.0):
        """
        Initialize UCB bandit
        
        Args:
            c: Exploration parameter (higher = more exploration)
        """
        self.c = c
        self.arm_counts = {}     # arm_id -> selection count
        self.arm_rewards = {}    # arm_id -> total reward
        self.total_counts = 0    # Total selections made
    
    def ucb_value(self, arm_id):
        """
        Calculate UCB value for an arm
        
        UCB = average_reward + c * sqrt(ln(total_selections) / arm_selections)
        
        Args:
            arm_id: Arm identifier
        
        Returns:
            UCB value (higher = should select this arm)
        """
        if arm_id not in self.arm_counts or self.arm_counts[arm_id] == 0:
            # Unexplored arms have infinite UCB (optimistic initialization)
            return float('inf')
        
        # Average reward
        avg_reward = self.arm_rewards[arm_id] / self.arm_counts[arm_id]
        
        # Confidence bonus
        confidence = self.c * np.sqrt(
            np.log(self.total_counts) / self.arm_counts[arm_id]
        )
        
        return avg_reward + confidence
    
    def select_arm(self, available_arms, top_k=2):
        """
        Select k arms with highest UCB values
        
        Args:
            available_arms: List of arm IDs
            top_k: Number of arms to select
        
        Returns:
            List of selected arm IDs
        """
        if not available_arms:
            return []
        
        top_k = min(top_k, len(available_arms))
        
        # Calculate UCB for all arms
        arm_ucbs = [(arm, self.ucb_value(arm)) for arm in available_arms]
        
        # Sort by UCB and select top k
        arm_ucbs.sort(key=lambda x: x[1], reverse=True)
        selected = [arm for arm, _ in arm_ucbs[:top_k]]
        
        return selected
    
    def update(self, arm_id, reward):
        """
        Update arm statistics
        
        Args:
            arm_id: ID of selected arm
            reward: Reward received (0 to 1)
        """
        if arm_id not in self.arm_counts:
            self.arm_counts[arm_id] = 0
            self.arm_rewards[arm_id] = 0.0
        
        self.arm_counts[arm_id] += 1
        self.arm_rewards[arm_id] += reward
        self.total_counts += 1
    
    def get_arm_stats(self, arm_id):
        """Get statistics for an arm"""
        if arm_id not in self.arm_counts:
            return {'count': 0, 'total_reward': 0, 'avg_reward': 0, 'ucb': float('inf')}
        
        count = self.arm_counts[arm_id]
        total = self.arm_rewards[arm_id]
        
        return {
            'count': count,
            'total_reward': total,
            'avg_reward': total / count if count > 0 else 0,
            'ucb': self.ucb_value(arm_id)
        }


class ThompsonSampling:
    """
    Thompson Sampling bandit
    Bayesian approach to exploration-exploitation
    """
    
    def __init__(self, alpha=1.0, beta=1.0):
        """
        Initialize Thompson Sampling
        
        Args:
            alpha: Prior successes (Beta distribution)
            beta: Prior failures (Beta distribution)
        """
        self.alpha_prior = alpha
        self.beta_prior = beta
        self.arm_successes = {}  # arm_id -> success count
        self.arm_failures = {}   # arm_id -> failure count
    
    def select_arm(self, available_arms, top_k=2):
        """
        Select arms using Thompson Sampling
        Samples from Beta distribution for each arm
        
        Args:
            available_arms: List of arm IDs
            top_k: Number of arms to select
        
        Returns:
            List of selected arm IDs
        """
        if not available_arms:
            return []
        
        top_k = min(top_k, len(available_arms))
        
        # Sample from Beta distribution for each arm
        arm_samples = []
        for arm in available_arms:
            # Get parameters
            successes = self.arm_successes.get(arm, 0) + self.alpha_prior
            failures = self.arm_failures.get(arm, 0) + self.beta_prior
            
            # Sample from Beta(successes, failures)
            sample = np.random.beta(successes, failures)
            arm_samples.append((arm, sample))
        
        # Select top k by sampled value
        arm_samples.sort(key=lambda x: x[1], reverse=True)
        selected = [arm for arm, _ in arm_samples[:top_k]]
        
        return selected
    
    def update(self, arm_id, reward):
        """
        Update arm statistics
        
        Args:
            arm_id: ID of selected arm
            reward: Reward (1 for success, 0 for failure)
        """
        if arm_id not in self.arm_successes:
            self.arm_successes[arm_id] = 0
            self.arm_failures[arm_id] = 0
        
        if reward > 0.5:  # Consider reward > 0.5 as success
            self.arm_successes[arm_id] += 1
        else:
            self.arm_failures[arm_id] += 1
    
    def get_arm_stats(self, arm_id):
        """Get statistics for an arm"""
        if arm_id not in self.arm_successes:
            return {'successes': 0, 'failures': 0, 'win_rate': 0}
        
        successes = self.arm_successes[arm_id]
        failures = self.arm_failures[arm_id]
        total = successes + failures
        
        return {
            'successes': successes,
            'failures': failures,
            'win_rate': successes / total if total > 0 else 0
        }


class ContextualBandit:
    """
    Contextual bandit that considers user state
    Uses linear model: reward = user_context · arm_features
    """
    
    def __init__(self, context_dim, learning_rate=0.1):
        """
        Initialize contextual bandit
        
        Args:
            context_dim: Dimension of context vector
            learning_rate: Learning rate for updates
        """
        self.context_dim = context_dim
        self.learning_rate = learning_rate
        self.arm_weights = {}  # arm_id -> weight vector
    
    def predict_reward(self, arm_id, context):
        """
        Predict expected reward for arm given context
        
        Args:
            arm_id: Arm identifier
            context: Context vector (user state)
        
        Returns:
            Expected reward
        """
        if arm_id not in self.arm_weights:
            # Initialize with zeros
            self.arm_weights[arm_id] = np.zeros(self.context_dim)
        
        # Linear prediction
        reward = np.dot(self.arm_weights[arm_id], context)
        return reward
    
    def select_arm(self, available_arms, context, top_k=2, epsilon=0.1):
        """
        Select arms given context
        
        Args:
            available_arms: List of arm IDs
            context: Context vector
            top_k: Number of arms to select
            epsilon: Exploration probability
        
        Returns:
            List of selected arm IDs
        """
        if not available_arms:
            return []
        
        top_k = min(top_k, len(available_arms))
        
        # Epsilon-greedy exploration
        if np.random.random() < epsilon:
            # Explore
            selected = np.random.choice(available_arms, size=top_k, replace=False)
            return list(selected)
        else:
            # Exploit: predict rewards
            arm_rewards = [
                (arm, self.predict_reward(arm, context))
                for arm in available_arms
            ]
            
            # Select top k
            arm_rewards.sort(key=lambda x: x[1], reverse=True)
            selected = [arm for arm, _ in arm_rewards[:top_k]]
            
            return selected
    
    def update(self, arm_id, context, reward):
        """
        Update arm weights based on observed reward
        
        Args:
            arm_id: ID of selected arm
            context: Context vector used
            reward: Observed reward
        """
        if arm_id not in self.arm_weights:
            self.arm_weights[arm_id] = np.zeros(self.context_dim)
        
        # Predict current reward
        predicted = self.predict_reward(arm_id, context)
        
        # Gradient update
        error = reward - predicted
        self.arm_weights[arm_id] += self.learning_rate * error * context


if __name__ == "__main__":
    # Test reinforcement learning module
    print("Testing Reinforcement Learning Module")
    print("=" * 50)
    
    # Test Multi-Armed Bandit
    print("\n1. Epsilon-Greedy Multi-Armed Bandit:")
    bandit = MultiArmedBandit(exploration_rate=0.2)
    
    arms = [1, 2, 3, 4, 5]
    
    # Simulate 100 selections
    for _ in range(100):
        selected = bandit.select_arm(arms, top_k=2)
        # Arm 3 is "best" - give it higher reward
        for arm in selected:
            reward = 1.0 if arm == 3 else 0.3
            bandit.update(arm, reward)
    
    print("Arm statistics after 100 selections:")
    for arm in arms:
        stats = bandit.get_arm_stats(arm)
        print(f"  Arm {arm}: {stats}")
    
    # Test UCB Bandit
    print("\n2. UCB Bandit:")
    ucb = UCBBandit(c=2.0)
    
    # Simulate selections
    for _ in range(50):
        selected = ucb.select_arm(arms, top_k=1)
        for arm in selected:
            reward = 1.0 if arm == 3 else 0.3
            ucb.update(arm, reward)
    
    print("Arm statistics after 50 selections:")
    for arm in arms:
        stats = ucb.get_arm_stats(arm)
        print(f"  Arm {arm}: count={stats['count']}, avg={stats['avg_reward']:.2f}, ucb={stats['ucb']:.2f}")
    
    # Test Thompson Sampling
    print("\n3. Thompson Sampling:")
    ts = ThompsonSampling()
    
    # Simulate selections
    for _ in range(50):
        selected = ts.select_arm(arms, top_k=1)
        for arm in selected:
            reward = 1 if arm == 3 else 0
            ts.update(arm, reward)
    
    print("Arm statistics after 50 selections:")
    for arm in arms:
        stats = ts.get_arm_stats(arm)
        print(f"  Arm {arm}: successes={stats['successes']}, failures={stats['failures']}, win_rate={stats['win_rate']:.2f}")
    
    # Test Contextual Bandit
    print("\n4. Contextual Bandit:")
    context_bandit = ContextualBandit(context_dim=5)
    
    # Sample context (user preferences)
    context = np.array([0.8, 0.2, 0.5, 0.9, 0.3])
    
    # Simulate selections
    for _ in range(30):
        selected = context_bandit.select_arm(arms, context, top_k=1)
        for arm in selected:
            # True reward depends on context-arm interaction
            true_reward = 0.5 + 0.3 * (arm == 3) + 0.2 * context[0]
            context_bandit.update(arm, context, true_reward)
    
    print("Predicted rewards given context:")
    for arm in arms:
        pred = context_bandit.predict_reward(arm, context)
        print(f"  Arm {arm}: {pred:.3f}")
    
    print("\nReinforcement learning module working correctly!")
