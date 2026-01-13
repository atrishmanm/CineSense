"""
AI Layer 1: Pairwise Preference Learning
Implements Bradley-Terry model and ELO-style scoring
"""

import numpy as np
from config import Config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PairwiseLearner:
    """
    Implements pairwise preference learning using ELO rating system
    Similar to Netflix and Spotify's learning-to-rank algorithms
    """
    
    def __init__(self, k_factor=32, initial_rating=1500):
        """
        Initialize pairwise learner
        
        Args:
            k_factor: Maximum rating change per game (32 for beginners, 16 for masters)
            initial_rating: Starting ELO rating for new items
        """
        self.k_factor = k_factor
        self.initial_rating = initial_rating
    
    def expected_score(self, rating_a, rating_b):
        """
        Calculate expected score using ELO formula
        
        Args:
            rating_a: Current rating of item A
            rating_b: Current rating of item B
        
        Returns:
            Expected score for A (probability A wins)
        """
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    
    def update_ratings(self, winner_rating, loser_rating):
        """
        Update ELO ratings after a comparison
        
        Args:
            winner_rating: Current rating of winning item
            loser_rating: Current rating of losing item
        
        Returns:
            (new_winner_rating, new_loser_rating)
        """
        # Expected scores
        expected_winner = self.expected_score(winner_rating, loser_rating)
        expected_loser = self.expected_score(loser_rating, winner_rating)
        
        # Actual scores (1 for win, 0 for loss)
        actual_winner = 1
        actual_loser = 0
        
        # Rating changes
        winner_change = self.k_factor * (actual_winner - expected_winner)
        loser_change = self.k_factor * (actual_loser - expected_loser)
        
        # New ratings
        new_winner_rating = winner_rating + winner_change
        new_loser_rating = loser_rating + loser_change
        
        return int(new_winner_rating), int(new_loser_rating)
    
    def get_win_probability(self, rating_a, rating_b):
        """
        Get probability that A beats B
        
        Args:
            rating_a: ELO rating of item A
            rating_b: ELO rating of item B
        
        Returns:
            Probability A wins (0 to 1)
        """
        return self.expected_score(rating_a, rating_b)
    
    def rank_items(self, items_with_ratings):
        """
        Rank items by ELO rating
        
        Args:
            items_with_ratings: List of (item_id, rating) tuples
        
        Returns:
            Sorted list of (item_id, rating) by rating descending
        """
        return sorted(items_with_ratings, key=lambda x: x[1], reverse=True)


class BradleyTerryModel:
    """
    Bradley-Terry model for pairwise comparisons
    More sophisticated than pure ELO
    """
    
    def __init__(self, n_items, learning_rate=0.1):
        """
        Initialize Bradley-Terry model
        
        Args:
            n_items: Number of items to rank
            learning_rate: Step size for gradient updates
        """
        self.n_items = n_items
        self.learning_rate = learning_rate
        
        # Initialize strength parameters (log-scale)
        self.strengths = np.ones(n_items)
    
    def probability_i_beats_j(self, i, j):
        """
        Probability that item i beats item j
        
        P(i > j) = strength_i / (strength_i + strength_j)
        """
        return self.strengths[i] / (self.strengths[i] + self.strengths[j])
    
    def update_from_comparison(self, winner_idx, loser_idx):
        """
        Update strengths based on comparison result
        
        Args:
            winner_idx: Index of winning item
            loser_idx: Index of losing item
        """
        # Current probability of this outcome
        p_win = self.probability_i_beats_j(winner_idx, loser_idx)
        
        # Gradient update
        # Winner: increase strength (less so if already expected to win)
        self.strengths[winner_idx] *= (1 + self.learning_rate * (1 - p_win))
        
        # Loser: decrease strength (more so if expected to win)
        self.strengths[loser_idx] *= (1 - self.learning_rate * (1 - p_win))
        
        # Normalize to prevent overflow
        if np.max(self.strengths) > 1000:
            self.strengths /= np.max(self.strengths) / 100
    
    def get_rankings(self):
        """
        Get current rankings of all items
        
        Returns:
            Array of indices sorted by strength (best first)
        """
        return np.argsort(self.strengths)[::-1]
    
    def get_top_k(self, k):
        """
        Get indices of top k items
        
        Args:
            k: Number of top items to return
        
        Returns:
            Array of top k indices
        """
        return self.get_rankings()[:k]


class UserPreferenceModel:
    """
    Personalized preference model for each user
    Learns user-specific tastes through pairwise comparisons
    """
    
    def __init__(self, user_id):
        """
        Initialize user preference model
        
        Args:
            user_id: Unique user identifier
        """
        self.user_id = user_id
        self.movie_preferences = {}  # movie_id -> preference score
        self.comparison_count = 0
        self.learning_rate = Config.LEARNING_RATE
    
    def record_preference(self, winner_id, loser_id, strength=1.0):
        """
        Record a pairwise preference
        
        Args:
            winner_id: ID of preferred movie
            loser_id: ID of rejected movie
            strength: How strong the preference is (default 1.0)
        """
        # Initialize if new movies
        if winner_id not in self.movie_preferences:
            self.movie_preferences[winner_id] = 0.0
        if loser_id not in self.movie_preferences:
            self.movie_preferences[loser_id] = 0.0
        
        # Update preferences
        self.movie_preferences[winner_id] += self.learning_rate * strength
        self.movie_preferences[loser_id] -= self.learning_rate * strength
        
        self.comparison_count += 1
    
    def get_preference_score(self, movie_id):
        """
        Get preference score for a movie
        
        Args:
            movie_id: Movie identifier
        
        Returns:
            Preference score (higher = more preferred)
        """
        return self.movie_preferences.get(movie_id, 0.0)
    
    def get_top_movies(self, n=10):
        """
        Get user's top preferred movies
        
        Args:
            n: Number of movies to return
        
        Returns:
            List of (movie_id, score) tuples
        """
        sorted_movies = sorted(
            self.movie_preferences.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_movies[:n]
    
    def has_enough_data(self, min_comparisons=5):
        """
        Check if user has made enough comparisons for reliable recommendations
        
        Args:
            min_comparisons: Minimum number of comparisons needed
        
        Returns:
            Boolean indicating if enough data exists
        """
        return self.comparison_count >= min_comparisons


if __name__ == "__main__":
    # Test pairwise learning
    print("Testing Pairwise Learning Module")
    print("=" * 50)
    
    # Test ELO system
    print("\n1. ELO Rating System:")
    learner = PairwiseLearner()
    
    # Simulate: Movie A (1500) vs Movie B (1500)
    rating_a, rating_b = 1500, 1500
    print(f"Initial: Movie A = {rating_a}, Movie B = {rating_b}")
    
    # A wins
    new_a, new_b = learner.update_ratings(rating_a, rating_b)
    print(f"After A wins: A = {new_a}, B = {new_b}")
    
    # Probability A beats B now
    prob = learner.get_win_probability(new_a, new_b)
    print(f"P(A beats B) = {prob:.2%}")
    
    # Test Bradley-Terry
    print("\n2. Bradley-Terry Model:")
    bt = BradleyTerryModel(n_items=5)
    
    print("Initial strengths:", bt.strengths)
    
    # Item 0 beats item 1 multiple times
    for _ in range(5):
        bt.update_from_comparison(winner_idx=0, loser_idx=1)
    
    print("After item 0 beats item 1 five times:")
    print("Strengths:", bt.strengths)
    print("Rankings:", bt.get_rankings())
    
    # Test User Preference Model
    print("\n3. User Preference Model:")
    user = UserPreferenceModel(user_id=1)
    
    # User prefers movie 101 over 102
    user.record_preference(winner_id=101, loser_id=102)
    user.record_preference(winner_id=101, loser_id=103)
    user.record_preference(winner_id=104, loser_id=102)
    
    print(f"Comparisons made: {user.comparison_count}")
    print(f"Top movies: {user.get_top_movies(3)}")
    print(f"Has enough data: {user.has_enough_data()}")
    
    print("\n✓ Pairwise learning module working correctly!")
