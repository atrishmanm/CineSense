"""
Advanced AI Features for Enhanced Recommendation Intelligence
Implements: Latent representations, implicit signals, probabilistic decisions,
memory/forgetting, and natural language explanations
"""

import numpy as np
from sklearn.decomposition import TruncatedSVD, PCA
from sklearn.preprocessing import StandardScaler
from scipy.special import softmax
from config import Config
import logging
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LatentSpaceEncoder:
    """
    Feature 1: Move from Features to Latent Representations
    Compresses high-dimensional features into dense latent vectors
    using PCA/SVD (actual representation learning)
    """
    
    def __init__(self, n_components=Config.LATENT_DIM):
        self.n_components = n_components
        self.encoder = TruncatedSVD(n_components=n_components, random_state=42)
        self.scaler = StandardScaler()
        self.is_fitted = False
    
    def fit(self, feature_vectors):
        """
        Learn latent representation from movie features
        
        Args:
            feature_vectors: array of shape (n_movies, n_features)
        """
        if len(feature_vectors) < self.n_components:
            logger.warning(f"Not enough samples ({len(feature_vectors)}) for {self.n_components} components")
            self.n_components = max(2, len(feature_vectors) // 2)
            self.encoder = TruncatedSVD(n_components=self.n_components, random_state=42)
        
        # Standardize features
        scaled = self.scaler.fit_transform(feature_vectors)
        
        # Learn latent space
        self.encoder.fit(scaled)
        self.is_fitted = True
        
        explained_var = self.encoder.explained_variance_ratio_.sum()
        logger.info(f"Latent space encoder fitted: {self.n_components} dimensions, "
                   f"{explained_var:.2%} variance explained")
    
    def transform(self, feature_vector):
        """
        Transform a feature vector into latent space
        
        Args:
            feature_vector: array of shape (n_features,)
        
        Returns:
            latent_vector: array of shape (n_components,)
        """
        if not self.is_fitted:
            logger.warning("Encoder not fitted, returning original vector")
            return feature_vector
        
        # Reshape and scale
        scaled = self.scaler.transform(feature_vector.reshape(1, -1))
        
        # Project to latent space
        latent = self.encoder.transform(scaled)
        
        return latent.flatten()
    
    def inverse_transform(self, latent_vector):
        """Reconstruct feature vector from latent space"""
        if not self.is_fitted:
            return latent_vector
        
        reconstructed = self.encoder.inverse_transform(latent_vector.reshape(1, -1))
        return self.scaler.inverse_transform(reconstructed).flatten()


class ImplicitSignalProcessor:
    """
    Feature 2: Learn from Implicit Signals
    Processes behavioral signals beyond explicit choices:
    - Hover time
    - Skips
    - Repeated views
    - Session abandonment
    """
    
    def __init__(self):
        self.signal_weights = Config.IMPLICIT_SIGNALS
    
    def calculate_implicit_reward(self, interaction_data):
        """
        Calculate reward from implicit signals
        
        Args:
            interaction_data: dict with keys like 'hover_time', 'was_skipped', etc.
        
        Returns:
            float: implicit reward score
        """
        reward = 0.0
        
        # Hover time (normalized)
        if 'hover_time' in interaction_data:
            hover_seconds = interaction_data['hover_time']
            # Normalize: 0-5 seconds → 0-1 score
            hover_score = min(hover_seconds / 5.0, 1.0)
            reward += hover_score * self.signal_weights['hover_time']
        
        # Skip penalty
        if interaction_data.get('was_skipped', False):
            reward += self.signal_weights['skip_penalty']
        
        # Repeat view bonus
        if interaction_data.get('repeat_view', False):
            reward += self.signal_weights['repeat_view']
        
        # Session abandonment penalty
        if interaction_data.get('session_abandoned', False):
            reward += self.signal_weights['session_abandon']
        
        return reward
    
    def enrich_interaction(self, base_reward, implicit_data):
        """
        Combine explicit choice with implicit signals
        
        Args:
            base_reward: 1.0 for chosen, 0.0 for rejected
            implicit_data: dict of implicit signals
        
        Returns:
            enriched_reward: float in [0, 1] range
        """
        implicit_reward = self.calculate_implicit_reward(implicit_data)
        
        # Combine: base reward is primary, implicit is adjustment
        enriched = base_reward + (implicit_reward * 0.3)  # 30% weight to implicit
        
        # Clip to valid range
        return np.clip(enriched, 0.0, 1.0)


class ProbabilisticSelector:
    """
    Feature 3: Probabilistic Decision-Making
    Uses softmax to select movies probabilistically rather than always picking highest score
    Makes system feel more human and exploratory
    """
    
    def __init__(self, temperature=Config.SOFTMAX_TEMPERATURE):
        self.temperature = temperature
    
    def select_with_probability(self, scores, top_k=1):
        """
        Select items using softmax probability distribution
        
        Args:
            scores: array of scores for each item
            top_k: number of items to select
        
        Returns:
            indices of selected items
        """
        # Convert scores to probabilities
        probabilities = softmax(np.array(scores) / self.temperature)
        
        # Sample based on probability
        selected_indices = np.random.choice(
            len(scores),
            size=min(top_k, len(scores)),
            replace=False,
            p=probabilities
        )
        
        return selected_indices
    
    def get_distribution(self, scores):
        """Get probability distribution over items"""
        return softmax(np.array(scores) / self.temperature)


class TemporalMemoryManager:
    """
    Feature 4: Memory & Forgetting
    Implements temporal decay - recent interactions matter more
    Gives illusion of evolving personality
    """
    
    def __init__(self, decay_factor=Config.TEMPORAL_DECAY_FACTOR, 
                 memory_window=Config.INTERACTION_MEMORY_WINDOW):
        self.decay_factor = decay_factor
        self.memory_window = memory_window
    
    def apply_temporal_weights(self, interactions):
        """
        Apply temporal decay to interactions
        
        Args:
            interactions: list of dicts with 'timestamp' and 'data'
        
        Returns:
            weighted_interactions: same list with added 'temporal_weight'
        """
        if not interactions:
            return []
        
        # Sort by timestamp (most recent first)
        sorted_interactions = sorted(
            interactions,
            key=lambda x: x.get('timestamp', datetime.now()),
            reverse=True
        )
        
        # Calculate weights
        weighted = []
        for i, interaction in enumerate(sorted_interactions):
            if i < self.memory_window:
                # Full weight for recent interactions
                weight = 1.0
            else:
                # Exponential decay for older interactions
                age = i - self.memory_window
                weight = np.exp(-age * (1 - self.decay_factor))
            
            interaction['temporal_weight'] = weight
            weighted.append(interaction)
        
        return weighted
    
    def compute_weighted_preference(self, recent_vector, past_vector):
        """
        Combine recent and past preferences with decay
        
        Formula: new_pref = decay_factor * recent + (1 - decay_factor) * past
        """
        return (self.decay_factor * recent_vector + 
                (1 - self.decay_factor) * past_vector)


class NaturalLanguageExplainer:
    """
    Feature 5: Natural Language Explanations
    Generates human-readable explanations for recommendations
    Interpreting model behavior in natural language
    """
    
    def __init__(self, detail_level=Config.EXPLANATION_DETAIL_LEVEL):
        self.detail_level = detail_level
    
    def explain_recommendation(self, movie, user_profile, similarity_score, 
                              preference_factors):
        """
        Generate natural language explanation
        
        Args:
            movie: dict with movie info
            user_profile: user's preference summary
            similarity_score: content similarity score
            preference_factors: dict of contributing factors
        
        Returns:
            string explanation
        """
        explanations = []
        
        # Analyze genre preferences
        if 'preferred_genres' in preference_factors:
            genres = preference_factors['preferred_genres']
            if genres:
                genre_list = ', '.join(genres[:3])
                explanations.append(
                    f"I noticed you consistently prefer {genre_list} films"
                )
        
        # Analyze director/actor patterns
        if 'favorite_directors' in preference_factors:
            directors = preference_factors['favorite_directors']
            if directors:
                explanations.append(
                    f"This aligns with your taste for {directors[0]}'s style"
                )
        
        # Analyze pacing/tone
        if 'pacing_preference' in preference_factors:
            pacing = preference_factors['pacing_preference']
            if pacing == 'slow-burn':
                explanations.append(
                    "You seem to appreciate slow-burn narratives over action-heavy films"
                )
            elif pacing == 'fast-paced':
                explanations.append(
                    "This matches your preference for high-energy, fast-paced stories"
                )
        
        # Confidence level
        if similarity_score > 0.8:
            confidence = "strongly"
        elif similarity_score > 0.6:
            confidence = "moderately"
        else:
            confidence = "cautiously"
        
        # Compose final explanation
        if self.detail_level == 'high':
            intro = f"I'm {confidence} recommending this because: "
            return intro + ". ".join(explanations) + "."
        elif self.detail_level == 'medium':
            return explanations[0] + "." if explanations else "Based on your viewing patterns."
        else:
            return f"Match score: {similarity_score:.0%}"
    
    def explain_learning_progress(self, interaction_count, accuracy_trend):
        """
        Explain how the AI is learning the user's taste
        
        Args:
            interaction_count: number of comparisons made
            accuracy_trend: improving/stable/declining
        
        Returns:
            string explanation
        """
        if interaction_count < 5:
            return "I'm just getting to know your taste. A few more choices will help!"
        elif interaction_count < 20:
            if accuracy_trend == 'improving':
                return "I'm learning quickly - your preferences are becoming clearer."
            else:
                return "Still learning your unique taste. Keep comparing!"
        else:
            if accuracy_trend == 'improving':
                return "I have a strong understanding of what you enjoy now."
            else:
                return "Your taste is nuanced - I'm refining my recommendations."
    
    def generate_taste_summary(self, user_profile):
        """
        Generate a personality-like summary of user's taste
        
        Args:
            user_profile: dict with aggregated preferences
        
        Returns:
            string summary in natural language
        """
        summaries = []
        
        # Genre personality
        if 'top_genres' in user_profile:
            genres = user_profile['top_genres'][:3]
            genre_str = ', '.join(genres[:-1]) + f" and {genres[-1]}" if len(genres) > 1 else genres[0]
            summaries.append(f"You're drawn to {genre_str}")
        
        # Era preference
        if 'avg_release_year' in user_profile:
            year = user_profile['avg_release_year']
            if year < 1990:
                summaries.append("with a love for classic cinema")
            elif year > 2010:
                summaries.append("favoring modern filmmaking")
        
        # Rating standards
        if 'avg_rating_preference' in user_profile:
            rating = user_profile['avg_rating_preference']
            if rating > 7.5:
                summaries.append("You have high standards - critically acclaimed films resonate with you")
            elif rating < 6.5:
                summaries.append("You appreciate hidden gems and underrated films")
        
        return ". ".join(summaries) + "." if summaries else "Your taste profile is still forming."


# Factory function to get all advanced AI components
def get_advanced_ai_suite():
    """
    Returns all advanced AI components as a dict
    """
    return {
        'latent_encoder': LatentSpaceEncoder(),
        'implicit_processor': ImplicitSignalProcessor(),
        'probabilistic_selector': ProbabilisticSelector(),
        'memory_manager': TemporalMemoryManager(),
        'nlg_explainer': NaturalLanguageExplainer()
    }
