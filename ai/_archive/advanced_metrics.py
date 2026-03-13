"""
Advanced Recommendation Metrics
Beyond RMSE: diversity, novelty, serendipity, coverage, and personalization
"""

import numpy as np
from typing import List, Dict, Set, Tuple, Optional
from collections import defaultdict, Counter
import logging
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import jaccard
import math

logger = logging.getLogger(__name__)


class AdvancedRecommendationMetrics:
    """
    Complete metrics suite for recommendation systems:
    
    Accuracy Metrics:
    - RMSE, MAE, R²
    - Precision@K, Recall@K, F1@K
    - MAP (Mean Average Precision)
    - NDCG (Normalized Discounted Cumulative Gain)
    
    Beyond-Accuracy Metrics:
    - Diversity (Intra-list diversity)
    - Novelty (Item popularity)
    - Serendipity (Unexpected relevance)
    - Coverage (Catalog coverage)
    - Personalization (User-specific recommendations)
    """
    
    def __init__(self, item_popularity: Optional[Dict[int, int]] = None):
        """
        Initialize metrics calculator
        
        Args:
            item_popularity: Dictionary of item_id -> popularity count
        """
        self.item_popularity = item_popularity or {}
        
        # For novelty calculation
        if item_popularity:
            total_interactions = sum(item_popularity.values())
            self.item_probability = {
                item_id: count / total_interactions
                for item_id, count in item_popularity.items()
            }
        else:
            self.item_probability = {}
    
    # ===== ACCURACY METRICS =====
    
    def rmse(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Root Mean Squared Error"""
        return np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    def mae(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Error"""
        return np.mean(np.abs(y_true - y_pred))
    
    def r_squared(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """R² Score"""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    def precision_at_k(
        self,
        recommendations: List[int],
        relevant_items: Set[int],
        k: int
    ) -> float:
        """
        Precision@K: Proportion of recommended items that are relevant
        
        Args:
            recommendations: List of recommended item IDs
            relevant_items: Set of relevant item IDs
            k: Number of recommendations to consider
            
        Returns:
            Precision score (0-1)
        """
        if k == 0 or len(recommendations) == 0:
            return 0.0
        
        top_k = recommendations[:k]
        relevant_recommended = sum(1 for item in top_k if item in relevant_items)
        return relevant_recommended / k
    
    def recall_at_k(
        self,
        recommendations: List[int],
        relevant_items: Set[int],
        k: int
    ) -> float:
        """
        Recall@K: Proportion of relevant items that are recommended
        
        Args:
            recommendations: List of recommended item IDs
            relevant_items: Set of relevant item IDs
            k: Number of recommendations to consider
            
        Returns:
            Recall score (0-1)
        """
        if len(relevant_items) == 0:
            return 0.0
        
        top_k = recommendations[:k]
        relevant_recommended = sum(1 for item in top_k if item in relevant_items)
        return relevant_recommended / len(relevant_items)
    
    def f1_at_k(
        self,
        recommendations: List[int],
        relevant_items: Set[int],
        k: int
    ) -> float:
        """F1@K: Harmonic mean of precision and recall"""
        precision = self.precision_at_k(recommendations, relevant_items, k)
        recall = self.recall_at_k(recommendations, relevant_items, k)
        
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def average_precision(
        self,
        recommendations: List[int],
        relevant_items: Set[int]
    ) -> float:
        """
        Average Precision: Average of precision values at each relevant item
        
        Used in MAP@K calculation
        """
        if len(relevant_items) == 0:
            return 0.0
        
        precisions = []
        num_relevant = 0
        
        for k, item_id in enumerate(recommendations, 1):
            if item_id in relevant_items:
                num_relevant += 1
                precision = num_relevant / k
                precisions.append(precision)
        
        if len(precisions) == 0:
            return 0.0
        
        return sum(precisions) / len(relevant_items)
    
    def mean_average_precision(
        self,
        all_recommendations: List[List[int]],
        all_relevant_items: List[Set[int]]
    ) -> float:
        """
        MAP (Mean Average Precision): Average of AP across all users
        
        Args:
            all_recommendations: List of recommendation lists
            all_relevant_items: List of relevant item sets
            
        Returns:
            MAP score (0-1)
        """
        if len(all_recommendations) == 0:
            return 0.0
        
        aps = [
            self.average_precision(recs, relevant)
            for recs, relevant in zip(all_recommendations, all_relevant_items)
        ]
        
        return np.mean(aps)
    
    def ndcg_at_k(
        self,
        recommendations: List[int],
        relevance_scores: Dict[int, float],
        k: int
    ) -> float:
        """
        NDCG@K (Normalized Discounted Cumulative Gain)
        
        Considers both relevance and ranking position
        
        Args:
            recommendations: List of recommended item IDs
            relevance_scores: Dictionary of item_id -> relevance score
            k: Number of recommendations to consider
            
        Returns:
            NDCG score (0-1)
        """
        if k == 0:
            return 0.0
        
        # DCG (Discounted Cumulative Gain)
        dcg = 0.0
        for i, item_id in enumerate(recommendations[:k], 1):
            rel = relevance_scores.get(item_id, 0.0)
            dcg += (2 ** rel - 1) / math.log2(i + 1)
        
        # IDCG (Ideal DCG)
        ideal_relevance = sorted(relevance_scores.values(), reverse=True)[:k]
        idcg = sum(
            (2 ** rel - 1) / math.log2(i + 1)
            for i, rel in enumerate(ideal_relevance, 1)
        )
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    # ===== DIVERSITY METRICS =====
    
    def intra_list_diversity(
        self,
        recommendations: List[int],
        item_features: Dict[int, np.ndarray]
    ) -> float:
        """
        Intra-list Diversity: Average dissimilarity between items
        
        Args:
            recommendations: List of recommended item IDs
            item_features: Dictionary of item_id -> feature vector
            
        Returns:
            Diversity score (0-1, higher is more diverse)
        """
        if len(recommendations) < 2:
            return 0.0
        
        # Get feature vectors
        features = []
        for item_id in recommendations:
            if item_id in item_features:
                features.append(item_features[item_id])
        
        if len(features) < 2:
            return 0.0
        
        # Calculate pairwise similarities
        features_matrix = np.array(features)
        similarities = cosine_similarity(features_matrix)
        
        # Average dissimilarity (1 - similarity)
        n = len(features)
        total_dissimilarity = 0.0
        count = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                total_dissimilarity += (1 - similarities[i, j])
                count += 1
        
        return total_dissimilarity / count if count > 0 else 0.0
    
    def genre_diversity(
        self,
        recommendations: List[int],
        item_genres: Dict[int, Set[str]]
    ) -> float:
        """
        Genre Diversity: Number of unique genres in recommendations
        
        Args:
            recommendations: List of recommended item IDs
            item_genres: Dictionary of item_id -> set of genres
            
        Returns:
            Genre diversity score (0-1)
        """
        all_genres = set()
        
        for item_id in recommendations:
            if item_id in item_genres:
                all_genres.update(item_genres[item_id])
        
        # Normalize by typical number of genres (e.g., 20)
        max_genres = 20
        return min(len(all_genres) / max_genres, 1.0)
    
    # ===== NOVELTY METRICS =====
    
    def novelty(self, recommendations: List[int]) -> float:
        """
        Novelty: Average unpopularity of recommended items
        
        Recommends less popular (more novel) items
        
        Args:
            recommendations: List of recommended item IDs
            
        Returns:
            Novelty score (0-10+, higher is more novel)
        """
        if not self.item_probability:
            logger.warning("Item popularity not set, novelty = 0")
            return 0.0
        
        novelties = []
        for item_id in recommendations:
            prob = self.item_probability.get(item_id, 1e-10)
            novelty = -math.log2(prob)
            novelties.append(novelty)
        
        return np.mean(novelties) if novelties else 0.0
    
    def long_tail_percentage(
        self,
        recommendations: List[int],
        long_tail_threshold: int = 100
    ) -> float:
        """
        Percentage of long-tail items (popularity < threshold)
        
        Args:
            recommendations: List of recommended item IDs
            long_tail_threshold: Popularity threshold for long-tail
            
        Returns:
            Percentage (0-100)
        """
        if not self.item_popularity:
            return 0.0
        
        long_tail_count = sum(
            1 for item_id in recommendations
            if self.item_popularity.get(item_id, 0) < long_tail_threshold
        )
        
        return (long_tail_count / len(recommendations) * 100) if recommendations else 0.0
    
    # ===== SERENDIPITY METRICS =====
    
    def serendipity(
        self,
        recommendations: List[int],
        relevant_items: Set[int],
        obvious_items: Set[int]
    ) -> float:
        """
        Serendipity: Proportion of relevant but unexpected items
        
        Args:
            recommendations: List of recommended item IDs
            relevant_items: Set of relevant items (liked by user)
            obvious_items: Set of obvious items (e.g., popular in user's genres)
            
        Returns:
            Serendipity score (0-1)
        """
        serendipitous = [
            item for item in recommendations
            if item in relevant_items and item not in obvious_items
        ]
        
        return len(serendipitous) / len(recommendations) if recommendations else 0.0
    
    # ===== COVERAGE METRICS =====
    
    def catalog_coverage(
        self,
        all_recommendations: List[List[int]],
        catalog_size: int
    ) -> float:
        """
        Catalog Coverage: Percentage of catalog items recommended
        
        Args:
            all_recommendations: All recommendation lists
            catalog_size: Total number of items in catalog
            
        Returns:
            Coverage percentage (0-100)
        """
        recommended_items = set()
        for recs in all_recommendations:
            recommended_items.update(recs)
        
        return (len(recommended_items) / catalog_size * 100) if catalog_size > 0 else 0.0
    
    def gini_coefficient(self, recommendation_counts: Dict[int, int]) -> float:
        """
        Gini Coefficient: Inequality in item recommendation distribution
        
        0 = perfect equality (all items recommended equally)
        1 = perfect inequality (few items get all recommendations)
        
        Args:
            recommendation_counts: Dictionary of item_id -> recommendation count
            
        Returns:
            Gini coefficient (0-1)
        """
        if not recommendation_counts:
            return 0.0
        
        counts = sorted(recommendation_counts.values())
        n = len(counts)
        
        cumsum = np.cumsum(counts)
        return (2 * np.sum((n + 1 - i) * count for i, count in enumerate(counts, 1)) / 
                (n * np.sum(counts))) - (n + 1) / n
    
    # ===== PERSONALIZATION METRICS =====
    
    def personalization(
        self,
        all_recommendations: List[List[int]]
    ) -> float:
        """
        Personalization: How different recommendations are across users
        
        Args:
            all_recommendations: List of recommendation lists for different users
            
        Returns:
            Personalization score (0-1, higher is more personalized)
        """
        if len(all_recommendations) < 2:
            return 0.0
        
        # Calculate pairwise Jaccard distances
        total_distance = 0.0
        count = 0
        
        for i in range(len(all_recommendations)):
            for j in range(i + 1, len(all_recommendations)):
                set_i = set(all_recommendations[i])
                set_j = set(all_recommendations[j])
                
                # Jaccard similarity
                intersection = len(set_i & set_j)
                union = len(set_i | set_j)
                similarity = intersection / union if union > 0 else 0.0
                
                # Jaccard distance (dissimilarity)
                distance = 1 - similarity
                total_distance += distance
                count += 1
        
        return total_distance / count if count > 0 else 0.0
    
    # ===== COMPREHENSIVE EVALUATION =====
    
    def evaluate_recommendations(
        self,
        recommendations: List[int],
        relevant_items: Set[int],
        item_features: Optional[Dict[int, np.ndarray]] = None,
        item_genres: Optional[Dict[int, Set[str]]] = None,
        relevance_scores: Optional[Dict[int, float]] = None,
        k: int = 10
    ) -> Dict[str, float]:
        """
        Complete evaluation with all metrics
        
        Args:
            recommendations: List of recommended item IDs
            relevant_items: Set of relevant items
            item_features: Item feature vectors (for diversity)
            item_genres: Item genres (for genre diversity)
            relevance_scores: Relevance scores (for NDCG)
            k: Number of top recommendations to evaluate
            
        Returns:
            Dictionary of metric name -> value
        """
        metrics = {}
        
        # Accuracy metrics
        metrics['precision@k'] = self.precision_at_k(recommendations, relevant_items, k)
        metrics['recall@k'] = self.recall_at_k(recommendations, relevant_items, k)
        metrics['f1@k'] = self.f1_at_k(recommendations, relevant_items, k)
        metrics['average_precision'] = self.average_precision(recommendations, relevant_items)
        
        if relevance_scores:
            metrics['ndcg@k'] = self.ndcg_at_k(recommendations, relevance_scores, k)
        
        # Diversity
        if item_features:
            metrics['diversity'] = self.intra_list_diversity(recommendations, item_features)
        
        if item_genres:
            metrics['genre_diversity'] = self.genre_diversity(recommendations, item_genres)
        
        # Novelty
        if self.item_popularity:
            metrics['novelty'] = self.novelty(recommendations)
            metrics['long_tail_%'] = self.long_tail_percentage(recommendations)
        
        return metrics
    
    def print_metrics(self, metrics: Dict[str, float]):
        """Pretty print metrics"""
        print("\n" + "=" * 60)
        print("RECOMMENDATION METRICS")
        print("=" * 60)
        
        # Group metrics
        accuracy = {k: v for k, v in metrics.items() if any(x in k for x in ['precision', 'recall', 'f1', 'ndcg', 'average'])}
        diversity = {k: v for k, v in metrics.items() if 'diversity' in k}
        novelty = {k: v for k, v in metrics.items() if 'novelty' in k or 'tail' in k}
        other = {k: v for k, v in metrics.items() if k not in accuracy and k not in diversity and k not in novelty}
        
        if accuracy:
            print("\nAccuracy:")
            for k, v in accuracy.items():
                print(f"  {k}: {v:.4f}")
        
        if diversity:
            print("\nDiversity:")
            for k, v in diversity.items():
                print(f"  {k}: {v:.4f}")
        
        if novelty:
            print("\nNovelty:")
            for k, v in novelty.items():
                print(f"  {k}: {v:.4f}")
        
        if other:
            print("\nOther:")
            for k, v in other.items():
                print(f"  {k}: {v:.4f}")
        
        print("=" * 60)


# Example usage
if __name__ == '__main__':
    print("Advanced Recommendation Metrics Demo")
    print("=" * 60)
    
    # Create sample data
    item_popularity = {i: 1000 - i * 10 for i in range(100)}  # Items 0-99
    
    metrics_calc = AdvancedRecommendationMetrics(item_popularity)
    
    # Scenario 1: Good personalized recommendations
    print("\nScenario 1: Personalized Recommendations")
    recommendations = [5, 12, 23, 45, 67, 89, 34, 56, 78, 90]
    relevant_items = {5, 12, 45, 67, 89, 34, 78}
    
    # Create dummy features
    np.random.seed(42)
    item_features = {i: np.random.rand(50) for i in range(100)}
    
    item_genres = {
        5: {'Action', 'Sci-Fi'},
        12: {'Drama', 'Romance'},
        23: {'Comedy'},
        45: {'Action', 'Adventure'},
        67: {'Thriller', 'Mystery'},
        89: {'Documentary'},
        34: {'Horror'},
        56: {'Animation'},
        78: {'Drama', 'History'},
        90: {'Western'}
    }
    
    relevance_scores = {item: 5.0 if item in relevant_items else 3.0 for item in recommendations}
    
    results = metrics_calc.evaluate_recommendations(
        recommendations=recommendations,
        relevant_items=relevant_items,
        item_features=item_features,
        item_genres=item_genres,
        relevance_scores=relevance_scores,
        k=10
    )
    
    metrics_calc.print_metrics(results)
    
    # Scenario 2: Popular but not diverse
    print("\n\nScenario 2: Popular Recommendations (Low Novelty)")
    popular_recommendations = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]  # Most popular
    relevant_popular = {0, 2, 5, 7}
    
    results2 = metrics_calc.evaluate_recommendations(
        recommendations=popular_recommendations,
        relevant_items=relevant_popular,
        item_features=item_features,
        k=10
    )
    
    metrics_calc.print_metrics(results2)
    
    # Coverage across multiple users
    print("\n\nCatalog Coverage Analysis")
    all_recommendations = [
        [5, 12, 23, 45, 67],
        [7, 15, 28, 50, 72],
        [3, 18, 33, 55, 77],
        [10, 20, 30, 40, 60]
    ]
    
    coverage = metrics_calc.catalog_coverage(all_recommendations, catalog_size=100)
    print(f"Catalog Coverage: {coverage:.2f}%")
    
    personalization = metrics_calc.personalization(all_recommendations)
    print(f"Personalization: {personalization:.4f}")
    
    print("\n✓ Advanced metrics system operational")
    print("\nAvailable metrics:")
    print("  • Precision, Recall, F1, MAP, NDCG")
    print("  • Intra-list diversity")
    print("  • Novelty & long-tail coverage")
    print("  • Serendipity")
    print("  • Catalog coverage & Gini coefficient")
    print("  • Personalization")
