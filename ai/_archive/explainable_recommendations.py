"""
Explainable AI Recommendations
Uses SHAP (SHapley Additive exPlanations) to explain model predictions
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# Try to import SHAP (optional dependency)
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    logger.warning("SHAP not installed. Install with: pip install shap")
    SHAP_AVAILABLE = False


class ExplainableRecommender:
    """
    Generate human-readable explanations for movie recommendations
    """
    
    def __init__(self, model, feature_names: List[str]):
        """
        Args:
            model: PyTorch recommendation model
            feature_names: Names of input features
        """
        self.model = model
        self.feature_names = feature_names
        self.explainer = None
        
        if not SHAP_AVAILABLE:
            logger.warning("SHAP not available, using fallback explanation method")
    
    def initialize_explainer(self, background_data: torch.Tensor):
        """
        Initialize SHAP explainer with background dataset
        
        Args:
            background_data: Sample of training data for SHAP baseline
        """
        if not SHAP_AVAILABLE:
            return
        
        try:
            logger.info("Initializing SHAP explainer...")
            self.explainer = shap.DeepExplainer(self.model, background_data)
            logger.info("✓ SHAP explainer initialized")
        except Exception as e:
            logger.error(f"Failed to initialize SHAP: {e}")
            self.explainer = None
    
    def explain_recommendation(
        self,
        user_id: int,
        movie_id: int,
        content_features: torch.Tensor,
        plot_embedding: Optional[torch.Tensor] = None,
        movie_metadata: Optional[Dict] = None
    ) -> Dict:
        """
        Generate explanation for a recommendation
        
        Returns:
            Dictionary with prediction, explanation, and feature importances
        """
        # Get prediction
        with torch.no_grad():
            if plot_embedding is not None:
                inputs = {
                    'user_ids': torch.LongTensor([user_id]),
                    'movie_ids': torch.LongTensor([movie_id]),
                    'content_features': content_features.unsqueeze(0),
                    'plot_embeddings': plot_embedding.unsqueeze(0)
                }
                prediction = self.model(**inputs).item()
            else:
                prediction = self.model(
                    torch.LongTensor([user_id]),
                    torch.LongTensor([movie_id]),
                    content_features.unsqueeze(0)
                ).item()
        
        # Generate explanation
        if SHAP_AVAILABLE and self.explainer is not None:
            explanation = self._shap_explanation(content_features, movie_metadata)
        else:
            explanation = self._fallback_explanation(content_features, movie_metadata)
        
        return {
            'predicted_rating': round(prediction, 2),
            'explanation': explanation['text'],
            'top_features': explanation['features'],
            'confidence': self._calculate_confidence(prediction)
        }
    
    def _shap_explanation(
        self, 
        content_features: torch.Tensor,
        movie_metadata: Optional[Dict]
    ) -> Dict:
        """Generate SHAP-based explanation"""
        try:
            # Calculate SHAP values
            shap_values = self.explainer.shap_values(content_features.unsqueeze(0))
            
            # Get feature importance
            feature_importance = []
            for i, (name, value) in enumerate(zip(self.feature_names, shap_values[0])):
                feature_importance.append({
                    'feature': name,
                    'importance': float(value),
                    'contribution': 'positive' if value > 0 else 'negative'
                })
            
            # Sort by absolute importance
            feature_importance.sort(key=lambda x: abs(x['importance']), reverse=True)
            
            # Generate text explanation
            text = self._generate_text_explanation(
                feature_importance[:5],
                movie_metadata
            )
            
            return {
                'text': text,
                'features': feature_importance[:10]
            }
        
        except Exception as e:
            logger.error(f"SHAP explanation failed: {e}")
            return self._fallback_explanation(content_features, movie_metadata)
    
    def _fallback_explanation(
        self,
        content_features: torch.Tensor,
        movie_metadata: Optional[Dict]
    ) -> Dict:
        """
        Simple gradient-based explanation when SHAP unavailable
        """
        # Use gradient to estimate feature importance
        content_features.requires_grad = True
        
        # Forward pass
        output = self.model(
            torch.LongTensor([0]),  # placeholder
            torch.LongTensor([0]),  # placeholder
            content_features.unsqueeze(0)
        )
        
        # Backward pass
        output.backward()
        
        # Get gradients
        gradients = content_features.grad.abs().numpy()
        
        # Map to feature names
        feature_importance = []
        for i, (name, grad) in enumerate(zip(self.feature_names, gradients)):
            feature_importance.append({
                'feature': name,
                'importance': float(grad),
                'contribution': 'positive' if content_features[i] > 0 else 'negative'
            })
        
        feature_importance.sort(key=lambda x: x['importance'], reverse=True)
        
        text = self._generate_text_explanation(
            feature_importance[:5],
            movie_metadata
        )
        
        return {
            'text': text,
            'features': feature_importance[:10]
        }
    
    def _generate_text_explanation(
        self,
        top_features: List[Dict],
        movie_metadata: Optional[Dict]
    ) -> str:
        """
        Convert feature importances to natural language
        """
        reasons = []
        
        for feature_info in top_features:
            feature = feature_info['feature']
            contribution = feature_info['contribution']
            
            # Map features to readable explanations
            if 'genre' in feature.lower():
                genre = feature.replace('genre_', '').title()
                if contribution == 'positive':
                    reasons.append(f"you enjoy {genre} movies")
                else:
                    reasons.append(f"it's not typically {genre}")
            
            elif 'year' in feature.lower():
                if contribution == 'positive':
                    reasons.append("it's from a time period you like")
                else:
                    reasons.append("it's from a different era")
            
            elif 'rating' in feature.lower():
                if contribution == 'positive':
                    reasons.append("it has high ratings")
                else:
                    reasons.append("ratings are mixed")
            
            elif 'popularity' in feature.lower():
                if contribution == 'positive':
                    reasons.append("many people enjoyed it")
                else:
                    reasons.append("it's a hidden gem")
            
            elif 'director' in feature.lower():
                if contribution == 'positive':
                    reasons.append("the director matches your taste")
            
            elif 'cast' in feature.lower():
                if contribution == 'positive':
                    reasons.append("the cast aligns with your preferences")
        
        # Build explanation
        if not reasons:
            explanation = "This movie matches your viewing patterns"
        elif len(reasons) == 1:
            explanation = f"Recommended because {reasons[0]}"
        elif len(reasons) == 2:
            explanation = f"Recommended because {reasons[0]} and {reasons[1]}"
        else:
            explanation = f"Recommended because {reasons[0]}, {reasons[1]}, and {reasons[2]}"
        
        # Add movie-specific info if available
        if movie_metadata:
            title = movie_metadata.get('title', 'this movie')
            year = movie_metadata.get('year', '')
            if year:
                explanation = f"'{title}' ({year}) - " + explanation
            else:
                explanation = f"'{title}' - " + explanation
        
        return explanation
    
    def _calculate_confidence(self, prediction: float) -> str:
        """
        Calculate confidence level for prediction
        
        Returns:
            'high', 'medium', or 'low'
        """
        # Heuristic based on prediction value
        if prediction >= 4.5:
            return 'high'
        elif prediction >= 3.5:
            return 'medium'
        else:
            return 'low'
    
    def compare_movies(
        self,
        movie1_features: torch.Tensor,
        movie2_features: torch.Tensor,
        movie1_metadata: Dict,
        movie2_metadata: Dict
    ) -> Dict:
        """
        Explain why one movie was recommended over another
        """
        # Get predictions for both
        with torch.no_grad():
            pred1 = self.model(
                torch.LongTensor([0]),
                torch.LongTensor([0]),
                movie1_features.unsqueeze(0)
            ).item()
            
            pred2 = self.model(
                torch.LongTensor([0]),
                torch.LongTensor([0]),
                movie2_features.unsqueeze(0)
            ).item()
        
        # Calculate differences
        feature_diff = (movie1_features - movie2_features).abs()
        
        # Find most different features
        top_diffs = []
        for i, (name, diff) in enumerate(zip(self.feature_names, feature_diff)):
            if diff > 0.1:  # Significant difference
                top_diffs.append({
                    'feature': name,
                    'difference': float(diff),
                    'movie1_value': float(movie1_features[i]),
                    'movie2_value': float(movie2_features[i])
                })
        
        top_diffs.sort(key=lambda x: x['difference'], reverse=True)
        
        # Generate comparison text
        if pred1 > pred2:
            better = movie1_metadata.get('title', 'Movie 1')
            worse = movie2_metadata.get('title', 'Movie 2')
            diff_text = f"{better} is recommended over {worse}"
        else:
            better = movie2_metadata.get('title', 'Movie 2')
            worse = movie1_metadata.get('title', 'Movie 1')
            diff_text = f"{better} is recommended over {worse}"
        
        return {
            'comparison': diff_text,
            'prediction_diff': abs(pred1 - pred2),
            'key_differences': top_diffs[:5]
        }


class SimpleExplainer:
    """
    Lightweight explanation system without SHAP
    Uses rule-based heuristics
    """
    
    def __init__(self):
        pass
    
    def explain(
        self,
        prediction: float,
        movie_metadata: Dict,
        user_history: List[Dict]
    ) -> str:
        """
        Generate simple rule-based explanation
        """
        reasons = []
        
        # Genre matching
        movie_genres = set(movie_metadata.get('genres', []))
        if user_history:
            user_genres = set()
            for movie in user_history:
                user_genres.update(movie.get('genres', []))
            
            common_genres = movie_genres & user_genres
            if common_genres:
                genres_str = ', '.join(list(common_genres)[:2])
                reasons.append(f"it's {genres_str}, which you enjoy")
        
        # Rating-based
        rating = movie_metadata.get('vote_average', 0)
        if rating >= 8.0:
            reasons.append("it has excellent reviews")
        elif rating >= 7.0:
            reasons.append("it's highly rated")
        
        # Popularity
        popularity = movie_metadata.get('popularity', 0)
        if popularity > 100:
            reasons.append("it's very popular")
        
        # Year
        year = movie_metadata.get('year', 0)
        if year >= 2020:
            reasons.append("it's a recent release")
        
        # Build explanation
        title = movie_metadata.get('title', 'This movie')
        
        if not reasons:
            return f"{title} matches your viewing patterns."
        elif len(reasons) == 1:
            return f"{title} is recommended because {reasons[0]}."
        else:
            return f"{title} is recommended because {', '.join(reasons[:-1])}, and {reasons[-1]}."


# Example usage
if __name__ == '__main__':
    print("Explainable Recommender Module")
    print("=" * 60)
    
    # Example without actual model
    feature_names = [
        'genre_Action', 'genre_Comedy', 'genre_Drama',
        'year', 'vote_average', 'popularity',
        'director_quality', 'cast_quality'
    ]
    
    movie_metadata = {
        'title': 'Inception',
        'year': 2010,
        'genres': ['Action', 'Sci-Fi', 'Thriller'],
        'vote_average': 8.8
    }
    
    simple_explainer = SimpleExplainer()
    user_history = [
        {'title': 'The Matrix', 'genres': ['Action', 'Sci-Fi']},
        {'title': 'Interstellar', 'genres': ['Sci-Fi', 'Drama']}
    ]
    
    explanation = simple_explainer.explain(4.5, movie_metadata, user_history)
    print(f"\nExplanation: {explanation}")
