"""
Test Advanced AI Features - Enhanced Recommendation Intelligence
Demonstrates: Latent space, probabilistic selection, temporal memory, NLG explanations
"""

import sys
import numpy as np
from ai.advanced_ai import (
    LatentSpaceEncoder, ImplicitSignalProcessor, ProbabilisticSelector,
    TemporalMemoryManager, NaturalLanguageExplainer
)
from datetime import datetime, timedelta


def test_latent_space():
    """Test Feature 1: Latent Representations"""
    print("\n" + "="*60)
    print("FEATURE 1: Latent Space Encoding (Dense Representations)")
    print("="*60)
    
    # Create sample movie vectors (55-dimensional)
    movie_vectors = np.random.rand(100, 55)
    
    encoder = LatentSpaceEncoder(n_components=32)
    encoder.fit(movie_vectors)
    
    # Transform a movie to latent space
    test_movie = np.random.rand(55)
    latent = encoder.transform(test_movie)
    
    print(f"✓ Original dimensions: {len(test_movie)}")
    print(f"✓ Latent dimensions: {len(latent)}")
    print(f"✓ Compression ratio: {len(test_movie)/len(latent):.1f}x")
    print(f"✓ Sample latent vector: {latent[:5]}")
    print("\nThese numbers have no individual meaning - they're learned representations!")
    print("   This is how neural systems 'think' - in abstract feature space.")


def test_implicit_signals():
    """Test Feature 2: Implicit Signal Processing"""
    print("\n" + "="*60)
    print("FEATURE 2: Implicit Signals (Beyond Explicit Choices)")
    print("="*60)
    
    processor = ImplicitSignalProcessor()
    
    # Test different interaction scenarios
    scenarios = [
        {
            'name': 'Engaged User',
            'data': {'hover_time': 4.5, 'was_skipped': False, 'repeat_view': True},
            'base_reward': 1.0
        },
        {
            'name': 'Quick Skip',
            'data': {'hover_time': 0.5, 'was_skipped': True},
            'base_reward': 0.0
        },
        {
            'name': 'Abandoned Session',
            'data': {'hover_time': 2.0, 'session_abandoned': True},
            'base_reward': 1.0
        }
    ]
    
    for scenario in scenarios:
        enriched = processor.enrich_interaction(
            scenario['base_reward'],
            scenario['data']
        )
        print(f"\n{scenario['name']}:")
        print(f"  Base reward: {scenario['base_reward']:.2f}")
        print(f"  Enriched reward: {enriched:.2f}")
        print(f"  Signals: {scenario['data']}")
    
    print("\nThe AI learns from behavior, not just explicit clicks!")
    print("   Modern systems track your interaction patterns.")


def test_probabilistic_selection():
    """Test Feature 3: Probabilistic Decision-Making"""
    print("\n" + "="*60)
    print("FEATURE 3: Probabilistic Selection (Human-like Inconsistency)")
    print("="*60)
    
    selector = ProbabilisticSelector(temperature=0.8)
    
    # Movie scores
    movie_scores = [0.9, 0.85, 0.75, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05]
    movie_names = [f"Movie {i+1}" for i in range(len(movie_scores))]
    
    print("\nMovie Scores:")
    for name, score in zip(movie_names, movie_scores):
        print(f"  {name}: {score:.2f}")
    
    # Get probability distribution
    probabilities = selector.get_distribution(movie_scores)
    
    print("\nSelection Probabilities (using softmax):")
    for name, prob in zip(movie_names, probabilities):
        bar = "█" * int(prob * 50)
        print(f"  {name}: {prob:.2%} {bar}")
    
    # Simulate 100 selections
    print("\n100 Probabilistic Selections:")
    selection_counts = {}
    for _ in range(100):
        selected = selector.select_with_probability(movie_scores, top_k=1)
        movie_name = movie_names[selected[0]]
        selection_counts[movie_name] = selection_counts.get(movie_name, 0) + 1
    
    for name in movie_names[:5]:  # Top 5
        count = selection_counts.get(name, 0)
        bar = "█" * (count // 2)
        print(f"  {name}: {count} times {bar}")
    
    print("\nTop movie doesn't ALWAYS win - system explores!")
    print("   This prevents filter bubbles and feels more human.")


def test_temporal_memory():
    """Test Feature 4: Memory & Forgetting"""
    print("\n" + "="*60)
    print("FEATURE 4: Temporal Memory (Recent > Past)")
    print("="*60)
    
    manager = TemporalMemoryManager(decay_factor=0.7)
    
    # Create interactions with timestamps
    now = datetime.now()
    interactions = []
    for i in range(10):
        interactions.append({
            'movie_id': i,
            'timestamp': now - timedelta(days=i*7),  # Weekly intervals
            'data': f'Interaction {i}'
        })
    
    # Apply temporal weights
    weighted = manager.apply_temporal_weights(interactions)
    
    print("\nTemporal Weights (decay_factor=0.7):")
    print("  Recent interactions get full weight (1.0)")
    print("  Older interactions fade exponentially\n")
    
    for i, inter in enumerate(weighted[:7]):
        age_days = (now - inter['timestamp']).days
        weight = inter['temporal_weight']
        bar = "█" * int(weight * 30)
        print(f"  {age_days:3d} days ago: {weight:.3f} {bar}")
    
    # Test preference blending
    recent_pref = np.array([1.0, 0.0, 0.0])  # Loves action now
    past_pref = np.array([0.0, 0.0, 1.0])    # Loved romance before
    
    blended = manager.compute_weighted_preference(recent_pref, past_pref)
    
    print(f"\nPreference Evolution:")
    print(f"  Past preferences:   {past_pref}")
    print(f"  Recent preferences: {recent_pref}")
    print(f"  Blended (70% new):  {blended}")
    
    print("\nAI adapts to your changing taste!")
    print("   Formula: 0.7 * recent + 0.3 * past")
    print("   This creates the illusion of 'remembering' and 'evolving'.")


def test_natural_language():
    """Test Feature 5: Natural Language Explanations"""
    print("\n" + "="*60)
    print("FEATURE 5: Natural Language Explanations")
    print("="*60)
    
    explainer = NaturalLanguageExplainer(detail_level='medium')
    
    # Sample movie
    movie = {
        'title': 'Blade Runner 2049',
        'genres': 'Sci-Fi, Drama',
        'directors': 'Denis Villeneuve'
    }
    
    # Sample user profile
    user_profile = {
        'preferred_genres': ['Sci-Fi', 'Drama', 'Thriller']
    }
    
    # Sample factors
    factors = {
        'preferred_genres': ['Sci-Fi', 'Drama'],
        'favorite_directors': ['Denis Villeneuve'],
        'pacing_preference': 'slow-burn'
    }
    
    explanation = explainer.explain_recommendation(
        movie, user_profile, similarity_score=0.87, preference_factors=factors
    )
    
    print(f"\n🎬 Recommended Movie: {movie['title']}")
    print(f"Match Score: 87%")
    print(f"💬 Explanation:")
    print(f"   \"{explanation}\"")
    
    # Test taste summary
    taste_profile = {
        'top_genres': ['Sci-Fi', 'Drama', 'Thriller'],
        'avg_release_year': 2015,
        'avg_rating_preference': 8.2
    }
    
    summary = explainer.generate_taste_summary(taste_profile)
    
    print(f"\nYour Taste Personality:")
    print(f"   \"{summary}\"")
    
    # Learning progress explanations
    scenarios = [
        (3, 'improving'),
        (15, 'improving'),
        (50, 'stable')
    ]
    
    print(f"\nLearning Progress Messages:")
    for count, trend in scenarios:
        msg = explainer.explain_learning_progress(count, trend)
        print(f"   After {count} interactions: \"{msg}\"")
    
    print("\nThis is NOT canned text - it interprets model behavior!")
    print("   You're not lying - you're translating math into language.")
    print("   This is Natural Language Generation (NLG).")


def main():
    """Run all advanced AI feature tests"""
    print("\n" + "="*60)
    print("ADVANCED AI FEATURES - ENHANCED INTELLIGENCE")
    print("="*60)
    print("\nDemonstrating 5 key features that make AI feel 'smart':")
    print("  1. Latent Space Encoding (Dense representations)")
    print("  2. Implicit Signal Processing (Behavioral learning)")
    print("  3. Probabilistic Selection (Human-like exploration)")
    print("  4. Temporal Memory (Forgetting old tastes)")
    print("  5. Natural Language Explanations")
    
    test_latent_space()
    test_implicit_signals()
    test_probabilistic_selection()
    test_temporal_memory()
    test_natural_language()
    
    print("\n" + "="*60)
    print("ALL ADVANCED FEATURES DEMONSTRATED")
    print("="*60)
    print("\nSummary:")
    print("  • Moved from symbolic → latent representations")
    print("  • Learning from behavior, not just clicks")
    print("  • Probabilistic decisions prevent filter bubbles")
    print("  • Temporal decay makes AI 'forget' old tastes")
    print("  • NLG makes recommendations feel conversational")
    print("\nThis is REAL AI - not just database queries!")
    print("="*60 + "\n")


if __name__ == "__main__":
    sys.exit(main())
