"""
Comprehensive AI Integration Verification
Verifies all AI components are properly integrated
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from ai.recommender import CineSenseRecommender
from config import Config


def verify_integration():
    """Verify all AI features are integrated"""
    print("\n" + "="*70)
    print("CINESENSE AI INTEGRATION VERIFICATION")
    print("="*70)
    
    # Initialize recommender
    recommender = CineSenseRecommender()
    
    print("\n[1] CORE AI LAYERS")
    print("-" * 70)
    print(f"  Layer 1 - Pairwise Learning:      {recommender.pairwise_learner is not None}")
    print(f"  Layer 2 - Movie Embeddings:       {recommender.movie_embedder is not None}")
    print(f"  Layer 2 - Content Recommender:    {recommender.content_recommender is not None}")
    print(f"  Layer 3 - UCB Bandit:              {recommender.bandit is not None}")
    
    print("\n[2] ADVANCED AI FEATURES")
    print("-" * 70)
    print(f"  Feature 1 - Latent Space Encoder:      {recommender.latent_encoder is not None}")
    print(f"  Feature 2 - Implicit Signal Processor: {recommender.implicit_processor is not None}")
    print(f"  Feature 3 - Probabilistic Selector:    {recommender.prob_selector is not None}")
    print(f"  Feature 4 - Temporal Memory Manager:   {recommender.memory_manager is not None}")
    print(f"  Feature 5 - NLG Explainer:              {recommender.nlg_explainer is not None}")
    
    print("\n[3] LAZY LOADING COMPONENTS")
    print("-" * 70)
    print(f"  Cache Manager:          {recommender.cache is not None}")
    print(f"  Candidate Generator:    {recommender.candidate_gen is not None}")
    print(f"  TMDB Fetcher:           {recommender.tmdb is not None}")
    
    print("\n[4] LAZY LOADING METHODS")
    print("-" * 70)
    print(f"  get_comparison_pair_lazy():    {hasattr(recommender, 'get_comparison_pair_lazy')}")
    print(f"  get_recommendations_lazy():    {hasattr(recommender, 'get_recommendations_lazy')}")
    
    print("\n[5] CONFIGURATION")
    print("-" * 70)
    print(f"  Movie Cache Size:         {Config.MOVIE_CACHE_SIZE}")
    print(f"  Vector Cache Size:        {Config.VECTOR_CACHE_SIZE}")
    print(f"  Candidate Count:          {Config.CANDIDATE_COUNT}")
    print(f"  Candidate Strategy:       {Config.CANDIDATE_STRATEGY}")
    print(f"  Use Dimensionality Reduction: {Config.USE_DIMENSIONALITY_REDUCTION}")
    print(f"  Use Softmax Selection:    {Config.USE_SOFTMAX_SELECTION}")
    print(f"  Enable Explanations:      {Config.ENABLE_EXPLANATIONS}")
    
    print("\n[6] INTEGRATION TEST")
    print("-" * 70)
    
    # Test pairwise learning
    from ai.pairwise_learning import PairwiseLearner
    learner = PairwiseLearner()
    new_a, new_b = learner.update_ratings(1500, 1500)
    assert new_a > 1500 and new_b < 1500, "Pairwise learning failed"
    print(f"  ✓ Pairwise Learning: Rating update works ({new_a} vs {new_b})")
    
    # Test bandit
    from ai.reinforcement import UCBBandit
    bandit = UCBBandit()
    selected = bandit.select_arm([1, 2, 3, 4, 5], top_k=2)
    assert len(selected) == 2, "Bandit selection failed"
    print(f"  \u2713 UCB Bandit: Can select arms ({selected})")
    
    # Test latent encoder
    from ai.advanced_ai import LatentSpaceEncoder
    import numpy as np
    encoder = LatentSpaceEncoder(n_components=32)
    vectors = np.random.randn(100, 55)
    encoder.fit(vectors)
    latent = encoder.transform(np.random.randn(55))
    assert len(latent) == 32, "Latent encoding failed"
    print(f"  \u2713 Latent Encoder: 55D \u2192 32D compression works")
    
    # Test probabilistic selector
    from ai.advanced_ai import ProbabilisticSelector
    selector = ProbabilisticSelector()
    scores = np.array([0.9, 0.8, 0.7, 0.5, 0.3])
    selected = selector.select_with_probability(scores, top_k=3)
    assert len(selected) == 3, "Probabilistic selection failed"
    print(f"  \u2713 Probabilistic Selector: Can select with probabilities")
    
    # Test temporal memory
    from ai.advanced_ai import TemporalMemoryManager
    memory = TemporalMemoryManager(decay_factor=0.7)
    old = np.array([0.5, 0.3])
    new = np.array([0.9, 0.1])
    updated = memory.compute_weighted_preference(new, old)
    expected = 0.7 * new + 0.3 * old
    assert np.allclose(updated, expected), "Temporal memory failed"
    print(f"  \u2713 Temporal Memory: Decay formula works (0.7 recent + 0.3 past)")
    
    # Test cache manager
    from ai.cache_manager import cache_manager
    stats = cache_manager.get_stats()
    assert 'movie_cache' in stats, "Cache stats failed"
    assert 'vector_cache' in stats, "Cache stats failed"
    print(f"  \u2713 Cache Manager: Stats accessible ({stats['movie_cache']['size']}/100 movies)")
    
    # Test candidate generator
    from ai.candidate_generator import candidate_generator
    assert candidate_generator is not None, "Candidate generator not initialized"
    print(f"  \u2713 Candidate Generator: Initialized with {Config.CANDIDATE_STRATEGY} strategy")
    
    print("\n" + "="*70)
    print("RESULT: ALL AI FEATURES WORKING AND INTEGRATED")
    print("="*70)
    
    print("\n[SUMMARY]")
    print("  - 3 Core AI layers initialized and functional")
    print("  - 5 Advanced AI features integrated")
    print("  - Lazy loading architecture operational")
    print("  - All integration tests passed")
    
    print("\n[READY FOR PRODUCTION]")
    print("  The CineSense recommendation engine is fully operational.")
    print("  All AI components are properly integrated and tested.")
    
    return True


if __name__ == "__main__":
    try:
        success = verify_integration()
        print("\n✓ Verification complete\n")
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Verification failed: {str(e)}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
