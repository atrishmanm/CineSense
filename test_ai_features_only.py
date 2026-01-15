"""
AI Features Integration Test (No Server Required)
Tests all AI components without needing Flask running
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database.db_manager import db
from ai.recommender import recommender
from ai.cache_manager import cache_manager
from ai.candidate_generator import CandidateGenerator
from tmdb.fetcher import TMDBFetcher

print("\n" + "#"*60)
print("CINESENSE - AI FEATURES INTEGRATION TEST")
print("#"*60)

# Test 1: Database
print("\n[TEST 1] Database Connection")
try:
    count = db.get_movie_count()
    print(f"✓ PASS - Database has {count} movies")
except Exception as e:
    print(f"✗ FAIL - {e}")

# Test 2: Pairwise Learning
print("\n[TEST 2] AI Layer 1 - Pairwise Learning")
try:
    pair = recommender.get_comparison_pair(user_id=1)
    if pair and len(pair) == 2:
        print(f"✓ PASS - Generated pair: {pair[0]['title']} vs {pair[1]['title']}")
    else:
        print("✗ FAIL - No pair generated")
except Exception as e:
    print(f"✗ FAIL - {e}")

# Test 3: Embeddings
print("\n[TEST 3] AI Layer 2 - Content-Based Embeddings")
try:
    movies = db.get_random_movies(limit=2)
    if movies and len(movies) >= 2:
        emb1 = recommender.movie_embedder.get_embedding(movies[0])
        emb2 = recommender.movie_embedder.get_embedding(movies[1])
        if emb1 is not None and emb2 is not None:
            print(f"✓ PASS - Embedding dim: {len(emb1)}")
        else:
            print("✗ FAIL - Embeddings not generated")
    else:
        print("✗ FAIL - Not enough movies")
except Exception as e:
    print(f"✗ FAIL - {e}")

# Test 4: Reinforcement Learning
print("\n[TEST 4] AI Layer 3 - Reinforcement Learning (UCB)")
try:
    arm = recommender.bandit.select_arm()
    recommender.bandit.update(arm, 0.8)
    print(f"✓ PASS - RL bandit selected arm {arm}, reward updated")
except Exception as e:
    print(f"✗ FAIL - {e}")

# Test 5: Latent Space
print("\n[TEST 5] Advanced Feature - Latent Space Encoding")
try:
    if recommender.latent_encoder:
        movies = db.get_random_movies(limit=5)
        embeddings = [recommender.movie_embedder.get_embedding(m) for m in movies]
        embeddings = [e for e in embeddings if e is not None]
        if embeddings:
            reduced = recommender.latent_encoder.reduce_dimensionality(embeddings, n_components=10)
            print(f"✓ PASS - Reduced {len(embeddings[0])} -> {len(reduced[0])} dimensions")
        else:
            print("✗ FAIL - No embeddings")
    else:
        print("⊘ SKIP - Latent space encoding disabled in config")
except Exception as e:
    print(f"✗ FAIL - {e}")

# Test 6: Implicit Signals
print("\n[TEST 6] Advanced Feature - Implicit Signal Processing")
try:
    recommender.implicit_processor.record_signal(user_id=1, movie_id=100, signal_type='view', value=120)
    print("✓ PASS - Implicit signal recorded")
except Exception as e:
    print(f"✗ FAIL - {e}")

# Test 7: Probabilistic Selection
print("\n[TEST 7] Advanced Feature - Probabilistic Selection")
try:
    if recommender.prob_selector:
        candidates = db.get_random_movies(limit=10)
        scores = np.random.rand(len(candidates))
        selected_idx = recommender.prob_selector.select(scores, temperature=1.0)
        print(f"✓ PASS - Selected index {selected_idx} probabilistically")
    else:
        print("⊘ SKIP - Probabilistic selection disabled in config")
except Exception as e:
    print(f"✗ FAIL - {e}")

# Test 8: Temporal Memory
print("\n[TEST 8] Advanced Feature - Temporal Memory")
try:
    recommender.memory_manager.update_weights(user_id=1, recent_interactions=5)
    print("✓ PASS - Temporal weights updated")
except Exception as e:
    print(f"✗ FAIL - {e}")

# Test 9: NLG Explanations
print("\n[TEST 9] Advanced Feature - Natural Language Explanations")
try:
    if recommender.nlg_explainer:
        movie = db.get_random_movies(limit=1)[0]
        explanation = recommender.nlg_explainer.generate_explanation(movie, user_preferences={'genres': ['Action']})
        print(f"✓ PASS - Generated explanation for '{movie['title']}'")
        print(f"  '{explanation[:80]}...'")
    else:
        print("⊘ SKIP - NLG explanations disabled in config")
except Exception as e:
    print(f"✗ FAIL - {e}")

# Test 10: Cache Manager
print("\n[TEST 10] Lazy Loading - Cache Manager")
try:
    stats = cache_manager.get_stats()
    print(f"✓ PASS - Movies: {stats['movie_cache']['size']}/{stats['movie_cache']['max_size']}, "
          f"Vectors: {stats['vector_cache']['size']}/{stats['vector_cache']['max_size']}")
except Exception as e:
    print(f"✗ FAIL - {e}")

# Test 11: Candidate Generator
print("\n[TEST 11] Lazy Loading - Candidate Generator")
try:
    generator = CandidateGenerator()
    candidates = generator.generate_candidates(user_id=1, limit=50)
    print(f"✓ PASS - Generated {len(candidates)} candidates")
except Exception as e:
    print(f"✗ FAIL - {e}")

# Test 12: TMDB Stream
print("\n[TEST 12] Lazy Loading - TMDB Infinite Stream")
try:
    fetcher = TMDBFetcher()
    data = fetcher.get_popular_movies(page=5)
    if data and 'results' in data:
        print(f"✓ PASS - Fetched page 5: {len(data['results'])} movies")
        print(f"  Total available: {data.get('total_results', 'Unknown')} movies across {data.get('total_pages', 'Unknown')} pages")
    else:
        print("✗ FAIL - No data from TMDB")
except Exception as e:
    print(f"✗ FAIL - {e}")

# Test 13: Lazy Recommendations
print("\n[TEST 13] Lazy Recommendations Integration")
try:
    recs = recommender.get_recommendations(user_id=1, limit=10)
    print(f"✓ PASS - Generated {len(recs)} lazy recommendations")
except Exception as e:
    print(f"✗ FAIL - {e}")

# Test 14: Lazy Comparison Pair
print("\n[TEST 14] Lazy Comparison Pair Generation")
try:
    pair = recommender.get_comparison_pair(user_id=1)
    if pair and len(pair) == 2:
        print(f"✓ PASS - Lazy pair: {pair[0]['title']} vs {pair[1]['title']}")
    else:
        print("✗ FAIL - No lazy pair")
except Exception as e:
    print(f"✗ FAIL - {e}")

print("\n" + "="*60)
print("ALL AI FEATURES TESTED")
print("="*60)
print("\nNote: To test infinite scroll API, make sure Flask is running")
print("and check browser console logs when clicking 'Load More'")
