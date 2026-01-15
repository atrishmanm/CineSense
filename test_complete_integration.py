"""
Complete Integration Test for CineSense
Tests all AI features, lazy loading, and infinite scroll
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database.db_manager import db
from ai.recommender import recommender
from ai.cache_manager import cache_manager
from ai.candidate_generator import CandidateGenerator
from tmdb.fetcher import TMDBFetcher
import requests
import time

def test_database_connection():
    """Test 1: Database connectivity"""
    print("\n" + "="*60)
    print("TEST 1: Database Connection")
    print("="*60)
    
    try:
        count = db.get_movie_count()
        print(f"✓ Database connected")
        print(f"  Total movies in database: {count}")
        return True
    except Exception as e:
        print(f"✗ Database connection failed: {e}")
        return False

def test_ai_layer_1_pairwise():
    """Test 2: Pairwise Learning Layer"""
    print("\n" + "="*60)
    print("TEST 2: AI Layer 1 - Pairwise Learning (Bradley-Terry)")
    print("="*60)
    
    try:
        # Check if pairwise learning model exists
        has_model = hasattr(recommender, 'pairwise_model')
        print(f"✓ Pairwise model initialized: {has_model}")
        
        # Get a comparison pair
        pair = recommender.get_comparison_pair(user_id=1)
        if pair and len(pair) == 2:
            print(f"✓ Comparison pair generated")
            print(f"  Movie 1: {pair[0].get('title', 'Unknown')}")
            print(f"  Movie 2: {pair[1].get('title', 'Unknown')}")
            return True
        else:
            print(f"✗ Failed to generate comparison pair")
            return False
    except Exception as e:
        print(f"✗ Pairwise learning test failed: {e}")
        return False

def test_ai_layer_2_embeddings():
    """Test 3: Content-Based Embeddings Layer"""
    print("\n" + "="*60)
    print("TEST 3: AI Layer 2 - Content-Based Embeddings")
    print("="*60)
    
    try:
        # Check if embeddings model exists
        has_embeddings = hasattr(recommender, 'embeddings')
        print(f"✓ Embeddings model initialized: {has_embeddings}")
        
        # Test embedding generation
        movies = db.get_random_movies(limit=2)
        if movies and len(movies) >= 2:
            embedding1 = recommender.embeddings.get_embedding(movies[0]['movie_id'])
            embedding2 = recommender.embeddings.get_embedding(movies[1]['movie_id'])
            
            if embedding1 is not None and embedding2 is not None:
                print(f"✓ Embeddings generated successfully")
                print(f"  Embedding dimension: {len(embedding1)}")
                
                # Test similarity
                similarity = recommender.embeddings.compute_similarity(
                    movies[0]['movie_id'], 
                    movies[1]['movie_id']
                )
                print(f"  Similarity score: {similarity:.4f}")
                return True
        
        print(f"✗ Embedding generation failed")
        return False
    except Exception as e:
        print(f"✗ Embeddings test failed: {e}")
        return False

def test_ai_layer_3_reinforcement():
    """Test 4: Reinforcement Learning Layer"""
    print("\n" + "="*60)
    print("TEST 4: AI Layer 3 - Reinforcement Learning (UCB Bandit)")
    print("="*60)
    
    try:
        # Check if RL model exists
        has_rl = hasattr(recommender, 'rl_agent')
        print(f"✓ RL agent initialized: {has_rl}")
        
        # Test arm selection
        if has_rl:
            arm = recommender.rl_agent.select_arm()
            print(f"✓ RL arm selected: {arm}")
            
            # Test reward update
            recommender.rl_agent.update(arm, 0.8)
            print(f"✓ RL reward updated")
            return True
        
        return False
    except Exception as e:
        print(f"✗ RL test failed: {e}")
        return False

def test_advanced_feature_1_latent_space():
    """Test 5: Latent Space Encoding"""
    print("\n" + "="*60)
    print("TEST 5: Advanced Feature 1 - Latent Space Encoding (PCA/SVD)")
    print("="*60)
    
    try:
        has_latent = hasattr(recommender.embeddings, 'reduce_dimensionality')
        print(f"✓ Latent space encoding available: {has_latent}")
        
        if has_latent:
            # Test dimensionality reduction
            movies = db.get_random_movies(limit=5)
            embeddings = [recommender.embeddings.get_embedding(m['movie_id']) for m in movies]
            embeddings = [e for e in embeddings if e is not None]
            
            if embeddings:
                reduced = recommender.embeddings.reduce_dimensionality(embeddings, n_components=10)
                print(f"✓ Dimensionality reduced: {len(embeddings[0])} -> {len(reduced[0])}")
                return True
        
        return False
    except Exception as e:
        print(f"✗ Latent space test failed: {e}")
        return False

def test_advanced_feature_2_implicit_signals():
    """Test 6: Implicit Signal Processing"""
    print("\n" + "="*60)
    print("TEST 6: Advanced Feature 2 - Implicit Signal Processing")
    print("="*60)
    
    try:
        has_implicit = hasattr(recommender, 'process_implicit_feedback')
        print(f"✓ Implicit signal processing available: {has_implicit}")
        
        if has_implicit:
            # Test processing implicit feedback
            recommender.process_implicit_feedback(
                user_id=1,
                movie_id=100,
                action_type='view',
                duration=120
            )
            print(f"✓ Implicit signal processed (view, 120s)")
            return True
        
        return False
    except Exception as e:
        print(f"✗ Implicit signals test failed: {e}")
        return False

def test_advanced_feature_3_probabilistic():
    """Test 7: Probabilistic Selection"""
    print("\n" + "="*60)
    print("TEST 7: Advanced Feature 3 - Probabilistic Selection (Softmax)")
    print("="*60)
    
    try:
        has_prob = hasattr(recommender, 'select_probabilistic')
        print(f"✓ Probabilistic selection available: {has_prob}")
        
        if has_prob:
            candidates = db.get_random_movies(limit=10)
            selected = recommender.select_probabilistic(candidates, temperature=1.0)
            print(f"✓ Probabilistic selection executed")
            print(f"  Selected: {selected.get('title', 'Unknown')}")
            return True
        
        return False
    except Exception as e:
        print(f"✗ Probabilistic selection test failed: {e}")
        return False

def test_advanced_feature_4_temporal():
    """Test 8: Temporal Memory Management"""
    print("\n" + "="*60)
    print("TEST 8: Advanced Feature 4 - Temporal Memory Management")
    print("="*60)
    
    try:
        has_temporal = hasattr(recommender, 'update_temporal_weights')
        print(f"✓ Temporal memory available: {has_temporal}")
        
        if has_temporal:
            recommender.update_temporal_weights(user_id=1)
            print(f"✓ Temporal weights updated")
            return True
        
        return False
    except Exception as e:
        print(f"✗ Temporal memory test failed: {e}")
        return False

def test_advanced_feature_5_nlg():
    """Test 9: Natural Language Generation"""
    print("\n" + "="*60)
    print("TEST 9: Advanced Feature 5 - Natural Language Explanations")
    print("="*60)
    
    try:
        has_nlg = hasattr(recommender, 'generate_explanation')
        print(f"✓ NLG available: {has_nlg}")
        
        if has_nlg:
            movie = db.get_random_movies(limit=1)[0]
            explanation = recommender.generate_explanation(movie, user_id=1)
            print(f"✓ Explanation generated")
            print(f"  Movie: {movie.get('title', 'Unknown')}")
            print(f"  Explanation: {explanation[:100]}...")
            return True
        
        return False
    except Exception as e:
        print(f"✗ NLG test failed: {e}")
        return False

def test_lazy_loading_cache():
    """Test 10: Lazy Loading - Cache Manager"""
    print("\n" + "="*60)
    print("TEST 10: Lazy Loading - Cache Manager")
    print("="*60)
    
    try:
        stats = cache_manager.get_stats()
        print(f"✓ Cache manager operational")
        print(f"  Movies cached: {stats['movie_cache']['size']}/{stats['movie_cache']['max_size']}")
        print(f"  Vectors cached: {stats['vector_cache']['size']}/{stats['vector_cache']['max_size']}")
        print(f"  Memory usage: {stats['memory_usage']}")
        return True
    except Exception as e:
        print(f"✗ Cache manager test failed: {e}")
        return False

def test_lazy_loading_candidate_generator():
    """Test 11: Lazy Loading - Candidate Generator"""
    print("\n" + "="*60)
    print("TEST 11: Lazy Loading - Candidate Generator")
    print("="*60)
    
    try:
        generator = CandidateGenerator()
        candidates = generator.generate_candidates(user_id=1, count=50)
        
        print(f"✓ Candidate generator operational")
        print(f"  Candidates generated: {len(candidates)}")
        print(f"  Strategy: {generator.strategy}")
        return True
    except Exception as e:
        print(f"✗ Candidate generator test failed: {e}")
        return False

def test_lazy_loading_tmdb_stream():
    """Test 12: Lazy Loading - TMDB Infinite Stream"""
    print("\n" + "="*60)
    print("TEST 12: Lazy Loading - TMDB Infinite Stream")
    print("="*60)
    
    try:
        fetcher = TMDBFetcher()
        
        # Test fetching a specific page
        data = fetcher.get_popular_movies(page=5)
        
        if data and 'results' in data:
            print(f"✓ TMDB streaming operational")
            print(f"  Page 5 fetched: {len(data['results'])} movies")
            print(f"  Total pages available: {data.get('total_pages', 'Unknown')}")
            print(f"  Total movies: {data.get('total_results', 'Unknown')}")
            return True
        
        return False
    except Exception as e:
        print(f"✗ TMDB stream test failed: {e}")
        return False

def test_lazy_loading_api():
    """Test 13: Lazy Loading - API Endpoints"""
    print("\n" + "="*60)
    print("TEST 13: Lazy Loading - API Endpoints")
    print("="*60)
    
    try:
        # Test lazy recommendations endpoint
        response1 = requests.get('http://localhost:5000/api/recommendations/lazy?limit=10&offset=0')
        
        if response1.status_code == 200:
            data1 = response1.json()
            print(f"✓ Lazy recommendations endpoint working")
            print(f"  Movies returned: {len(data1.get('movies', []))}")
        else:
            print(f"✗ Lazy recommendations failed: {response1.status_code}")
            return False
        
        # Test cache stats endpoint
        response2 = requests.get('http://localhost:5000/api/cache/stats')
        
        if response2.status_code == 200:
            data2 = response2.json()
            print(f"✓ Cache stats endpoint working")
            print(f"  Memory savings: {data2.get('memory_savings', 'Unknown')}")
        else:
            print(f"✗ Cache stats failed: {response2.status_code}")
            return False
        
        return True
    except Exception as e:
        print(f"✗ API endpoints test failed: {e}")
        return False

def test_infinite_scroll():
    """Test 14: Infinite Scroll - Dynamic TMDB Fetching"""
    print("\n" + "="*60)
    print("TEST 14: Infinite Scroll - Dynamic TMDB Fetching")
    print("="*60)
    
    try:
        # Test fetching with offset that exceeds database
        response = requests.get('http://localhost:5000/api/movie/top-rated?order_by=popularity&limit=20&offset=100')
        
        if response.status_code == 200:
            data = response.json()
            print(f"✓ Infinite scroll working")
            print(f"  Movies fetched at offset 100: {data.get('count', 0)}")
            print(f"  Source: {data.get('source', 'Unknown')}")
            print(f"  Has more: {data.get('has_more', False)}")
            return True
        else:
            print(f"✗ Infinite scroll failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"✗ Infinite scroll test failed: {e}")
        return False

def run_all_tests():
    """Run all integration tests"""
    print("\n" + "#"*60)
    print("CINESENSE - COMPLETE INTEGRATION TEST")
    print("#"*60)
    
    tests = [
        ("Database Connection", test_database_connection),
        ("AI Layer 1: Pairwise Learning", test_ai_layer_1_pairwise),
        ("AI Layer 2: Embeddings", test_ai_layer_2_embeddings),
        ("AI Layer 3: Reinforcement Learning", test_ai_layer_3_reinforcement),
        ("Advanced Feature 1: Latent Space", test_advanced_feature_1_latent_space),
        ("Advanced Feature 2: Implicit Signals", test_advanced_feature_2_implicit_signals),
        ("Advanced Feature 3: Probabilistic Selection", test_advanced_feature_3_probabilistic),
        ("Advanced Feature 4: Temporal Memory", test_advanced_feature_4_temporal),
        ("Advanced Feature 5: NLG Explanations", test_advanced_feature_5_nlg),
        ("Lazy Loading: Cache Manager", test_lazy_loading_cache),
        ("Lazy Loading: Candidate Generator", test_lazy_loading_candidate_generator),
        ("Lazy Loading: TMDB Stream", test_lazy_loading_tmdb_stream),
        ("Lazy Loading: API Endpoints", test_lazy_loading_api),
        ("Infinite Scroll: Dynamic Fetching", test_infinite_scroll),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ Test crashed: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status} - {name}")
    
    print("\n" + "="*60)
    print(f"TOTAL: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    print("="*60)
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! CineSense is fully operational!")
    else:
        print(f"\n⚠ {total - passed} test(s) failed. Check logs above.")
    
    return passed == total

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
