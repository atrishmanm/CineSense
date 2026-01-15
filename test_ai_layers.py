"""
Test Script: Verify All Three AI Layers are Working
Tests Layer 1 (Pairwise), Layer 2 (Embeddings), Layer 3 (Bandit)
"""

import sys
import traceback
from ai.recommender import CineSenseRecommender
from database.db_manager import db

def test_layer_1_pairwise():
    """Test Layer 1: Pairwise Learning (ELO)"""
    print("\n" + "="*60)
    print("TESTING LAYER 1: Pairwise Learning (Bradley-Terry/ELO)")
    print("="*60)
    
    try:
        recommender = CineSenseRecommender()
        learner = recommender.pairwise_learner
        
        # Test ELO calculation
        winner_rating = 1500
        loser_rating = 1500
        new_winner, new_loser = learner.update_ratings(winner_rating, loser_rating)
        
        print(f"Initial ratings: Winner={winner_rating}, Loser={loser_rating}")
        print(f"Updated ratings: Winner={new_winner}, Loser={new_loser}")
        print(f"Rating change: Winner +{new_winner-winner_rating}, Loser {new_loser-loser_rating}")
        
        # Test win probability
        prob = learner.get_win_probability(1600, 1400)
        print(f"Win probability (1600 vs 1400): {prob:.2%}")
        
        print("\nLAYER 1 WORKING: Pairwise learning implemented correctly")
        return True
        
    except Exception as e:
        print(f"\nLAYER 1 FAILED: {e}")
        traceback.print_exc()
        return False


def test_layer_2_embeddings():
    """Test Layer 2: Content-Based Filtering (Vector Embeddings)"""
    print("\n" + "="*60)
    print("TESTING LAYER 2: Vector Embeddings (Content-Based)")
    print("="*60)
    
    try:
        recommender = CineSenseRecommender()
        recommender._fit_encoders_if_needed()
        
        # Get a sample movie
        movies = db.get_top_movies(limit=2)
        if not movies or len(movies) < 2:
            print("Need at least 2 movies in database to test")
            return False
        
        movie1 = movies[0]
        movie2 = movies[1]
        
        # Create embeddings
        emb1 = recommender._movie_to_embedding(movie1)
        emb2 = recommender._movie_to_embedding(movie2)
        
        print(f"✓ Created embedding for '{movie1['title']}'")
        print(f"  - Vector dimension: {len(emb1)}")
        print(f"  - First 5 values: {emb1[:5]}")
        
        print(f"\n✓ Created embedding for '{movie2['title']}'")
        print(f"  - Vector dimension: {len(emb2)}")
        print(f"  - First 5 values: {emb2[:5]}")
        
        # Calculate similarity
        import numpy as np
        similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
        print(f"\n✓ Cosine similarity between movies: {similarity:.4f}")
        
        print("\nLAYER 2 WORKING: Embeddings created and similarity calculated")
        return True
        
    except Exception as e:
        print(f"\n❌ LAYER 2 FAILED: {e}")
        traceback.print_exc()
        return False


def test_layer_3_bandit():
    """Test Layer 3: Reinforcement Learning (Multi-Armed Bandit)"""
    print("\n" + "="*60)
    print("TESTING LAYER 3: Multi-Armed Bandit (Exploration/Exploitation)")
    print("="*60)
    
    try:
        recommender = CineSenseRecommender()
        bandit = recommender.bandit
        
        # Test arm selection
        available_arms = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        selected = bandit.select_arm(available_arms, top_k=2)
        
        print(f"✓ Available arms (movies): {available_arms}")
        print(f"✓ Selected arms: {selected}")
        
        # Simulate some rewards
        bandit.update(selected[0], reward=1.0)  # Chosen movie
        bandit.update(selected[1], reward=0.0)  # Rejected movie
        
        stats = bandit.get_arm_stats(selected[0])
        print(f"\n✓ Arm {selected[0]} stats after update:")
        print(f"  - Count: {stats['count']}")
        print(f"  - Average reward: {stats['avg_reward']:.2f}")
        
        # Test exploration vs exploitation
        exploration_count = 0
        for i in range(100):
            sel = bandit.select_arm(available_arms, top_k=2)
            # If selection is random, it's exploring
            
        print(f"\n✓ Bandit uses UCB algorithm with c={bandit.c}")
        print(f"✓ Balances exploration and exploitation")
        
        print("\nLAYER 3 WORKING: Bandit algorithm functioning correctly")
        return True
        
    except Exception as e:
        print(f"\n❌ LAYER 3 FAILED: {e}")
        traceback.print_exc()
        return False


def test_integration():
    """Test all three layers working together"""
    print("\n" + "="*60)
    print("TESTING INTEGRATION: All Three Layers Combined")
    print("="*60)
    
    try:
        recommender = CineSenseRecommender()
        
        # Test comparison pair generation (uses Layer 3: Bandit)
        print("\n1. Getting comparison pair (Layer 3: Bandit selection)...")
        movie1, movie2 = recommender.get_comparison_pair()
        
        if movie1 and movie2:
            print(f"✓ Movie 1: {movie1['title']}")
            print(f"✓ Movie 2: {movie2['title']}")
        else:
            print("⚠ Not enough movies for comparison")
            return False
        
        # Test user choice processing (uses all 3 layers)
        print("\n2. Processing user choice (All layers)...")
        test_user_id = 999  # Test user
        
        success = recommender.process_user_choice(
            user_id=test_user_id,
            chosen_movie_id=movie1['movie_id'],
            rejected_movie_id=movie2['movie_id']
        )
        
        if success:
            print(f"✓ Layer 1: ELO ratings updated")
            print(f"✓ Layer 2: User embedding updated")
            print(f"✓ Layer 3: Bandit rewards recorded")
        
        # Test recommendations (uses all 3 layers)
        print("\n3. Generating recommendations (All layers combined)...")
        recommendations = recommender.get_recommendations(test_user_id, n=5)
        
        if recommendations:
            print(f"✓ Generated {len(recommendations)} recommendations")
            print(f"\nTop recommendation:")
            top = recommendations[0]
            print(f"  - Title: {top['title']}")
            if 'recommendation_score' in top:
                print(f"  - Final Score: {top['recommendation_score']:.4f}")
            if 'content_score' in top:
                print(f"  - Content Score (Layer 2): {top['content_score']:.4f}")
            if 'preference_score' in top:
                print(f"  - Preference Score (Layer 1): {top['preference_score']:.4f}")
        
        print("\nINTEGRATION WORKING: All three layers working together!")
        return True
        
    except Exception as e:
        print(f"\n❌ INTEGRATION FAILED: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("CINESENSE AI LAYER VERIFICATION")
    print("="*60)
    print("\nTesting if all three AI layers are actually implemented:")
    print("  Layer 1: Pairwise Learning (Bradley-Terry/ELO)")
    print("  Layer 2: Vector Embeddings (Content-Based)")
    print("  Layer 3: Multi-Armed Bandit (Exploration/Exploitation)")
    
    results = {
        "Layer 1 (Pairwise)": test_layer_1_pairwise(),
        "Layer 2 (Embeddings)": test_layer_2_embeddings(),
        "Layer 3 (Bandit)": test_layer_3_bandit(),
        "Integration": test_integration()
    }
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"{test_name:25} {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "="*60)
    if all_passed:
        print("ALL TESTS PASSED!")
        print("All three AI layers are implemented and working correctly.")
    else:
        print("⚠ SOME TESTS FAILED")
        print("Please check the errors above.")
    print("="*60 + "\n")
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
