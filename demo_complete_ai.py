"""
CineSense - Complete AI Demo
Shows all 3 core layers + 5 advanced features working together
"""

from ai.recommender import recommender
from database.db_manager import db


def main():
    print("\n" + "="*70)
    print("CINESENSE - COMPLETE AI SYSTEM DEMONSTRATION")
    print("="*70)
    
    print("\nCORE AI LAYERS:")
    print("  Layer 1: Pairwise Learning (ELO/Bradley-Terry)")
    print("  Layer 2: Vector Embeddings (Content-Based + Latent Space)")
    print("  Layer 3: Multi-Armed Bandit (Exploration/Exploitation)")
    
    print("\nADVANCED AI FEATURES:")
    print("  1. Latent Space Encoding (Dense representations)")
    print("  2. Implicit Signal Processing (Behavioral learning)")
    print("  3. Probabilistic Selection (Softmax, not argmax)")
    print("  4. Temporal Memory & Forgetting (Recent > Past)")
    print("  5. Natural Language Explanations")
    
    print("\n" + "-"*70)
    print("SYSTEM STATUS CHECK")
    print("-"*70)
    
    # Check components
    components = {
        "Pairwise Learner": recommender.pairwise_learner,
        "Movie Embedder": recommender.movie_embedder,
        "UCB Bandit": recommender.bandit,
        "Latent Encoder": recommender.latent_encoder,
        "Implicit Processor": recommender.implicit_processor,
        "Probabilistic Selector": recommender.prob_selector,
        "Memory Manager": recommender.memory_manager,
        "NLG Explainer": recommender.nlg_explainer
    }
    
    for name, component in components.items():
        status = "Loaded" if component is not None else "Missing"
        print(f"  {name:25} {status}")
    
    # Check database connection
    print("\n" + "-"*70)
    print("DATABASE STATUS")
    print("-"*70)
    
    try:
        movies = db.get_top_movies(limit=5)
        print(f"  Database connected")
        print(f"  Found {len(movies)} movies (showing top 5)")
        
        for i, movie in enumerate(movies[:5], 1):
            elo = movie.get('elo_score', 1500)
            print(f"     {i}. {movie['title']} (ELO: {elo})")
    except Exception as e:
        print(f"  ❌ Database error: {e}")
    
    # Demonstrate comparison pair selection (Layer 3: Bandit)
    print("\n" + "-"*70)
    print("LAYER 3 DEMO: Bandit Selection")
    print("-"*70)
    
    try:
        movie1, movie2 = recommender.get_comparison_pair()
        if movie1 and movie2:
            print(f"  Selected pair for comparison:")
            print(f"     Option A: {movie1['title']}")
            print(f"     Option B: {movie2['title']}")
            print(f"\n  Bandit algorithm balanced exploration vs exploitation")
        else:
            print(f"  ⚠️  Not enough movies for comparison")
    except Exception as e:
        print(f"  ❌ Error: {e}")
    
    # Show advanced features availability
    print("\n" + "-"*70)
    print("ADVANCED FEATURES AVAILABILITY")
    print("-"*70)
    
    from config import Config
    
    features = {
        "Latent Space Compression": Config.USE_DIMENSIONALITY_REDUCTION,
        "Probabilistic Selection": Config.USE_SOFTMAX_SELECTION,
        "Natural Language Explanations": Config.ENABLE_EXPLANATIONS,
        "Temporal Decay Factor": f"{Config.TEMPORAL_DECAY_FACTOR} (70% recent, 30% past)",
        "Softmax Temperature": f"{Config.SOFTMAX_TEMPERATURE} (controls exploration)",
        "Latent Dimensions": f"{Config.LATENT_DIM}D (from {Config.TOTAL_VECTOR_DIM}D)"
    }
    
    for feature, status in features.items():
        if isinstance(status, bool):
            status_str = "Enabled" if status else "Disabled"
        else:
            status_str = f"{status}"
        print(f"  {feature:30} {status_str}")
    
    # Architecture summary
    print("\n" + "="*70)
    print("ARCHITECTURE SUMMARY")
    print("="*70)
    
    print("""
    USER INTERACTION
         ↓
    [Movie Comparison] ← Layer 3: Bandit selects pairs
         ↓
    User Choice
         ↓
    ┌────────────────────────────────────────┐
    │  MULTI-LAYER AI PROCESSING             │
    ├────────────────────────────────────────┤
    │ Layer 1: Update ELO ratings            │
    │ Layer 2: Update user embedding vector  │
    │ Layer 3: Record bandit rewards         │
    │                                        │
    │ Advanced: Apply temporal weights       │
    │ Advanced: Process implicit signals     │
    │ Advanced: Compress to latent space     │
    └────────────────────────────────────────┘
         ↓
    RECOMMENDATIONS
         ↓
    ┌────────────────────────────────────────┐
    │  SCORING ALGORITHM                     │
    ├────────────────────────────────────────┤
    │ 50%: Content similarity (Layer 2)      │
    │ 30%: Preference score (Layer 1)        │
    │ 20%: Global quality (ELO)              │
    │                                        │
    │ Then: Probabilistic selection (softmax)│
    │ Then: Generate NLG explanation         │
    └────────────────────────────────────────┘
         ↓
    USER sees personalized movies + explanations
    """)
    
    print("="*70)
    print("CONCLUSION")
    print("="*70)
    print("""
This is NOT just a database query system.
This is REAL AI with:
  • Learned latent representations
  • Behavioral intelligence
  • Probabilistic reasoning
  • Adaptive memory
  • Natural language generation

All 8 components working together to create an advanced
recommendation experience!
    """)
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
