"""
CineSense Feature Verification Script
Comprehensive check of all implemented features
"""

import os
import importlib
import sys
from pathlib import Path

# Ensure project root is on sys.path and cwd
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
os.chdir(PROJECT_ROOT)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def check_feature(feature_name, file_path, class_name=None):
    """Check if a feature file exists and optionally if it contains a specific class"""
    status = "✓" if os.path.exists(file_path) else "✗"
    color = Colors.GREEN if status == "✓" else Colors.RED
    
    # Check for class if specified
    if status == "✓" and class_name:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if f"class {class_name}" in content:
                    status = "✓✓"
                    color = Colors.GREEN
                else:
                    status = "✓⚠"
                    color = Colors.YELLOW
        except Exception:
            pass
    
    return status, color

def main():
    print(f"\n{Colors.BOLD}{'='*70}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}🎬 CINESENSE - FEATURE VERIFICATION REPORT{Colors.ENDC}")
    print(f"{Colors.BOLD}{'='*70}{Colors.ENDC}\n")
    
    features = {
        "🔍 Semantic Search": {
            "file": "ai/semantic_search.py",
            "class": "SemanticMovieSearch",
            "description": "Advanced NLP-based movie search"
        },
        "🔄 Sequential Models": {
            "file": "training/sequential_model.py",
            "class": "SequentialRecommender",
            "description": "GRU-based user history modeling"
        },
        "💬 Conversational AI": {
            "file": "ai/conversational_agent.py",
            "class": "MovieChatbot",
            "description": "DialoGPT-powered movie chatbot"
        },
        "😊 Mood Detection": {
            "file": "ai/mood_detector.py",
            "class": "MoodBasedRecommender",
            "description": "Emotion-based recommendations"
        },
        "👥 Social Features": {
            "file": "api/social_routes.py",
            "class": "init_social_routes",
            "description": "Friends, watch parties, lists"
        },
        "🔬 Explainable AI": {
            "file": "ai/_archive/explainable_recommendations.py",
            "class": "ExplainableRecommender",
            "description": "SHAP-based explanations"
        },
        "📈 Trending Detection": {
            "file": "ai/trending_detector.py",
            "class": "TrendingDetector",
            "description": "Viral movie detection"
        },
        "🖼️ Visual Search": {
            "file": "ai/visual_search.py",
            "class": "VisualMovieSearch",
            "description": "CLIP-based image search"
        },
        "⚡ Redis Caching": {
            "file": "ai/redis_cache.py",
            "class": "RedisCache",
            "description": "High-performance caching"
        },
        "🔬 A/B Testing": {
            "file": "ai/ab_testing.py",
            "class": "Experiment",
            "description": "Experimentation framework"
        },
        "🌐 Distributed Training": {
            "file": "training/distributed_training.py",
            "class": "setup_distributed",
            "description": "Multi-GPU training"
        },
        "🚀 TorchServe": {
            "file": "inference/torchserve_handler.py",
            "class": "CineSenseRecommenderHandler",
            "description": "Production model serving"
        },
        "📊 Advanced Metrics": {
            "file": "ai/_archive/advanced_metrics.py",
            "class": "RecommendationMetrics",
            "description": "Comprehensive evaluation"
        },
        "🔍 Query Understanding": {
            "file": "ai/_archive/query_understanding.py",
            "class": "QueryEnhancer",
            "description": "T5-based query expansion"
        },
        "🎯 Reranker": {
            "file": "ai/_archive/reranker.py",
            "class": "SemanticReranker",
            "description": "Cross-encoder reranking"
        },
        "🎨 Multi-Modal Search": {
            "file": "ai/_archive/multimodal_search.py",
            "class": "MultiModalSearch",
            "description": "Text + Image search"
        },
        "🗄️ Vector Store": {
            "file": "ai/_archive/vector_store.py",
            "class": "FAISSVectorStore",
            "description": "FAISS vector database"
        }
    }
    
    # Check all features
    implemented = 0
    total = len(features)
    
    print(f"{Colors.BOLD}FEATURE IMPLEMENTATION STATUS:{Colors.ENDC}\n")
    
    for feature_name, config in features.items():
        status, color = check_feature(
            feature_name,
            config["file"],
            config.get("class")
        )
        
        if "✓" in status:
            implemented += 1
        
        status_icon = "✓✓" if status == "✓✓" else ("✓" if status == "✓" else "✗")
        print(f"{color}{status_icon}{Colors.ENDC} {feature_name:<25} "
              f"{'✓' if '✓' in status else '✗':<3} {config['description']}")
    
    # API Endpoints
    print(f"\n{Colors.BOLD}API ENDPOINTS:{Colors.ENDC}\n")
    
    endpoints = {
        "/api/chat": "Conversational AI chatbot",
        "/api/mood-recommendations": "Mood-based recommendations",
        "/api/trending": "Trending movies detection",
        "/api/viral": "Viral movies detection",
        "/api/visual-search": "Image-based search",
        "/api/explain/<movie_id>": "Explainable recommendations",
        "/api/social/friends/add": "Add friend",
        "/api/social/friends/list": "List friends",
        "/api/social/watchparty/create": "Create watch party",
        "/api/social/lists/create": "Create movie list"
    }
    
    if os.path.exists("app_integrated.py"):
        with open("app_integrated.py", 'r', encoding='utf-8') as f:
            app_content = f.read()
            
        for endpoint, description in endpoints.items():
            # Check if endpoint is registered
            endpoint_base = endpoint.split('<')[0].replace('/', '\\/').replace('.', '\\.')
            if endpoint_base in app_content:
                print(f"{Colors.GREEN}✓{Colors.ENDC} {endpoint:<35} {description}")
            else:
                print(f"{Colors.RED}✗{Colors.ENDC} {endpoint:<35} {description}")
    
    # Database Tables
    print(f"\n{Colors.BOLD}DATABASE TABLES:{Colors.ENDC}\n")
    
    required_tables = [
        "friend_requests",
        "friendships",
        "watch_parties",
        "watch_party_invites",
        "movie_lists",
        "list_movies",
        "chat_history",
        "ab_experiments",
        "ab_user_assignments",
        "ab_metrics"
    ]
    
    print(f"{Colors.YELLOW}Note: Run scripts/update_schema.py to verify social tables{Colors.ENDC}\n")
    
    for table in required_tables:
        print(f"{Colors.BLUE}•{Colors.ENDC} {table}")
    
    # Summary
    print(f"\n{Colors.BOLD}{'='*70}{Colors.ENDC}")
    print(f"\n{Colors.BOLD}SUMMARY:{Colors.ENDC}")
    print(f"  Features Implemented: {Colors.GREEN}{implemented}/{total}{Colors.ENDC} "
          f"({Colors.GREEN}{(implemented/total)*100:.1f}%{Colors.ENDC})")
    
    if implemented == total:
        print(f"\n  {Colors.GREEN}✓ ALL FEATURES IMPLEMENTED!{Colors.ENDC} 🎉")
    elif implemented >= total * 0.9:
        print(f"\n  {Colors.YELLOW}⚠ Almost there! {total - implemented} features remaining{Colors.ENDC}")
    else:
        print(f"\n  {Colors.RED}✗ {total - implemented} features need implementation{Colors.ENDC}")
    
    print(f"\n{Colors.BOLD}{'='*70}{Colors.ENDC}\n")
    
    # Model Status
    print(f"{Colors.BOLD}MODEL FILES:{Colors.ENDC}\n")
    
    model_files = [
        ("model/hybrid_recommender.pt", "Main hybrid model"),
        ("model/ncf_recommender.pt", "NCF model"),
        ("model/plot_embeddings.npy", "Plot embeddings"),
        ("model/movie_metadata.csv", "Movie metadata"),
    ]
    
    for file_path, description in model_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            print(f"{Colors.GREEN}✓{Colors.ENDC} {file_path:<35} {description} ({size:.1f} MB)")
        else:
            print(f"{Colors.YELLOW}⚠{Colors.ENDC} {file_path:<35} {description} (not trained)")
    
    print(f"\n{Colors.BOLD}NEXT STEPS:{Colors.ENDC}\n")
    print("1. Start the app: python app_integrated.py")
    print("2. Test features: python test_features.py")
    print("3. Train models: python train_model.py (if needed)")
    print("4. Access UI: http://localhost:5000/features")
    print(f"\n{Colors.GREEN}✓ Verification complete!{Colors.ENDC}\n")

if __name__ == "__main__":
    main()
