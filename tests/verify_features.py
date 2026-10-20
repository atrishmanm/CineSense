"""
CineSense Feature Verification Script
Comprehensive check of all implemented features
"""

import os
import re
import sys
from pathlib import Path

# Windows consoles default to cp1252; the report uses emoji, so force UTF-8.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# Path baked in (audit I1.6c): os.chdir used to run at import time, silently
# moving the process CWD as a side effect of import. The chdir is moved into
# main() below; PROJECT_ROOT is just a constant here.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

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
    # os.chdir + sys.path setup done here, inside main(), per audit I1.6c
    # (importing this module must not mutate the process CWD).
    os.chdir(PROJECT_ROOT)
    sys.path.insert(0, str(PROJECT_ROOT))

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
        }
    }
    
    # Check all features
    implemented = 0
    warnings = 0
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
        if status == "✓⚠":
            warnings += 1
        
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
        "/api/social/friends/add": "Add friend",
        "/api/social/friends/list": "List friends",
        "/api/social/watchparty/create": "Create watch party",
        "/api/social/lists/create": "Create movie list"
    }

    # Audit I1.6a: the previous check mangled the endpoint string before
    # searching (`'/api/chat'.replace('/', '\\/')` -> `\/api\/chat`), which
    # never occurs in source, so every endpoint printed ✗ forever. Route
    # registration is a quoted literal (e.g. `@app.route('/api/chat', ...)`),
    # so the honest check searches for the QUOTED path and accepts both quote
    # styles. Blueprint modules (api/social_routes.py,
    # url_prefix='/api/social') register only the SUFFIX, so url_prefix
    # declarations are parsed and the endpoint's remainder is checked too.
    # Also scans api/routes.py; verdicts are counted toward the exit code
    # (audit I1.6b) instead of being printed-only.
    endpoint_failures = 0
    app_files = ["app_integrated.py", "api/routes.py", "api/social_routes.py"]
    available_sources = [p for p in app_files if os.path.exists(p)]

    prefix_re = re.compile(r"url_prefix\s*=\s*(['\"])([^'\"]*)\1")

    if not available_sources:
        print(f"{Colors.YELLOW}⚠{Colors.ENDC} No app source found "
              f"({', '.join(app_files)}) -- endpoints unverifiable")
        endpoint_failures += len(endpoints)
    else:
        sources: dict[str, tuple[str, str]] = {}
        for p in available_sources:
            with open(p, "r", encoding="utf-8") as f:
                content = f.read()
            m = prefix_re.search(content)
            prefix = m.group(2).rstrip("/") if m else ""
            sources[p] = (content, prefix)

        for endpoint, description in endpoints.items():
            path = endpoint.split('<')[0]
            registered = False
            for content, prefix in sources.values():
                if prefix and path.startswith(prefix):
                    # Blueprint route literal: '/friends/add', ...
                    if (("'" + path[len(prefix):] + "'") in content
                            or ('"' + path[len(prefix):] + '"') in content):
                        registered = True
                        break
                # App-level route literal: '/api/chat', ...
                if (("'" + path + "'") in content
                        or ('"' + path + '"') in content):
                    registered = True
                    break
            if registered:
                print(f"{Colors.GREEN}✓{Colors.ENDC} {endpoint:<35} {description}")
            else:
                endpoint_failures += 1
                print(f"{Colors.RED}✗{Colors.ENDC} {endpoint:<35} {description}")

        if endpoint_failures:
            print(f"\n{Colors.RED}{endpoint_failures} endpoint(s) not found in "
                  f"{', '.join(available_sources)}{Colors.ENDC}")
    
    # Database Tables
    print(f"\n{Colors.BOLD}DATABASE TABLES:{Colors.ENDC}\n")
    
    required_tables = [
        "friend_requests",
        "friendships",
        "watch_parties",
        "watch_party_invites",
        "collaborative_lists",
        "collaborative_list_items",
        "collaborative_list_members",
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
    if warnings:
        print(f"  {Colors.YELLOW}Warnings (file present, expected class missing): {warnings}{Colors.ENDC}")
    
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
    print("2. Test features: python tests/test_features.py")
    print("3. Train models: python scripts/train_model.py (if needed)")
    print("4. Access UI: http://localhost:5000/features")

    # Non-zero exit when ANY verification failed so callers/CI can detect
    # regressions (audit I1.6b): previously the exit code counted only the
    # feature-file checks, so "all 9 endpoint checks print ✗" still exited 0
    # while claiming success.
    failed_checks = (total - implemented) + endpoint_failures
    if failed_checks == 0:
        print(f"\n{Colors.GREEN}✓ Verification complete!{Colors.ENDC}\n")
        return 0
    print(f"\n{Colors.RED}✗ Verification FAILED: {failed_checks} check(s) did not pass "
          f"({total - implemented} feature, {endpoint_failures} endpoint).{Colors.ENDC}\n")
    return 1


if __name__ == "__main__":
    sys.exit(main())
