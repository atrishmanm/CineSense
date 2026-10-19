"""
Comprehensive AI Component Testing for CineSense
Tests all AI features with model downloading and warmup.

These tests require a live server on BASE_URL. When the server is not
reachable they are skipped (not failed) so `pytest -q` stays green offline.
"""

import os
import sys
import time

import pytest
import requests

BASE_URL = "http://localhost:5000"
PROBE_TIMEOUT = 2.0


def _server_up(timeout=PROBE_TIMEOUT) -> bool:
    """Probe the server with a short timeout (never at import time)."""
    try:
        return requests.get(BASE_URL, timeout=timeout).status_code == 200
    except Exception:
        return False


@pytest.fixture(autouse=True)
def _require_live_server():
    """Skip every test in this module when the live server is unreachable."""
    if not _server_up():
        pytest.skip("live server at localhost:5000 not running")


class Colors:
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'


def print_result(test_name: str, success: bool, details: str = "", time_taken: float = 0):
    """Print formatted test result"""
    status = f"{Colors.GREEN}✓{Colors.ENDC}" if success else f"{Colors.RED}✗{Colors.ENDC}"
    time_str = f" ({time_taken:.2f}s)" if time_taken > 0 else ""
    print(f"{status} {test_name}{time_str}")
    if details:
        print(f"   → {details}")


def test_server_health() -> None:
    """Test if server is running"""
    response = requests.get(BASE_URL, timeout=5)
    assert response.status_code == 200


def test_conversational_ai():
    """Test chatbot with warmup"""
    print(f"\n{Colors.BOLD}💬 Testing Conversational AI...{Colors.ENDC}")
    print(f"{Colors.YELLOW}   Note: First request may take 30-60s (downloading DialoGPT model){Colors.ENDC}\n")

    test_queries = [
        "Recommend me a thrilling sci-fi movie",
        "What's a good comedy for tonight?",
        "I want something with plot twists",
    ]

    for i, query in enumerate(test_queries, 1):
        start_time = time.time()
        response = requests.post(
            f"{BASE_URL}/api/chat",
            json={"message": query},
            timeout=120,  # Long timeout for first request
        )
        elapsed = time.time() - start_time
        assert response.status_code == 200, (
            f"chat query {i} returned {response.status_code}")
        data = response.json()
        assert isinstance(data, dict), "chat response must be a JSON object"
        assert "response" in data or "message" in data, "chat response missing text field"
        bot_response = str(data.get("response", data.get("message", "")))
        print_result(f"Query {i}: '{query[:50]}...'", True,
                     f"Response: {bot_response[:100]}...", elapsed)


def test_mood_recommendations():
    """Test mood-based recommendations"""
    print(f"\n{Colors.BOLD}😊 Testing Mood Detection...{Colors.ENDC}\n")

    moods = ["happy", "sad", "excited", "bored", "romantic"]

    for mood in moods:
        start_time = time.time()
        response = requests.post(
            f"{BASE_URL}/api/mood-recommendations",
            json={"mood": mood, "limit": 5},
            timeout=30,
        )
        elapsed = time.time() - start_time
        assert response.status_code == 200, (
            f"mood '{mood}' returned {response.status_code}")
        data = response.json()
        assert "recommendations" in data, "mood response missing 'recommendations'"
        assert isinstance(data["recommendations"], list)
        print_result(f"Mood: {mood}", True,
                     f"Found {len(data['recommendations'])} recommendations", elapsed)


def test_trending_detection():
    """Test trending movies detection"""
    print(f"\n{Colors.BOLD}📈 Testing Trending Detection...{Colors.ENDC}\n")

    start_time = time.time()
    response = requests.get(f"{BASE_URL}/api/trending?limit=10", timeout=30)
    elapsed = time.time() - start_time
    assert response.status_code == 200, f"trending returned {response.status_code}"
    data = response.json()
    assert "trending" in data, "trending response missing 'trending'"
    trending = data["trending"]
    assert isinstance(trending, list)
    print_result("Trending Movies", True,
                 f"Found {len(trending)} trending movies", elapsed)

    if trending:
        assert "title" in trending[0], "trending items must expose a title"
        print(f"\n   {Colors.BLUE}Top 3 Trending:{Colors.ENDC}")
        for i, movie in enumerate(trending[:3], 1):
            title = movie.get("title", "Unknown")
            score = movie.get("trending_score", 0)
            print(f"      {i}. {title} (score: {score:.2f})")


def test_viral_detection():
    """Test viral detection"""
    print(f"\n{Colors.BOLD}🔥 Testing Viral Detection...{Colors.ENDC}\n")

    start_time = time.time()
    response = requests.get(f"{BASE_URL}/api/viral?limit=10", timeout=30)
    elapsed = time.time() - start_time
    assert response.status_code == 200, f"viral returned {response.status_code}"
    data = response.json()
    assert "viral" in data, "viral response missing 'viral'"
    assert isinstance(data["viral"], list)
    print_result("Viral Movies", True,
                 f"Found {len(data['viral'])} viral movies", elapsed)


# 1x1 transparent PNG, used to exercise the base64 image path.
_TINY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8z8BQ"
    "DwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def test_visual_search():
    """Test visual search capability with a real request"""
    print(f"\n{Colors.BOLD}🖼️  Testing Visual Search...{Colors.ENDC}")
    print(f"{Colors.YELLOW}   Note: First request may take 40-60s (downloading CLIP model){Colors.ENDC}\n")

    # Missing image must be rejected with a validation error.
    missing = requests.post(f"{BASE_URL}/api/visual-search", json={}, timeout=30)
    assert missing.status_code == 400, (
        f"visual-search without image expected 400, got {missing.status_code}")

    # With an image the endpoint must respond with JSON (results or error).
    response = requests.post(
        f"{BASE_URL}/api/visual-search",
        json={"image": _TINY_PNG_B64, "top_k": 3},
        timeout=120,
    )
    assert response.status_code in (200, 500, 503), (
        f"visual-search returned unexpected {response.status_code}")
    data = response.json()
    assert isinstance(data, dict)
    assert "results" in data or "error" in data, "visual-search response missing results/error"
    print_result("Visual Search Endpoint", True,
                 f"Endpoint responded (status {response.status_code})")


def test_semantic_search():
    """Test semantic search"""
    print(f"\n{Colors.BOLD}🔍 Testing Semantic Search...{Colors.ENDC}\n")

    queries = [
        "indian spy thriller",
        "time loop movie",
        "movie about artificial intelligence",
    ]

    for query in queries:
        start_time = time.time()
        response = requests.get(
            f"{BASE_URL}/search",
            params={"q": query},
            timeout=30,
        )
        elapsed = time.time() - start_time
        assert response.status_code == 200, (
            f"search '{query}' returned {response.status_code}")
        assert response.text, "search page returned an empty body"
        print_result(f"Query: '{query}'", True, "Search page loaded", elapsed)


def test_social_features():
    """Test social features endpoints"""
    print(f"\n{Colors.BOLD}👥 Testing Social Features...{Colors.ENDC}\n")

    endpoints = [
        ("/api/social/friends/list", "Friends List"),
        ("/api/social/friends/requests", "Friend Requests"),
        ("/api/social/watchparty/upcoming", "Watch Parties"),
    ]

    for endpoint, name in endpoints:
        response = requests.get(f"{BASE_URL}{endpoint}", timeout=10)
        assert response.status_code in (200, 401), (
            f"{endpoint} returned unexpected {response.status_code}")
        # 401 = Auth required (correct behavior for an anonymous probe)
        label = "Protected (Auth required) ✓" if response.status_code == 401 else "Accessible"
        print_result(name, True, label)


def test_model_checkpoints():
    """Check model files exist (fails when artifacts are missing)"""
    print(f"\n{Colors.BOLD}🧠 Checking Model Files...{Colors.ENDC}\n")

    models = [
        ("model/hybrid_recommender.pt", "Hybrid Model"),
        ("model/ncf_recommender.pt", "NCF Model"),
        ("model/plot_embeddings.npy", "Plot Embeddings"),
        ("model/movie_metadata.csv", "Movie Metadata"),
    ]

    for file_path, name in models:
        assert os.path.exists(file_path), f"missing model artifact: {file_path}"
        size_mb = os.path.getsize(file_path) / (1024 * 1024)
        print_result(name, True, f"{size_mb:.1f} MB")


def run_comprehensive_test():
    """Run all tests."""
    print(f"\n{Colors.BOLD}{'='*70}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.BLUE}🎬 CINESENSE - COMPREHENSIVE AI TESTING{Colors.ENDC}")
    print(f"{Colors.BOLD}{'='*70}{Colors.ENDC}")

    # Check server first
    print(f"\n{Colors.BOLD}🌐 Checking Server Status...{Colors.ENDC}\n")
    if not _server_up():
        print_result("Server Status", False, "Server not running!")
        print(f"\n{Colors.RED}❌ Start the server first: python app_integrated.py{Colors.ENDC}\n")
        return 1

    print_result("Server Status", True, f"Running on {BASE_URL}")

    checks = [
        ("Model Checkpoints", test_model_checkpoints),
        ("Semantic Search", test_semantic_search),
        ("Mood Recommendations", test_mood_recommendations),
        ("Trending Detection", test_trending_detection),
        ("Viral Detection", test_viral_detection),
        ("Conversational AI", test_conversational_ai),
        ("Visual Search", test_visual_search),
        ("Social Features", test_social_features),
    ]

    failures = 0
    for name, check in checks:
        try:
            check()
        except Exception as e:  # noqa: BLE001 - CLI harness reports all failures
            failures += 1
            print_result(name, False, str(e))
        time.sleep(1)  # Rate limiting

    # Summary
    print(f"\n{Colors.BOLD}{'='*70}{Colors.ENDC}")
    print(f"\n{Colors.BOLD}✨ TESTING COMPLETE!{Colors.ENDC}\n")
    if failures:
        print(f"{Colors.RED}✗ {failures} check(s) failed{Colors.ENDC}")
    else:
        print(f"{Colors.GREEN}✓ All AI components are integrated and operational{Colors.ENDC}")
    print(f"\n{Colors.BOLD}Access Points:{Colors.ENDC}")
    print(f"   • Main App: {Colors.BLUE}{BASE_URL}/{Colors.ENDC}")
    print(f"   • Features Showcase: {Colors.BLUE}{BASE_URL}/features{Colors.ENDC}")
    print(f"   • Chat UI: {Colors.BLUE}{BASE_URL}/chat-ui{Colors.ENDC}")
    print(f"   • Search: {Colors.BLUE}{BASE_URL}/search{Colors.ENDC}")
    print(f"\n{Colors.YELLOW}Note: AI features need 30-60s warmup on first request (model download){Colors.ENDC}")
    print(f"{Colors.BOLD}{'='*70}{Colors.ENDC}\n")
    return 1 if failures else 0


if __name__ == "__main__":
    try:
        sys.exit(run_comprehensive_test())
    except KeyboardInterrupt:
        print(f"\n\n{Colors.YELLOW}⚠️  Testing interrupted by user{Colors.ENDC}\n")
        sys.exit(130)
