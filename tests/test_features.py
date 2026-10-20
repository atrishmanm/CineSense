"""
CineSense Feature Testing Script
Test all integrated features with example API calls.

These tests require a live server on BASE_URL. When the server is not
reachable they are skipped (not failed) so `pytest -q` stays green offline.
"""

import sys
from time import sleep

import pytest
import requests

BASE_URL = "http://localhost:5000"
PROBE_TIMEOUT = 2.0

# Audit I1.3: tag this module so `pytest -m live` / `-m "not live"` filters
# work without PytestUnknownMarkWarning (marker registered in tests/conftest.py).
pytestmark = pytest.mark.live


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


def print_result(feature, status, details=""):
    symbol = "✅" if status else "❌"
    print(f"{symbol} {feature}")
    if details:
        print(f"   → {details}")


def test_server():
    """Test if server is running"""
    response = requests.get(BASE_URL, timeout=5)
    assert response.status_code == 200
    print_result("Server Status", True, "CineSense is running")


def test_chat():
    """Test conversational AI"""
    print("\n🤖 Testing Conversational AI...")
    response = requests.post(
        f"{BASE_URL}/api/chat",
        json={"message": "Recommend me a thrilling sci-fi movie"},
        timeout=120,
    )
    assert response.status_code == 200, f"chat returned {response.status_code}"
    data = response.json()
    assert isinstance(data, dict), "chat response must be a JSON object"
    assert "response" in data or "message" in data, "chat response missing text field"
    print_result("Chatbot", True, f"Response: {str(data.get('response', data))[:100]}...")


def test_mood_recs():
    """Test mood-based recommendations"""
    print("\n😊 Testing Mood Recommendations...")
    response = requests.post(
        f"{BASE_URL}/api/mood-recommendations",
        json={"mood": "excited", "limit": 5},
        timeout=60,
    )
    assert response.status_code == 200, f"mood endpoint returned {response.status_code}"
    data = response.json()
    assert isinstance(data, dict)
    assert "recommendations" in data, "mood response missing 'recommendations'"
    assert isinstance(data["recommendations"], list)
    print_result("Mood Detection", True, f"Found {len(data['recommendations'])} movies for 'excited' mood")


def test_trending():
    """Test trending detection"""
    print("\n📈 Testing Trending Movies...")
    response = requests.get(f"{BASE_URL}/api/trending?limit=10", timeout=60)
    assert response.status_code == 200, f"trending returned {response.status_code}"
    data = response.json()
    assert isinstance(data, dict)
    assert "trending" in data, "trending response missing 'trending'"
    assert isinstance(data["trending"], list)
    print_result("Trending Detection", True, f"Found {len(data['trending'])} trending movies")


def test_viral():
    """Test viral detection"""
    print("\n🔥 Testing Viral Detection...")
    response = requests.get(f"{BASE_URL}/api/viral?limit=10", timeout=60)
    assert response.status_code == 200, f"viral returned {response.status_code}"
    data = response.json()
    assert isinstance(data, dict)
    assert "viral" in data, "viral response missing 'viral'"
    assert isinstance(data["viral"], list)
    print_result("Viral Detection", True, f"Found {len(data['viral'])} viral movies")


def test_social_endpoints():
    """Test social feature endpoints"""
    print("\n👥 Testing Social Features...")

    # These require authentication, so they must be either protected (401)
    # or return a JSON object (200).
    endpoints = [
        '/api/social/friends/list',
        '/api/social/friends/requests',
        '/api/social/watchparty/upcoming',
    ]

    for endpoint in endpoints:
        response = requests.get(f"{BASE_URL}{endpoint}", timeout=10)
        assert response.status_code in (200, 401), (
            f"{endpoint} returned unexpected {response.status_code}")
        if response.status_code == 200:
            assert isinstance(response.json(), dict), f"{endpoint} did not return JSON"
            print_result(f"Social: {endpoint}", True, "Endpoint accessible")
        else:
            print_result(f"Social: {endpoint}", True, "Endpoint protected (Auth required)")


def test_core_routes():
    """Test core application routes"""
    print("\n🎬 Testing Core Routes...")

    routes = {
        '/': 'Home Page',
        '/search': 'Search',
        '/compare': 'Compare',
    }

    for route, name in routes.items():
        response = requests.get(f"{BASE_URL}{route}", timeout=10)
        assert response.status_code == 200, f"{route} returned {response.status_code}"
        assert response.text, f"{route} returned an empty body"
        print_result(name, True, "Page loads successfully")


def main():
    print("=" * 70)
    print("🎬 CINESENSE - FEATURE TESTING SUITE")
    print("=" * 70)

    if not _server_up():
        print("\n❌ Server not running. Start with: python app_integrated.py")
        return 1

    print("\nTesting all features... (This may take 1-2 minutes)\n")

    checks = [
        ("Server Status", test_server),
        ("Core Routes", test_core_routes),
        ("Chatbot", test_chat),
        ("Mood Detection", test_mood_recs),
        ("Trending Detection", test_trending),
        ("Viral Detection", test_viral),
        ("Social Features", test_social_endpoints),
    ]

    failures = 0
    for name, check in checks:
        try:
            check()
        except Exception as e:  # noqa: BLE001 - CLI harness reports all failures
            failures += 1
            print_result(name, False, str(e))
        sleep(1)  # Prevent overwhelming the server

    print("\n" + "=" * 70)
    print("✨ Testing Complete!")
    print("=" * 70)
    print("\n📊 Summary:")
    print(f"   • Checks run: {len(checks)}")
    print(f"   • Failures: {failures}")
    print("\n🌐 Access the app at: http://localhost:5000")
    print("💬 Try the chat UI at: http://localhost:5000/chat-ui")

    if failures:
        print("\n❌ CineSense feature checks FAILED!")
        return 1
    print("\n🎉 CineSense is fully operational!")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\n⚠️  Testing interrupted by user")
        sys.exit(130)
