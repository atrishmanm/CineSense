"""Endpoint smoke tests for CineSense (live server required).

These tests require a running server on BASE (default http://localhost:5000).
When the server is unreachable they are skipped (not failed) so `pytest -q`
stays green offline. Can still be run directly as a script:
`python tests/test_endpoints.py` (exits non-zero on failure).
"""

import sys

import pytest
import requests

BASE = "http://localhost:5000"
PROBE_TIMEOUT = 2.0

# Audit I1.3: tag this module so `pytest -m live` / `-m "not live"` filters
# work without PytestUnknownMarkWarning (marker registered in tests/conftest.py).
pytestmark = pytest.mark.live


def _server_up(timeout=PROBE_TIMEOUT) -> bool:
    """Probe the server with a short timeout (never at import time)."""
    try:
        return requests.get(BASE, timeout=timeout).status_code == 200
    except Exception:
        return False


@pytest.fixture(autouse=True)
def _require_live_server():
    """Skip every test in this module when the live server is unreachable."""
    if not _server_up():
        pytest.skip("live server at localhost:5000 not running")


def test_home_page():
    """Test 1: Home page loads"""
    response = requests.get(f"{BASE}/", timeout=30)
    assert response.status_code == 200
    assert "CineSense" in response.text


def test_trending_api():
    """Test 2: Trending API returns JSON"""
    response = requests.get(f"{BASE}/api/trending", timeout=60)
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)
    assert "trending" in data or "movies" in data


def test_chat_api():
    """Test 3: Chat API responds to a message"""
    response = requests.post(
        f"{BASE}/api/chat",
        json={"message": "recommend me a good sci-fi movie"},
        timeout=120,
    )
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)
    assert "response" in data or "message" in data


def test_mood_api():
    """Test 4: Mood recommendations API responds"""
    response = requests.post(
        f"{BASE}/api/mood-recommendations",
        json={"mood": "happy", "user_id": 1},
        timeout=60,
    )
    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, dict)
    assert "recommendations" in data or "movies" in data


def test_chat_ui_page():
    """Test 5: Chat UI page loads"""
    response = requests.get(f"{BASE}/chat-ui", timeout=30)
    assert response.status_code == 200
    assert "chat" in response.text.lower() or "CineSense" in response.text


def test_friends_page():
    """Test 6: Friends page loads (may redirect to login)"""
    response = requests.get(f"{BASE}/friends", timeout=30, allow_redirects=True)
    assert response.status_code in (200, 302, 401)


def test_features_page():
    """Test 7: Features page loads"""
    response = requests.get(f"{BASE}/features", timeout=30)
    assert response.status_code == 200
    assert response.text


def main() -> int:
    """CLI harness: run every endpoint check and report failures."""
    if not _server_up():
        print(f"Server not running at {BASE}. Start with: python app_integrated.py")
        return 1

    checks = [
        test_home_page,
        test_trending_api,
        test_chat_api,
        test_mood_api,
        test_chat_ui_page,
        test_friends_page,
        test_features_page,
    ]

    failures = 0
    for check in checks:
        try:
            check()
            print(f"PASS {check.__name__}")
        except Exception as e:  # noqa: BLE001 - CLI harness reports all failures
            failures += 1
            print(f"FAIL {check.__name__}: {e}")

    print(f"\n{'='*50}")
    if failures:
        print(f"{failures}/{len(checks)} checks FAILED")
        return 1
    print("All tests complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
