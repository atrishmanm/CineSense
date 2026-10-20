"""
Shared pytest configuration for the CineSense test suite.

Marker registration (audit I1.3): the three live-server modules
(test_features.py, test_endpoints.py, test_ai_comprehensive.py) are tagged
``live`` via ``pytestmark = pytest.mark.live`` so they can be filtered with
``pytest -m live`` / ``pytest -m "not live"`` without raising a
``PytestUnknownMarkWarning``.

Offline behaviour is unchanged: each live module keeps its autouse
fixture that skips its tests unless the server on localhost:5000 answers,
so a plain ``pytest -q`` still stays green without a server.
"""

def pytest_configure(config):
    # pytest.ini-style registration done here (pytest.ini lives outside the
    # tests/ ownership boundary of this fix wave).
    config.addinivalue_line(
        "markers",
        "live: requires a running CineSense server on localhost:5000 "
        "(auto-skipped when the server is unreachable)",
    )
