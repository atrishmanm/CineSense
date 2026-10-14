"""
CineSense High Availability & Resilience Engine
================================================

Implements:
1. Circuit Breaker Pattern (Hystrix-style state transitions: CLOSED -> OPEN -> HALF_OPEN).
2. Deadline Context & Timeout Budget Propagation for SLA enforcement.
3. Token Bucket Rate Limiting with burst allowance.
4. Idempotency Key Registry with TTL to prevent duplicate state updates.
"""

from __future__ import annotations

import threading
import time
from enum import Enum
from typing import Any, Callable, Dict, Optional, Tuple


class CircuitState(Enum):
    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


class CircuitBreakerOpenException(Exception):
    """Raised when an operation is attempted while the circuit breaker is OPEN."""
    pass


class CircuitBreaker:
    """
    Guards external dependencies and heavy AI components against cascade failures.
    """

    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout_sec: float = 10.0,
        half_open_success_threshold: int = 2,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.recovery_timeout_sec = recovery_timeout_sec
        self.half_open_success_threshold = half_open_success_threshold

        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: float = 0.0
        self._lock = threading.Lock()

    def execute(self, fn: Callable[[], Any], fallback_fn: Optional[Callable[[], Any]] = None) -> Any:
        now = time.time()
        with self._lock:
            if self.state == CircuitState.OPEN:
                if now - self.last_failure_time >= self.recovery_timeout_sec:
                    # Transition to HALF_OPEN for exploratory probe
                    self.state = CircuitState.HALF_OPEN
                    self.success_count = 0
                else:
                    if fallback_fn:
                        return fallback_fn()
                    raise CircuitBreakerOpenException("Circuit breaker is OPEN -- operation blocked")

        try:
            result = fn()
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            if fallback_fn:
                return fallback_fn()
            raise e

    def _on_success(self) -> None:
        with self._lock:
            if self.state == CircuitState.HALF_OPEN:
                self.success_count += 1
                if self.success_count >= self.half_open_success_threshold:
                    self.state = CircuitState.CLOSED
                    self.failure_count = 0
            elif self.state == CircuitState.CLOSED:
                self.failure_count = 0

    def _on_failure(self) -> None:
        with self._lock:
            self.failure_count += 1
            self.last_failure_time = time.time()
            if self.failure_count >= self.failure_threshold:
                self.state = CircuitState.OPEN


class DeadlineContext:
    """
    Propagates SLA deadlines across recommendation pipeline stages.
    """

    def __init__(self, timeout_ms: float = 25.0) -> None:
        self.start_time = time.perf_counter()
        self.deadline_sec = timeout_ms / 1000.0

    def time_remaining_ms(self) -> float:
        elapsed = time.perf_counter() - self.start_time
        remaining = max(0.0, (self.deadline_sec - elapsed) * 1000.0)
        return remaining

    def is_expired(self) -> bool:
        return self.time_remaining_ms() <= 0.0


class TokenBucketRateLimiter:
    """
    Token bucket rate limiter with smooth replenishment.
    """

    def __init__(self, rate_per_sec: float = 100.0, capacity: float = 100.0) -> None:
        self.rate = rate_per_sec
        self.capacity = capacity
        self.tokens = capacity
        self.last_check = time.time()
        self._lock = threading.Lock()

    def allow_request(self, tokens_needed: float = 1.0) -> bool:
        with self._lock:
            now = time.time()
            elapsed = now - self.last_check
            self.last_check = now
            # Replenish tokens
            self.tokens = min(self.capacity, self.tokens + elapsed * self.rate)
            if self.tokens >= tokens_needed:
                self.tokens -= tokens_needed
                return True
            return False


class IdempotencyRegistry:
    """
    Guarantees state-changing endpoints (e.g., ratings/interactions) execute at most once.
    """

    def __init__(self, ttl_sec: float = 300.0) -> None:
        self.ttl_sec = ttl_sec
        self._keys: Dict[str, Tuple[float, Any]] = {}  # key -> (expiry, response)
        self._lock = threading.Lock()

    def check_or_set(self, idempotency_key: str, execution_fn: Callable[[], Any]) -> Tuple[bool, Any]:
        """
        Returns (was_cached, result).
        If already executed, returns (True, cached_result).
        Otherwise executes execution_fn and stores with TTL.
        """
        now = time.time()
        with self._lock:
            # Purge expired
            expired = [k for k, (exp, _) in self._keys.items() if exp < now]
            for k in expired:
                del self._keys[k]

            if idempotency_key in self._keys:
                return True, self._keys[idempotency_key][1]

        # Execute
        result = execution_fn()

        with self._lock:
            self._keys[idempotency_key] = (now + self.ttl_sec, result)

        return False, result
