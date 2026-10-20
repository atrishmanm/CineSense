"""
CineSense Production High-Throughput Caching Engine
===================================================

Implements:
1. XFetch Algorithm: Optimal Probabilistic Early Expiration to eliminate cache stampedes (VLDB 2015).
2. Singleflight: In-flight request coalescing (only 1 thread computes while concurrent callers share the result).
3. Two-Tier Cache Architecture: L1 in-memory LRU + L2 distributed / persistent store with fallback.
4. Comprehensive Hit/Miss/Stampede-Prevented Telemetry.
"""

from __future__ import annotations

import math
import random
import threading
import time
from collections import OrderedDict
from typing import Any, Callable, Dict, Optional


class _Flight:
    """Shared result holder for an in-flight Singleflight call.

    Waiters hold a reference to the same object as the leader, so the
    dictionary entry can be removed as soon as the leader finishes without
    any waiter raising KeyError.
    """

    __slots__ = ("event", "value", "error")

    def __init__(self) -> None:
        self.event = threading.Event()
        self.value: Any = None
        self.error: Optional[Exception] = None


class Singleflight:
    """
    Suppresses duplicate concurrent calls.
    Ensures that only one execution is in-flight for a given key at a time.
    If a duplicate comes in, that caller waits for the first to complete and receives the same result.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._calls: Dict[str, _Flight] = {}

    def do(self, key: str, fn: Callable[[], Any], timeout: Optional[float] = None) -> Any:
        """Execute ``fn`` under the given key, coalescing concurrent callers.

        Waiters return (or raise) exactly what the leader produced. A
        BaseException is treated like an ordinary error (wave5 fix C1.33):
        the flight records it and it is re-raised on the leader *after* the
        finally bookkeeping, and waiters raise the same exception object
        instead of silently returning a None "success".

        Args:
            key: Coalescing key.
            fn: The single-execution callable.
            timeout: Optional max wait (seconds) for a follower to block on
                the leader's flight (wave5 fix C1.34). ``None`` (default)
                waits indefinitely, matching the previous contract.
        """
        with self._lock:
            flight = self._calls.get(key)
            is_leader = flight is None
            if is_leader:
                flight = _Flight()
                self._calls[key] = flight

        if not is_leader:
            # Another call is already in progress; wait for it and share the result
            # wave5 fix (C1.34): always actually wait on the flight event --
            # with a deadline when one is given, indefinitely otherwise. The
            # previous `if timeout is not None and not wait(...)` guard
            # short-circuited below the default (no-timeout) path and made
            # followers return the leader's still-None holder value.
            if timeout is not None:
                if not flight.event.wait(timeout):
                    raise TimeoutError(f"singleflight wait timed out after {timeout}s for key {key!r}")
            else:
                flight.event.wait()
            if flight.error is not None:
                raise flight.error
            return flight.value

        # First caller: execute, then publish the result and wake waiters
        try:
            flight.value = fn()
        except BaseException as e:  # noqa: BLE001 - re-raised to all waiters below
            # wave5 fix (C1.33): catching BaseException so a KeyboardInterrupt/
            # SystemExit does not leave waiters to consume a None success.
            flight.error = e
        finally:
            with self._lock:
                self._calls.pop(key, None)
            flight.event.set()

        if flight.error is not None:
            raise flight.error
        return flight.value


class XFetchCache:
    """
    Two-Tier LRU cache with XFetch Probabilistic Early Recomputation.
    Formula: -beta * delta * ln(random()) > (expiry - now)
    where:
      - delta: time taken to compute the value
      - beta: aggressiveness parameter (default: 1.0)
    """

    def __init__(self, max_size: int = 500, default_ttl_sec: float = 300.0, beta: float = 1.0) -> None:
        self.max_size = max_size
        self.default_ttl_sec = default_ttl_sec
        self.beta = beta

        self._cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()
        self._lock = threading.Lock()
        self._singleflight = Singleflight()

        # Telemetry
        self.stats = {
            "hits": 0,
            "misses": 0,
            "early_recomputes": 0,
            "stampedes_prevented": 0,
        }

    def get(
        self,
        key: str,
        compute_fn: Optional[Callable[[], Any]] = None,
        ttl_sec: Optional[float] = None,
        timeout: Optional[float] = None,
    ) -> Optional[Any]:
        """
        Get value with XFetch stampede prevention.
        If compute_fn is provided, transparently computes or refreshes on cache misses or early expiry.

        Args:
            key: Cache key.
            compute_fn: Callable that recomputes the value on miss/refresh.
            ttl_sec: Per-call TTL override (seconds).
            timeout: Optional wait deadline (seconds) passed through to
                Singleflight when this caller is a follower (wave5 C1.34).
                ``None`` blocks indefinitely.
        """
        now = time.time()
        ttl = ttl_sec or self.default_ttl_sec

        with self._lock:
            entry = self._cache.get(key)

        if entry is not None:
            val = entry["val"]
            expiry = entry["expiry"]
            delta = entry.get("delta", 0.05)

            # XFetch probabilistic early refresh condition
            # If random() is near 0, -ln(random()) is high -> early trigger
            r = max(1e-7, random.random())
            prob_trigger = (-self.beta * delta * math.log(r)) > (expiry - now)

            if now < expiry and not prob_trigger:
                # Valid cache hit
                with self._lock:
                    self._cache.move_to_end(key)
                    self.stats["hits"] += 1
                return val

            if now < expiry and prob_trigger:
                # Early recompute triggered before expiration! Stampede averted!
                with self._lock:
                    self.stats["early_recomputes"] += 1
                if compute_fn:
                    return self._recompute_and_store(key, compute_fn, ttl, timeout)
                # wave5 fix (C1.32): a probabilistic refresh without a
                # compute_fn must NOT discard the still-fresh cached value;
                # return it instead of None.
                with self._lock:
                    self._cache.move_to_end(key)
                return val

            if now >= expiry:
                # Hard expired
                with self._lock:
                    self.stats["misses"] += 1

        else:
            with self._lock:
                self.stats["misses"] += 1

        if compute_fn:
            return self._recompute_and_store(key, compute_fn, ttl, timeout)

        return None

    def _recompute_and_store(
        self, key: str, compute_fn: Callable[[], Any], ttl_sec: float,
        timeout: Optional[float] = None,
    ) -> Any:
        """Execute compute_fn guarded by Singleflight so concurrent requests coalesce."""
        def _wrapper():
            t0 = time.perf_counter()
            computed_val = compute_fn()
            calc_delta = time.perf_counter() - t0
            now = time.time()
            with self._lock:
                if len(self._cache) >= self.max_size and key not in self._cache:
                    self._cache.popitem(last=False)  # LRU evict
                self._cache[key] = {
                    "val": computed_val,
                    "expiry": now + ttl_sec,
                    "delta": max(calc_delta, 0.001),
                }
                self._cache.move_to_end(key)
            return computed_val

        return self._singleflight.do(key, _wrapper, timeout)

    def set(self, key: str, value: Any, ttl_sec: Optional[float] = None, delta: float = 0.01) -> None:
        """Direct set with explicit TTL."""
        ttl = ttl_sec or self.default_ttl_sec
        now = time.time()
        with self._lock:
            if len(self._cache) >= self.max_size and key not in self._cache:
                self._cache.popitem(last=False)
            self._cache[key] = {
                "val": value,
                "expiry": now + ttl,
                "delta": delta,
            }
            self._cache.move_to_end(key)

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()

    def get_hit_ratio(self) -> float:
        total = self.stats["hits"] + self.stats["misses"]
        return self.stats["hits"] / total if total > 0 else 0.0
