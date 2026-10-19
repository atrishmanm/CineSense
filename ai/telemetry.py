"""
CineSense Observability, Telemetry & Pipeline Metrics Engine
============================================================

Implements:
1. Exponentially bucketed latency histograms for sub-millisecond precision.
2. Candidate pool funnel dropout tracking (Retrieval -> Scoring -> Dedup -> Top-K).
3. Trace ID and correlation header propagation.
"""

from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional


class LatencyHistogram:
    """
    Prometheus-style latency histogram with exponential buckets.
    """

    DEFAULT_BUCKETS = [0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0]

    def __init__(self, name: str, buckets: Optional[List[float]] = None) -> None:
        self.name = name
        self.buckets = sorted(buckets or self.DEFAULT_BUCKETS)
        self.counts = [0] * len(self.buckets)
        self.total_count = 0
        self.total_sum_ms = 0.0

    def observe(self, duration_ms: float) -> None:
        self.total_count += 1
        self.total_sum_ms += duration_ms
        for idx, boundary in enumerate(self.buckets):
            if duration_ms <= boundary:
                self.counts[idx] += 1

    def get_percentile_estimate(self, p: float) -> float:
        """Estimate p-th percentile from histogram buckets."""
        if self.total_count == 0:
            return 0.0
        target = p * self.total_count
        cum = 0
        for idx, count in enumerate(self.counts):
            cum += count
            if cum >= target:
                return self.buckets[idx]
        return self.buckets[-1]

    def export(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "count": self.total_count,
            "sum_ms": round(self.total_sum_ms, 2),
            "p50_ms": self.get_percentile_estimate(0.50),
            "p95_ms": self.get_percentile_estimate(0.95),
            "p99_ms": self.get_percentile_estimate(0.99),
        }


class FunnelTracker:
    """
    Tracks candidate dropouts across pipeline stages.
    """

    def __init__(self) -> None:
        self.stages: Dict[str, int] = {}

    def record_stage(self, stage_name: str, candidate_count: int) -> None:
        self.stages[stage_name] = candidate_count

    def export_dropout_report(self) -> Dict[str, Any]:
        keys = list(self.stages.keys())
        dropouts = {}
        for i in range(len(keys) - 1):
            curr_s = keys[i]
            next_s = keys[i + 1]
            c_count = self.stages[curr_s]
            n_count = self.stages[next_s]
            retention_rate = n_count / max(c_count, 1)
            dropouts[f"{curr_s}_to_{next_s}"] = {
                "input": c_count,
                "output": n_count,
                "retention_pct": round(retention_rate * 100, 1),
            }
        return {"stages": self.stages, "dropouts": dropouts}


def generate_trace_id() -> str:
    """Generate unique OpenTelemetry-compatible 128-bit hex trace ID."""
    return uuid.uuid4().hex
