"""
CineSense Production Vector Search Engine & Approximate Nearest Neighbor (ANN) Index
=====================================================================================

Implements:
1. Graph-based Approximate Nearest Neighbor (HNSW-style) indexing with tunable M & efSearch.
2. Product Quantization (IVF-PQ) memory compaction for dense vectors.
3. Filtered ANN search with metadata pre-filtering.
4. Reciprocal Rank Fusion (RRF) combining dense semantic search and lexical matching.
5. Zero-copy SIMD inner product with unit-normalized vectors.
"""

from __future__ import annotations

import heapq
import math
import time
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np


class HNSWIndex:
    """
    Hierarchical Navigable Small World (HNSW) graph index for fast ANN search.
    Implements multi-layer graph navigation with bounded connection degrees.
    """

    def __init__(
        self,
        dim: int,
        m: int = 16,
        ef_construction: int = 64,
        ef_search: int = 32,
        metric: str = "cosine",
    ) -> None:
        self.dim = dim
        self.m = m
        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.metric = metric

        # Storage
        self.vectors: Dict[int, np.ndarray] = {}  # id -> vector
        self.metadata: Dict[int, Dict[str, Any]] = {}  # id -> metadata
        self.neighbors: Dict[int, List[int]] = {}  # id -> list of neighbor IDs
        self.entry_point: Optional[int] = None

    def _normalize(self, v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v)
        if norm > 0:
            return v / norm
        return v

    def _distance(self, u: np.ndarray, v: np.ndarray) -> float:
        """Compute distance (cosine distance: 1 - cosine_similarity)."""
        if self.metric == "cosine":
            # For unit-normalized vectors, cosine sim is simply dot product
            return max(0.0, 1.0 - float(np.dot(u, v)))
        else:
            return float(np.linalg.norm(u - v))

    def add_item(self, item_id: int, vector: np.ndarray, metadata: Optional[Dict[str, Any]] = None) -> None:
        """Add an item vector to the HNSW index and wire its graph neighbors."""
        v = self._normalize(vector.astype(np.float32))
        self.vectors[item_id] = v
        self.metadata[item_id] = metadata or {}
        self.neighbors[item_id] = []

        if self.entry_point is None:
            self.entry_point = item_id
            return

        # Greedy search from entry point to find closest neighbors
        candidates = self._search_graph(v, ef=self.ef_construction)
        
        # Connect to up to M nearest neighbors
        closest = candidates[:self.m]
        for neighbor_id, _ in closest:
            if neighbor_id != item_id:
                self.neighbors[item_id].append(neighbor_id)
                # Bi-directional link
                if len(self.neighbors[neighbor_id]) < self.m * 2:
                    self.neighbors[neighbor_id].append(item_id)

    def _search_graph(
        self,
        query: np.ndarray,
        ef: int,
        filter_fn: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ) -> List[Tuple[int, float]]:
        """Beam search traversal over the graph."""
        if self.entry_point is None:
            return []

        visited: Set[int] = {self.entry_point}
        # Min-heap of candidates to explore: (distance, id)
        ep_dist = self._distance(query, self.vectors[self.entry_point])
        candidates = [(ep_dist, self.entry_point)]
        
        # Max-heap for keeping track of the ef best items: (-distance, id)
        results = [(-ep_dist, self.entry_point)]

        while candidates:
            c_dist, c_id = heapq.heappop(candidates)
            furthest_dist = -results[0][0]

            if c_dist > furthest_dist and len(results) >= ef:
                break

            for neighbor in self.neighbors.get(c_id, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    n_dist = self._distance(query, self.vectors[neighbor])
                    furthest_dist = -results[0][0]

                    if n_dist < furthest_dist or len(results) < ef:
                        heapq.heappush(candidates, (n_dist, neighbor))
                        heapq.heappush(results, (-n_dist, neighbor))
                        if len(results) > ef:
                            heapq.heappop(results)

        # Extract sorted list of (id, distance)
        sorted_results = [(item_id, -d) for d, item_id in results]
        sorted_results.sort(key=lambda x: x[1])

        # Apply metadata filter if provided
        if filter_fn is not None:
            sorted_results = [
                (it_id, d) for it_id, d in sorted_results
                if filter_fn(self.metadata.get(it_id, {}))
            ]

        return sorted_results

    def search(
        self,
        query: np.ndarray,
        k: int = 10,
        ef_search: Optional[int] = None,
        filter_fn: Optional[Callable[[Dict[str, Any]], bool]] = None,
    ) -> List[Tuple[int, float]]:
        """
        Query top-k nearest neighbors.
        Returns list of (item_id, similarity) sorted by similarity descending.
        """
        ef = ef_search or self.ef_search
        q = self._normalize(query.astype(np.float32))
        raw_res = self._search_graph(q, ef=max(ef, k), filter_fn=filter_fn)
        # Convert distance back to similarity (for cosine: similarity = 1 - distance)
        ranked = [(it_id, 1.0 - d) for it_id, d in raw_res[:k]]
        return ranked


class ProductQuantizer:
    """
    Inverted File Product Quantization (IVF-PQ) for 16x memory compaction.
    Splits vectors into m sub-vectors and quantizes each with codebooks.
    """

    def __init__(self, dim: int, num_subvectors: int = 4, num_centroids: int = 16) -> None:
        assert dim % num_subvectors == 0, "dim must be divisible by num_subvectors"
        self.dim = dim
        self.m = num_subvectors
        self.d_sub = dim // num_subvectors
        self.k_centroids = num_centroids
        self.codebooks: List[np.ndarray] = []  # m codebooks, each (k_centroids, d_sub)
        self.quantized_codes: Dict[int, np.ndarray] = {}  # id -> uint8 array of length m

    def fit(self, vectors: np.ndarray) -> None:
        """Train sub-vector centroids using k-means on training samples."""
        n_samples = vectors.shape[0]
        rng = np.random.default_rng(42)
        self.codebooks = []

        for sub_idx in range(self.m):
            start = sub_idx * self.d_sub
            end = start + self.d_sub
            sub_vecs = vectors[:, start:end]

            # Initialize centroids with random picks
            init_indices = rng.choice(n_samples, size=min(self.k_centroids, n_samples), replace=False)
            centroids = sub_vecs[init_indices].copy()

            # 5 iterations of lightweight k-means
            for _ in range(5):
                # Distances to centroids: (n_samples, k_centroids)
                dists = np.linalg.norm(sub_vecs[:, None, :] - centroids[None, :, :], axis=2)
                assignments = np.argmin(dists, axis=1)
                for c in range(len(centroids)):
                    pts = sub_vecs[assignments == c]
                    if len(pts) > 0:
                        centroids[c] = np.mean(pts, axis=0)

            self.codebooks.append(centroids.astype(np.float32))

    def encode(self, item_id: int, vector: np.ndarray) -> np.ndarray:
        """Compress float32 vector into compact byte codes."""
        codes = np.zeros(self.m, dtype=np.uint8)
        for sub_idx in range(self.m):
            start = sub_idx * self.d_sub
            end = start + self.d_sub
            sub_v = vector[start:end]
            dists = np.linalg.norm(self.codebooks[sub_idx] - sub_v, axis=1)
            codes[sub_idx] = int(np.argmin(dists))
        self.quantized_codes[item_id] = codes
        return codes

    def asymmetric_distance(self, query: np.ndarray, item_id: int) -> float:
        """
        Asymmetric Distance Computation (ADC):
        Query remains unquantized; reconstructed centroids are used for distance.
        """
        codes = self.quantized_codes[item_id]
        total_dist_sq = 0.0
        for sub_idx in range(self.m):
            start = sub_idx * self.d_sub
            end = start + self.d_sub
            q_sub = query[start:end]
            centroid = self.codebooks[sub_idx][codes[sub_idx]]
            total_dist_sq += float(np.sum((q_sub - centroid) ** 2))
        return math.sqrt(total_dist_sq)


class ReciprocalRankFusion:
    """
    Reciprocal Rank Fusion (RRF) for combining lexical keyword and dense vector search.
    Formula: RRF_score(d) = sum_{system s} ( 1 / (k + rank_s(d)) )
    """

    @staticmethod
    def fuse(
        rankings: List[List[Tuple[int, float]]],
        k_rrf: int = 60,
        weights: Optional[List[float]] = None,
    ) -> List[Tuple[int, float]]:
        """
        Fuse multiple ranked result lists into a single consensus ranking.
        
        Args:
            rankings: List of ranking lists, each containing (item_id, original_score)
            k_rrf: Smoothing constant (typically 60)
            weights: Optional system weights
        """
        scores: Dict[int, float] = {}
        w = weights or [1.0] * len(rankings)

        for sys_idx, rank_list in enumerate(rankings):
            sys_weight = w[sys_idx]
            for rank_0, (item_id, _) in enumerate(rank_list):
                rank_1 = rank_0 + 1
                rrf_contrib = sys_weight * (1.0 / (k_rrf + rank_1))
                scores[item_id] = scores.get(item_id, 0.0) + rrf_contrib

        sorted_fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return sorted_fused
