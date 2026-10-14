"""
CineSense Semantic Retrieval & Search Benchmark
================================================

Compares:
1. Pure Keyword (Lexical title + genre matching)
2. Dense Embedding (Cosine similarity over semantic representation)
3. Hybrid Retrieval (Linear blend of lexical and semantic match)

Evaluated on a human-curated intent query set containing:
- Specific titles & partial matches
- Conceptual / thematic searches ("space opera rebellion", "mafia family drama")
- Mood & genre queries ("hilarious romantic comedy", "creepy psychological horror")
- Difficult / descriptive queries ("toys coming to life", "minnesota snowy crime thriller")

Metrics:
- Recall@10
- MRR@10 (Mean Reciprocal Rank)
- NDCG@10
- Zero-Result Rate (%)
- Query Latency: p50 and p95 in milliseconds
"""

from __future__ import annotations

import math
import re
import time
from typing import Dict, List, Set, Tuple, Any
import numpy as np

from benchmarks.data_loader import Movie


# Curated labeled query benchmark: (query, relevant_keywords_or_genres, expected_title_substrings)
LABELED_QUERIES: List[Tuple[str, List[str], List[str]]] = [
    ("animated toys coming to life", ["animation", "children"], ["toy story"]),
    ("space opera galactic empire rebellion", ["sci-fi", "action", "adventure"], ["star wars", "empire strikes back", "return of the jedi"]),
    ("italian mafia family crime saga", ["crime", "drama"], ["godfather"]),
    ("minnesota dark comedy crime kidnapping", ["crime", "drama"], ["fargo"]),
    ("cyberpunk virtual reality future", ["sci-fi", "action"], ["matrix", "terminator", "blade runner", "twelve monkeys"]),
    ("romantic comedy love story paris", ["romance", "comedy"], ["french kiss", "before sunrise", "sabrina"]),
    ("underwater monster shark thriller", ["thriller", "action"], ["jaws", "abyss"]),
    ("courtroom legal drama justice", ["drama"], ["12 angry men", "to kill a mockingbird", "few good men", "time to kill"]),
    ("world war 2 epic invasion drama", ["war", "drama", "action"], ["saving private ryan", "schindler's list", "platoon"]),
    ("time travel paradox sci-fi adventure", ["sci-fi", "adventure"], ["back to the future", "terminator", "twelve monkeys"]),
    ("psychological serial killer detective thriller", ["mystery", "thriller"], ["silence of the lambs", "se7en", "seven"]),
    ("dinosaur theme park science goes wrong", ["adventure", "sci-fi"], ["jurassic park", "lost world"]),
    ("underdog boxer redemption champion", ["drama"], ["rocky"]),
    ("haunted house spooky supernatural horror", ["horror"], ["shining", "poltergeist", "psycho"]),
    ("quirky indie comedy high school", ["comedy"], ["clueless", "ferris bueller"]),
    ("secret agent spy mission explosion", ["action", "thriller"], ["goldeneye", "mission: impossible", "die hard"]),
    ("western gunslinger bounty hunter desert", ["western"], ["unforgiven", "good, the bad and the ugly"]),
    ("fantasy quest ring kingdom sword", ["fantasy", "adventure"], ["princess bride", "dragonheart"]),
    ("epic heist bank robbery crew", ["crime", "action"], ["heat", "usual suspects", "reservoir dogs"]),
    ("superhero vigilante protecting city dark", ["action", "crime"], ["batman", "crow"]),
]


def _tokenize(text: str) -> Set[str]:
    return set(re.findall(r"\b[a-z0-9]+\b", text.lower()))


def run_semantic_benchmark(movies: Dict[int, Movie], k: int = 10) -> Dict[str, Any]:
    print("\n" + "=" * 72)
    print("SEMANTIC RETRIEVAL & SEARCH BENCHMARK")
    print("=" * 72)

    # Genre vocabulary
    genre_names = [
        "action", "adventure", "animation", "children", "comedy", "crime",
        "documentary", "drama", "fantasy", "film-noir", "horror", "musical",
        "mystery", "romance", "sci-fi", "thriller", "war", "western", "unknown"
    ]
    genre_to_idx = {g: i for i, g in enumerate(genre_names)}

    # Pre-index movies: tokens and dense genre vector
    movie_tokens: Dict[int, Set[str]] = {}
    movie_dense: Dict[int, np.ndarray] = {}
    for mid, m in movies.items():
        t = _tokenize(m.title)
        for g_idx, val in enumerate(m.genres):
            if val > 0 and g_idx < len(genre_names):
                t.add(genre_names[g_idx])
        movie_tokens[mid] = t
        movie_dense[mid] = m.genres.astype(np.float32)

    methods = ["Lexical / Keyword Only", "Dense Semantic Embedding", "Hybrid (Keyword + Dense)"]
    method_metrics: Dict[str, Dict[str, Any]] = {}

    for method in methods:
        recalls, mrrs, ndcgs, latencies = [], [], [], []
        zero_results = 0

        for query, intent_genres, ground_truth_titles in LABELED_QUERIES:
            t0 = time.perf_counter()
            q_tokens = _tokenize(query)

            # Build query dense representation from intent genres and query text
            q_vec = np.zeros(len(genre_names), dtype=np.float32)
            for g in intent_genres:
                if g in genre_to_idx:
                    q_vec[genre_to_idx[g]] = 1.0
            for w in q_tokens:
                if w in genre_to_idx:
                    q_vec[genre_to_idx[w]] = 1.0
            q_norm = np.linalg.norm(q_vec)

            # Define relevance ground truth in the catalog
            rel_mids: Set[int] = set()
            for mid, m in movies.items():
                m_title_lower = m.title.lower()
                if any(gt in m_title_lower for gt in ground_truth_titles):
                    rel_mids.add(mid)

            if not rel_mids:
                continue

            scores: List[Tuple[int, float]] = []
            for mid, m in movies.items():
                lexical_score = 0.0
                if method in ["Lexical / Keyword Only", "Hybrid (Keyword + Dense)"]:
                    overlap = len(q_tokens & movie_tokens[mid])
                    lexical_score = overlap / len(q_tokens) if q_tokens else 0.0

                dense_score = 0.0
                if method in ["Dense Semantic Embedding", "Hybrid (Keyword + Dense)"]:
                    m_norm = np.linalg.norm(movie_dense[mid])
                    if q_norm > 0 and m_norm > 0:
                        dense_score = float(np.dot(q_vec, movie_dense[mid]) / (q_norm * m_norm))

                if method == "Lexical / Keyword Only":
                    s = lexical_score
                elif method == "Dense Semantic Embedding":
                    s = dense_score
                else:  # Hybrid
                    s = 0.50 * lexical_score + 0.50 * dense_score

                if s > 0:
                    scores.append((mid, s))

            elapsed_ms = (time.perf_counter() - t0) * 1000
            latencies.append(elapsed_ms)

            scores.sort(key=lambda x: x[1], reverse=True)
            top_recs = [mid for mid, _ in scores[:k]]

            if not top_recs:
                zero_results += 1
                recalls.append(0.0)
                mrrs.append(0.0)
                ndcgs.append(0.0)
                continue

            # Recall@K
            hits = sum(1 for mid in top_recs if mid in rel_mids)
            recalls.append(hits / len(rel_mids))

            # MRR@K
            mrr = 0.0
            for rank, mid in enumerate(top_recs, 1):
                if mid in rel_mids:
                    mrr = 1.0 / rank
                    break
            mrrs.append(mrr)

            # NDCG@K
            dcg = sum(1.0 / math.log2(i + 2) for i, mid in enumerate(top_recs) if mid in rel_mids)
            idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(rel_mids), k)))
            ndcgs.append(dcg / idcg if idcg > 0 else 0.0)

        lat_arr = np.array(latencies)
        method_metrics[method] = {
            f"recall@{k}": float(np.mean(recalls)),
            f"mrr@{k}": float(np.mean(mrrs)),
            f"ndcg@{k}": float(np.mean(ndcgs)),
            "zero_result_rate": float(zero_results / len(LABELED_QUERIES)),
            "p50_latency_ms": float(np.percentile(lat_arr, 50)),
            "p95_latency_ms": float(np.percentile(lat_arr, 95)),
            "n_queries": len(LABELED_QUERIES),
        }

        print(f"  {method:28s} | Recall@{k}={method_metrics[method][f'recall@{k}']:.4f}  "
              f"MRR@{k}={method_metrics[method][f'mrr@{k}']:.4f}  "
              f"NDCG@{k}={method_metrics[method][f'ndcg@{k}']:.4f} | "
              f"ZeroRes={method_metrics[method]['zero_result_rate']:.1%} | "
              f"p95 Latency={method_metrics[method]['p95_latency_ms']:.2f}ms")

    return method_metrics
