"""
CineSense Diversity-Aware Discovery & Explainability Benchmark
==============================================================

Evaluates:
1. Diversity Reranking Trade-off:
   - Evaluates Maximal Marginal Relevance (MMR) with lambda in [0.0, 0.2, 0.5, 0.8]
   - Measures the trade-off between Ranking Relevance (NDCG@10), Intra-List Diversity (ILD), Novelty, and Catalog Coverage.
2. Explainability Metrics:
   - Explanation Coverage (% of recommended items with complete score breakdown and attribution)
   - Attribution Consistency (whether stated components match score composition)
"""

from __future__ import annotations

from typing import Dict, List, Set, Any
import numpy as np

from benchmarks.data_loader import Movie
from benchmarks.metrics import (
    ndcg_at_k,
    recall_at_k,
    catalog_coverage,
    mean_self_information,
    intra_list_diversity,
)
from benchmarks.recommenders import HybridRecommender


def run_diversity_and_explanation_benchmark(
    movies: Dict[int, Movie],
    splits: Dict[str, Any],
    user_train_items: Dict[int, Set[int]],
    test_relevant: Dict[int, Set[int]],
    item_pop: Dict[int, float],
    item_feats: Dict[int, np.ndarray],
    k: int = 10,
    lambdas: List[float] = [0.0, 0.2, 0.5, 0.8],
) -> Dict[str, Any]:
    print("\n" + "=" * 72)
    print("DIVERSITY-AWARE DISCOVERY & EXPLAINABILITY ABLATION")
    print("=" * 72)

    eval_users = [u for u in sorted(splits["eval_users"]) if u in test_relevant and test_relevant[u]]
    catalog = splits["catalog"]

    # Base Hybrid Recommender
    rec = HybridRecommender(0.50, 0.30, 0.20, relevance_threshold=4.0)
    rec.fit(splits["train"], movies)

    results_by_lambda: Dict[str, Dict[str, Any]] = {}

    for lam in lambdas:
        recs_by_user: Dict[int, List[int]] = {}

        for uid in eval_users:
            cands = catalog - user_train_items.get(uid, set())
            if not cands:
                continue

            # Stage 1: score candidates (sorted for deterministic tie order)
            scored = [(it, rec.score_item(uid, it)) for it in sorted(cands)]
            scored.sort(key=lambda x: (-x[1], x[0]))

            if lam == 0.0:
                top_items = [it for it, _ in scored[:k]]
            else:
                # Greedily apply MMR
                selected: List[int] = []
                pool = [it for it, _ in scored[:50]]  # candidate pool of 50
                score_dict = dict(scored[:50])

                if pool:
                    best = pool[0]
                    selected.append(best)
                    pool.remove(best)

                    while pool and len(selected) < k:
                        best_cand = None
                        best_mmr = float("-inf")
                        for cand in pool:
                            rel_s = score_dict[cand]
                            # Compute max similarity to selected
                            c_feat = item_feats[cand]
                            c_norm = np.linalg.norm(c_feat)
                            max_sim = 0.0
                            if c_norm > 0:
                                for s in selected:
                                    s_feat = item_feats[s]
                                    s_norm = np.linalg.norm(s_feat)
                                    if s_norm > 0:
                                        sim = float(np.dot(c_feat, s_feat) / (c_norm * s_norm))
                                        if sim > max_sim:
                                            max_sim = sim
                            mmr_val = (1.0 - lam) * rel_s - lam * max_sim
                            if mmr_val > best_mmr:
                                best_mmr = mmr_val
                                best_cand = cand

                        if best_cand is not None:
                            selected.append(best_cand)
                            pool.remove(best_cand)
                        else:
                            break
                top_items = selected[:k]

            recs_by_user[uid] = top_items

        # Compute ranking metrics and diversity metrics
        ndcgs, recs = [], []
        for uid, rec_list in recs_by_user.items():
            rel = test_relevant[uid]
            ndcgs.append(ndcg_at_k(rec_list, rel, k))
            recs.append(recall_at_k(rec_list, rel, k))

        cov = catalog_coverage(recs_by_user, len(catalog))
        ild = intra_list_diversity(recs_by_user, item_feats, k)
        nov = mean_self_information(recs_by_user, item_pop, k)

        label = f"lambda={lam:.1f}" + (" (No Rerank)" if lam == 0.0 else "")
        res = {
            "lambda": lam,
            f"ndcg@{k}": float(np.mean(ndcgs)),
            f"recall@{k}": float(np.mean(recs)),
            "intra_list_diversity": float(ild),
            "novelty": float(nov),
            "catalog_coverage": float(cov),
            # Audit H1.9: the previous "explanation_coverage": 1.0 and
            # "attribution_consistency": 1.0 were hardcoded, never-measured
            # literals (reports advertised "100% explanation coverage").
            # Removed -- these quantities are not measured by this benchmark.
        }
        results_by_lambda[label] = res
        print(f"  {label:24s} | NDCG@{k}={res[f'ndcg@{k}']:.4f}  Recall@{k}={res[f'recall@{k}']:.4f} | "
              f"Diversity={res['intra_list_diversity']:.4f}  Novelty={res['novelty']:.2f}  "
              f"Coverage={res['catalog_coverage']:.1%}")

    return results_by_lambda
