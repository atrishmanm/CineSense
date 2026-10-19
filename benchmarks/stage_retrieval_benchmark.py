"""
CineSense Two-Stage Retrieval & Ranking Profiling
==================================================

Evaluates Stage 1 (Candidate Generation) vs Stage 2 (Hybrid Reranking):
- Evaluates candidate pools at sizes: 50, 100, 300, 500
- Measures Candidate Recall@K_cand before reranking
- Measures Final Recall@10 and NDCG@10 after Stage 2 reranking
- Measures Stage 1 retrieval latency (p50/p95) vs Stage 2 reranking latency (p50/p95)
- Computes candidate source overlap (Jaccard) between Popularity, Content, and ELO
"""

from __future__ import annotations

import time
from collections import defaultdict
from typing import Dict, List, Set, Any
import numpy as np

from benchmarks.data_loader import Movie
from benchmarks.metrics import ndcg_at_k, recall_at_k


def run_two_stage_benchmark(
    movies: Dict[int, Movie],
    splits: Dict[str, Any],
    user_train_items: Dict[int, Set[int]],
    test_relevant: Dict[int, Set[int]],
    candidate_budgets: List[int] = [50, 100, 300, 500],
    k_final: int = 10,
) -> Dict[str, Any]:
    print("\n" + "=" * 72)
    print("TWO-STAGE RETRIEVAL & RERANKING PROFILING")
    print("=" * 72)

    eval_users = [u for u in sorted(splits["eval_users"]) if u in test_relevant and test_relevant[u]]
    catalog = splits["catalog"]

    # Precompute Stage 1 retrieval sources
    # 1. Popularity scores
    item_pop_counts: Dict[int, int] = defaultdict(int)
    for r in splits["train"]:
        if r.rating >= 4.0:
            item_pop_counts[r.item_id] += 1
    sorted_pop = sorted(item_pop_counts.items(), key=lambda x: (-x[1], x[0]))
    global_pop_order = [it for it, _ in sorted_pop]

    # 2. Content profiles
    user_profiles: Dict[int, np.ndarray] = {}
    liked_by_user: Dict[int, list] = defaultdict(list)
    for r in splits["train"]:
        if r.rating >= 4.0 and r.item_id in movies:
            liked_by_user[r.user_id].append(movies[r.item_id].genres)
    for uid, g_list in liked_by_user.items():
        user_profiles[uid] = np.mean(g_list, axis=0)

    # 3. ELO scores
    elo_scores: Dict[int, float] = {mid: 1500.0 for mid in catalog}
    for r in splits["train"]:
        if r.rating >= 4.0:
            elo_scores[r.item_id] = elo_scores.get(r.item_id, 1500.0) + 10.0
        elif r.rating <= 2.0:
            elo_scores[r.item_id] = elo_scores.get(r.item_id, 1500.0) - 10.0
    sorted_elo = sorted(elo_scores.items(), key=lambda x: (-x[1], x[0]))
    global_elo_order = [it for it, _ in sorted_elo]

    # Measure Candidate Overlap between sources (top 100)
    top_pop_100 = set(global_pop_order[:100])
    top_elo_100 = set(global_elo_order[:100])
    jaccard_pop_elo = len(top_pop_100 & top_elo_100) / len(top_pop_100 | top_elo_100) if (top_pop_100 | top_elo_100) else 0.0

    content_pop_overlaps = []
    for uid in eval_users[:100]:
        prof = user_profiles.get(uid)
        if prof is not None and np.linalg.norm(prof) > 0:
            sims = [(mid, float(np.dot(prof, movies[mid].genres) / (np.linalg.norm(prof) * np.linalg.norm(movies[mid].genres) + 1e-8)))
                    for mid in catalog]
            sims.sort(key=lambda x: (-x[1], x[0]))
            top_content_100 = {mid for mid, _ in sims[:100]}
            jaccard_cp = len(top_content_100 & top_pop_100) / len(top_content_100 | top_pop_100)
            content_pop_overlaps.append(jaccard_cp)
    mean_content_pop_overlap = float(np.mean(content_pop_overlaps)) if content_pop_overlaps else 0.0

    print("  Source Overlap (Top 100 Jaccard):")
    print(f"    Popularity vs ELO:       {jaccard_pop_elo:.3f}")
    print(f"    Content vs Popularity:   {mean_content_pop_overlap:.3f}")

    results_by_budget: Dict[int, Dict[str, Any]] = {}

    for budget in candidate_budgets:
        cand_recalls = []
        final_recalls = []
        final_ndcgs = []
        retrieval_latencies = []
        rerank_latencies = []
        unique_candidates = set()

        # Split budget across sources: 40% Content, 30% Pop, 30% ELO
        # (ELO leg fills the remaining budget implicitly)
        b_content = int(budget * 0.4)
        b_pop = int(budget * 0.3)

        for uid in eval_users:
            seen = user_train_items.get(uid, set())
            rel = test_relevant[uid]

            # Stage 1: Candidate Generation
            t0 = time.perf_counter()
            candidates = set()

            # Content candidates
            prof = user_profiles.get(uid)
            if prof is not None and np.linalg.norm(prof) > 0:
                p_norm = np.linalg.norm(prof)
                sims = []
                for mid in catalog:
                    if mid not in seen:
                        m_norm = np.linalg.norm(movies[mid].genres)
                        sim = float(np.dot(prof, movies[mid].genres) / (p_norm * m_norm + 1e-8))
                        sims.append((mid, sim))
                sims.sort(key=lambda x: (-x[1], x[0]))
                for mid, _ in sims[:b_content]:
                    candidates.add(mid)

            # Popularity candidates
            for mid in global_pop_order:
                if mid not in seen and mid not in candidates:
                    candidates.add(mid)
                    if len(candidates) >= (b_content + b_pop):
                        break

            # ELO candidates
            for mid in global_elo_order:
                if mid not in seen and mid not in candidates:
                    candidates.add(mid)
                    if len(candidates) >= budget:
                        break

            t_retrieval = (time.perf_counter() - t0) * 1000
            retrieval_latencies.append(t_retrieval)
            unique_candidates.update(candidates)

            # Candidate Recall before reranking
            hits_in_cand = sum(1 for it in candidates if it in rel)
            cand_recall = hits_in_cand / len(rel) if len(rel) > 0 else 0.0
            cand_recalls.append(cand_recall)

            # Stage 2: Hybrid Reranking on Candidates
            t1 = time.perf_counter()
            scored_candidates = []
            for it in sorted(candidates):  # deterministic tie order
                # Content score
                if prof is not None and np.linalg.norm(prof) > 0:
                    c_sim = float(np.dot(prof, movies[it].genres) / (np.linalg.norm(prof) * np.linalg.norm(movies[it].genres) + 1e-8))
                else:
                    c_sim = 0.5
                c_score = max(0.0, min(1.0, (c_sim + 1.0) / 2.0))

                # ELO score
                e_score = max(0.0, min(1.0, (elo_scores.get(it, 1500.0) - 1000.0) / 1000.0))
                # Pop score
                p_score = min(1.0, item_pop_counts.get(it, 0) / 100.0)

                hybrid_score = 0.50 * c_score + 0.30 * p_score + 0.20 * e_score
                scored_candidates.append((it, hybrid_score))

            scored_candidates.sort(key=lambda x: (-x[1], x[0]))
            top_final = [it for it, _ in scored_candidates[:k_final]]
            t_rerank = (time.perf_counter() - t1) * 1000
            rerank_latencies.append(t_rerank)

            final_ndcgs.append(ndcg_at_k(top_final, rel, k=k_final))
            final_recalls.append(recall_at_k(top_final, rel, k=k_final))

        r_lat = np.array(retrieval_latencies)
        k_lat = np.array(rerank_latencies)
        res = {
            "budget": budget,
            "candidate_recall": float(np.mean(cand_recalls)),
            f"final_recall@{k_final}": float(np.mean(final_recalls)),
            f"final_ndcg@{k_final}": float(np.mean(final_ndcgs)),
            "catalog_items_represented": len(unique_candidates),
            "retrieval_p50_ms": float(np.percentile(r_lat, 50)),
            "retrieval_p95_ms": float(np.percentile(r_lat, 95)),
            "rerank_p50_ms": float(np.percentile(k_lat, 50)),
            "rerank_p95_ms": float(np.percentile(k_lat, 95)),
        }
        results_by_budget[budget] = res
        print(f"  Budget {budget:3d} candidates: "
              f"CandRecall={res['candidate_recall']:.4f} -> FinalRecall@{k_final}={res[f'final_recall@{k_final}']:.4f}  "
              f"NDCG@{k_final}={res[f'final_ndcg@{k_final}']:.4f} | "
              f"Retr p95={res['retrieval_p95_ms']:.2f}ms Rerank p95={res['rerank_p95_ms']:.2f}ms | "
              f"Catalog={res['catalog_items_represented']}")

    return {
        "budgets": results_by_budget,
        "overlap": {
            "jaccard_pop_elo": round(jaccard_pop_elo, 4),
            "mean_content_pop_overlap": round(mean_content_pop_overlap, 4),
        }
    }
