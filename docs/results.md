# CineSense — Test Results

> Canonical command: `python benchmarks/run_all_benchmarks.py`
> (per-user temporal 70/10/20 split, seed 42, full-catalog candidates, K=10).
> Relevance = rating ≥ 4.0. Artifacts: `benchmarks/results/benchmark_report.md`,
> `benchmark_results.json`, `metrics.csv`.
> Part of the [README](../README.md) doc set.

## 1. Full-catalog ranking, ML-100K (reproduced locally, Oct 2026)

| Strategy | NDCG@10 | Hit@10 | Coverage | Novelty |
|----------|--------|-------|----------|---------|
| Popularity | **0.0939** | **0.396** | 4.0% | 1.44 |
| Content | 0.0281 | 0.170 | 26.3% | 5.32 |
| Preference+ELO | 0.0245 | 0.178 | 2.2% | 3.30 |
| Hybrid (50/30/20) | 0.0304 | 0.199 | 21.8% | 4.46 |
| Hybrid+Popularity (RRF) | 0.0632 | 0.317 | — | — |

Paired bootstrap 95% CIs (n=1000) confirm the gaps to Popularity are
significant (p<0.05). Honest reading: **popularity wins head-ranking** on
MovieLens-100K. The RRF popularity prior closes two-thirds of the gap
(Hybrid+Pop NDCG 0.063 vs hybrid 0.030) — recommended as the production
default blend. The pure hybrid's value is coverage (5.5×), novelty, and
personalization — don't claim top-10 wins.

## 2. Ablation (validation → test)

Validation best: ELO-heavy 0.0314, Equal 0.0309, Pref-heavy 0.0309.
Test (top-3): **Equal 0.0337**, Pref-heavy 0.0337, ELO-heavy 0.0294.
No-content collapses to the ELO-only floor — content carries the hybrid.

## 3. Cold start

By activity tier (Hybrid): heavy (51+) NDCG 0.047/Hit 0.336, moderate 0.017/0.091,
light 0.012/0.033. Simulated new users, 200-user cohort, refit per k:

| History | Popularity | Hybrid | Hybrid+Pop |
|---------|-----------|--------|-----------|
| k=0 (no history) | 0.052 / 0.275 | 0.024 / 0.170 | 0.043 / 0.230 |
| k=2 | 0.052 / 0.275 | 0.022 / 0.160 | 0.041 / 0.285 |
| k=5 | 0.052 / 0.275 | 0.026 / 0.195 | 0.047 / 0.260 |

(NDCG@10 / Hit@10.) Cold start is popularity-dominant, as expected — the
takeaway is the fallback order: with no history, serve popularity; blend in
the hybrid as interactions accrue. Hybrid+Pop is the single best compromise.

## 4. Retrieval funnel + latency

| Budget | CandRecall | Final NDCG@10 | Retrieval p95 | Rerank p95 |
|--------|-----------|--------------|--------------|-----------|
| 50 | 0.205 | **0.0811** | 5.7ms | 0.6ms |
| 300 | 0.615 | 0.0703 | 6.3ms | 3.0ms |
| 500 | 0.771 | 0.0711 | 5.9ms | 4.2ms |

Note: budget-50 rerank (0.081) beats the flat hybrid (0.030) — the ELO-heavy
rerank path carries a popularity-like prior. This motivates the
Hybrid+Popularity variant above. Cache: warm p95 <0.01ms.

## 5. Diversity / semantic (torch-free stand-ins)

MMR λ=0.2: NDCG 0.0312, diversity 0.33 (vs 0.20 unreranked), coverage 18%,
explanation coverage 100%. Semantic search: lexical Recall@10 0.088;
"dense" is a genre-vector stand-in scoring ~0 on title-based ground truth
(expected — genre vectors carry no title signal); production 768-d
embeddings are untested in this harness.

## 6. Offline torch models (NOT reproduced locally, NOT wired into the app)

| Artifact | Raw RMSE | ROC-AUC | Note |
|----------|---------|--------|------|
| `models/kaggle_t4x2` (100K Deep V2) | 0.949 | 0.774 | no local trainer reproduces it; use `training/train_100k.py` for reproducible 100K numbers |
| `models/kaggle_t4x2_full32m` (Deep V3) | 0.77 | 0.8352 | 32M ratings; rerun 2026-10-14 on Kaggle Dual T4 (kernel v5, log kept) — reproduces prior 0.7678/0.8355 within noise; ranking used a sampled protocol |
| `models/kaggle_t4x2_3m_subset` (Deep V3) | 0.801 | 0.823 | 3M-rating subset, formerly mislabeled "32M" |

DL scoring is OFF by default (`USE_DL_SCORING=0`); these are research
artifacts. Load check (Oct 2026, torch 2.10): the full-32M checkpoint loads
into `training/deep_v3.py:CineSenseDeepV3Full32M` with **0 missing /
0 unexpected keys** and a sane forward pass. The 32M metrics were reproduced
by a fresh Kaggle rerun (2026-10-14, raw RMSE 0.77 vs prior 0.7678) with the
log kept in `models/kaggle_t4x2_full32m/`.

## 7. Reproducible local training

```bash
python training/train_100k.py --epochs 15   # → model/mf_100k.npz + results/train_100k_metrics.json
python benchmarks/run_all_benchmarks.py     # full suite → benchmarks/results/
python benchmarks/run_comprehensive_evaluation.py  # regression-heavy eval → all_metrics.json
pytest -q                                   # unit/integration (DB-free subset passes)
```

## 8. Test suite

`pytest -q` collects **88 tests** across 10 `test_*.py` files in `tests/`.
Latest verified run: **64 passed, 24 skipped** (Oct 2026) — skips are
live-server-gated, optional-dependency, and offline-skip tests.

[<- Back to README](../README.md)
