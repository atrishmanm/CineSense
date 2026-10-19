# CineSense — AI Movie Recommendations

Pairwise-comparison recommender (Flask + MySQL 8.0 + PyTorch) on MovieLens + TMDB.
Asks "which movie do you prefer?", learns ELO taste, serves hybrid neural + content picks.

## Docs (3 files, merged 2026-10-19)

- [Project Reference](docs/PROJECT.md) — overview, architecture, methods, database & API, setup & deployment (merged: overview + architecture + methods + database-api + setup)
- [Test Results](docs/RESULTS.md) — reproduced benchmarks (popularity NDCG@10 0.094 beats hybrid 0.030; hybrid wins coverage 5.5×)
- [Audit Report 2026-10-19](docs/AUDIT-2026-10-19.md) — full audit ledger: 651 curated + 192 machine-scan + 26 addendum + 20 wave-4 findings, all fixed statuses, plus the complete original findings lists
- [Model Artifacts](models/README.md) — every checkpoint: producer, consumer, reproduction

## Quick start

```bash
pip install -r requirements.txt
cp .env.example .env   # set DB_* + TMDB_API_KEY
mysql -u root -p cinesense < database/schema.sql
python app_integrated.py   # → http://localhost:5000
```

## Key numbers (reproduced, not claimed)

| Item | Value |
|------|-------|
| Benchmark harness | `python benchmarks/run_all_benchmarks.py` — temporal split, full catalog, bootstrap CIs |
| Best local ranker | Popularity NDCG@10 0.094 / Hit@10 0.40 (hybrid: 0.030 / 0.20, 5.5× coverage) |
| Offline torch (32M, Kaggle rerun 2026-10-14) | Raw RMSE 0.77, ROC-AUC 0.8352 — research artifact, not in serving path |
| Serving | 300-candidate funnel → top 20, retrieval p95 ~6ms |
| Tests | 83 test functions; latest run **65 passed / 24 skipped / 0 failed** |
| Lint | pyflakes **0 findings** repo-wide |

Torch checkpoints are offline-only (`USE_DL_SCORING=0` by default).

## Codebase (verified 2026-10-19)

149 tracked files · 96 Python files · 28,114 Python LOC · 36 tables / 24 views /
24 procedures / 8 functions / 11 triggers · 69 URL rules (55 `/api` endpoints).

## References

- [MovieLens](https://grouplens.org/datasets/movielens/) · [TMDB API](https://developer.themoviedb.org/docs) · [NCF (He et al., 2017)](https://arxiv.org/abs/1708.05031) · [Sentence-BERT](https://arxiv.org/abs/1908.10084)
