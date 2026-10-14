# CineSense — AI Movie Recommendations

Pairwise-comparison recommender (Flask + MySQL 8.0 + PyTorch) on MovieLens + TMDB.
Asks "which movie do you prefer?", learns ELO taste, serves hybrid neural + content picks.

## Docs

- [Overview](docs/overview.md) — what it is, problems, objectives
- [Architecture](docs/architecture.md) — system design, stack, modules
- [Data & Methods](docs/methods.md) — datasets, pipeline, models
- [Database & API](docs/database-api.md) — schema (28 tables, 24 views, 24 procedures, 8 functions, 11 triggers) + 55 `/api` endpoints
- [Setup & Deployment](docs/setup.md) — install, run, deploy
- [Test Results](docs/results.md) — reproduced benchmarks (popularity NDCG@10 0.094 beats hybrid 0.030; hybrid wins coverage 5.5×)

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
| Offline torch (32M, Kaggle) | Raw RMSE 0.768, ROC-AUC 0.836 — research artifact, not in serving path |
| Serving | 300-candidate funnel → top 20, retrieval p95 ~6ms |
| Tests | 76 tests, DB-free subset 19 passed / 2 skipped |

Torch checkpoints are offline-only (`USE_DL_SCORING=0` by default).

## References

- [MovieLens](https://grouplens.org/datasets/movielens/) · [TMDB API](https://developer.themoviedb.org/docs) · [NCF (He et al., 2017)](https://arxiv.org/abs/1708.05031) · [Sentence-BERT](https://arxiv.org/abs/1908.10084)
