# CineSense — System Architecture

> Verified against the repo (Oct 2026). Corrects stale counts from earlier docs.
> Part of the [README](../README.md) doc set.

## High-level diagram

```
Browser (15 Jinja templates + static/css + static/js)
        │  fetch()/AJAX
        ▼
app_integrated.py :: create_app()   ← single entry point (no app.py)
 ├── Web routes: / /compare /movie/<id> /search /login /signup
 │               /profile /category/<name> /monitor /chat-ui
 │               /features /friends
 ├── Blueprint: api/routes.py  (mounted under /api)
 ├── Blueprint: api/social_routes.py (friends, watch parties, lists)
        │
        ├── database/db_manager.py → MySQL 8.0 pool (10 conns)
        ├── ai/recommender.py      → master orchestrator
        └── tmdb/fetcher.py        → TMDB API fallback
```

## Tech stack (from requirements.txt / config.py)

| Layer | Technology |
|-------|-----------|
| Backend | Flask 3.0, Gunicorn 21.2, Flask-CORS, python-dotenv |
| Database | MySQL 8.0, mysql-connector-python 8.2 (pool), PyMySQL fallback |
| Deep learning | PyTorch ≥2.1, sentence-transformers ≥2.2.2, transformers ≥4.35 |
| ML / data | NumPy, scikit-learn, SciPy, pandas, tqdm, faiss-cpu |
| Serving extras | redis 5.0, Pillow, shap, torchserve/model-archiver |
| Frontend | HTML5 + CSS3 + vanilla JS + Jinja2 (no SPA framework) |
| Deploy | Render (render.yaml) + Procfile + runtime.txt |

## Request flow (recommendations)

```
GET /api/recommendations
  1. Load user vector (user_embeddings)
  2. Retrieve ~300 candidates (config CANDIDATE_COUNT=300, mixed strategy)
     ├── genre affinity slice
     ├── popularity / trending slice
     └── UCB exploration slice
  3. Score: content similarity + ELO/pairwise + (optional) DL score
     (DL scoring OFF by default: USE_DL_SCORING=0, DL_SCORE_WEIGHT=0.15;
      torch checkpoints in models/ are offline research artifacts)
  4. Rerank + diversify → FINAL_COUNT=20
  5. Log to recommendation_log, return JSON
```

## Modules (actual files)

- `ai/` — 32 modules: `recommender` (orchestrator), `pairwise_learning` (ELO),
  `two_tower_ncf`, `hybrid_model`, `neumf_scorer`,
  `reinforcement` (UCB), `embeddings` (55-dim), `semantic_search`,
  `candidate_generator`, `multi_stage_ranker`, `diversity_engine`,
  `vector_search`, `reranker`/`cross-encoder` equivalents, `mood_detector`,
  `conversational_agent`, `visual_search`, `trending_detector`,
  `cache_manager`, `xfetch_cache`, `redis_cache`, `resilience`,
  `feature_store`, `telemetry`, `drift_monitor`, `ab_testing`,
  `ab_experimentation`, `evaluation`, `inference_pipeline`,
  `training_pipeline`, `netflix_recommender`, `content_pipeline`,
  `data_preprocessor`
- `training/` — 11 files: `models`, `advanced_models`, `advanced_models_v2`,
  `deep_v3`, `losses` (Focal/Weighted/Ranking/Combined/BPR/Listwise), `train`,
  `train_100k`, `advanced_training`, `sequential_model`,
  `distributed_training`, `colab_train_100k.ipynb`
- `api/` — `routes.py` (34 routes), `social_routes.py`
- `preprocessing/` — `merge_datasets.py`, `feature_engineering.py`
- `benchmarks/` — 12 scripts incl. `run_comprehensive_evaluation.py`, `run_32m_benchmarks.py`
- `templates/` — 15 pages; `static/` — `css/style.css`, `js/main.js`, `js/lazy_loading.js`
- `serving/` — TorchServe handler; `inference/` — production inference

## Config highlights (config.py)

- 55-dim features = 20 genre + 10 director + 20 actor + 5 metadata
- ELO start 1500, LR 0.1, exploration 0.2, latent 32, temporal decay 0.7, memory window 50
- Cache: 100 movies / 500 vectors; batch 20

[<- Back to README](../README.md) · Next: [Data & Methods](methods.md)
