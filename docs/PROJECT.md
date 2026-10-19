# CineSense — Project Reference

Single reference doc (merged from the former overview / architecture /
methods / database-api / setup docs on 2026-10-19). Everything factual that
lived in those files is here.

## 1. What it is (Overview)

**CineSense** is an AI-powered movie recommendation platform. Instead of 1–5
star ratings, it learns taste through **pairwise comparisons** ("Which movie
do you prefer?"), backed by a normalized MySQL database and hybrid ML models.

Core techniques:

- **Pairwise preference learning** — Bradley-Terry / ELO updates from comparisons
- **Neural collaborative filtering** — Deep V2/V3 models (DCN-v2 + FM, trained on Kaggle Dual T4 GPUs)
- **Content-based filtering** — Sentence-Transformer plot embeddings + 55-dim genre/director/actor/metadata vectors
- **Exploration** — UCB multi-armed bandit over genres
- **Two-stage serving** — candidate retrieval (300) → hybrid rerank → top 20

Problem → Solution:

| Problem | Solution in CineSense |
|---------|----------------------|
| Cold start (no history) | Content embeddings understand movies with zero interactions |
| Rating bias (my 3★ ≠ your 3★) | Pairwise choices need no absolute scale |
| Filter bubbles | UCB bandit + DPP diversity reranking force exploration |
| Scale (score everything per user) | Funnel: full catalog → 300 candidates → 20 results |

Objectives: full-stack Flask + MySQL + PyTorch app; normalized DB with views,
procedures, functions, triggers; hybrid AI (collaborative + content + pairwise
+ bandit); train/evaluate on MovieLens-100K and MovieLens-32M (+ TMDB
metadata); real-time learning from comparisons; responsive UI (search,
compare, recommendations, social); cloud deployment (Render + Gunicorn).

## 2. System Architecture

Verified against the repo (Oct 2026); counts corrected from stale earlier docs.

High-level diagram:

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

Tech stack (from requirements.txt / config.py):

| Layer | Technology |
|-------|-----------|
| Backend | Flask 3.0, Gunicorn 21.2, Flask-CORS, python-dotenv |
| Database | MySQL 8.0, mysql-connector-python 8.2 (pool), PyMySQL fallback |
| Deep learning | PyTorch ≥2.1, sentence-transformers ≥2.2.2, transformers ≥4.35 |
| ML / data | NumPy, scikit-learn, SciPy, pandas, tqdm, faiss-cpu |
| Serving extras | redis 5.0, Pillow, shap, torchserve/model-archiver |
| Frontend | HTML5 + CSS3 + vanilla JS + Jinja2 (no SPA framework) |
| Deploy | Render (render.yaml) + Procfile + runtime.txt |

Request flow (recommendations):

```
GET /api/recommendations
  1. Load user vector (user_embeddings)
  2. Retrieve ~300 candidates (CANDIDATE_COUNT=300, mixed strategy)
     ├── genre affinity slice
     ├── popularity / trending slice
     └── UCB exploration slice
  3. Score: content similarity + ELO/pairwise + (optional) DL score
     (DL scoring OFF by default: USE_DL_SCORING=0, DL_SCORE_WEIGHT=0.15;
      torch checkpoints in models/ are offline research artifacts)
  4. Rerank + diversify → FINAL_COUNT=20
  5. Log to recommendation_log, return JSON
```

Modules (actual files):

- `ai/` — 32 modules: `recommender` (orchestrator), `pairwise_learning` (ELO),
  `two_tower_ncf`, `hybrid_model`, `neumf_scorer`, `reinforcement` (UCB),
  `embeddings` (55-dim), `semantic_search`, `candidate_generator`,
  `multi_stage_ranker`, `diversity_engine`, `vector_search`, `reranker`/
  `cross-encoder` equivalents, `mood_detector`, `conversational_agent`,
  `visual_search`, `trending_detector`, `cache_manager`, `xfetch_cache`,
  `redis_cache`, `resilience`, `feature_store`, `telemetry`, `drift_monitor`,
  `ab_testing`, `ab_experimentation`, `evaluation`, `inference_pipeline`,
  `training_pipeline`, `netflix_recommender`, `content_pipeline`,
  `data_preprocessor`
- `training/` — 11 files: `models`, `advanced_models`, `advanced_models_v2`,
  `deep_v3`, `losses` (Focal/Weighted/Ranking/Combined/BPR/Listwise), `train`,
  `train_100k`, `advanced_training`, `sequential_model`,
  `distributed_training`, `colab_train_100k.ipynb`
- `api/` — `routes.py` (34 routes), `social_routes.py`
- `preprocessing/` — `merge_datasets.py`, `feature_engineering.py`
- `benchmarks/` — 12 scripts incl. `run_comprehensive_evaluation.py`,
  `run_32m_benchmarks.py`
- `templates/` — 15 pages; `static/` — `css/style.css`, `js/main.js`,
  `js/lazy_loading.js`
- `serving/` — TorchServe handler; `inference/` — production inference

Config highlights (config.py): 55-dim features = 20 genre + 10 director +
20 actor + 5 metadata; ELO start 1500 (INITIAL_ELO_SCORE), ELO ranking floor
1000 (ELO_BASE, normalization only), LR 0.1, exploration 0.2, latent 32,
temporal decay 0.7, memory window 50; caches 100 movies / 500 vectors;
batch 20.

## 3. Data & Methods

### 3.1 Datasets

| Dataset | Size | Key fields |
|---------|------|-----------|
| MovieLens-100K (`ml-100k/`) | 100K ratings, 943 users, 1,682 movies | userId, movieId, rating (1–5), timestamp + movies/links/tags |
| MovieLens-32M | ~32M ratings, ~200K users, ~87K movies | same schema, used for Deep V3 |
| TMDB | ~100K movies (`TMDB_movie_dataset_v11.csv`) | title, overview, genres, popularity, votes, dates, posters, cast/crew via API |

Merge key: `links.csv` (MovieLens movieId → TMDB tmdbId), inner-joined in
`preprocessing/merge_datasets.py`.

### 3.2 Preprocessing (`preprocessing/`)

1. **Merge** — ratings + links + TMDB metadata → unified table
2. **Encode** — raw user/movie IDs → contiguous ints for embedding layers
3. **Embed plots** — `all-MiniLM-L6-v2` → 384-dim vectors (`model/plot_embeddings.npy`)
4. **55-dim movie vectors** — 20 genre (multi-hot) + 10 director + 20 actor + 5 metadata (normalized rating, popularity, year, runtime, votes)
5. **Split** — leakage-safe per-user 70/10/20 (train/val/test, seed 42); train items masked from candidates at eval

Reproducible local training: `python training/train_100k.py --epochs 15`
(numpy BiasedMF, test RMSE 0.995 / NDCG@10 0.053, full catalog) — the
counterpart to the offline Kaggle torch checkpoints in `models/`.

### 3.3 Methods

**Pairwise ELO** (`ai/pairwise_learning.py`, DB procedure
`record_user_interaction`): every movie starts at 1500. A preferred win moves
points from loser to winner:
`E_A = 1 / (1 + 10^((R_B − R_A)/400))`, `R_A' = R_A + K·(S_A − E_A)`, K = 32.

**Two-Tower NCF** (`ai/two_tower_ncf.py`, `training/models.py`): user tower
and movie tower → 32-dim vectors → dot product → rating. Deep V2/V3 add
DCN-v2 cross layers + Factorization Machines + multi-head attention +
Highway MLP (`training/advanced_models_v2.py`; the Kaggle checkpoints in
`models/` are offline artifacts — see `docs/RESULTS.md`).

**Content-based** (`ai/embeddings.py`, `ai/semantic_search.py`): cosine
similarity over 55-dim vectors + 384/768-dim transformer plot embeddings
(`all-MiniLM-L6-v2` / `all-mpnet-base-v2`). Handles cold start.

**UCB bandit** (`ai/reinforcement.py`): `UCB_i = x̄_i + c·√(ln N / n_i)`,
c = 2.0 — each genre is an arm, forcing exploration of unseen genres.

**Serving ranker** (`ai/multi_stage_ranker.py`, `ai/candidate_generator.py`):
Stage 1: multi-source retrieval (~300) → Stage 2: hybrid weighted scoring →
Stage 3: probability calibration → Stage 4: DPP diversity rerank → top 20.

**Extras**: semantic/hybrid search, mood→genre mapping (7 emotions), visual
CLIP search, DialoGPT chatbot, trending velocity/acceleration scorer,
XFetch/Redis cache, singleflight anti-stampede, A/B framework, SHAP
explanations.

### 3.4 Losses (`training/losses.py`)

FocalMSE, WeightedMSE, Ranking, Combined, BPR, Listwise.

## 4. Database & API

Counts verified directly from `database/schema.sql` (Oct 2026). Earlier docs
said "16 tables / 12 views / 8 procedures / 4 functions / 5 triggers" — stale.

| Object | In `schema.sql` | Live (MySQL 8.0.44, verified Oct 2026) |
|--------|---------------|------------------------------------------|
| Tables | **36** | 42 (schema.sql now includes the 7 social tables + `schema_migrations`; live DB also has a few app-created extras) |
| Views | **24** | 24 ✓ |
| Stored procedures | **24** | 25 (24 + `get_semantic_search_results` from social setup) — all create cleanly after the LOCK TABLES fix |
| Functions | **8** | 8 ✓ |
| Triggers | **11** | 11 ✓ |
| Standalone CREATE INDEX | 10 (+ 54 inline KEY/INDEX + 3 FULLTEXT) | — |

LOCK TABLES fix (Oct 2026): three procedures used `LOCK TABLES`, which is
illegal inside MySQL stored programs — they silently failed to create (live
DB had only 22 procedures). Rewrote with row-level `FOR SHARE` /
transactions; verified all three create and run on MySQL 8.0.44
(`scripts/apply_lock_fix.py`). MariaDB note: 22/24 procedures load there.

Authoritative file: `database/schema.sql`. Access via `database/db_manager.py`
(singleton, 10-connection pool, parameterized calls). Migrations:
`database/run_migration.py`.

### Tables (36)

Core: `users`, `movies`, `genres`, `directors`, `actors`, `movie_genres`,
`movie_directors`, `movie_actors`, `user_interactions`, `user_embeddings`,
`movie_embeddings`, `user_preferences`, `movie_keywords`, `search_history`,
`cache_stats`, `candidate_generation_log`, `recommendation_log`, `watchlist`,
`movie_reviews`, `genre_win_counts`.
Extended (`IF NOT EXISTS`): `keywords`, `semantic_embeddings`, `search_logs`,
`user_moods`, `recommendation_feedback`, `ab_test_assignments`,
`ab_test_metrics`, `model_versions`.
Social (added to schema.sql Oct 2026): `friend_requests`, `friendships`,
`watch_parties`, `watch_party_invites`, `collaborative_lists`,
`collaborative_list_items`, `collaborative_list_members`.
Migration bookkeeping: `schema_migrations`.

### Views (24)

`movie_details`, `user_stats`, `comprehensive_movie_view`,
`advanced_user_stats`, `genre_popularity_view`, `director_performance_view`,
`actor_performance_view`, `daily_activity_stats`, `trending_movies_view`,
`elite_movies`, `intersect_movies_emulation`, `except_movies_emulation`,
`user_network_view`, `power_users`, `movie_search_index`, `search_analytics`,
`user_mood_trends`, `recommendation_performance`, `user_review_summary`,
`watchlist_details`, `genre_comparison_stats`, `recommendation_effectiveness`,
`cached_movies`, `lazy_loading_stats`.

### Procedures (24) — key ones

`record_user_interaction` (compare + ELO), `get_personalized_recommendations`,
`search_movies_advanced`, `update_user_preferences`, `get_comparison_pair`,
`update_movie_elo`, `transactional_elo_update`, `get_mood_recommendations`,
`assign_ab_test_variant`, `log_recommendation_feedback`,
`update_cache_stats`, `cleanup_temporary_movies`, plus maintenance/locking
helpers.

### Functions (8)

`calculate_user_similarity`, `get_movie_diversity_score`,
`has_user_interacted_with_movie`, `calculate_cache_hit_rate`,
`get_popularity_tier`, `jaccard_genre_similarity`, `get_user_taste_profile`,
`weighted_recommendation_score`.

### Triggers (11)

Validation on movie insert/update, user-init on signup, interaction
bookkeeping (`after_interaction_insert`, genre-win counts), search logging,
review/watchlist sync, access tracking.

### API

Base: blueprint `api/routes.py` under `/api` (34 routes) +
`api/social_routes.py` (14) + feature routes in `app_integrated.py` (7) =
**55 under `/api`** (69 URL rules total incl. pages). Don't claim "60+ REST
routes."

| Method | Endpoint | Purpose |
|--------|----------|---------|
| POST | `/api/user/signup`, `/api/user/login`, `/api/user/logout` | auth |
| GET | `/api/user/profile`, `/api/user/preferences`, `/api/user/interactions`, `/api/user/compared-movies` | user data |
| GET | `/api/recommendations`, `/api/recommendations/lazy` | personalized recs |
| GET | `/api/compare`, `/api/compare/lazy` | comparison pair |
| POST | `/api/feedback` | submit comparison |
| GET | `/api/movie/<id>`, `/api/movie/<id>/similar`, `/api/movie/by-genre`, `/api/movie/search`, `/api/movie/top-rated` | movies |
| POST | `/api/search/ai` | hybrid AI search |
| GET/POST/PUT/DELETE | `/api/watchlist…` | watchlist CRUD |
| GET/POST | `/api/reviews…` | reviews |
| GET | `/api/analytics/genre-stats`, `/api/analytics/genre-audit`, `/api/analytics/user-profile`, `/api/analytics/movie-score/<id>` | analytics |
| GET | `/api/stats`, `/api/cache/stats`, `/api/cache/monitor`, `/api/cache-stats` | stats/cache |
| POST | `/api/chat`, `/api/mood-recommendations`, `/api/visual-search`, `/api/text-to-image-search` | AI features |
| GET | `/api/trending`, `/api/viral` | trending |
| GET | `/api/test/tmdb` | TMDB health check |

Pages: `/`, `/compare`, `/movie/<id>`, `/search`, `/login`, `/signup`,
`/profile`, `/category/<name>`, `/monitor`, `/chat-ui`, `/features`,
`/friends`.

## 5. Setup & Deployment

Verified entry point: `app_integrated.py` (there is no `app.py`).

### Prerequisites

- Python 3.10+ (`runtime.txt` pins `python-3.10.0`; note `render.yaml` says
  `PYTHON_VERSION=3.11.0` — align these before deploying)
- MySQL 8.0+, TMDB API key (free at themoviedb.org/settings/api)
- Optional: Redis (caching), CUDA GPU (5–10× inference)

### Install

```bash
git clone https://github.com/your-repo/CineSense.git
cd CineSense
pip install -r requirements.txt
cp .env.example .env   # then set DB_* + TMDB_API_KEY
```

### Database

```bash
mysql -u root -p -e "CREATE DATABASE cinesense CHARACTER SET utf8mb4;"
mysql -u root -p cinesense < database/schema.sql
python database/run_migration.py
```

`database/schema.sql` is the single authoritative schema (36 tables,
24 views, 24 procedures, 8 functions, 11 triggers).

### Run

```bash
python app_integrated.py            # → http://localhost:5000
gunicorn "app_integrated:create_app()" --workers 2 --timeout 120   # prod
```

Key pages: `/` `/compare` `/search` `/chat-ui` `/features` `/friends`
`/monitor`.

### Train / benchmark

```bash
python benchmarks/run_all_benchmarks.py   # canonical suite → benchmarks/results/
python training/train_100k.py --epochs 15  # reproducible 100K trainer → model/mf_100k.npz
python benchmarks/run_comprehensive_evaluation.py  # regression-heavy eval
pytest -q                                 # unit/integration tests
```

Kaggle Dual-T4 training: push `kaggle_runner/`, check kernel status, pull
weights into `models/` (see `docs/RESULTS.md`).

### Deploy (Render)

`Procfile`: `web: gunicorn "app_integrated:create_app()"`. `render.yaml`:
build `pip install -r requirements.txt`, start with 2 workers, 120s timeout.
Set `SECRET_KEY` (generate), `TMDB_API_KEY`, `DB_*` in dashboard.

### Troubleshooting

| Symptom | Fix |
|---------|-----|
| DB connection fails | MySQL running? `.env` creds? database created? |
| Redis errors | Optional — app runs without it, just slower; start `redis-server` |
| TMDB 401 | invalid key — refresh at themoviedb.org/settings/api |
| Slow first start | ~2GB HuggingFace models download once, then cached |
| AI endpoints 503 | models lazy-load; wait 30–60s after cold start |
| OOM | default is `faiss-cpu`; lower batch size |

## 6. Model artifacts

See `models/README.md` for every checkpoint: what produced it, what consumes
it, and how to (re)produce it. Key contract: `model/` artifacts are
gitignored (reproducible locally via `training/train_100k.py`); the Kaggle
Deep V3 checkpoints under `models/` are the only models shipped with the
repository. All loaders fail loudly with actionable messages when an artifact
is missing.
