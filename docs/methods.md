# CineSense — Data & Methods

> Datasets, preprocessing, and every model family in one place.
> Part of the [README](../README.md) doc set.

## 1. Datasets

| Dataset | Size | Key fields |
|---------|------|-----------|
| MovieLens-100K (`ml-100k/`) | 100K ratings, 943 users, 1,682 movies | userId, movieId, rating (1–5), timestamp + movies/links/tags |
| MovieLens-32M | ~32M ratings, ~200K users, ~87K movies | same schema, used for Deep V3 |
| TMDB | ~100K movies (`TMDB_movie_dataset_v11.csv`) | title, overview, genres, popularity, votes, dates, posters, cast/crew via API |

Merge key: `links.csv` (MovieLens movieId → TMDB tmdbId), inner-joined in
`preprocessing/merge_datasets.py`.

## 2. Preprocessing (`preprocessing/`)

1. **Merge** — ratings + links + TMDB metadata → unified table
2. **Encode** — raw user/movie IDs → contiguous ints for embedding layers
3. **Embed plots** — `all-MiniLM-L6-v2` → 384-dim vectors (`model/plot_embeddings.npy`)
4. **55-dim movie vectors** — 20 genre (multi-hot) + 10 director + 20 actor + 5 metadata (normalized rating, popularity, year, runtime, votes)
5. **Split** — leakage-safe per-user 70/10/20 (train/val/test, seed 42); train items masked from candidates at eval

Reproducible local training: `python training/train_100k.py --epochs 15`
(numpy BiasedMF, test RMSE 0.995 / NDCG@10 0.053, full catalog) — the
counterpart to the offline Kaggle torch checkpoints in `models/`.

## 3. Methods

### Pairwise ELO (`ai/pairwise_learning.py`, DB procedure `record_user_interaction`)

Every movie starts at 1500. A preferred win moves points from loser to winner:

`E_A = 1 / (1 + 10^((R_B − R_A)/400))`, `R_A' = R_A + K·(S_A − E_A)`, K = 32.

### Two-Tower NCF (`ai/two_tower_ncf.py`, `training/models.py`)

User tower and movie tower → 32-dim vectors → dot product → rating.
Deep V2/V3 add DCN-v2 cross layers + Factorization Machines + multi-head
attention + Highway MLP (`training/advanced_models_v2.py`; the Kaggle
checkpoints in `models/` are offline artifacts — see [Results](results.md)).

### Content-based (`ai/embeddings.py`, `ai/semantic_search.py`)

Cosine similarity over 55-dim vectors + 384/768-dim transformer plot
embeddings (`all-MiniLM-L6-v2` / `all-mpnet-base-v2`). Handles cold start.

### UCB bandit (`ai/reinforcement.py`)

`UCB_i = x̄_i + c·√(ln N / n_i)`, c = 2.0 — each genre is an arm, forcing
exploration of unseen genres.

### Serving ranker (`ai/multi_stage_ranker.py`, `ai/candidate_generator.py`)

Stage 1: multi-source retrieval (~300) → Stage 2: hybrid weighted scoring →
Stage 3: probability calibration → Stage 4: DPP diversity rerank → top 20.

### Extras

Semantic/hybrid search, mood→genre mapping (7 emotions), visual CLIP search,
DialoGPT chatbot, trending velocity/acceleration scorer, XFetch/Redis cache,
singleflight anti-stampede, A/B framework, SHAP explanations.

## 4. Losses (`training/losses.py`)

FocalMSE, WeightedMSE, Ranking, Combined, BPR, Listwise.

[<- Back to README](../README.md) · Next: [Database & API](database-api.md)
