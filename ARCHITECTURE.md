# CineSense — Architecture & Technical Documentation

Comprehensive technical reference for the CineSense AI recommendation platform. Covers system design, AI pipeline, data flow, database schema, API surface, deployment, and configuration.

---

## Table of Contents

- [System Overview](#system-overview)
- [AI Architecture](#ai-architecture)
  - [Layer 1: Preference Learning](#layer-1-preference-learning-elo--bradley-terry)
  - [Layer 2: Content Embeddings](#layer-2-content-embeddings)
  - [Layer 3: Exploration via Reinforcement Learning](#layer-3-exploration-via-reinforcement-learning)
  - [Layer 4: Advanced AI Features](#layer-4-advanced-ai-features)
  - [Layer 5: Deep Learning Ensemble](#layer-5-deep-learning-ensemble)
  - [Scoring Formula](#scoring-formula)
- [Deep Learning Models](#deep-learning-models)
  - [Two-Tower NCF](#two-tower-ncf)
  - [Hybrid Content-Aware Model](#hybrid-content-aware-model)
  - [NeuMF V2 Ensemble](#neumf-v2-ensemble)
  - [Training Pipeline](#training-pipeline)
- [Lazy Loading Architecture](#lazy-loading-architecture)
  - [Sliding Window Cache](#sliding-window-cache)
  - [Candidate Generation](#candidate-generation)
  - [Memory Comparison](#memory-comparison)
- [Data Pipeline](#data-pipeline)
  - [Data Sources](#data-sources)
  - [Preprocessing Flow](#preprocessing-flow)
  - [Content Ingestion Pipeline](#content-ingestion-pipeline)
- [Database Design](#database-design)
  - [Schema Overview](#schema-overview)
  - [Core Tables](#core-tables)
  - [Lazy Loading Tables](#lazy-loading-tables)
  - [Views & Stored Procedures](#views--stored-procedures)
  - [Migrations](#migrations)
- [REST API Reference](#rest-api-reference)
  - [User Endpoints](#user-endpoints)
  - [Recommendation Endpoints](#recommendation-endpoints)
  - [Movie Endpoints](#movie-endpoints)
  - [Cache Monitoring](#cache-monitoring)
- [Frontend](#frontend)
  - [Pages](#pages)
  - [Lazy Loading Manager](#lazy-loading-manager-js)
- [Configuration Reference](#configuration-reference)
- [Deployment](#deployment)
  - [Render](#render)
  - [Heroku](#heroku)
- [Project Statistics](#project-statistics)

---

## System Overview

CineSense is a full-stack movie recommendation system built on Flask, MySQL, and PyTorch. It replaces traditional star-rating input with pairwise comparisons ("which movie do you prefer?") and combines five AI layers into a unified scoring engine.

```
┌──────────────────────────────────────────────────────────────────────┐
│                         FRONTEND (Jinja2 + JS)                       │
│   index · compare · detail · search · profile · category · monitor   │
└────────────────────────────────┬─────────────────────────────────────┘
                                 │ HTTP
┌────────────────────────────────▼─────────────────────────────────────┐
│                        FLASK API  (api/routes.py)                    │
│   /api/recommendations · /api/compare · /api/feedback · /api/search  │
└────────────┬───────────────────┬────────────────────┬────────────────┘
             │                   │                    │
   ┌─────────▼──────┐  ┌────────▼────────┐  ┌────────▼────────┐
   │  AI ENGINE     │  │  DATABASE       │  │  TMDB API       │
   │  (ai/)         │  │  (MySQL 8.0+)   │  │  (tmdb/)        │
   │                │  │                 │  │                 │
   │ 5 AI Layers    │  │ 11 tables       │  │ Lazy streaming  │
   │ 13-model DL    │  │ 2 views         │  │ 1M+ movies      │
   │ ensemble       │  │ stored procs    │  │                 │
   └────────────────┘  └─────────────────┘  └─────────────────┘
```

### Request Flow (Recommendation)

```
User Request
    → Candidate Generator (300 candidates: 40% genre, 30% popularity, 20% explore, 10% cache)
    → Lazy Embedding (on-demand 55D feature vectors)
    → AI Scoring (50% content + 30% preference + 20% ELO + optional 15% DL)
    → Diversity Reranking (genre constraints)
    → Top 20 returned with explanations
```

---

## AI Architecture

The recommendation engine (`ai/recommender.py`, ~1100 lines) integrates five distinct AI layers into a single `CineSenseRecommender` class.

### Layer 1: Preference Learning (ELO + Bradley-Terry)

**Module:** `ai/pairwise_learning.py`

When a user picks one movie over another:
- Both movies' ELO scores are updated (k-factor = 32)
- The user's preference vector is adjusted toward the winner's features
- A Bradley-Terry model estimates pairwise probabilities via gradient descent

This eliminates the need for explicit 1-5 star ratings — every comparison teaches the system.

### Layer 2: Content Embeddings

**Module:** `ai/embeddings.py`

Every movie is represented as a 55-dimensional feature vector:

| Component | Dimensions | Method |
|---|---|---|
| Genres | 20 | Multi-hot encoding |
| Directors | 10 | Frequency-weighted encoding |
| Actors | 20 | Frequency-weighted encoding |
| Metadata | 5 | Normalized (rating, popularity, year, etc.) |

User preferences become a vector of the same shape. Recommendations use **cosine similarity** between user and movie vectors.

**Latent Space Compression:** PCA/SVD compresses vectors from 55D → 32D for dense learned representations.

### Layer 3: Exploration via Reinforcement Learning

**Module:** `ai/reinforcement.py`

Balances exploitation (similar to what the user likes) vs. exploration (new genres/directors):

| Algorithm | Description |
|---|---|
| ε-Greedy | Random exploration with probability ε |
| UCB (Upper Confidence Bound) | Explores under-sampled genres |
| Thompson Sampling | Bayesian probability-based selection |
| Contextual Bandit | Linear model using movie features as context |

Default: UCB + softmax probabilistic selection (temperature = 0.8).

### Layer 4: Advanced AI Features

**Module:** `ai/advanced_ai.py`

| Feature | Description |
|---|---|
| Latent Space Encoding | PCA/SVD for dimensionality reduction |
| Implicit Signal Processing | Learns from hover time, skips, repeat views, session abandonment |
| Probabilistic Selection | Softmax distribution avoids always picking the top-scored movie |
| Temporal Memory | Exponential decay: `0.7 × recent + 0.3 × past` |
| Natural Language Explanations | Generates human-readable reasoning for each recommendation |

### Layer 5: Deep Learning Ensemble

**Module:** `ai/neumf_scorer.py`

A 13-model NeuMF mega-ensemble provides genre-affinity scoring:
- **Phase 1:** 5× NeuMF_Genre models (user + movie + genre features)
- **Phase 2:** 8× NeuMF_V2 models (+ TMDB metadata, keyword features, user demographics)
- **Stacking:** Ridge regression + scipy-optimized weights → final ensemble prediction

Achieved **RMSE: 0.8932** on MovieLens-100K evaluation.

### Scoring Formula

```
final_score = 0.50 × content_similarity
            + 0.30 × user_preference_alignment
            + 0.20 × normalized_elo_score
            + 0.15 × dl_genre_affinity  (optional, when DL models loaded)
```

---

## Deep Learning Models

### Two-Tower NCF

**Module:** `ai/two_tower_ncf.py`

```
User ID → User Embedding (128D) → Dense(256→128→64) → User Vector (64D)
                                                            ↓
                                                      Dot Product → Score
                                                            ↑
Movie ID → Movie Embedding (128D) → Dense(256→128→64) → Movie Vector (64D)
```

Also includes GMF (Generalized Matrix Factorization) and MLP variants.

### Hybrid Content-Aware Model

**Module:** `ai/hybrid_model.py`

Extends Two-Tower NCF with content features:
- **MovieContentEncoder:** SentenceTransformer (`all-MiniLM-L6-v2`) encodes plot text → 768D
- **ContentEmbeddingLayer:** Projects 768D → embedding dimension
- **HybridMovieTower:** Fuses CF embedding + content embedding → joint representation
- Cold-start capable: can recommend movies with zero interaction history

### NeuMF V2 Ensemble

**Module:** `ai/neumf_scorer.py` + `colab_train_100k.ipynb`

Multi-phase training:
1. **Phase 1 (NeuMF_Genre):** 5 models × different seeds, user/movie embeddings + genre side features
2. **Phase 2 (NeuMF_V2):** 8 models with TMDB continuous features, keyword TF-IDF, user demographics
3. **Phase 3 (Stacking):** Ridge regression over out-of-fold predictions, scipy weight optimization

### Training Pipeline

**Module:** `ai/training_pipeline.py`

| Feature | Details |
|---|---|
| Optimizer | Adam / AdamW |
| LR Scheduling | ReduceLROnPlateau, Cosine Annealing |
| Early Stopping | Patience-based with best-model checkpointing |
| Gradient Clipping | Max norm = 1.0 |
| Mixed Precision | Optional AMP (automatic mixed precision) |
| Evaluation | RMSE, Hit@10, NDCG@10, Precision@K, Recall@K, MAP@K |

**Training commands:**
```bash
python train_model.py --sample --epochs 5           # Quick test (100K samples)
python train_model.py --model-type hybrid --epochs 20  # Full hybrid training
python run_pipeline.py                               # Complete pipeline
```

---

## Lazy Loading Architecture

**Problem:** Loading all movies into memory causes 54MB+ usage and limits content to the database.

**Solution:** Stream movies from TMDB on demand, cache in a sliding window, persist only on interaction.

### Sliding Window Cache

**Module:** `ai/cache_manager.py`

| Component | Capacity | Eviction |
|---|---|---|
| `SlidingWindowCache` (movies) | 100 items | LRU (OrderedDict) |
| `VectorCache` (feature vectors) | 500 items | LRU |

Auto-refills when cache drops below 30% capacity.

### Candidate Generation

**Module:** `ai/candidate_generator.py`

Default `mixed` strategy generates 300 candidates:

| Source | Percentage | Method |
|---|---|---|
| Genre-based | 40% | TMDB discover by user's top genres |
| Popularity-based | 30% | TMDB popular/top-rated movies |
| Exploration | 20% | Random genres, decades, languages |
| Cache-based | 10% | Already-cached movies |

For pairwise comparisons: 50% known-preference movies + 50% exploration.

### Memory Comparison

| Metric | Traditional | Lazy Loading |
|---|---|---|
| Memory usage | ~54 MB | ~700 KB |
| Reduction | — | **77×** |
| Content pool | ~10K (DB) | 1M+ (TMDB API) |
| Cold-cache response | N/A | ~200ms |
| Warm-cache response | N/A | ~50ms |

---

## Data Pipeline

### Data Sources

| Source | Records | Usage |
|---|---|---|
| MovieLens-32M | 32M ratings, 87K movies | Collaborative filtering training data |
| TMDB API | 1M+ movies | Metadata, posters, plot summaries, real-time streaming |
| TMDB Dataset (Kaggle) | 930K movies | Offline TMDB metadata for training |

### Preprocessing Flow

```
MovieLens ratings.csv + movies.csv + links.csv
    ↓ merge_datasets.py
    + TMDB_movie_dataset_v11.csv
    ↓
data/merged.csv (ratings with TMDB metadata)
    ↓ feature_engineering.py
data/encoded.csv (label-encoded user/movie IDs)
model/mappings.pkl (encoder mappings)
model/plot_embeddings.npy (768D BERT embeddings)
    ↓ training pipeline
model/ncf_recommender.pt
model/hybrid_recommender.pt
```

### Content Ingestion Pipeline

**Module:** `ai/content_pipeline.py`

Runs as a background thread in the Flask app:

| Schedule | Action |
|---|---|
| Daily | Incremental fetch of new popular/top-rated movies |
| Weekly | Full refresh of trending & genre-diverse content |
| Every 6 hours | Trending movies update |
| On-demand | Fetches when cache inventory drops below minimum |

---

## Database Design

**Engine:** MySQL 8.0+ (InnoDB, UTF-8MB4)  
**Normalization:** 3NF  
**Schema:** `database/schema.sql`

### Schema Overview

```
┌──────────┐    ┌──────────────┐    ┌────────────────────┐
│  USERS   │───→│ INTERACTIONS │←───│      MOVIES        │
└──────────┘    └──────────────┘    └────────────────────┘
                                           │    │    │
                                    ┌──────┘    │    └──────┐
                                    ▼           ▼           ▼
                              MOVIE_GENRES  MOVIE_DIRS  MOVIE_ACTORS
                                    │           │           │
                                    ▼           ▼           ▼
                                 GENRES    DIRECTORS     ACTORS

┌───────────────┐  ┌──────────────────┐  ┌─────────────┐
│ USER_EMBEDDINGS│  │ MOVIE_EMBEDDINGS │  │ CACHE_STATS │
└───────────────┘  └──────────────────┘  └─────────────┘
```

### Core Tables

| Table | Purpose | Key Columns |
|---|---|---|
| `users` | User accounts | `user_id`, `username`, `email`, `password_hash`, `interaction_count` |
| `movies` | Movie catalog | `movie_id`, `tmdb_id`, `title`, `overview`, `tmdb_rating`, `elo_score`, `poster_path` |
| `genres` | Genre taxonomy | `genre_id`, `name` (19 pre-seeded) |
| `directors` | Director records | `director_id`, `name`, `tmdb_id` |
| `actors` | Actor records | `actor_id`, `name`, `tmdb_id` |
| `movie_genres` | Movie ↔ Genre junction | `movie_id`, `genre_id` |
| `movie_directors` | Movie ↔ Director junction | `movie_id`, `director_id` |
| `movie_actors` | Movie ↔ Actor junction | `movie_id`, `actor_id`, `character_name`, `cast_order` |
| `user_interactions` | Pairwise comparison history | `user_id`, `movie_1_id`, `movie_2_id`, `chosen_movie_id`, `session_id` |
| `user_embeddings` | User preference vectors | `user_id`, `feature_index`, `feature_value` |
| `movie_embeddings` | Movie feature vectors | `movie_id`, `feature_index`, `feature_value` |

### Lazy Loading Tables

Added via `migrations/001_lazy_loading_migration.sql`:

| Table | Purpose |
|---|---|
| `cache_stats` | Tracks cache hit rates, refill counts, evictions over time |
| `candidate_generation_log` | Logs candidate generation strategy, counts, timing |

Additional columns on `movies`: `movie_source`, `is_persisted`, `last_accessed`, `access_count`  
Additional columns on `user_interactions`: `interaction_type`, `is_lazy_loaded`, `cache_hit`

### Views & Stored Procedures

| Name | Type | Purpose |
|---|---|---|
| `movie_details` | View | Complete movie info with metadata |
| `user_stats` | View | User activity aggregates |
| `cached_movies` | View | Top 100 by last_accessed |
| `lazy_loading_stats` | View | Aggregate cache statistics |
| `update_movie_elo` | Stored Proc | Atomic ELO score update |
| `cleanup_temporary_movies(days)` | Stored Proc | Remove non-persisted movies older than N days |
| `persist_movie(id)` | Stored Proc | Mark movie as permanently stored |

### Migrations

```bash
python database/run_migration.py
```

Tracks applied migrations in `schema_migrations` table. Prevents duplicate execution.

---

## REST API Reference

Base URL: `http://localhost:5000`

### User Endpoints

| Method | Path | Body / Params | Response |
|---|---|---|---|
| `POST` | `/api/user/signup` | `{username, email, password}` | `{user_id, message}` |
| `POST` | `/api/user/login` | `{username, password}` | `{user_id, username}` |
| `POST` | `/api/user/logout` | — | `{message}` |
| `GET` | `/api/user/profile` | — | `{user, stats, preferences}` |

### Recommendation Endpoints

| Method | Path | Params | Response |
|---|---|---|---|
| `GET` | `/api/recommendations` | — | `{movies[], explanations[]}` |
| `GET` | `/api/recommendations/lazy` | `?limit=20&strategy=mixed` | `{movies[], cacheStats}` |
| `GET` | `/api/featured` | — | `{movie}` (hero banner) |
| `GET` | `/api/compare` | — | `{movie1, movie2}` |
| `GET` | `/api/compare/lazy` | — | `{movie1, movie2, cacheStats}` |
| `POST` | `/api/feedback` | `{winner_id, loser_id}` | `{message, updated_scores}` |

### Movie Endpoints

| Method | Path | Params | Response |
|---|---|---|---|
| `GET` | `/api/movie/<id>` | — | `{movie, similar[]}` |
| `GET` | `/api/movie/search` | `?q=query` | `{results[]}` |
| `GET` | `/api/search/ai` | `?q=query` | `{results[]}` (semantic search) |
| `GET` | `/api/movie/by-genre/<genre>` | — | `{movies[]}` |
| `GET` | `/api/movie/top-rated` | — | `{movies[]}` |
| `GET` | `/api/stats` | — | `{total_movies, total_users, ...}` |

### Cache Monitoring

| Method | Path | Response |
|---|---|---|
| `GET` | `/api/cache/stats` | `{movie_count, vector_count, hit_rates}` |
| `GET` | `/api/cache/monitor` | Detailed real-time metrics |

---

## Frontend

### Pages

| Route | Template | Description |
|---|---|---|
| `/` | `index.html` | Home with recommendation carousels |
| `/compare` | `compare.html` | Side-by-side movie comparison |
| `/movie/<id>` | `detail.html` | Movie detail with similar titles |
| `/search` | `search.html` | AI-powered search |
| `/profile` | `profile.html` | User preferences & history |
| `/login` | `login.html` | Authentication |
| `/signup` | `signup.html` | Registration |
| `/category/<name>` | `category.html` | Genre/category browsing |
| `/monitor` | `cache_monitor.html` | Real-time cache dashboard |

### Lazy Loading Manager (JS)

`static/js/lazy_loading.js` provides the frontend cache integration:

```javascript
// Get comparison pair (auto-selects lazy vs standard)
const { movie1, movie2, cacheStats } = await LazyLoadingManager.getComparisonPair();

// Get recommendations with strategy
const { movies, strategy } = await LazyLoadingManager.getRecommendations(20, 'mixed');

// Toggle lazy loading
LazyLoadingManager.toggleLazyLoading(true);

// Change candidate strategy: 'mixed' | 'genre' | 'popularity' | 'exploration'
LazyLoadingManager.setStrategy('genre');
```

---

## Configuration Reference

All settings in `config.py`:

| Category | Setting | Default | Description |
|---|---|---|---|
| **Flask** | `SECRET_KEY` | env | Session encryption key |
| **Flask** | `PORT` | 5000 | Server port |
| **Database** | `DB_CONFIG` | env | MySQL connection dict |
| **TMDB** | `TMDB_API_KEY` | env | API key (free from themoviedb.org) |
| **AI** | `LEARNING_RATE` | 0.1 | ELO learning rate |
| **AI** | `EXPLORATION_RATE` | 0.2 | Bandit exploration probability |
| **AI** | `INITIAL_ELO_SCORE` | 1500 | Starting ELO for new movies |
| **AI** | `LATENT_DIM` | 32 | Compressed vector dimension |
| **AI** | `SOFTMAX_TEMPERATURE` | 0.8 | Selection temperature |
| **AI** | `TEMPORAL_DECAY_FACTOR` | 0.7 | Weight for recent interactions |
| **Cache** | `MOVIE_CACHE_SIZE` | 100 | Max movies in sliding window |
| **Cache** | `VECTOR_CACHE_SIZE` | 500 | Max cached feature vectors |
| **Cache** | `CACHE_REFILL_THRESHOLD` | 0.3 | Refill when ≤30% full |
| **Candidates** | `CANDIDATE_COUNT` | 300 | Candidates before ranking |
| **Candidates** | `CANDIDATE_STRATEGY` | `mixed` | Default generation strategy |
| **Pipeline** | `FINAL_RECOMMENDATION_COUNT` | 20 | Top-K returned to user |
| **DL** | `USE_DL_SCORING` | True | Enable NeuMF ensemble scoring |

---

## Deployment

### Render

Configuration in `render.yaml`:

```yaml
services:
  - type: web
    name: cinesense-web
    runtime: python
    plan: free
    buildCommand: pip install -r requirements.txt
    startCommand: gunicorn app:app --workers 2 --timeout 120
```

1. Push to GitHub
2. Create Web Service on [Render](https://render.com)
3. Connect repository, add environment variables
4. Deploy

### Heroku

```
Procfile: web: gunicorn app:app
runtime.txt: python-3.10.0
```

### External MySQL

Use a free MySQL host (e.g., Aiven, PlanetScale, Railway) and set `DB_HOST`, `DB_PORT`, `DB_USER`, `DB_PASSWORD`, `DB_NAME` in environment variables.

---

## Project Statistics

| Metric | Value |
|---|---|
| Python source files | ~32 |
| Lines of Python | ~8,500+ |
| Lines of SQL | ~625 |
| AI modules | 16 |
| DL models in ensemble | 13 |
| REST API endpoints | 16 |
| HTML templates | 11 |
| Database tables | 11 + 2 (lazy loading) |
| Feature vector dimensions | 55 (compressed to 32) |
| BERT embedding dimensions | 768 |
| Content pool | 1M+ movies (TMDB) |
| Memory footprint | ~700 KB (lazy loading) |

---

## Security

| Feature | Implementation |
|---|---|
| Password storage | bcrypt hash via `werkzeug.security` |
| SQL injection prevention | Parameterized queries throughout |
| Session management | Flask session with random secret key |
| CORS | Configurable via Flask-CORS |
| API key protection | Environment variables (`.env`) |
