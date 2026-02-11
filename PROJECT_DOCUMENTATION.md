# CineSense — Complete Project Documentation

> **DBMS Project — AI-Based Movie Recommendation System**
> Built with Flask, MySQL 8.0, PyTorch, and Sentence Transformers
> Datasets: MovieLens 100K + TMDB 100K

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Problem Statement](#2-problem-statement)
3. [Objectives](#3-objectives)
4. [System Architecture](#4-system-architecture)
5. [Tech Stack](#5-tech-stack)
6. [Data Sources & Datasets](#6-data-sources--datasets)
7. [Data Preprocessing Pipeline](#7-data-preprocessing-pipeline)
8. [AI/ML Components — Detailed Breakdown](#8-aiml-components--detailed-breakdown)
9. [Backend — Flask Application](#9-backend--flask-application)
10. [Frontend — User Interface](#10-frontend--user-interface)
11. [API Documentation](#11-api-documentation)
12. [Database Design Overview](#12-database-design-overview)
13. [Deployment & DevOps](#13-deployment--devops)
14. [Module-Wise File Documentation](#14-module-wise-file-documentation)
15. [How It All Works Together](#15-how-it-all-works-together)
16. [Future Enhancements](#16-future-enhancements)

---

## 1. Introduction

**CineSense** is an AI-powered movie recommendation platform that learns user preferences through pairwise movie comparisons. Unlike traditional rating-based systems (1–5 stars), CineSense uses a **"Which movie do you prefer?"** approach — the same method used by Netflix for its internal ranking systems.

The platform combines multiple AI techniques:
- **Neural Collaborative Filtering (NeuMF)** for learning user-movie interaction patterns
- **Content-Based Filtering** using Transformer embeddings for understanding movie content (plots, genres)
- **ELO Rating System** for ranking movies based on pairwise comparison outcomes
- **Multi-Armed Bandit (UCB)** for smart exploration of new content
- **Hybrid Deep Learning** that fuses collaborative and content signals

The entire system is backed by a **fully normalized MySQL 8.0 database** (Third Normal Form) with stored procedures, views, indexes, and a connection-pooled Python backend.

---

## 2. Problem Statement

Traditional movie recommendation systems suffer from:

1. **Cold-start problem** — new users or movies have no interaction history
2. **Rating bias** — users rate inconsistently (one person's 3-star is another's 4-star)
3. **Limited expressiveness** — a 1–5 scale cannot capture nuanced preferences
4. **Filter bubbles** — systems only recommend what users already like
5. **Scalability** — scoring all movies for every user is computationally expensive

**CineSense addresses these by:**

| Problem | Solution |
|---------|----------|
| Cold start | Content-based Transformer embeddings understand movies without user data |
| Rating bias | Pairwise comparisons remove absolute scale — only relative preference matters |
| Limited expressiveness | ELO system captures fine-grained preference orderings |
| Filter bubbles | UCB Bandit ensures exploration of unseen genres/movies |
| Scalability | Candidate generation funnel: 100K → 500 candidates → 20 recommendations |

---

## 3. Objectives

1. Build a full-stack movie recommendation web application
2. Design and implement a normalized (3NF) MySQL database for movie and user data
3. Integrate multiple AI/ML models for hybrid recommendations
4. Use 100K datasets from both TMDB and MovieLens for training and serving
5. Implement real-time preference learning through pairwise comparisons
6. Provide a responsive web UI with search, comparison, and recommendation features
7. Deploy the application to a cloud platform (Render)

---

## 4. System Architecture

### High-Level Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         USER (Browser)                           │
│         index.html | compare.html | search.html | etc.          │
└────────────────────────────┬─────────────────────────────────────┘
                             │  HTTP Requests (AJAX/Fetch)
                             ▼
┌──────────────────────────────────────────────────────────────────┐
│                  FLASK APPLICATION (app.py)                       │
│  ┌─────────────────────┐  ┌──────────────────────────────────┐  │
│  │  Web Routes         │  │  API Routes (api/routes.py)      │  │
│  │  / (index)          │  │  POST /api/user/signup           │  │
│  │  /movie/<id>        │  │  POST /api/user/login            │  │
│  │  /compare           │  │  GET  /api/movies/top            │  │
│  │  /search            │  │  GET  /api/recommend             │  │
│  │  /profile           │  │  POST /api/compare               │  │
│  └─────────────────────┘  └──────────────┬───────────────────┘  │
└──────────────────────────────────────────┼───────────────────────┘
                                           │
                    ┌──────────────────────┼──────────────────────┐
                    ▼                      ▼                      ▼
          ┌─────────────────┐   ┌──────────────────┐   ┌─────────────────┐
          │  MySQL Database │   │    AI Engine      │   │   TMDB API      │
          │  (db_manager)   │   │  (recommender.py) │   │  (fetcher.py)   │
          │                 │   │                   │   │                 │
          │ • users         │   │ • Pairwise ELO    │   │ • Popular movies│
          │ • movies        │   │ • NeuMF Ensemble  │   │ • Movie details │
          │ • genres        │   │ • Content Embed.  │   │ • Cast/crew     │
          │ • actors        │   │ • UCB Bandit      │   │ • Posters       │
          │ • directors     │   │ • Candidate Gen   │   │ • Search        │
          │ • interactions  │   │ • Cache Manager   │   │                 │
          │ • embeddings    │   │ • Inference Pipe.  │   │                 │
          └─────────────────┘   └──────────────────┘   └─────────────────┘
```

### Data Flow for a Recommendation Request

```
User clicks "Get Recommendations"
         │
         ▼
1. Flask receives GET /api/recommend
         │
         ▼
2. Load user's preference vector from user_embeddings table
         │
         ▼
3. Candidate Generation (300-500 movies)
   ├── Genre-based: movies from user's top genres
   ├── Popularity-based: trending/highly-rated movies
   └── Exploration: random movies from unseen genres (UCB Bandit)
         │
         ▼
4. Scoring each candidate
   ├── ELO score (from pairwise comparisons)
   ├── Content similarity (cosine similarity of embeddings)
   ├── NeuMF ensemble score (13-model neural prediction)
   └── Temporal/implicit signals (recency, hover time)
         │
         ▼
5. Hybrid ranking: weighted combination of all scores
         │
         ▼
6. Diversification: ensure genre/era variety
         │
         ▼
7. Return top 20 recommendations as JSON
```

---

## 5. Tech Stack

### Backend

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| Web Framework | Flask | 3.0.0 | HTTP server, routing, templating |
| WSGI Server | Gunicorn | 21.2.0 | Production HTTP server |
| CORS | Flask-CORS | 4.0.0 | Cross-origin support |
| Environment | python-dotenv | 1.0.0 | `.env` file loading |
| HTTP Client | Requests | 2.31.0 | TMDB API calls |
| Scheduling | Schedule | 1.2.0 | Periodic content pipeline tasks |

### Database

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| RDBMS | MySQL | 8.0+ | Relational data storage |
| Python Driver | mysql-connector-python | 8.2.0 | Connection pooling, parameterized queries |
| Alt Driver | PyMySQL | 1.1.0 | Fallback connectivity |

### AI / Machine Learning

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| Deep Learning | PyTorch | 2.1+ | Model training & inference |
| NLP Embeddings | sentence-transformers | 2.2.2+ | Movie content encoding (384-dim) |
| Transformers | HuggingFace transformers | 4.35+ | Pre-trained language models |
| Numerical | NumPy | 1.26+ | Vector math, linear algebra |
| ML Toolkit | Scikit-learn | 1.3+ | PCA, evaluation metrics |
| Scientific | SciPy | 1.11+ | Sparse matrices, optimization |
| DataFrames | Pandas | 2.1+ | Dataset manipulation |
| Progress Bars | tqdm | 4.66+ | Training progress display |

### Frontend

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Markup | HTML5 | Page structure |
| Styling | CSS3 | Visual design |
| Interactivity | Vanilla JavaScript | AJAX calls, DOM manipulation |
| Templating | Jinja2 | Server-side rendering |

---

## 6. Data Sources & Datasets

### 6.1 MovieLens 100K Dataset

**Source:** [GroupLens Research, University of Minnesota](https://grouplens.org/datasets/movielens/)

The MovieLens-100K dataset is a subset extracted from the MovieLens-32M collection.

| File | Records | Columns | Description |
|------|---------|---------|-------------|
| `ratings.csv` | 100,000 | userId, movieId, rating, timestamp | User movie ratings (0.5–5.0 scale) |
| `movies.csv` | ~62,000 | movieId, title, genres | Movie titles with pipe-separated genres |
| `links.csv` | ~62,000 | movieId, imdbId, tmdbId | Cross-platform ID mappings |
| `tags.csv` | ~100,000 | userId, movieId, tag, timestamp | Free-text tags by users |

**Key characteristics:**
- Ratings are on a 0.5–5.0 scale (half-star increments)
- Each user has rated at least 20 movies
- Genres are pipe-separated (e.g., `Action|Adventure|Sci-Fi`)

### 6.2 TMDB 100K Dataset

**Source:** [The Movie Database (TMDB) API v3](https://www.themoviedb.org/)

| File | Records | Key Fields |
|------|---------|------------|
| `TMDB_movie_dataset_v11.csv` | ~100,000 | id, title, overview, genres, popularity, vote_average, vote_count, release_date, poster_path, backdrop_path, revenue, budget, production_companies |

**Key characteristics:**
- Rich metadata: plot summaries (overview), posters, backdrops
- Popularity scores based on TMDB user engagement
- Cast and crew information available via API
- 19 standardized genres

### 6.3 Data Merging Strategy

The two datasets are merged using `links.csv` which maps MovieLens `movieId` to TMDB `tmdbId`:

```
MovieLens ratings.csv ──┐
                        ├── links.csv (movieId → tmdbId) ──→ merged.csv
TMDB_movie_dataset.csv ─┘
```

The merge is performed in `preprocessing/merge_datasets.py` using an inner join on TMDB ID, producing a unified dataset with MovieLens ratings enriched with TMDB metadata (overview, poster, popularity, genres, etc.).

---

## 7. Data Preprocessing Pipeline

### Step 1: Dataset Merging (`preprocessing/merge_datasets.py`)

```python
# Simplified merge logic
movielens_ratings = pd.read_csv('data/movie-lens_ml-32m/ratings.csv')
tmdb_movies = pd.read_csv('data/tmdb/TMDB_movie_dataset_v11.csv')
links = pd.read_csv('data/movie-lens_ml-32m/links.csv')

# Map MovieLens movieId → TMDB tmdbId
merged = movielens_ratings.merge(links, on='movieId')
merged = merged.merge(tmdb_movies, left_on='tmdbId', right_on='id')
merged.to_csv('data/merged.csv', index=False)
```

### Step 2: Feature Engineering (`preprocessing/feature_engineering.py`)

1. **User/Movie ID Encoding**: Map raw IDs to contiguous integers (0, 1, 2, ...) for neural network input
2. **Plot Embeddings**: Generate 384-dimensional vectors from movie overviews using Sentence Transformers (all-MiniLM-L6-v2)
3. **Output**: `data/encoded.csv` and `model/plot_embeddings.npy`

### Step 3: Model Training (`training/train.py`)

1. Load preprocessed data
2. Create train/test splits (80/20)
3. Train NCF model (user embedding × movie embedding → rating prediction)
4. Train Hybrid model (NCF + content embeddings)
5. Evaluate with RMSE, Hit Rate@10, NDCG@10
6. Save checkpoints to `model/` directory

---

## 8. AI/ML Components — Detailed Breakdown

### 8.1 Pairwise Learning with ELO Rating (`ai/pairwise_learning.py`)

The ELO system, originally designed for chess, is adapted for movie ranking:

- Every movie starts with an ELO score of **1500**
- When a user prefers Movie A over Movie B:
  - Movie A's ELO increases
  - Movie B's ELO decreases
  - The magnitude depends on the expected outcome (an upset causes a larger change)

**Formula:**

$$E_A = \frac{1}{1 + 10^{(R_B - R_A) / 400}}$$

$$R_A' = R_A + K \cdot (S_A - E_A)$$

Where:
- $E_A$ = expected score for movie A
- $R_A, R_B$ = current ELO ratings
- $K$ = K-factor (32 in CineSense)
- $S_A$ = actual outcome (1 for win, 0 for loss)

### 8.2 Neural Collaborative Filtering (`training/models.py`, `ai/two_tower_ncf.py`)

The NCF model learns latent representations of users and movies:

```
User ID ──→ [Embedding Layer] ──→ User Vector (32-dim)
                                          │
                                          ├──→ [Element-wise Product] ──→ GMF
                                          │
Movie ID ──→ [Embedding Layer] ──→ Movie Vector (32-dim)
                                          │
                                          ├──→ [Concatenation] ──→ MLP ──→ MLP Output
                                          │
                                   GMF + MLP Output ──→ [Fusion Layer] ──→ Rating
```

**Two-Tower Variant** (`ai/two_tower_ncf.py`):
- **User Tower**: Deep NN (user features → 64 → 32 dims)
- **Movie Tower**: Deep NN (movie features → 64 → 32 dims)
- **Scoring**: Dot product of tower outputs

### 8.3 Content-Based Filtering (`ai/embeddings.py`, `ai/hybrid_model.py`)

Movies are encoded as **55-dimensional feature vectors**:

| Component | Dimensions | Encoding Method |
|-----------|-----------|-----------------|
| Genres | 20 | Multi-hot encoding |
| Directors | 10 | Frequency-based encoding |
| Actors | 20 | Frequency-based encoding |
| Metadata | 5 | Normalized (rating, popularity, year, runtime, votes) |

**Hybrid Model** adds **Transformer-based content understanding**:
- Uses `all-MiniLM-L6-v2` (a Sentence-BERT model) to encode movie overviews into **384-dimensional** dense vectors
- These content embeddings capture semantic meaning of movie plots
- Enables "If you liked Inception's mind-bending plot, try Interstellar" type recommendations

### 8.4 NeuMF Ensemble Scorer (`ai/neumf_scorer.py`)

A **13-model mega-ensemble** trained on MovieLens-100K:
- Multiple NeuMF variants with different hyperparameters
- Combined predictions via weighted averaging
- Provides **genre affinity scores** transferable to new movies
- Achieves RMSE = 0.8932 on MovieLens-100K test set

### 8.5 Reinforcement Learning — UCB Bandit (`ai/reinforcement.py`)

The **Upper Confidence Bound (UCB)** algorithm treats each genre/movie category as a "bandit arm":

$$UCB_i = \bar{x}_i + c\sqrt{\frac{\ln(N)}{n_i}}$$

Where:
- $\bar{x}_i$ = average reward for arm $i$ (user satisfaction with genre)
- $N$ = total number of recommendations made
- $n_i$ = number of times genre $i$ was recommended
- $c$ = exploration parameter (2.0 in CineSense)

This ensures unpopular or unseen genres still get recommended occasionally, breaking filter bubbles.

### 8.6 Advanced AI Features (`ai/advanced_ai.py`)

| Feature | Description |
|---------|-------------|
| **Latent Space Compression** | PCA/SVD reduces feature vectors from 55→32 dimensions for efficiency |
| **Implicit Signals** | Hover time (+0.15), skip penalty (−0.2), repeat view (+0.3), session abandon (−0.1) |
| **Softmax Selection** | Temperature-controlled probabilistic selection (T=0.8) instead of deterministic argmax |
| **Temporal Memory** | Recent interactions weighted 70%, older ones 30%. Last 50 interactions at full weight |
| **NL Explanations** | Human-readable explanations: "Recommended because you enjoy sci-fi thrillers by Christopher Nolan" |

### 8.7 Candidate Generation Pipeline (`ai/candidate_generator.py`)

Netflix-style funnel approach:

```
Total Movie Pool (100K)
         │
         ▼ Candidate Generation (300-500 movies)
    ┌────┼────┐
    ▼    ▼    ▼
  Genre  Pop  Exploration
  (50%)  (30%)  (20%)
    └────┼────┘
         ▼ Scoring (NeuMF + ELO + Content)
         │
         ▼ Ranking & Diversification
         │
         ▼ Top 20 Recommendations
```

---

## 9. Backend — Flask Application

### 9.1 Application Factory (`app.py`)

The Flask app uses the **factory pattern**:
1. Creates Flask instance
2. Configures secret key, session type, JSON provider
3. Enables CORS for cross-origin requests
4. Registers API blueprint (`api/routes.py`)
5. Starts content pipeline in a daemon thread
6. Defines web routes (home, detail, compare, search, profile, login, signup, category, monitor)
7. Registers error handlers (404, 500)
8. Injects user context into all templates

### 9.2 Configuration (`config.py`)

Centralized configuration using environment variables with defaults:

| Category | Key Settings |
|----------|-------------|
| **Flask** | SECRET_KEY, PORT (5000), DEBUG mode |
| **Database** | host, port, user, password, database name, charset (utf8mb4) |
| **TMDB** | API key, base URL, image base URL |
| **AI** | Learning rate (0.1), exploration rate (0.2), initial ELO (1500) |
| **Features** | Genre dim (20), Director dim (10), Actor dim (20), Metadata dim (5) |
| **Advanced** | Latent dim (32), temporal decay (0.7), softmax temperature (0.8) |
| **Cache** | Movie cache (100), vector cache (500), refill threshold (30%) |
| **Candidates** | Count (300), strategy (mixed), known ratio (0.5) |

### 9.3 API Routes (`api/routes.py`)

933 lines of RESTful endpoints organized into:

- **User Endpoints**: signup, login, logout, profile, preferences analysis
- **Movie Endpoints**: top movies, search, detail, genre filtering, recommendations
- **Comparison Endpoints**: get pair for comparison, submit comparison result, update ELO
- **Cache Endpoints**: cache stats, monitoring metrics
- **Content Pipeline**: trigger content updates

### 9.4 Database Manager (`database/db_manager.py`)

**Singleton pattern** with MySQL connection pooling:
- Pool size: 10 connections
- Context managers for safe connection/cursor handling
- Automatic commit/rollback on success/failure
- Operations: User CRUD, Movie CRUD, Genre/Director/Actor linking, Interaction recording, Embedding storage, Statistics queries

---

## 10. Frontend — User Interface

### 10.1 Templates (Jinja2)

| Template | Purpose |
|----------|---------|
| `base.html` | Base layout with navigation bar, footer, session-aware user display |
| `index.html` | Home page with recommendation carousels (trending, top-rated, for you) |
| `detail.html` | Movie detail page with poster, overview, cast, director, similar movies |
| `compare.html` | Side-by-side movie comparison — user clicks their preference |
| `search.html` | Search with instant results (multi-tier relevance scoring) |
| `login.html` | Login form with validation |
| `signup.html` | Registration form |
| `profile.html` | User statistics, genre preferences chart, interaction history |
| `category.html` | Browse movies by genre (action, thriller, comedy, etc.) |
| `cache_monitor.html` | Real-time cache performance dashboard |
| `404.html` | Custom error page |

### 10.2 Static Assets

- **`style.css`**: Full application styling — responsive grid layouts, movie cards, navigation, comparison UI
- **`main.js`**: Core logic — API calls via fetch(), DOM manipulation, event handling, movie card rendering
- **`lazy_loading.js`**: Infinite scroll / lazy loading — loads more movies as user scrolls down

---

## 11. API Documentation

### User Endpoints

| Method | Endpoint | Request Body | Response |
|--------|----------|-------------|----------|
| POST | `/api/user/signup` | `{username, email, password}` | `{user_id, username}` (201) |
| POST | `/api/user/login` | `{username, password}` | `{user_id, username}` (200) |
| POST | `/api/user/logout` | — | `{message}` (200) |
| GET | `/api/user/profile` | — | `{user_id, username, interaction_count, ...}` |
| GET | `/api/user/preferences` | — | `{genre_preferences, favorite_directors, ...}` |

### Movie Endpoints

| Method | Endpoint | Query Params | Response |
|--------|----------|-------------|----------|
| GET | `/api/movies/top` | `limit, offset, order_by, media_type` | `[{movie_id, title, genres, ...}]` |
| GET | `/api/movies/search` | `q, limit` | `[{movie with relevance_score}]` |
| GET | `/api/movies/<id>` | — | `{full movie details}` |
| GET | `/api/movies/genre/<name>` | `limit, offset, media_type` | `[{movies in genre}]` |

### Recommendation Endpoints

| Method | Endpoint | Description | Response |
|--------|----------|-------------|----------|
| GET | `/api/recommend` | Get personalized AI recommendations | `[{movie, score, explanation}]` |
| GET | `/api/compare/pair` | Get two movies for comparison | `{movie_1, movie_2}` |
| POST | `/api/compare` | Submit comparison: `{movie_1_id, movie_2_id, chosen_id}` | `{updated elo scores}` |

---

## 12. Database Design Overview

The database is fully documented in [DATABASE_DOCUMENTATION.md](DATABASE_DOCUMENTATION.md).

**Summary:**
- **11 tables** in Third Normal Form (3NF)
- **2 views**: `movie_details`, `user_stats`
- **2 stored procedures**: `update_user_interaction_count`, `update_movie_elo`
- **Engine**: InnoDB (for foreign key and transaction support)
- **Charset**: utf8mb4 (full Unicode including emojis)
- **Key tables**: `users`, `movies`, `genres`, `directors`, `actors`, `movie_genres`, `movie_directors`, `movie_actors`, `user_interactions`, `user_embeddings`, `movie_embeddings`

---

## 13. Deployment & DevOps

### Production Stack

| Component | Configuration |
|-----------|--------------|
| **WSGI Server** | Gunicorn with multiple workers |
| **Platform** | Render (cloud hosting) |
| **Procfile** | `web: gunicorn app:create_app()` |
| **Runtime** | Python 3.11 (specified in `runtime.txt`) |

### Deployment Files

| File | Purpose |
|------|---------|
| `Procfile` | Gunicorn startup command |
| `render.yaml` | Render platform service definition |
| `runtime.txt` | Python version specification |
| `requirements.txt` | Dependency list for `pip install` |

---

## 14. Module-Wise File Documentation

### Root Files

| File | Lines | Description |
|------|-------|-------------|
| `app.py` | 189 | Flask app factory — creates app, registers blueprints, defines routes, starts content pipeline |
| `config.py` | 105 | All configuration — DB, TMDB, AI params, cache sizes, feature dims |
| `requirements.txt` | 25 | Pip dependencies |
| `run_pipeline.py` | — | Orchestrates: merge → features → train |
| `train_model.py` | — | CLI trainer: `python train_model.py --epochs 50 --model-type hybrid` |
| `quick_start.py` | — | Validates data files and dependencies before training |

### `ai/` Module (16 files)

| File | Lines | Role in the System |
|------|-------|--------------------|
| `recommender.py` | 1098 | **Master orchestrator** — combines all AI layers into a single `CineSenseRecommender` class |
| `hybrid_model.py` | 435 | Two-Tower + Transformer hybrid model definition |
| `two_tower_ncf.py` | — | User Tower + Movie Tower NCF architecture |
| `neumf_scorer.py` | — | 13-model ensemble scorer (NeuMF + Genre affinity) |
| `pairwise_learning.py` | — | ELO rating system (Bradley-Terry model) |
| `reinforcement.py` | — | UCB Multi-Armed Bandit for exploration |
| `embeddings.py` | — | 55-dim movie vectors + UserEmbedding + ContentBasedRecommender |
| `advanced_ai.py` | — | Latent space, implicit signals, NLG, temporal memory |
| `cache_manager.py` | — | LRU sliding-window cache (50-100 movies) |
| `candidate_generator.py` | — | Genre/popularity/exploration candidate funnel |
| `content_pipeline.py` | — | Scheduled TMDB content ingestion |
| `data_preprocessor.py` | — | Data loading, merging, encoding for training |
| `evaluation.py` | — | RMSE, Hit Rate@K, NDCG@K metrics |
| `inference_pipeline.py` | — | Production: embed → candidates → score → rank → diversify |
| `netflix_recommender.py` | — | Flask integration wrapper for DL inference |
| `training_pipeline.py` | — | Adam + LR scheduler + early stopping + checkpointing |

### `api/` Module

| File | Lines | Description |
|------|-------|-------------|
| `routes.py` | 933 | All REST API endpoints: users, movies, recommendations, comparisons, cache |

### `database/` Module

| File | Lines | Description |
|------|-------|-------------|
| `schema.sql` | 320 | Full DDL: 11 tables, 2 views, 2 stored procedures, indexes |
| `db_manager.py` | 509 | Singleton + connection pooling + all CRUD operations |
| `run_migration.py` | — | Applies SQL migrations with version tracking |
| `001_lazy_loading_migration.sql` | 314 | Adds lazy loading, cache tracking, interaction types |

### `preprocessing/` Module

| File | Description |
|------|-------------|
| `merge_datasets.py` | Inner-join MovieLens + TMDB via `links.csv` |
| `feature_engineering.py` | Label-encode IDs + generate 384-dim plot embeddings |

### `scripts/` Module

| File | Description |
|------|-------------|
| `fetch_tmdb_data.py` | Fetch ~100K movies: TMDB API → MySQL |
| `fetch_tv_series.py` | Fetch TV series: TMDB API → MySQL |
| `update_tv_posters.py` | Fix poster URL format for TV series |

### `training/` Module

| File | Description |
|------|-------------|
| `models.py` | PyTorch model classes: `NCF`, `HybridRecommender` |
| `train.py` | Training loop: epochs, metrics, early stopping, checkpointing |
| `advanced_models.py` | `AttentionLayer`, `DeepNCF` (attention-based, RMSE < 0.75) |
| `advanced_training.py` | `FocalLoss`, LR warmup, negative sampling, data augmentation |

### `inference/` Module

| File | Description |
|------|-------------|
| `recommend.py` | Production inference: ensemble + legacy + popular fallback |

### `tmdb/` Module

| File | Description |
|------|-------------|
| `fetcher.py` | TMDB API client: popular, top-rated, search, details, credits |

### `templates/` (11 files)

| File | Page |
|------|------|
| `base.html` | Layout wrapper (nav, footer, session) |
| `index.html` | Home with recommendation carousels |
| `detail.html` | Movie details + similar movies |
| `compare.html` | Pairwise comparison UI |
| `search.html` | Search with autocomplete |
| `login.html` | Login form |
| `signup.html` | Registration form |
| `profile.html` | User stats & preferences |
| `category.html` | Genre browsing |
| `cache_monitor.html` | Cache dashboard |
| `404.html` | Error page |

### `static/` (3 files)

| File | Description |
|------|-------------|
| `css/style.css` | Full application CSS |
| `js/main.js` | Core JS: API calls, DOM, events |
| `js/lazy_loading.js` | Infinite scroll |

---

## 15. How It All Works Together

### User Journey

1. **New User** visits CineSense → sees popular & trending movies (cold start handled by content embeddings)
2. **Signs Up** → account created in `users` table
3. **Browses Movies** → movies fetched from `movie_details` view
4. **Goes to Compare** → gets random movie pairs from `movies` table
5. **Makes Comparisons** → interaction saved to `user_interactions`, ELO scores updated via `update_movie_elo` stored procedure
6. **Gets Recommendations** → AI engine generates candidates, scores them using ELO + content + NeuMF, returns personalized top 20
7. **Continues Using** → preference model refines with each interaction, temporal memory weighs recent choices higher

### System Startup Flow

```
python app.py
    │
    ├── create_app()
    │     ├── Load config from config.py + .env
    │     ├── Initialize Flask, CORS, JSON provider
    │     ├── Register API blueprint (routes.py)
    │     ├── Start content pipeline (background thread)
    │     ├── Define web routes
    │     └── Return configured app
    │
    └── app.run(host='0.0.0.0', port=5000)
          │
          ├── First request triggers:
          │     ├── DatabaseManager singleton (10-connection pool)
          │     ├── CineSenseRecommender initialization
          │     │     ├── PairwiseLearner (ELO)
          │     │     ├── MovieEmbedding (55-dim vectors)
          │     │     ├── UCBBandit (exploration)
          │     │     ├── LatentSpaceEncoder (PCA 55→32)
          │     │     ├── SlidingWindowCache (100 slots)
          │     │     ├── CandidateGenerator
          │     │     └── NeuMF Ensemble Scorer (13 models)
          │     └── TMDBFetcher (API client)
          │
          └── Ready to serve requests
```

---

## 16. Future Enhancements

1. **Graph Neural Networks** — model user-movie-genre relationships as a knowledge graph
2. **Real-time A/B testing** — test different recommendation strategies on user segments
3. **Multi-modal embeddings** — incorporate movie trailers (video) and soundtracks (audio)
4. **Federated learning** — train on user data without centralizing it
5. **Social recommendations** — "Users similar to you also liked..."
6. **Explanation dashboard** — visual breakdown of why each movie was recommended
7. **Mobile application** — React Native or Flutter frontend
8. **Advanced caching** — Redis for sub-millisecond embedding lookups
