# CineSense

AI-powered movie recommendation platform that learns your taste through pairwise comparisons and delivers personalized suggestions using a multi-layered intelligence stack — from ELO-based preference learning to a 13-model deep learning ensemble.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-3.0-green?logo=flask)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red?logo=pytorch)
![MySQL](https://img.shields.io/badge/MySQL-8.0+-orange?logo=mysql&logoColor=white)

## Features

- **Pairwise Comparison Engine** — Pick which movie you prefer; the AI learns the rest
- **Multi-Layer AI** — ELO scoring → content embeddings → reinforcement learning → deep learning ensemble
- **Explainable Recommendations** — Natural language explanations for every suggestion
- **Two-Tower Neural Network** — User Tower + Movie Tower architecture (Netflix/YouTube-style)
- **Hybrid Content-Aware Model** — Collaborative filtering + BERT-based plot understanding
- **Lazy Loading Architecture** — 77× memory reduction via sliding-window cache + TMDB streaming
- **Infinite Content** — Access to 1M+ movies through real-time TMDB API integration
- **Production-Ready** — Candidate generation, diversity reranking, cache monitoring dashboard

## Quick Start

```bash
# Clone & setup
git clone https://github.com/<your-username>/CineSense.git
cd CineSense
python -m venv .venv && .venv\Scripts\activate   # Windows
pip install -r requirements.txt

# Configure environment
cp .env.example .env   # Edit with your DB & TMDB credentials

# Initialize database
mysql -u root -p < database/schema.sql
python database/run_migration.py

# Run
python app.py
```

Open [http://localhost:5000](http://localhost:5000)

### Environment Variables

Create a `.env` file:

```env
DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASSWORD=your_password
DB_NAME=cinesense
TMDB_API_KEY=your_tmdb_api_key    # Free from themoviedb.org
SECRET_KEY=your_secret_key
```

### Data Setup

Download and place in `data/`:

| Dataset | Path | Source |
|---|---|---|
| MovieLens 32M | `data/movie-lens_ml-32m/` | [grouplens.org](https://grouplens.org/datasets/movielens/32m/) |
| TMDB Movies | `data/tmdb/TMDB_movie_dataset_v11.csv` | [kaggle.com](https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies) |

## Model Training

```bash
# Quick test (~10 min, 100K samples)
python train_model.py --sample --epochs 5

# Full training (~4-6 hrs CPU, ~1 hr GPU)
python train_model.py --model-type hybrid --epochs 20

# Complete pipeline (merge → features → train)
python run_pipeline.py
```

Training notebook: [`colab_train_100k.ipynb`](colab_train_100k.ipynb) — runs on Google Colab with GPU.

## Tech Stack

| Layer | Technologies |
|---|---|
| **Backend** | Flask, Gunicorn, MySQL 8.0+ |
| **AI/ML** | PyTorch, Sentence-Transformers (BERT), scikit-learn, NumPy |
| **Models** | Two-Tower NCF, Hybrid CF+Content, NeuMF V2 Ensemble (13 models) |
| **Frontend** | Jinja2 templates, vanilla JS, CSS |
| **Data** | MovieLens-32M, TMDB API |
| **Deploy** | Render, Heroku (Procfile + gunicorn) |

## Project Structure

```
CineSense/
├── app.py                    # Flask application factory
├── config.py                 # Centralized configuration
├── train_model.py            # CLI training entry point
├── run_pipeline.py           # End-to-end pipeline orchestrator
├── quick_start.py            # Interactive setup wizard
│
├── ai/                       # AI engine (16 modules)
│   ├── recommender.py        # Main recommendation engine
│   ├── two_tower_ncf.py      # Two-Tower NCF architecture
│   ├── hybrid_model.py       # Content-aware hybrid model
│   ├── neumf_scorer.py       # 13-model NeuMF ensemble
│   ├── inference_pipeline.py # Production inference pipeline
│   ├── pairwise_learning.py  # ELO + Bradley-Terry model
│   ├── reinforcement.py      # Multi-armed bandits
│   ├── embeddings.py         # Content-based feature vectors
│   ├── cache_manager.py      # Sliding-window LRU cache
│   ├── candidate_generator.py# Candidate generation
│   ├── content_pipeline.py   # Automated content ingestion
│   └── ...                   # + 5 more modules
│
├── api/routes.py             # REST API endpoints
├── database/                 # MySQL schema, migrations, DB manager
├── training/                 # Model architectures & training loops
├── inference/                # Production model serving
├── preprocessing/            # Data merging & feature engineering
├── tmdb/fetcher.py           # TMDB API integration
├── templates/                # Jinja2 HTML templates
└── static/                   # CSS + JavaScript
```

## API Overview

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/api/user/signup` | Create account |
| `POST` | `/api/user/login` | Login |
| `GET` | `/api/recommendations` | Personalized recommendations |
| `GET` | `/api/recommendations/lazy` | Lazy-loaded recommendations |
| `GET` | `/api/compare` | Pairwise comparison pair |
| `POST` | `/api/feedback` | Submit user preference |
| `GET` | `/api/movie/<id>` | Movie details |
| `GET` | `/api/movie/search?q=` | Search movies |
| `GET` | `/api/cache/stats` | Cache performance metrics |

> Full technical documentation: [ARCHITECTURE.md](ARCHITECTURE.md)

## License

MIT
