# CineSense — AI-Powered Movie Recommendation Platform

**An intelligent movie recommendation system powered by deep learning, pairwise preference learning, and reinforcement learning — built with a fully normalized MySQL database and 100K+ movie records from TMDB and MovieLens datasets.**

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | Flask 3.0 (Python) |
| Database | MySQL 8.0+ (3NF, 11 tables) |
| Deep Learning | PyTorch 2.1+ (NCF, Two-Tower, NeuMF ×13 ensemble) |
| NLP / Embeddings | Sentence-Transformers (all-MiniLM-L6-v2, 384-dim) |
| ML Libraries | NumPy, Scikit-learn, Pandas |
| External API | TMDB API v3 |
| Frontend | HTML5, CSS3, JavaScript, Jinja2 |
| Deployment | Gunicorn + Render |

## Key Features

- **Pairwise Comparison** — "Which movie do you prefer?" interactions that learn user taste in real-time via ELO ratings (Bradley-Terry model)
- **13-Model NeuMF Ensemble** — Neural Collaborative Filtering trained on MovieLens-100K (RMSE = 0.8932)
- **Content-Based Filtering** — Sentence Transformer embeddings for cold-start handling
- **Reinforcement Learning** — UCB Multi-Armed Bandit for exploration vs exploitation
- **55-Dimension Feature Vectors** — 20 genre + 10 director + 20 actor + 5 metadata features
- **Normalized Database** — 11 tables in 3NF with stored procedures, views, and 19 indexes

## Setup

```bash
# Clone & install
git clone https://github.com/your-repo/CineSense.git
cd CineSense
pip install -r requirements.txt

# Configure .env
DB_HOST=localhost
DB_PORT=3306
DB_USER=root
DB_PASSWORD=your_password
DB_NAME=cinesense
TMDB_API_KEY=your_tmdb_api_key

# Initialize database & train models
mysql -u root -p < database/schema.sql
python database/run_migration.py
python scripts/fetch_tmdb_data.py
python run_pipeline.py

# Run
python app.py
```

App runs at `http://localhost:5000`.

## Documentation

| Document | Contents |
|----------|----------|
| [PROJECT_DOCUMENTATION.md](PROJECT_DOCUMENTATION.md) | Full system architecture, AI/ML pipeline details, module-wise file descriptions, API endpoints, frontend pages, datasets, deployment, and more |
| [DATABASE_DOCUMENTATION.md](DATABASE_DOCUMENTATION.md) | ER model & diagram, relational schema design, normalization (1NF→3NF), table descriptions, DDL/DML commands, views, stored procedures, indexes |

## References

- [MovieLens Dataset — GroupLens](https://grouplens.org/datasets/movielens/)
- [TMDB API](https://developer.themoviedb.org/docs)
- [Neural Collaborative Filtering (He et al., 2017)](https://arxiv.org/abs/1708.05031)
- [Sentence-BERT (Reimers & Gurevych, 2019)](https://arxiv.org/abs/1908.10084)
