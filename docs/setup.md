# CineSense — Setup & Deployment

> Verified entry point: `app_integrated.py` (there is no `app.py`).
> Part of the [README](../README.md) doc set.

## Prerequisites

- Python 3.10+ (`runtime.txt` pins `python-3.10.0`; note `render.yaml` says
  `PYTHON_VERSION=3.11.0` — align these before deploying)
- MySQL 8.0+, TMDB API key (free at themoviedb.org/settings/api)
- Optional: Redis (caching), CUDA GPU (5–10× inference)

## Install

```bash
git clone https://github.com/your-repo/CineSense.git
cd CineSense
pip install -r requirements.txt
cp .env.example .env   # then set DB_* + TMDB_API_KEY
```

## Database

```bash
mysql -u root -p -e "CREATE DATABASE cinesense CHARACTER SET utf8mb4;"
mysql -u root -p cinesense < database/schema.sql
python database/run_migration.py
```

`database/schema.sql` is the single authoritative schema (36 tables, 24 views,
24 procedures, 8 functions, 11 triggers).

## Run

```bash
python app_integrated.py            # → http://localhost:5000
gunicorn "app_integrated:create_app()" --workers 2 --timeout 120   # prod
```

Key pages: `/` `/compare` `/search` `/chat-ui` `/features` `/friends` `/monitor`.

## Train / benchmark

```bash
python benchmarks/run_all_benchmarks.py   # canonical suite → benchmarks/results/
python training/train_100k.py --epochs 15  # reproducible 100K trainer → model/mf_100k.npz
python benchmarks/run_comprehensive_evaluation.py  # regression-heavy eval
pytest -q                                 # unit/integration tests
```

Kaggle Dual-T4 training: push `kaggle_runner/`, check kernel status, pull
weights into `models/` (see [Results](results.md)).

## Deploy (Render)

`Procfile`: `web: gunicorn "app_integrated:create_app()"`.
`render.yaml`: build `pip install -r requirements.txt`, start with 2 workers,
120s timeout. Set `SECRET_KEY` (generate), `TMDB_API_KEY`, `DB_*` in dashboard.

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| DB connection fails | MySQL running? `.env` creds? database created? |
| Redis errors | Optional — app runs without it, just slower; start `redis-server` |
| TMDB 401 | invalid key — refresh at themoviedb.org/settings/api |
| Slow first start | ~2GB HuggingFace models download once, then cached |
| AI endpoints 503 | models lazy-load; wait 30–60s after cold start |
| OOM | default is `faiss-cpu`; lower batch size |

[<- Back to README](../README.md) · Next: [Results](results.md)
