# CineSense — Overview

> What CineSense is, what problems it solves, and its objectives.
> Part of the [README](../README.md) doc set.

## What it is

**CineSense** is an AI-powered movie recommendation platform. Instead of 1–5 star
ratings, it learns taste through **pairwise comparisons** ("Which movie do you
prefer?"), backed by a normalized MySQL database and hybrid ML models.

Core techniques:

- **Pairwise preference learning** — Bradley-Terry / ELO updates from comparisons
- **Neural collaborative filtering** — Deep V2/V3 models (DCN-v2 + FM, trained on Kaggle Dual T4 GPUs)
- **Content-based filtering** — Sentence-Transformer plot embeddings + 55-dim genre/director/actor/metadata vectors
- **Exploration** — UCB multi-armed bandit over genres
- **Two-stage serving** — candidate retrieval (300) → hybrid rerank → top 20

## Problem → Solution

| Problem | Solution in CineSense |
|---------|----------------------|
| Cold start (no history) | Content embeddings understand movies with zero interactions |
| Rating bias (my 3★ ≠ your 3★) | Pairwise choices need no absolute scale |
| Filter bubbles | UCB bandit + DPP diversity reranking force exploration |
| Scale (score everything per user) | Funnel: full catalog → 300 candidates → 20 results |

## Objectives

1. Full-stack recommendation web app (Flask + MySQL + PyTorch)
2. Normalized MySQL schema with views, procedures, functions, triggers
3. Hybrid AI (collaborative + content + pairwise + bandit)
4. Train/evaluate on MovieLens-100K and MovieLens-32M (+ TMDB metadata)
5. Real-time learning from comparisons
6. Responsive UI: search, compare, recommendations, social
7. Cloud deployment (Render + Gunicorn)

## Next

- [Architecture](architecture.md) — how the system fits together
- [Data & Methods](methods.md) — datasets, pipeline, models
- [Results](results.md) — benchmark numbers

[<- Back to README](../README.md)
