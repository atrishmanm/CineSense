# CineSense — Database & API

> Counts verified directly from `database/schema.sql` (Oct 2026).
> Earlier docs said "16 tables / 12 views / 8 procedures / 4 functions / 5 triggers" — **stale**.
> Part of the [README](../README.md) doc set.

## Database (MySQL 8.0, InnoDB, utf8mb4)

| Object | In `schema.sql` | Live (MySQL 8.0.44, verified Oct 2026) |
|--------|---------------|------------------------------------------|
| Tables | **28** | 42 (28 core + 14 social/migration: friendships, watch parties, lists, chat, A/B, `schema_migrations`) |
| Views | **24** | 24 ✓ |
| Stored procedures | **24** | 25 (24 + `get_semantic_search_results` from social setup) — all create cleanly after the LOCK TABLES fix below |
| Functions | **8** | 8 ✓ |
| Triggers | **11** | 11 ✓ |
| Standalone CREATE INDEX | 4 (+ 54 inline KEY/INDEX + 3 FULLTEXT) | — |

> LOCK TABLES fix (Oct 2026): three procedures used `LOCK TABLES`, which is
> illegal inside MySQL stored programs — they silently failed to create
> (live DB had only 22 procedures). Rewrote with row-level `FOR SHARE` /
> transactions; verified all three create and run on MySQL 8.0.44
> (`scripts/apply_lock_fix.py`). MariaDB note: 22/24 procedures load there.

Authoritative file: `database/schema.sql`. Access via `database/db_manager.py`
(singleton, 10-connection pool, parameterized calls). Migrations:
`database/run_migration.py`.

### Tables (28)

Core: `users`, `movies`, `genres`, `directors`, `actors`,
`movie_genres`, `movie_directors`, `movie_actors`,
`user_interactions`, `user_embeddings`, `movie_embeddings`,
`user_preferences`, `movie_keywords`, `search_history`, `cache_stats`,
`candidate_generation_log`, `recommendation_log`, `watchlist`,
`movie_reviews`, `genre_win_counts`.
Extended (`IF NOT EXISTS`): `keywords`, `semantic_embeddings`, `search_logs`,
`user_moods`, `recommendation_feedback`, `ab_test_assignments`,
`ab_test_metrics`, `model_versions`.

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
`update_cache_stats`, `cleanup_temporary_movies`, plus maintenance/locking helpers.

### Functions (8)

`calculate_user_similarity`, `get_movie_diversity_score`,
`has_user_interacted_with_movie`, `calculate_cache_hit_rate`,
`get_popularity_tier`, `jaccard_genre_similarity`, `get_user_taste_profile`,
`weighted_recommendation_score`.

### Triggers (11)

Validation on movie insert/update, user-init on signup, interaction
bookkeeping (`after_interaction_insert`, genre-win counts), search logging,
review/watchlist sync, access tracking.

## API

Base: blueprint `api/routes.py` under `/api` (34 routes) + `api/social_routes.py`
(14) + feature routes in `app_integrated.py` (7) = **55 under `/api`**
(68 URL rules total incl. pages). Don't claim "60+ REST routes."

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
`/profile`, `/category/<name>`, `/monitor`, `/chat-ui`, `/features`, `/friends`.

[<- Back to README](../README.md) · Next: [Setup](setup.md)
