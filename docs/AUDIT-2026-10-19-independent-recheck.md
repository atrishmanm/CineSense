# CineSense — Independent Full Re-Audit (2026-10-19, Wave 5)

**Status: DRAFT — read-only audit. No code was changed.**

This is a fully independent re-read of the entire repository, conducted *after*
the wave 1–4 fix sessions recorded in `docs/AUDIT-2026-10-19.md`. Every file in
`app/`, `api/`, `ai/`, `database/`, `tmdb/`, `serving/`, `inference/`,
`training/`, `kaggle_runner/`, `preprocessing/`, `benchmarks/`, `scripts/`,
`tests/`, `templates/`, `static/`, plus root configs (`config.py`,
`app_integrated.py`, `requirements.txt`, `Procfile`, `render.yaml`,
`runtime.txt`, `.env.example`, `pytest.ini`, `README.md`) was read end-to-end
and cross-checked against callers, `database/schema.sql`, and
`benchmarks/results/*` artifacts. Eight independent area-audit passes were run
and their findings merged and de-duplicated here.

Method notes / honesty caveats:

- Each finding carries an exact `file:line` from **current** file content.
- The highest-impact findings were independently re-verified (execution or
  grep) before being listed here: the hybrid checkpoint key drift was
  reproduced by running `torch.load` + `load_state_dict`; `/api/cache-stats`,
  the review-`helpful_count` key, the `FIND_IN_SET` view separator and the dead
  `process_user_choice` path were grep-verified in current code.
- Findings that *contradict* the wave-4 ledger's claims ("pyflakes 0", "18 of
  20 fixed", README metrics) are called out explicitly.
- `FIND_IN_SET` semantics were re-derived by this writer (see C-2.5); one
  area-pass overstated it ("matches zero rows") — corrected below.

---

## 0. Executive summary

| Metric (verified now) | Value |
|---|---|
| Findings in this document | **~470 numbered** across sections A–N |
| Still-open guarantees-500 endpoints | 1 (`/api/cache-stats`) — **fixed in Wave 5, see §O** |
| Dead-on-arrival model-loading paths | 2 (hybrid checkpoint key drift; torchserve full-module pickle contract) — **both fixed in Wave 5, see §O** |
| Stored-XSS injection surfaces | 9+ (unescaped `innerHTML` interpolations) |
| Modules imported by nothing outside tests/benchmarks (dead in prod) | 14 modules + numerous classes |
| README claims contradicted by repo state | 4 (doc links → deleted files; test counts; endpoint count; "reproduced" benchmark metrics) |
| Schema object truth | 36 tables / 24 views / 24 procs / 8 funcs / 11 triggers (README table correct) |

**Top-10 most severe, in order:**

1. **[bug] model/**`hybrid_recommender.pt`/`hybrid_checkpoint.pt` cannot be
   loaded by `training/models.py::HybridRecommender` — state-dict key drift
   (`user_embedding.*`/`fusion.*` vs `user_emb.*`/`fc.*`). Every consumer
   crashes (F-1.1). Verified by execution.
2. **[bug] app_integrated.py:633 — `/api/cache-stats` always 500s**:
   `app.cache.get_cache_stats()` vs `RedisCache.get_stats()`.
3. **[security] templates/** — stored XSS: review text/username and friend
   names are interpolated into `innerHTML` unescaped on 9+ surfaces (J-21..25).
4. **[bug] database/db_manager.py:53-55 — connection pool-slot leak**: when a
   connection is dead, `close()` (which returns it to the pool) is skipped;
   after one MySQL blip the pool permanently exhausts (D-2.1).
5. **[bug] ai/recommender.py:308 — `process_user_choice` is never called** by
   any route, so bandit/embedding/preference learning never actually updates
   from user feedback in production (C-1.1).
6. **[bug] on-demand TMDB TV ingest inserts blank-title rows** (`title=''`,
   `release_year=NULL`) because TV payloads (`name`/`first_air_date`) are fed
   to `parse_movie_data` which reads `title`/`release_date` (C3.15/E1.1).
7. **[bug] benchmark harness contradicts its own results** —
   `run_32m_benchmarks.py` claims "bootstrap CIs"/emits `bootstrap_iterations`
   metadata with no bootstrap code in the file; stored reports reference
   outputs the current code cannot produce (H-18..20).
8. **[security] api/social_routes.py:59-75 — friend lists expose other users'
   e-mail PII to any logged-in user**; `_can_edit_list` grants viewers edit
   rights (IDOR/privilege escalation) (B-3, B-7).
9. **[inconsistency] preference term is provably constant in every benchmark**
   — `user_pref` only contains train-rated items, which are excluded from
   candidates; a 30% weight in the "hybrid" model is dead, making several
   ablation rows identical duplicates (H-2).
10. **[robustness] finding-queue truncation**: heavy models (CLIP,
    transformers, cross-encoder) download at request time with unlocked lazy
    loads — first requests can block for minutes and race double-loads
    (C-2.16/17, C-3.52-54).

> **Postscript (2026-10-21, fix campaign):** all ten items above are fixed in
> the Wave 5 fix campaign — verified by an executed load-forward smoke test
> (both legacy checkpoints), a live call-site change, XSS escaping across every
> enumerated surface, an unconditional pooled close, the `process_user_choice`
> wiring in `/api/feedback`, TV-payload normalization, honest bootstrap
> implementation, PII/authorization fixes, and one-time lazy-load warnings.
> Per-ID status is in **§O. Wave 5 fix campaign status** at the end of this
> document; detailed per-fix reports live in `scratch/wave5_fixes_*.md`.

---

# A. Root application + configuration (`app_integrated.py`, `config.py`)

## A-1. Broken endpoint / response-shape / error handling

- **A1.1 [bug]** `app_integrated.py:633` — `/api/cache-stats` calls
  `app.cache.get_cache_stats()`, but `RedisCache` only exposes `get_stats()`
  (`ai/redis_cache.py:375`). Every call raises `AttributeError`, swallowed into
  a 500 at :635-637. The endpoint is **guaranteed to 500 whenever Redis is
  available** (the only condition under which it runs), because the else-branch
  503s when Redis is absent. Verified by grep: `get_cache_stats` exists only on
  `ai/recommender.py:1184`.
- **A1.2 [bug]** `app_integrated.py:481-483` — `/api/mood-recommendations`
  catches model/DB failure and returns the fallback payload with **HTTP 200**,
  masking outages (contrast `/api/trending` which 503s on its final fallback).
- **A1.3 [bug]** `app_integrated.py:350-453` — `/api/chat` has no top-level
  try/except; an unexpected DB error renders the **HTML** 500 page while every
  other `/api/*` route returns a JSON error. The blueprint-level
  `api.errorhandler(500)` (`api/routes.py:1653-1655`) does not cover
  app-level routes.
- **A1.4 [bug]** `app_integrated.py:353` — `load_ai_models('chatbot')` loads
  `ConversationalRecommender` (heavy, blocking) on the first `/api/chat` hit,
  but nothing in the handler ever reads `app.ai_systems['chatbot']` (the chat
  logic is reimplemented inline). Pure dead weight + first-request latency
  spike; see also C-3.67.
- **A1.5 [robustness]** `app_integrated.py:508-515` — `/api/trending`: the
  detector-failure path returns 200 with popularity rows (silent degradation)
  while the outer-failure path returns 503 + empty list. Same endpoint, two
  opposite failure modes.
- **A1.6 [inconsistency]** `app_integrated.py:502-506, 565-577` —
  `/api/trending` and `/api/mood-recommendations` return two different row
  schemas under the same key depending on which branch answered (detector rows
  vs `movie_details` fallback rows); mood responses carry both `movies` and
  `recommendations` keys pointing at the same list.
- **A1.7 [inconsistency]** `app_integrated.py:545-554` — `/api/featured`
  attaches `explanation` only for logged-in users, and the explanation feature
  is otherwise unreachable from the frontend (see J-5).

## A-2. Server lifecycle / security posture

- **A2.1 [robustness]** `app_integrated.py:803-807` —
  `app.run(use_reloader=True)`: with the reloader, `create_app()` executes
  **twice** (parent + child). The content-pipeline thread (:337-344), its
  `schedule` jobs, the DB pool and the up-to-10k-movie semantic index build
  (:285-323) all run in the reloader's parent as well. The pipeline's
  `is_running` guard (`ai/content_pipeline.py:35-37`) is per-process only.
- **A2.2 [security]** `app_integrated.py:205` — `CORS(app)` with defaults =
  `Access-Control-Allow-Origin: *` on every route; no origin whitelist, no CSRF
  protection on POST/PUT/DELETE (relies only on default cookie SameSite).
- **A2.3 [security]** `app_integrated.py:804` — dev server hardcoded to
  `host='0.0.0.0'`.
- **A2.4 [hygiene]** `app_integrated.py:197` + `requirements.txt` —
  `SESSION_TYPE='filesystem'` is a **no-op**: `flask-session` is not a
  dependency; sessions are actually default signed-client-cookies. The setting
  misleads readers.
- **A2.5 [hygiene]** `app_integrated.py:198` — `JSON_SORT_KEYS` is a
  deprecated no-op in Flask 3 (should be `app.json.sort_keys`). Similarly,
  `config.py:15` derives `DEBUG` from `FLASK_ENV`, deprecated in Flask 3.
- **A2.6 [robustness]** `app_integrated.py:253-255` — `load_ai_models`
  check-then-load has no lock; under `--threads 4` (Procfile) concurrent first
  requests can load the same heavy model twice.
- **A2.7 [inconsistency]** `app_integrated.py:626` vs `api/routes.py:1472` —
  two overlapping cache-stats endpoints under different paths
  (`/api/cache-stats` broken per A1.1 and consumer-less per J-10;
  `/api/cache/stats` the one the frontend actually uses).
- **A2.8 [inconsistency]** `app_integrated.py:675` — `/profile` passes
  `logged_in`/`username` explicitly, duplicating the context processor values:
  every other route relies on the processor.
- **A2.9 [hygiene]** `app_integrated.py:699-702` — `/monitor` (cache
  dashboard) has no login gate; nav visibility for it in `base.html` is
  decorative.
- **A2.10 [hygiene]** `app_integrated.py:311` — SQL constant format-injection
  of `STARTUP_INDEX_LIMIT` (safe today, _format-into-SQL_ style); and the
  startup semantic query at :293-317 never selects `media_type`, so the
  chat UI's TV badge (`templates/chat.html:144`) can never render.
- **A2.11 [robustness]** `config.py:13` — per-process random `SECRET_KEY`
  fallback: any multi-worker/multi-process local run randomly logs users out
  (documented in config.py and accepted in wave 4 as dev-only; listed for
  completeness).

## A-3. Config cross-checks

- **A3.1 [hygiene]** `config.py:163-217` — 12 self-declared "reserved / not yet
  wired" constants (`CHECKPOINT_DIR`, `RECOMMENDATION_BATCH_SIZE`,
  `STORE_ONLY_INTERACTED`, `LAZY_EMBEDDING`, `EVICTION_STRATEGY`,
  `USE_CANDIDATE_GENERATION`, `FINAL_RECOMMENDATION_COUNT`,
  `PAIRWISE_KNOWN_RATIO`): dead configuration surface inviting misuse.
- **A3.2 [inconsistency]** `config.py:63-65` declares a "single source of
  truth for magic values" but `ai/redis_cache.py` (TTLs 86400/7200/1800/86400),
  `ai/xfetch_cache.py` (max_size=500, ttl=300, beta), `ai/drift_monitor.py`
  (0.10/0.25/0.15), `api/routes.py:1579-1580` (duplicated cache sizes 100/500),
  `static/js/lazy_loading.js:184-189` (/100, /500) and
  `templates/cache_monitor.html` all keep their own copies.
- **A3.3 [verified — clean]** Exhaustive `Config.*` grep across `ai/`: every
  referenced key exists in `config.py`; no missing-config findings.
- **A3.4 [inconsistency]** `.env.example` is missing three live env reads:
  `ENABLE_CROSS_ENCODER`, `CROSS_ENCODER_LOCAL_ONLY` (`ai/semantic_search.py`)
  and `TMDB_POSTER_BASE_URL` (`ai/visual_search.py`).
- **A3.5 [inconsistency]** `config.py:199` — `CANDIDATE_COUNT = 300` comment
  says "200-500 candidates": stale.
- **A3.6 [verified]** `.env` (with a real TMDB key/token and DB password) is
  **not** git-tracked (only `.env.example` is); flagged because the file
  carries production-looking credentials in plaintext on disk.

---

# B. API layer (`api/routes.py` 1655 lines, `api/social_routes.py` 821 lines)

## B-1. Auth, ownership, anti-abuse

- **B1.1 [security]** `api/social_routes.py:59-75, 736-789` —
  `_can_edit_list` returns `True` for **any** member and ignores the `role`
  column ('owner'/'editor'/'viewer', `schema.sql:495`). A viewer can add
  items, and via `ON DUPLICATE KEY UPDATE ... added_by = VALUES(added_by)`
  (:778-782) can **overwrite authorship/notes/position of other members'
  entries** (attribute forging on other users' collaborators' rows).
- **B1.2 [security]** `api/social_routes.py:182,287` —
  `/social/friends/requests` and `/social/friends/list` return other users'
  `u.email` (PII) to any logged-in user.
- **B1.3 [bug]** `api/routes.py:729-772` — `/api/feedback` has **no
  anti-replay/idempotency**: the same pair can be posted repeatedly (or an
  arbitrary pair never actually shown), each call inserting an interaction row
  and applying ±K ELO twice — easy ELO/leaderboard inflation.
- **B1.4 [bug]** `api/routes.py:43-45, 1069-1088` — helpful-vote dedupe is a
  per-process in-memory set; on hitting the cap it **clears all** dedupe state
  (:1074) and it evaporates on restart. No per-user vote record exists in DB
  (`movie_reviews` has only a counter) → vote inflation across
  restarts/workers (partial mitigation only).
- **B1.5 [robustness]** `api/routes.py:206-221` — signup duplicate pre-check
  has a TOCTOU race with the UNIQUE constraint: under race, a duplicate
  email/username returns 500 instead of the intended 409.
- **B1.6 [robustness — clean]** Verified: **no endpoint trusts a
  client-supplied user_id** (chat :357, mood :462, analytics, social all bind
  to session). No SQL injection path found anywhere (all `%s`-parameterized;
  `order_by` double-whitelisted, see D-1.3).

## B-2. Validation / contract drift

- **B2.1 [inconsistency]** `api/routes.py:519-520,546` —
  `/api/user/compared-movies` returns `200 {'movies': []}` without `count`
  for guests while logged-in shape is `{'movies','count'}`; clients cannot
  distinguish guest from empty history.
- **B2.2 [robustness]** `api/routes.py:69-79,563-584` — `_parse_offset` has no
  upper cap: `?offset=1000000` yields `get_recommendations(n=1000020)` — an
  unbounded fetch on the recommendation path.
- **B2.3 [robustness]** `api/routes.py:946,1045`, `social_routes.py:112,381` —
  `movie_id`/`friend_id` arrive unvalidated as ints; non-numeric values
  surface as generic 500s instead of 400. (`social_routes.py:459-463` does
  int-coerce — inconsistent validation style within the same file family.)
- **B2.4 [robustness]** `app_integrated.py` `_clamp_int` swallows bad input to
  a default (no 400), while `routes._parse_limit` 400s — two competing
  limit-parsing contracts in one app.
- **B2.5 [bug]** `api/routes.py:1191-1195` vs `schema.sql:556-588` —
  `/api/search/ai`'s `include_tv=False` filter is **ineffective**:
  `search_movies_advanced` returns `m.*` from `comprehensive_movie_view`,
  which has **no `media_type` column** (only `movie_details` does, :523), so
  the route-side default-at-:1194 logic cannot see TV rows and never filters
  them.
- **B2.6 [inconsistency]** `api/routes.py:985-1002` vs
  `db_manager.py:755-762` — PUT `/api/watchlist/<id>` validates `rating`
  (0-10) but `update_watchlist_status` only writes `user_rating` when
  `status=='completed'`; rating silently discarded for other statuses.
- **B2.7 [inconsistency]** `api/routes.py:1055-1057` — re-POSTing a review
  silently upsert-overwrites the previous review yet returns 201 "Review
  added" as if newly created.
- **B2.8 [inconsistency]** `api/routes.py:38 vs 1046-1049` —
  `VALID_SEARCH_TYPES` includes `'hybrid'` while the SP sanitizer collapses it
  to `'title'`; the route must remap (`:1368`) because the DB enum lacks
  'hybrid' ('title','storyline','semantic','advanced').

## B-3. Side effects, performance, information leaks

- **B3.1 [hygiene]** `api/routes.py:154-165` — `_ensure_movie_genres`
  performs INSERTs inside **GET** handlers (`/api/user/preferences`,
  `/api/user/compared-movies?include_genres=1`).
- **B3.2 [robustness]** `api/routes.py:115-137` — when movies lack genres,
  `_load_tmdb_genres_for_ids` synchronously scans a potentially
  hundreds-of-MB CSV (`data/tmdb/TMDB_movie_dataset_v11.csv`) **inside the
  request**, and `TMDB_GENRE_CACHE` (:23) is unbounded.
- **B3.3 [security]** `api/routes.py:778,1411-1414,1517-1520` — three
  endpoints return `'details': str(e)` in 500 bodies (internal exception text
  leaked to clients). (Other endpoints use generic messages; the wave-4 note
  that `/api/test/tmdb` was fixed holds.)
- **B3.4 [hygiene]** `api/routes.py:458,509,648,681,725,777,915,1318,1409,
  1516,1640` — `traceback.print_exc()` to stdout in 11 handlers (console
  noise; not routed through logging).
- **B3.5 [robustness]** `db_manager.record_interaction` (:459-471) swallows
  the SP's `SIGNAL 45000` validation ("chosen must be one of the pair") into a
  generic `False`; the route cannot distinguish validation failure from DB
  outage — all become the same 500.
- **B3.6 [robustness]** `api/social_routes.py:220-236` — `/friends/accept`
  does a bare INSERT; second accept hits `unique_friendship` → unhandled
  IntegrityError → 500 for an already-successful state.
- **B3.7 [inconsistency]** `api/social_routes.py:151-155` — resending a friend
  request **resurrects a rejected request** (`ON DUPLICATE KEY UPDATE
  status='pending'`); rejection is not sticky.
- **B3.8 [inconsistency]** `api/social_routes.py:551-592` — watch-party join
  flips a **declined** invite to 'accepted' (`status <> 'accepted'` includes
  `declined`); decline isn't sticky.
- **B3.9 [inconsistency]** `api/social_routes.py:239-270` — `/friends/remove`
  returns success even if no friendship existed (no 404 branch).
- **B3.10 [robustness]** `api/routes.py:1524-1560` — `/api/test/tmdb` is a
  permanent, unauthenticated, per-request `TMDBFetcher`-constructing TMDB
  relay usable to burn the TMDB quota.
- **B3.11 [hygiene]** `api/social_routes.py:803-817` —
  `init_social_routes` rebinds the module-global `db` to the passed manager
  even though the module already imports the same singleton (split-brain
  hazard if a different instance is ever passed).
- **B3.12 [hygiene]** `api/routes.py:244, 323` etc., `social_routes.py:342-356,
  507-517, 695-714` — `SELECT *` misuse across many handlers (aggregated view
  payloads where slim projections suffice).

## B-4. Route inventory (verified) & count claims

- **B4.1 [inconsistency]** Actual count: **69 registered routes = 56 `/api`
  endpoints (34 blueprint + 7 app-level + 15 social) + 13 pages**. The README/docs
  claim of "55 `/api` endpoints" is off by one (the two cache-stat routes and
  the one guaranteed-500 notwithstanding, no route is dead-but-counted —
  `/api/test/tmdb` and `/api/cache-stats` both are registered and counted).
- **B4.2 [verified]** Full inventory (method, path, location):
  34 blueprint endpoints (`/api/user/*` ×7, `/api/recommendations[lazy]` ×2,
  `/api/featured`, `/api/compare[lazy]` ×2, `/api/feedback`, `/api/movie/*` ×7,
  `/api/watchlist` ×4, `/api/reviews` ×3, `/api/analytics/*` ×4,
  `/api/search/ai`, `/api/stats`, `/api/cache/*` ×2, `/api/test/tmdb`),
  7 app-level (`/api/chat`, `/api/mood-recommendations`, `/api/trending`,
  `/api/viral`, `/api/visual-search`, `/api/text-to-image-search`,
  `/api/cache-stats`), 15 social (`/api/social/*`), 13 pages
  (`/`, `/compare`, `/movie/<id>`, `/login`, `/signup`, `/profile`, `/search`,
  `/category/<name>`, `/monitor`, `/chat-ui`, `/features`, `/friends`,
  `/favicon.ico`). No duplicate path/method conflicts; the only
  guaranteed-broken route is `/api/cache-stats` (A1.1).

---

# C. AI layer (`ai/`)

## C-1. Serving core: `ai/recommender.py` (57 KB), `candidate_generator.py`, caches, resilience

- **C1.1 [bug — CRITICAL]** `ai/recommender.py:308` — `process_user_choice`
  is **never called** (grep-verified; `/api/feedback` at `api/routes.py:752-760`
  calls `db.record_interaction` directly). The bandit update (:362-363),
  embedding persist (:357) and in-memory preference feed (:340-341) never run
  for real users: the bandit stays permanently cold, `_get_user_embedding`
  (:476) returns a zero vector and `norm_content_score` becomes a constant
  0.5 for every real user. Learning exists only via the UserPreferenceModel's
  DB warm-start replay path (:126-131), not via live feedback.
- **C1.2 [bug]** `ai/recommender.py:1128-1160` —
  `_fit_encoders_from_candidates` contaminates genre vocabulary with single
  characters: when a candidate's `genres` is the DB string `'Action,Drama'`
  (:1148 fallback), `for g in raw_genres` iterates **per character**
  ('A','c','t'…), passing the string-vs-int guard. Encoder vocabulary polluted
  with garbage tokens.
- **C1.3 [bug]** `ai/recommender.py:1021-1032, 1069-1076` — the lazy path
  feeds raw TMDB dicts (keys `genre_ids`, `vote_average`, `release_date`) into
  an embedding pipeline keyed on DB-schema fields (`genres`, `directors`,
  `actors`, `tmdb_rating`, `release_year`, per `ai/embeddings.py:54-92`):
  every candidate gets a nearly-identical default embedding; the "rank by
  content" step degenerates to insertion order. The production path has a
  converter (`_movie_to_embedding`, :269-306); the lazy path skips it.
- **C1.4 [bug]** `ai/recommender.py:1261-1267` — semantic keyword scoring
  double-counts partial matches: `word in overview` (+8) and
  `any(word in w for w in overview.split())` (+3) are the same condition →
  every hit scores 11, never 8 or 3.
- **C1.5 [bug]** `ai/recommender.py:1219, 1272` — the `'sci-fi'` genre boost
  compares against DB genre names, where the canonical value is `'Science
  Fiction'`: that branch is dead weighting.
- **C1.6 [inconsistency]** `ai/recommender.py:761-787 vs 504-537` — the two
  scoring paths emit differently-scaled `recommendation_score` (production:
  cosine clamped [0,1] + sigmoid preference; advanced: raw cosine [-1,1] +
  raw preference, only ELO shares the helper). Three ELO normalizations
  coexist repo-wide (this one, raw cosine path, and benchmark min-max in
  `benchmarks/recommenders.py:197`; plus a *fourth* incompatible variant in
  `stage_retrieval_benchmark.py:157-160`).
- **C1.7 [robustness]** `ai/recommender.py:291-295` — `_movie_to_embedding`
  raises on SQL `NULL` (`float(None)`); one bad row kills the whole
  recommendation call (falls back to popular movies at :568 silently).
- **C1.8 [robustness]** `ai/recommender.py:981-982, 1032, 1069` — movie
  identity key `m.get('id') or m.get('movie_id')` can be `None`, colliding all
  id-less movies under key `None` (duplicated, divergent key logic vs
  `candidate_generator._movie_key`, :25-35, which handles this).
- **C1.9 [inconsistency]** `ai/recommender.py:1041-1046, 1083` — lazy endpoint
  returns unscored dicts in cold-start/exception branches but scored dicts in
  the personalized branch, with `personalized: True` set by routes.py either
  way → heterogeneous response shape.
- **C1.10 [inconsistency]** `ai/recommender.py:729 vs 458` — two different
  personalization gates (raw DB interaction-row count vs
  `user_model.has_enough_data`); rows with `chosen_movie_id IS NULL` count
  toward the former, so the gates disagree.
- **C1.11 [inconsistency]** `ai/recommender.py:790-796` — softmax selection
  branch returns items in sampling order (not score-descending) while the
  deterministic branch is sorted; plus a dead second
  `USE_SOFTMAX_SELECTION` check.
- **C1.12 [robustness]** `ai/recommender.py:246-267, 818-844` — N+1 DB
  queries inside every scoring request (`_build_user_genre_vec` up to 50
  `get_movie_by_id` calls; `_build_temporal_user_vector` up to 20).
- **C1.13 [robustness]** `ai/recommender.py:694` — malformed explanation
  string when no liked titles resolve: `"Recommended because you enjoyed . "`.
- **C1.14 [hygiene]** `ai/recommender.py:1188-1318` — the entire 130-line
  `semantic_search` method is dead code (search endpoints use
  `app.semantic_search.hybrid_search`); its inline keyword table is a second,
  diverging search implementation.
- **C1.15 [hygiene]** `ai/recommender.py:1190` — stale docstring ("AI-powered
  semantic search using embeddings and NLP"); it is pure keyword scoring.
- **C1.16 [hygiene]** `ai/recommender.py:708-816, 904-918` —
  `get_advanced_recommendations` and `get_taste_summary` have zero callers
  (including tests); `apply_diversity_rerank` and `get_cache_stats` are
  called only by tests.
- **C1.17 [hygiene]** `ai/recommender.py:484` — `getattr(Config,
  'DL_SCORE_WEIGHT', 0.15)` guards a key that exists (`config.py:224`);
  masks future renames.
- **C1.18 [hygiene]** `ai/recommender.py:81` + `ai/candidate_generator.py:50`
  — two independent `TMDBFetcher` instances across the two singletons.
- **C1.19 [hygiene]** `ai/recommender.py:953-963` — inline method-level
  `import random`.
- **C1.20 [robustness]** `ai/recommender.py:632-634` +
  `ai/cache_manager.py:129` — cached movie dicts are shared mutable
  references; in-place mutators (e.g. diversity rerank writes
  `mmr_score`/`diversity_penalty`) silently mutate cache contents.
- **C1.21 [robustness]** `ai/recommender.py:1322` (+ `candidate_generator.py:
  385`, `cache_manager.py:305`) — module-import-time instantiation of
  DB-backed singletons: any import-time DB/TMDB hiccup breaks the module
  chain.
- **C1.22 [inconsistency]** `ai/recommender.py:966-974 vs 417-418` — the lazy
  compare path guards against duplicate selected movie ids; the
  `get_comparison_pair` bandit path does not (possible duplicated pair via
  arm-id duplication).
- **C1.23 [inconsistency]** `ai/candidate_generator.py:103-110 vs 76-101` —
  only the `'mixed'` strategy dedupes its channels; `genre`/`popularity`/
  `exploration` return raw concatenated results.
- **C1.24 [inconsistency]** `ai/candidate_generator.py:213-247` — the
  exploration branch ignores `Config.MAX_PAGES_PER_FETCH` (random page
  1-3, uncapped), while the popularity branch caps pages.
- **C1.25 [inconsistency]** `ai/candidate_generator.py:348-376` —
  `generate_pairwise_candidates` mixes DB rows (`movie_id`, `genres`,
  `tmdb_id`) and raw TMDB dicts (`id`, `genre_ids`) in the same 'known' list;
  the dedupe filter only sees DB-shape ids, so a TMDB-known movie can equal an
  exploratory candidate.
- **C1.26 [hygiene]** `ai/candidate_generator.py:251-260` —
  `_generate_from_cache` bypasses the class's injected-dependency pattern by
  reaching the module-global `cache_manager` (untestable).
- **C1.27 [inconsistency]** `ai/candidate_generator.py:118` — every
  `generate_candidates` call bulk-writes into the shared movie cache, evicting
  the LRU window and making the 10% "fast access" channel self-echoing
  (candidate pool reads back what it just inserted).
- **C1.28 [bug]** `ai/cache_manager.py:21-149, 152-228` —
  `SlidingWindowCache`/`VectorCache` have **no thread-safety** despite Flask
  threaded serving + `--threads 4` (concurrent `move_to_end`/`popitem`/
  counter +=  operations): contrast `ai/xfetch_cache.py`, which locks
  everything.
- **C1.29 [hygiene]** `ai/cache_manager.py:81` — `evicted_data` unpacked and
  unused; **:65-71** — `put()` refreshes LRU position but not `last_access`/
  `access_count` (stats stale for overwritten entries); **:131-135, 298-301** —
  `clear()`/`clear_all` wipe data but never reset cumulative inline
  `hits/misses`.
- **C1.30 [inconsistency]** `ai/cache_manager.py:144` — `total_access_count`
  counts accesses **plus insertions** (the `put()` seed of 1) — stat key
  misnamed; same skew for `avg_access_per_movie`.
- **C1.31 [inconsistency]** `ai/cache_manager.py:274-296` vs
  `api/routes.py:1577-1606` — flat keys `movie_max_size`/`vector_max_size` are
  requested but never produced by `get_stats()` (only satisfied because the
  route duplicates the 100/500 defaults as nested fallbacks). All other flat
  keys (`movie_count`, `movie_hits`, ...) align today.
- **C1.32 [bug]** `ai/xfetch_cache.py:131-157` — when the probabilistic
  refresh triggers without a `compute_fn`, a **valid cached value is discarded
  and None returned** (none of the branches return `entry["val"]`).
- **C1.33 [bug]** `ai/xfetch_cache.py:66-68` — `Singleflight.do` only catches
  `Exception`; a `BaseException` in `fn` leaves waiters to blindly return
  `flight.value` (None) as silent success.
- **C1.34 [robustness]** `ai/xfetch_cache.py:59` — waiters block indefinitely
  (no deadline, no cancellation).
- **C1.35 [hygiene]** `ai/xfetch_cache.py:88, 142` — stale "refresh
  asynchronously" comment (recompute is synchronous); module-level hardcoded
  defaults not from config; the module's whole API is wired only in
  benchmarks/tests, not production.
- **C1.36 [bug]** `ai/resilience.py:145-165` — `IdempotencyRegistry` has a
  check-then-act race: membership check and store are separate lock
  acquisitions with `execution_fn()` between → two concurrent same-key
  requests both execute, violating the documented at-most-once contract.
- **C1.37 [bug]** `ai/resilience.py:53-62` — OPEN-state `fallback_fn()`
  executes **while holding the breaker lock** (slow fallback stalls every
  other thread).
- **C1.38 [robustness]** `ai/resilience.py:54-65` — HALF_OPEN permits
  unbounded concurrent probes (no single-explorer gate) and `failure_count` is
  not reset on OPEN→HALF_OPEN, so a single probe failure instantly re-opens
  (`half_open_success_threshold` unreachable in the failure path).
- **C1.39 [hygiene]** `ai/resilience.py:60-72, 92-165` — fallback exceptions
  bypass breaker accounting; `DeadlineContext`, `TokenBucketRateLimiter`,
  `IdempotencyRegistry` are wired only in tests/benchmarks (the docstring's
  "High Availability & Resilience Engine" has no production wiring).
- **C1.40 [bug]** `ai/redis_cache.py:62 vs 65-70` — `self.default_ttl` is only
  assigned in the success branch of `__init__`; any future use before an
  availability check `AttributeError`s on failure-initialized instances.
- **C1.41 [robustness]** `ai/redis_cache.py:452-463` —
  `get_redis_cache(**kwargs)` silently drops kwargs when the singleton exists
  (different host/db/password requests are ignored without warning).
- **C1.42 [inconsistency]** `ai/redis_cache.py` — mixed serialization: pickle
  for recs/embeddings/search (:123,147,168,193,207,231,250,274) vs JSON for
  movies/sessions (:293,317,425,440). A `db_manager` movie dict containing
  `Decimal`/`datetime` fails `json.dumps` in `set_movie` (silently dropped)
  while the same dict pickles fine elsewhere.
- **C1.43 [robustness]** `ai/redis_cache.py:15,97-107` — the outage flag never
  resets after recovery: after the first incident, later outages are logged
  only at debug level (effectively silent forever). Also `is_available()`
  pings on **every** cache access (:92-107) — one extra network round-trip per
  get/set, no fast-fail state.
- **C1.44 [security]** `ai/redis_cache.py:123,168,207,250` — `pickle.loads`
  of Redis-stored bytes is an RCE vector on a shared/compromised store; the
  same class already uses the safer JSON path for movies/sessions
  (inconsistent risk posture).
- **C1.45 [hygiene]** `ai/redis_cache.py:187,226,269,312` — TTL magic numbers
  hardcoded (config claims single-source-of-truth).
- **C1.46 [hygiene]** `ai/reinforcement.py:14-107, 213-297, 300-397` —
  `MultiArmedBandit`, `ThompsonSampling`, `ContextualBandit` are unwired;
  only `UCBBandit` is used (module comment at :339 admits ContextualBandit is
  dead).
- **C1.47 [robustness]** `ai/reinforcement.py:110-210` — `UCBBandit` state is
  RAM-only with no persistence: every restart resets all arms to
  optimistic-inf exploration (the asymmetry the warm-start work was supposed
  to fix for the preference model still exists for the bandit).
- **C1.48 [hygiene]** `ai/reinforcement.py:53,364` — all default parameter
  values are bound at import time (runtime `Config` edits ignored); behavior
  undocumented.
- **C1.49 [hygiene]** `ai/telemetry.py:1-91` — the entire module is
  production-dead (only test_scale_architecture imports it).
- **C1.50 [bug]** `ai/telemetry.py:69-86` — `FunnelTracker.record_stage` can
  overwrite a stage value silently while dict-insertion order pins its
  position → `export_dropout_report` mixes counts from different runs.
- **C1.51 [inconsistency]** `ai/telemetry.py:31-57` — `LatencyHistogram` is
  unsynchronized (lost updates under concurrency); `export()` labels coarse
  bucket edges as exact p50/p95/p99 with no interpolation and no bucket for
  durations > 500 ms (counted in total, invisible in buckets).

## C-2. Search / mood / conversational / trending

### `ai/semantic_search.py`

- **C2.1 [bug]** Model/index dimension mismatch across cache generations: if
  `all-mpnet-base-v2` (768-d) fails, the fallback is `all-MiniLM-L6-v2`
  (384-d) (:41,49), but `load_cache()` (:439-459) rebuilds the FAISS index
  without validating vector dim vs loaded model → request-time dimension
  errors on every search, silently degrading `/movie/search` and `/api/chat`
  semantics.
- **C2.2 [robustness]** Model selection is nondeterministic (mpnet vs MiniLM
  depending on HF cache/network state) and the pickle cache is not versioned
  by model name/dim → same deploy can produce different score scales.
- **C2.3 [robustness]** When nothing passes `min_score`, :197-202 silently
  returns up to `top_k` below-threshold results as `match_type=
  'semantic_fallback'`, and no caller (routes.py:822,1246,1356;
  app_integrated.py:438) distinguishes them — below-threshold noise surfaces
  as usable results.
- **C2.4 [inconsistency]** `hybrid_search` hardcodes stage-1 top_k=50 /
  min_score=0.2 (:221), different from `search()` defaults (20/0.3) and not
  derived from the requested top_k → `hybrid_search(top_k=100)` can only ever
  rerank 50 candidates.
- **C2.5 [inconsistency]** Two divergent `final_score` formulas in one method:
  `semantic_weight·rel + (1-w)·kw` (:232-235) vs the hardcoded
  `0.5·rel + 0.2·kw + 0.3·rerank` (:252-256). The `semantic_weight` argument
  is ignored whenever the cross-encoder runs.
- **C2.6 [robustness]** `_load_cross_encoder()` (:416-437) is unlocked; two
  concurrent first-time calls can both download/load the model (request-time
  HF download blocks the request when local-only=0) and every request retries
  a failed load (no failure memoization). Redundant double enable-check
  (:238 vs :420).
- **C2.7 [security/robustness]** Cache save/load uses raw pickle (unsafe
  deserialization) without atomic write (:375-397, 442-451); embedding index
  never invalidates stale movie rows (edits in DB persist forever as stale
  titles/overviews); all data cached in one pickle.
- **C2.8 [hygiene]** `metadata.json` is written (:380,390-393) but read by
  nothing (dead artifact). `_extract_keywords` is punctuation-blind
  (:296-300): tokens like "indian," miss keyword matches.
- **C2.9 [robustness]** Empty index edge: `k = min(top_k*2,
  len(movie_data))` (:168) → k=0 errors in `faiss.IndexFlatIP.search` when the
  index was built from zero movies. FAISS build failures and
  `ImportError` collapse to the same silent slow path (:409-414).
- **C2.10 [guard mismatch]** `routes.py:1243` checks `movie_embeddings is
  None` but calls `hybrid_search`, which requires `movie_data` too — a
  partial cache load state passes the guard and IndexError's at request time
  (only the route try/except saves it). Two different guards on two different
  callers (:820 vs :1243).

### `ai/visual_search.py`

- **C2.11 [robustness]** With default `VISUAL_SEARCH_LOCAL_ONLY=0`, CLIP
  weights download synchronously inside the first `/api/visual-search` /
  `/api/text-to-image-search` request (:56-91). `_ensure_model_loaded` and
  `_ensure_index_ready` (:103-126) and `load_ai_models` in
  app_integrated.py:253-283 are all unlocked → duplicate downloads/double
  index builds under concurrent first requests.
- **C2.12 [bug]** Retry counters count attempts, not failures (`model_load_
  attempts += 1` before success can consume the budget; :64, 115).
- **C2.13 [robustness]** `index_movie_posters` (:128-162) downloads up to 500
  posters sequentially per build (5 s timeout each → tens of minutes worst
  case); partial indexes accepted silently (5/500 posters = "ready").
- **C2.14 [robustness]** `extract_dominant_colors` (:392-422) swallows all
  failures (missing `raise_for_status` — inconsistent with :172) →
  `search_by_color_palette` (:462-478) does 50 fetches + 50 KMeans per request
  over only the **first 50** indexed movies while the docstring claims " the
  indexed movies".
- **C2.15 [inconsistency]** Four different similarity keys on one class:
  `visual_similarity` (:251), `visual_match_score` (:305),
  `similarity_score` (:376), `color_similarity` (:475) — callers pass results
  through, so frontends must handle all four shapes; the same endpoint can
  return CLIP results (visual keys) or semantic-fallback results (score
  keys) with no marker.
- **C2.16 [inconsistency]** `search_by_uploaded_image` defaults `is_base64=
  False` (:208) but the alias `search_by_image` defaults `True` (:318) —
  divergent defaults for aliases; plus a duplicated third image-encode path
  (:232-237).
- **C2.17 [hygiene]** `find_similar_posters` (:326+) is dead code; the
  `__main__` demo block (:483-504) constructs the class with no db_manager,
  so it can never work as shown.

### `ai/vector_search.py` (dead module, but code claims vs reality)

- **C2.18 [hygiene]** The entire module is dead in production (imported only
  by `benchmarks/run_scale_benchmark.py:30` and
  `tests/test_scale_architecture.py:21`); two parallel ANN implementations
  coexist (this and semantic_search's FAISS).
- **C2.19 [hygiene]** Module header claims ("multi-layer graph navigation",
  "IVF-PQ compaction", "Zero-copy SIMD", "Filtered ANN with metadata
  pre-filtering") are not implemented: single-layer HNSW (:22-52), PQ without
  inverted files (:155-158), post-search filtering (:127-131).
- **C2.20 [bug]** HNSW link construction is first-come-first-served (:82-83):
  closer new neighbors can be dropped for earlier farther ones (recall
  erosion); `ef_search=0` silently falls back to default (:146); any
  non-'cosine' metric string silently becomes L2 (:54-60); PQ uint8 codes
  silently wrap for `num_centroids > 256` (:197-206); k-means has no
  empty-centroid reassignment, fixed 5 iterations, and O(n·k) intermediate
  matrices (:185-193).
- **C2.21 [bug]** `ReciprocalRankFusion.fuse` (:230-252) ignores the item
  scores it documents (list order only — wrong fusion for unsorted input);
  `weights=[]` silently becomes all-ones (falsy trap); short weights list →
  bare IndexError.

### `ai/embeddings.py`

- **C2.22 [hygiene]** `logging.basicConfig(INFO)` at module import (:11)
  mutates global logging config (repeated in 8 more ai/ modules — see C-3.129).
- **C2.23 [inconsistency]** `UserEmbedding.predict_preference` (:241-256)
  ranks by raw dot product while `ContentBasedRecommender` (:427) ranks by
  cosine; the user embedding is renormalized every 10th update (:236-239) —
  and renormalization can *amplify* when norm < 1 — so the two agree only
  sometimes.
- **C2.24 [hygiene]** Default `learning_rate=Config.LEARNING_RATE` frozen at
  import (:217); `find_similar_movies`/`recommend_for_user` without explicit
  candidates search only already-computed embeddings (:379,419) — almost
  always a no-op fallback; `add_movie` deprecated-dead (:345-352);
  `predict_preference` used only by its own `__main__`; `metadata_dim`
  assigned never read (:36).
- **C2.25 [bug]** Silent truncation/zero-padding on encoder/dim mismatch
  (:56-67) — wrong-but-shaped vectors, no assertion; year anchors unbounded
  > 2030 (:81-82) while co-located constants live outside config (:73-91).
- **C2.26 [robustness/OOM]** Per-pair sklearn `cosine_similarity` for every
  candidate (:95-110 called per candidate at :386,427) — O(N) sklearn
  overhead per request; `movie_embeddings` dict grows unbounded (no
  eviction; duplicates `cache_manager` vector role — two sources of truth);
  no lock on `get_or_create_embedding` (:296-343) under threaded serving.

### `ai/mood_detector.py`

- **C2.27 [hygiene — wasted model]** `self.sentiment_analyzer` (:98-102) loads
  a full RoBERTa sentiment classifier that **no method ever reads** (only
  assignment sites :98/117/129) — double memory/download/startup for zero
  function.
- **C2.28 [robustness]** transformers model download happens lazily inside
  the first request (:143) and blocks it; once a load attempt fails (even
  transiently), `models_load_attempted=True` sticks (:91-94) — permanent
  keyword fallback until restart, with no response marker to callers.
- **C2.29 [bug]** `detect_mood` indexes `self.emotion_detector(text)[0]`
  (:150) — shape is transformers-version-specific; a version change flips
  `[0]` to grab a single dict and every call raises (silently converted to
  fallback). No pin/comment protects the contract.
- **C2.30 [bug]** Emotion→mood mapping (:160-171) adds `'anger','frustration'
  → 'bored'`: an angry user is labeled *bored* and gets Horror ("engaging")
  results; `fear → 'anxious'` gives calming genres to someone asking for scary
  — the `scared` bucket (mood_mapping :69-74) is **unreachable from the model
  branch**, as are `thoughtful` and `nostalgic`. The model label set
  (j-hartmann: anger, disgust, fear, joy, neutral, sadness, surprise) leaves
  `disgust`/`neutral` unmapped — probability mass dropped pre-normalization.
- **C2.31 [bug — DB dead-query root]** `mood_detector` genre fetches are
  dead: `db_manager.get_movies_by_genre` (:391-401) runs
  `FIND_IN_SET(%s, md.genres)` against the `movie_details` view whose
  `GROUP_CONCAT ... SEPARATOR ', '` (`schema.sql:526` — comma+space, see
  D-1.9) only matches a genre when it is the **alphabetically-first entry**
  of a movie's list (FIND_IN_SET does not trim; `' Drama'` ≠ `Drama`). Every
  genre that is not first-in-order silently mismatches. Combined with the
  mood genre constants `'Musical'`/`'Sci-Fi'` (:34,46,58,76) which are not
  TMDB canonical names (`Music`, `Science Fiction`), mood recommendations
  almost always fall through to the popularity branch (:281-296) while the
  response still claims mood-derived results.
- **C2.32 [robustness]** Double inference per request: `get_mood_
  recommendations` calls `get_primary_mood` (→ `detect_mood`) then
  `detect_mood` again for mood_scores (:247-248). `user_id` accepted and
  ignored (:228-244). Primary-genre boost reads `next(iter(set))` —
  nondeterministic (:359-360). Failure defaults are indistinguishable from
  model output (no path marker key).

### `ai/conversational_agent.py`

- **C2.33 [hygiene — dead class]** `ConversationalRecommender` is instantiated
  (app_integrated.py:262) but never used; `/api/chat` reimplements the same
  logic inline. Repo-wide grep: the only reference is the instantiation.
- **C2.34 [bug/leak]** `user_sessions` append-only (:39-41), never read,
  never trimmed, not thread-safe; `user_preferences` (:24) never read;
  the constructor's `movie_recommender` dependency accepted and discarded.
- **C2.35 [bug]** Substring intent matching: `'hi' in message` matches "this",
  "history"… (:112); `movie`/`watch` substrings classify nearly everything as
  a recommendation query (:195-198) — implementing diverges from the live
  chat's `startswith` approach.
- **C2.36 [bug]** Its "recommendation" path always calls
  `_get_recommendation_movies({}, user_id)` with empty prefs (:95) — always
  yields top movies while labeling `type: 'recommendation'` "Based on what
  you told me".
- **C2.37 [inconsistency]** Intent ordering: trending check precedes mood
  (:63-92) — "I feel sad, show me trending" discards the mood; a **third** and
  **fourth** mood-keyword table (here :46-55, and local `mood_genre_map`
  :126-135) now coexist with mood_detector's two — mutually inconsistent.
- **C2.38 [robustness]** Genre aggregation try/except wraps the whole loop
  (:139-191): a failure on genre #3 discards partial results from #1-2.

### `ai/trending_detector.py`

- **C2.39 [bug]** "Recent velocity" is not recent: bucket counts are
  cumulative-within-window; `velocities[0]` (:87 comment "Most recent") is the
  **1h-6h-ago bucket ÷ 5h**; the freshest 0-1h signal is never computed.
  Same mislabel for accelerations[0] (:79-91).
- **C2.40 [robustness]** N+1 storms: `calculate_trending_score` = 5
  queries/movie (:60-67) × 100 movies per `/api/trending` request (~500
  queries); `predict_next_viral` +2/movie; `detect_viral_outbreak` 3/movie
  over every weekly-active movie.
- **C2.41 [inconsistency]** Score scale effectively unreachable relative to
  thresholds: raw scores realistically ≪ the 80/60/40/20 indicator bands
  (:96-104 vs :176-185), given bucket velocities divided by 5/18/48/96 h;
  `TRENDING_SCORE_SCALE=10` was never re-derived against these units.
- **C2.42 [hygiene]** Mojibake emoji literals (:177-184, 408 — `"ðŸ”¥
  VIRAL"` etc.) sent verbatim in API responses; hardcoded `LIMIT 100`
  (:149) ignores the `limit` param (Python-side slice after); duplicated
  double-comment at :102-103.
- **C2.43 [inconsistency]** Four sibling metrics shapes/scales:
  `trend_indicator` (rounded, :157-158), unrounded score (:304), unbounded
  `viral_score` (:250), `predicted_viral_probability` (:365).
- **C2.44 [robustness]** `detect_viral_outbreak` requires prior activity
  (`avg_daily > 0`, :241) — a first-today burst (the viral archetype) can
  never be flagged. `SELECT DISTINCT` + `GROUP BY` redundancy (:207-212,
  :284-297). Python-local `datetime.now()` vs MySQL `CURRENT_TIMESTAMP`
  timezone skew (:54,133,203-205,282,340-342 — no TZ normalization).
- **C2.45 [robustness]** All failures return `0.0`/`[]` with log-only
  (:110-112, 168-170, 258-260, 311-313, 376-378); the app-level fallback then
  silently serves popularity — DB outage is indistinguishable from "nothing
  trending". `time_diff > 0` guard at :75 is dead (windows fixed and
  increasing). Only `chosen_movie_id` counts as interaction versus the
  docstring's "interactions" (:19).
- **C2.46 [hygiene]** `__main__` demo imports the live DB singleton (:383) —
  `python ai/trending_detector.py` becomes a side-effectful smoke test.

## C-3. Remaining ai/ modules (A/B, advanced, content pipeline, offline island)

### `ai/ab_testing.py` + `ai/ab_experimentation.py`

- **C3.1 [bug]** `ab_testing._make_recommendation` dead branch (:374-379):
  `best_variant` is only set inside a significance check, so `elif
  is_significant:` is always true and the "not yet significant" fallback is
  unreachable.
- **C3.2 [bug]** Asymmetric gating (:311): the CTR chi-square requires
  control ≥ min_sample_size but never checks **treatment** impressions — a
  1-impression treatment can be tested.
- **C3.3 [hygiene]** `t_stat` unused in both t-tests (:326,336); the class's
  `db_manager` dependency stored and never read (:101-108); `stop_experiment`
  (:381) has no callers; `track_rating` accepts floats outside [1,5] and
  `track_watch_time` negatives (:236-262).
- **C3.4 [bug]** The whole framework is dead at runtime: loaded as
  `app.ai_systems['ab_test']` in the default feature list, but **no reader**
  of that key exists anywhere (grep-verified) — the import is a no-op.
- **C3.5 [inconsistency]** The two A/B engines still assign users to
  *different variants*: `md5("user:experiment")` (:197-199) vs
  `sha256("salt:experiment:user")` (`ab_experimentation.py:34-35`) with
  different hash + key order — the scale was unified in wave 4 (E8) but the
  hash/key contract remains divergent, contradicting the "so both bucketing
  systems agree" comment.
- **C3.6 [bug]** `ab_experimentation.py:69-75` — SRM p-value formula is exact
  only for df≤2; for ≥4 variants (df≥3) `exp(-chi2/2)` overstates
  significance ~3× (χ²=13.27, df=3: true p≈0.004 vs computed 0.0013). The
  "regularized incomplete gamma" comment is unimplemented; `detect_srm`
  doesn't validate/normalize `expected_ratios` (:57-62).
- **C3.7 [hygiene]** `ab_experimentation` module only imported by
  benchmarks/tests; docstrings claim MurmurHash3 and "Viewport Exposure
  Tracking" — neither implemented.
- **C3.8 [hygiene]** Stopped experiments keep accepting metrics counter
  updates (:218-222); inactive/missing experiments return 'control' without
  caching the assignment (:192-194); `Experiment.traffic_allocation` accepts
  negative values (:33-43).

### `ai/advanced_ai.py`

- **C3.9 [bug]** `LatentSpaceEncoder` is constructed in the live path
  (`recommender.py:72`) and gates on `is_fitted` (:757-759), but **nothing
  ever calls `.fit()`** — the entire latent-space branch silently never runs.
  When not fitted, `transform` returns the raw **full-dimension** vector with
  only a log warning (:76-78).
- **C3.10 [robustness]** `LatentSpaceEncoder.fit` guards samples but not
  n_features (TruncatedSVD requires n_components < n_features) (:50-53);
  `inverse_transform` never called.
- **C3.11 [hygiene — dead chain]** `ImplicitSignalProcessor.enrich_
  interaction`/`calculate_implicit_reward` (:110,143) are unreachable — the
  configured IMPLICIT_SIGNAL_* behavior is vacuous; `Probabilistic Selector.
  get_distribution` (:197-199) never called; `get_advanced_ai_suite`
  (:395-405) never called.
- **C3.12 [bug]** `TemporalMemoryManager` implements "temporal decay" by
  **list position**, not timestamp age (:203-248); its sort key
  `x.get('timestamp', datetime.now())` (:230) fails with TypeError on
  string/None timestamps mixed with datetimes.
- **C3.13 [inconsistency]** The decay `exp(-age·0.3)` is effectively no
  forgetting by interaction ~150 — vestigial effect versus the config's
  intent (:243).
- **C3.14 [hygiene]** Malformed fallback text for empty explanations on
  'high' detail (:324-326).

### `ai/content_pipeline.py`

- **C3.15 [bug]** On-demand **TV** ingest produces blank rows: TV payloads
  (`name`/`first_air_date`) are fed to `parse_movie_data` which reads
  `title`/`original_title`/`release_date` (`tmdb/fetcher.py:228-231`) →
  `title=''`, `release_year=NULL`, `original_title=''` rows inserted with
  `media_type='tv'` (:147-149 → :287). The manual script
  (`scripts/fetch_tv_series.py:114-135`) does it correctly; the automated
  pipeline is the broken one.
- **C3.16 [bug]** `movies` table collision: `media_type='tv'` set only after
  `parse`, but `insert_movie` in `scheduled_refresh` (:186) runs before it is
  distinguished → TV and movie data can collide on `movie_id == tmdb_id`
  in the single `movies` table.
- **C3.17 [bug]** Scheduler jobs pile up: `ContentPipeline.start()` appends
  `schedule.every().day…` to the **global** schedule object (:42-44);
  stop()/start() duplicates all three jobs (double fetch frequency), and
  `stop()` does not join/signal the in-flight scheduler thread.
- **C3.18 [robustness]** The scheduler thread has no try/except in its loop
  (:60-64) — an escaping exception kills it with `is_running` stuck `True`.
- **C3.19 [robustness]** Request-path concurrency: `fetch_on_demand`
  (`api/routes.py:1272-1273`, gunicorn `--threads 4`) shares the unpinned
  `requests.Session`-holding fetcher with the scheduler thread; DB writes
  from multiple threads uncoordinated; 6× `time.sleep(0.25)` inside the
  request path (:112,127,143,150,188,221,268).
- **C3.20 [inconsistency]** `ensure_content_availability(min_movies=1000)`
  default (:248) vs initial fetch's `500` (:360); refreshed-marker logic
  refreshes by `created_at < NOW()-7d` (:166-171) — the same old rows churn
  weekly forever (nothing marks "updated").
- **C3.21 [robustness]** `_store_credits` triggers a second full
  `get_movie_details` call per movie (~500 extra API calls at startup,
  ~2 min); `parse_credits`' person `popularity` is parsed then discarded
  (:331-334) leaving `directors/actors.popularity` NULL in DB.
- **C3.22 [robustness]** `fetch_on_demand` genre/year branch fetches only
  page 1 and returns `[]` on any error (:229-246).

### `ai/data_preprocessor.py` (offline island)

- **C3.23 [bug]** Global min-max normalization computed over the **entire**
  ratings frame before the temporal split (:129-147) — test statistics leak
  into training; `rating_mean/std` are saved but the min/max are not, so
  predictions can never be denormalized correctly.
- **C3.24 [bug]** Merge suffix collision: after `_ml/_tmdb` suffixing (:98),
  no bare `genres`/`title` column exists, so `prepare_movie_content`'s
  `row.get('genres','')` (:192,209) always defaults — genre text component
  permanently empty.
- **C3.25 [hygiene]** `astype(str)` on `tmdbId` turns NaN into the literal
  `'nan'` join key (:82); `merge_datasets()` before `load_datasets()` gives
  an opaque AttributeError (`self.links_df`, :70 vs :82-90);
  `save_processed_data(output_dir=...)` default bypasses `self.cache_dir`
  (:235,268); `__main__` runs the *full 32M-rating* preprocessing by default
  (:368-371).

### `ai/diversity_engine.py`

- **C3.26 [hygiene]** `CalibratedReranker` (:77-136) is never referenced by
  any code; the module itself is only wired in tests/benchmarks.
- **C3.27 [bug/perf]** DPP greedy recomputes `[item_ids.index(it) for it in
  selected]` O(n) per element (:65) — O(k·n²) overall, and `.index()`
  zeroes the wrong slot when ids contain duplicates (:65-67).
- **C3.28 [robustness]** Diversifier silently returns < k items when all
  scores degenerate (:55-57); KL calibration defaults missing scores to 0.5
  which can beat genuinely low-scoring items (:121).

### `ai/drift_monitor.py`

- **C3.29 [hygiene]** Module only imported by tests/benchmarks; asserts used
  for validation (:32); PSI crashes on constant baselines (duplicate
  quantile edges, :36-38); docstring promises rank correlation — only
  Jaccard implemented (:95-115); thresholds 0.10/0.25/0.15 hardcoded
  outside config (:50-88).

### `ai/evaluation.py`

- **C3.30 [bug — metrics inflated]** `evaluate_ranking` (:202-269) ranks
  using `sorted(user_predictions[uid])` where predictors correspond only to
  the items in that user's test batches (not  full catalog, not sampled
  negatives) — Hit/NDCG/Precision/Recall/MAP are conditional on per-user
  candidate sets and not comparable to any catalog-wide benchmark.
- **C3.31 [bug]** `precision@k` divides by constant `k` when fewer than k
  predictions exist (:118); users with zero positives included in means
  (:244-257); mutable default `k_values=[5,10,20]` (:202,312); module used
  only by the offline chain (:23 ← training_pipeline ← scripts/train_model).
- **C3.32 [bug/latent]** numpy float64 metrics dumped via `json.dump`
  (:92,120,151,190 → `training_pipeline.py:429-434`) — TypeError on many
  numpy versions; the path likely never ran end-to-end.

### `ai/feature_store.py`

- **C3.33 [hygiene]** Module only imported by tests/benchmarks.
- **C3.34 [bug]** `timestamp or time.time()` falsy-zero bug (:67); defaultdict
  writes for arbitrary queried user ids (:34-38,68,96-99) — unbounded growth
  on scan loops; `total_lifetime_interactions` capped by `maxlen=200`
  deques (:38,103).
- **C3.35 [inconsistency]** Positive-rating threshold 4.0 here (:72) vs 3.5
  in `data_preprocessor.create_implicit_feedback` (:218-228) — two
  definitions of "positive interaction".
- **C3.36 [robustness]** `point_in_time_join` silently drops interactions
  predating any snapshot with no count/log (:123-133); re-sorted history per
  interaction (:128).

### `ai/hybrid_model.py` + `ai/two_tower_ncf.py` (offline island)

- **C3.37 [bug/perf]** `ContentEmbeddingLayer.__init__` calls
  `movie_encoder.transform()` once per movie (:135-136) — O(n²) for large
  catalogs via LabelEncoder's full class-array validation.
- **C3.38 [inconsistency]** Sigmoid-over-L2-normalized-cosine compresses
  scores into ≈[0.27, 0.73] (:316-321; two_tower_ncf.py:198-205);
  `predict_rating * 5.0` (netflix_recommender.py:270-273) then can only
  produce ≈1.3-3.7 stars while claiming a 0-5 scale.
- **C3.39 [hygiene]** `hybrid_model`/`two_tower_ncf` are offline-only;
  `two_tower_ncf.GMF/MLP/get_model/predict_batch/compute_all_movie_scores`
  (:215-387) and three near-identical tower implementations
  (UserTower/MovieTower/HybridMovie.tower) are dead or triplicated, with an
  existing inconsistency (`MLP._init_weights` omits the bias zeroing the
  towers apply) (:349-355); `assert` for dim match (:294) vanishes under -O;
  BatchNorm1d + no `drop_last` = final size-1 batch crash risk (:41-107).
- **C3.40 [inconsistency]** Only `model_type='hybrid'` is ever exercised:
  training default and `create_hybrid_model` build their own
  `[256,128,64]` and ignore user-passed `hidden_dims` (training_pipeline
  default + hybrid_model.py:416-423).

### `ai/inference_pipeline.py` (labeled "Production Inference Pipeline" but offline-only)

- **C3.41 [bug]** `ProductionRecommender` builds `CandidateGenerator` with
  `ratings_df=None` (:33-39) → `_compute_popularity` never runs and the
  "popular" channel is literally the first k DataFrame rows (:54-58).
- **C3.42 [bug]** Candidate pool truncation is set-order nondeterministic
  (:125-141) — results not reproducible across runs/platforms.
- **C3.43 [bug]** Genre-similar channel is dead on the real dataset: the
  preprocessed frame has suffix-collided columns, so the `genres` check
  (:84-85) always fires and the channel always returns [] (root cause C3.24).
- **C3.44 [inconsistency]** `DiversityReranker.max_per_genre` is not a cap —
  soft penalty only (:169-227); penalty compounds past the cap and re-sorts
  arbitrarily; `sorted(genres)[0]` makes "primary genre" alphabetical,
  not semantic (:206); two config keys for the same knob
  (`INFERENCE_MAX_PER_GENRE` vs `INFERENCE_RECOMMEND_MAX_PER_GENRE`,
  method default vs caller, :169 vs :348, plus `recommend()` re-specifies
  both :347-348).
- **C3.45 [bug]** Unknown users silently map to `user_encoder`'s **first
  class** — another real user's latent vector as seed (:256-260).
- **C3.46 [bug/perf]** `_encode_movie_ids` LabelEncoder-transforms per id
  (:267-274) — O(n·classes); `batch_recommend` is purely sequential
  (:376-386); dead null-check after construction (:450-451); `movieId`-as-
  recency acknowledged inconsistency (:69-72).

### `ai/multi_stage_ranker.py`

- **C3.47 [hygiene]** Module only imported by tests/benchmarks; docstring
  claims a "3-Stage Pipeline" with retrieval + dedup stages that don't exist
  (:6,46-49).
- **C3.48 [inconsistency]** Two different "cold" defaults in one file:
  `item_impression_count=100` (:72) vs `impression_count` fallback 50 (:141)
  with threshold 20; `total_system_impressions` default 100_000 never
  supplied by `rank_stage2` (:99) — exploration bonus inflated for small
  traffic.
- **C3.49 [inconsistency]** Platt "probability" only correct for negative
  `PLATT_CALIBRATION_A` (config -2.5); nothing validates the sign (:25-41);
  `(elo-1000)/1000` squeezes quality differentiation at high ELO (:86);
  duck-typed preference model silently scores 0.0 when the method is
  missing (:139).

### `ai/netflix_recommender.py`

- **C3.50 [bug]** History harvesting takes `chosen_movie_id or movie_1_id`
  (:106-115): half of every pairwise event (the chosen-side movie_2) is
  omitted, and *lost* movies are logged as 3.0-rated "viewed" items.
- **C3.51 [inconsistency]** `num_candidates=500` hardcoded, third duplicate of
  the pool-size constant (:87 vs `Config.RECOMMEND_CANDIDATE_POOL`,
  `Config.INFERENCE_CANDIDATE_POOL`); module's docstring claims to be "the
  main interface for the Flask app" but is importable only from
  `scripts/quick_start.py:199` (live path is ai/recommender.py).
- **C3.52 [robustness]** Broad excepts return `[]` / default-2.5 rating
  indistinguishable from real results (:94-96, 275-277); `get_similar_movies`
  similarity over an arbitrary first-1000-row slice of the catalog
  (:186-187); module-level singleton `netflix_recommender` orphaned.

### `ai/neumf_scorer.py` (partially live)

- **C3.53 [hygiene]** `ridge_coefs`/`ridge_intercept` loaded, never used
  (:251-253).
- **C3.54 [inconsistency]** V1 vs V2 regularization differ: V1 includes
  user/movie-bias terms, V2 excludes them (:76-82 vs :143-146).
- **C3.55 [robustness]** `get_scorer()` singleton not thread-safe (double
  torch-load of a 13-model ensemble possible; :383-389); on load failure the
  caller receives a live-but-unhealthy scorer (is_loaded=False, no error).
- **C3.56 [security]** `torch.load(..., weights_only=False)` on checkpoints
  (:212,230) — arbitrary-code execution on untrusted checkpoints.
- **C3.57 [inconsistency]** Hardcoded 18-genre `ML100K_GENRES`/`zeros(18)`
  (:152-166,366) vs checkpoint-configured `n_genres` — misalignment silently
  propagates into `genre_affinity_score` for differently-trained
  checkpoints; `score()` scalar detection misses floats (:303).
- **C3.58 [inconsistency]** Docstrings invite "any movie (TMDB etc.)" usage
  for CF scoring that is valid only for ML-100K indices (:7,176-179); the
  mega-weights fallback silently resets to uniform on length mismatch
  (:245-249) — live serving distribution changes without error.

### `ai/pairwise_learning.py` (live)

- **C3.59 [bug]** `PairwiseLearner.update_ratings` truncates ratings with
  `int()` each update (:71) — systematic fractional-drift loss and
  asymmetric truncation toward zero for negative deltas.
- **C3.60 [bug]** BradleyTerry update is multiplicative-linear for the
  winner and exponential for the loser (:140 vs :144) — asymmetric gradient
  drift; the rescale preserves ratios so it doesn't fix it.
- **C3.61 [bug]** `UserPreferenceModel` has no decay or bound (:191-210) —
  preferences drift unboundedly, stale tastes never fade; `learning_rate`
  shares `Config.LEARNING_RATE` with an unrelated gradient-step semantic
  (:189 vs :105).
- **C3.62 [hygiene]** `__main__` unit-test code duplicates
  `tests/test_pairwise.py` (:257-305); `get_top_movies` ties depend on dict
  insertion order (:224-239).

### `ai/training_pipeline.py`

- **C3.63 [hygiene]** Early-stopping logged twice back-to-back (:63, :388);
  duplicate late imports (`preprocess_pipeline` at :118 and :121);
  `ReduceLROnPlateau(verbose=True)` deprecated (:262-268); `history['val_
  metrics']` initialized never appended (:92-96).
- **C3.64 [bug]** Cache-miss path silently runs the full multi-hour
  preprocessing (`preprocess_pipeline(sample_size=None)` on a typo'd dir,
  :112-122).
- **C3.65 [inconsistency]** Three hardcoded path sets (`'ai/models'`, `'ai/
  cache'`) diverge from `Config.MODEL_DIR='./model'` (:78,131-133,193,465 vs
  config.py:166); `use_implicit_feedback=True` produces "RMSE" on binary
  labels presented as rating RMSE (:330-332); `hidden_dims` silently ignored
  in hybrid mode (:210-215); split pickles rewritten without atomic write
  (:130-133).

### Cross-cutting C-3 wiring map (verified by grep)

- **C3.66 [hygiene — orphan modules]** Imported **only** by tests/benchmarks
  (never by `api/`/`app_integrated`/`serving`/`inference`):
  `ab_experimentation`, `diversity_engine`, `drift_monitor`, `feature_store`,
  `multi_stage_ranker`; imported only by `scripts/quick_start.py`:
  `netflix_recommender` (→ transitively `inference_pipeline`);
  offline research island reachable only via `scripts/train_model.py`:
  `data_preprocessor`, `evaluation`, `training_pipeline`, `hybrid_model`,
  `two_tower_ncf`. The "Production Inference Pipeline" header and the Flask-
  interface claim of `netflix_recommender` are false in current code.
- **C3.67 [hygiene]** `ab_testing` is imported by app_integrated but its
  `app.ai_systems['ab_test']` entry has no consumer.
- **C3.68 [hygiene]** `logging.basicConfig` at import time in 8+ ai/ modules
  (embeddings:11, advanced_ai:15, content_pipeline:15, data_preprocessor:15,
  evaluation:11, hybrid_model:25, pairwise_learning:10, training_pipeline:25,
  two_tower_ncf:16) — global logging side effect; the first import wins.
- **C3.69 [inconsistency]** `HybridRecommender(…)` and TwoTowerNCF are the
  only two model types; `model_type` branches in training/inference are
  reachable only via hand-edited checkpoints (see C3.40, G-1.6).

---

# D. Database layer (`database/db_manager.py`, `schema.sql`, `run_migration.py`, migrations)

## D-1. Query-vs-schema verification (verified columns: OK)

- **D1.1 [verified — clean]** Every table/column referenced by db_manager SQL
  resolves in schema.sql (users, movies(+media_type, +is_persisted/
  movie_source/access_count), genres, directors, actors, linkage tables,
  user_interactions(+chosen/rejected, session_id), embeddings, keywords,
  cache_stats, watchlist(+user_rating/watched_at), movie_reviews(+helpful_
  votes), all social tables, and the views/procedures/functions used).
  Stored-procedure call-site arities all match (`record_user_interaction` 5,
  `update_user_interaction_count` 1, `update_movie_elo` 3,
  `search_movies_advanced` 4, `get_personalized_recommendations` 2,
  `get_comparison_pair` 1, `update_user_preferences` 1, `update_cache_stats` 5,
  `generate_genre_audit` 0).
- **D1.2 [hygiene]** `db_manager.py:244, 327` — the `media_type IS NULL` arms
  are dead: the column is `NOT NULL DEFAULT 'movie'` (`schema.sql:84`). The
  clause documents a NULL contract the schema forbids (harmless; distinct
  from B2.5's *view*-column issue).
- **D1.3 [verified — clean]** `order_by`/`sort_by` injection: whitelisted at
  db_manager (:237-239, :339-345) **and** route layer (`routes.py:35,
  1427-1433`, fixed dict at app_integrated.py:686-697); LIMIT/OFFSET bound
  params; `FIND_IN_SET`/`LIKE` parameterized. No injection path.
- **D1.4 [hygiene]** `search_movies` lowercases and interpolates user-supplied
  `%`/`_` wildcards into `LIKE` (parameter-safe but pathological-pattern
  DoS-able, no length clamp; `search_term.lower()` AttributeError on None)
  (:266, 297-299).
- **D1.5 [robustness]** `get_random_movies` samples `LIMIT 1 OFFSET n` per row
  from a counted total (:207-233): O(limit×total) scans and a race under
  concurrent inserts silently returns fewer rows.
- **D1.6 [inconsistency]** `insert_movie` ODKU (:159-166) does not refresh
  `release_year`/`runtime`/`watch_link` on re-insert — stale/NULL year and
  runtime persist forever once inserted; `watch_link` is never populated by
  anything (schema reserves the column; fetcher always None).
- **D1.7 [hygiene]** `FIND_IN_SET` on the aggregated `movie_details` view
  (:323,395) — full view scan per request, indexless, and semantics degrade
  per D1.9; safe but slow and fragile if a genre name ever contains a comma.
- **D1.8 [robustness — latent]** `get_cursor` failure path (:73-93): if the
  `execute` wrapper installs but the `callproc` patch raises, `executed` is
  set to None while the execute wrapper remains installed → first real query
  raises `AttributeError` on `None.append` (corrupted fallback).
- **D1.9 [bug — root cause of the genre dead-query class]** The canonical
  `movie_details`/`comprehensive_movie_view` genre strings are built with
  `SEPARATOR ', '` (comma+space) (`schema.sql:526-528, 574-576`;
  `db_manager.py:273-275`), while FIND_IN_SET consumers assume
  comma-delimited tokens (no trim): only the alphabetically-first genre of a
  movie matches. Affects `get_movies_by_genre` (mood), `get_movies_by_
  genre_and_type` consumers, and the get_personalized_recommendations SP
  branches (`schema.sql:962,1184,1196`). (An area-pass claimed "matches zero
  rows"; re-derived: matches only exactly-first genres — still wrong for the
  majority of cases.)

## D-2. Connection pool / transaction modeling

- **D2.1 [bug — CRITICAL]** `db_manager.get_connection` finally-block
  (:53-55): `if connection and connection.is_connected(): connection.close()`.
  In mysql-connector 8.2.0, `PooledMySQLConnection.close()` is what returns
  the connection to the pool. If `is_connected()` is False (dead socket after
  a MySQL blip), `close()` is skipped → the pooled slot is **leaked
  permanently**; after `pool_size=10` (:32) leaks, every request dies with
  `PoolError` forever. Should be unconditional close.
- **D2.2 [robustness]** Non-`mysql.connector.Error` exceptions skip both
  commit and rollback (:48-52; `get_cursor` :107-110 relies on
  `pool_reset_session`) — write-integrity depends on re-auth, no explicit
  rollback.
- **D2.3 [robustness]** No reconnect/health policy: no `connection_timeout`,
  no pre-reuse ping (config.py:23-31) → stale "MySQL server has gone away"
  on idle workers instead of transparent reconnect.
- **D2.4 [robustness]** Singleton `__new__` has no lock (:21-25) — two
  simultaneous first imports can build two 10-connection pools; pool init
  happens at import time (:1036) so a down MySQL crashes every import.
- **D2.5 [hygiene]** Read/write commit heuristic (:99-102) treats any
  non-SELECT as a write (WITH…SELECT misclassified benignly);
  `cursor.executemany` is untracked → future writers silently uncommitted.
- **D2.6 [robustness]** Pool exhaustion is immediate (no wait/retry): the
  11th concurrent DB request under threaded Flask 500s.
- **D2.7 [inconsistency]** Transactions are never multi-statement: e.g.
  content_pipeline's insert_movie → N×insert_genre → N×link runs in separate
  pool connections — a crash mid-way leaves a movie with partial genres;
  no `transaction=True`/manual BEGIN anywhere.
- **D2.8 [robustness, from wave-4 doc'd state]** `get_cursor` conditional
  commit & kwarg tracking were fixed in E6 — re-verified honest except
  D1.8/D2.5 edge cases above.

## D-3. Schema objects, triggers, dead surface

- **D3.1 [inconsistency]** Duplicated interaction-count maintenance:
  `after_interaction_insert` trigger increments `users.interaction_count`
  (:1362-1365) *and* `record_user_interaction` immediately recomputes it via
  `update_user_interaction_count` (:793-805, called at :864) — the trigger's
  bump is wasted work and the `% 5` preference-refresh gate (:1367-1369)
  fires depending on ordering between the two mechanisms.
- **D3.2 [inconsistency]** Duplicated genre win/loss bookkeeping:
  materialized `genre_win_counts` table (:1938-1946) vs live
  `genre_comparison_stats` view (:1771-1788); db_manager reads the **view**
  (:861-866); the table has no consumer except `generate_genre_audit`
  (:2060-2063).
- **D3.3 [hygiene]** `db_manager.update_movie_elo` (:485-488) has **zero
  callers** (the double-ELO bug removed in wave 4 E1): dead API surface that
  still bumps `comparison_count` if invoked (duplicate-comparison hazard).
- **D3.4 [inconsistency]** `track_movie_access` trigger (:2314-2322) +
  `movies.last_accessed ON UPDATE CURRENT_TIMESTAMP` (:89) — **every**
  metadata update increments `access_count` (it counts updates, not
  accesses); `persist_on_interaction` (:2304-2311) rewrites the *compared*
  movies' `movie_source='user_interaction'` on every interaction, stomping
  real source values.
- **D3.5 [hygiene]** `after_search_history_insert` sets `@last_search_id`
  (:1421-1430) — no reader anywhere; `before_movie_insert/update_validate`
  re-validate `tmdb_rating` bounds already enforced by column CHECK (:94 vs
  :1374-1407); `log_recommendation_feedback` silently coerces unknown
  feedback to 'not_interested' and drifts `movies.popularity ±0.1/0.05`
  (:1690-1703) mixing thumbs with TMDB popularity.
- **D3.6 [robustness]** `transactional_elo_update` exception path
  (:2119-2124) does `ROLLBACK TO SAVEPOINT` + RELEASE but **no full
  ROLLBACK/COMMIT** — the transaction stays open on the session (locks held
  until pooled reset), unlike `cleanup_temporary_movies`/`batch_rescore`
  which do it right.
- **D3.7 [hygiene — dead schema surface]** No Python consumer exists for:
  tables `keywords`, `semantic_embeddings`, `search_logs`, `user_moods`,
  `recommendation_feedback`, `ab_test_assignments`, `ab_test_metrics`,
  `model_versions`, `candidate_generation_log`, `recommendation_log`, and
  ~15 demo views (`elite_movies`, `power_users`, `user_network_view`,
  `search_analytics`, `user_mood_trends`, `recommendation_effectiveness`,
  `cached_movies`, `lazy_loading_stats`, …).
- **D3.8 [hygiene]** Schema's own object-count summary SELECT sits mid-file
  (:1456-1457) before the late CREATEs (13 views, 12 procs, 4 funcs, late
  triggers) — reading it as a tally undercounts.
- **D3.9 [verified — corrected truth]** Mechanical count from schema.sql:
  **36 tables / 24 views / 24 procedures / 8 functions / 11 triggers**
  (including `schema_migrations`). README's numbers are exactly right; the
  older audit's "28 tables" was stale. Docs claims of "42 tables"/a 25th
  procedure `get_semantic_search_results` were stale (docs since
  consolidated).

## D-4. `ensure_social_tables()` / migrations / `run_migration.py`

- **D4.1 [inconsistency]** `ensure_social_tables` (:938-1024) matches
  schema.sql:423-500 structurally (ENUMs, defaults, keys) but **omits
  `COLLATE=utf8mb4_unicode_ci`** — app-created tables inherit the server
  default (MySQL 8's `utf8mb4_0900_ai_ci` on many hosts), which can
  break joins/unique-key comparisons against schema-built tables. (Wave-4's
  ENUM-gap migration is consistent across schema/db_manager/migration file —
  resolved.)
- **D4.2 [inconsistency — stale third DDL]** `scripts/update_schema.py:25-99`
  still ships a divergent social DDL: `friend_requests/friendships(user_id,
  friend_id)` (canonical uses `from_user_id/to_user_id`,
  `user_id_1/user_id_2`), `watch_parties.status` missing 'pending'/'active',
  invites missing 'pending', plus a nowherelse `movie_lists` table and
  wrong-named `ab_experiments`-family tables — running it on a legacy DB
  produces tables `social_routes.py` cannot use.
- **D4.3 [bug]** `run_migration.run_all_migrations` (:270-299): on a
  **fresh DB**, if `migrations/` has ≥1 file, schema.sql is never applied and
  the ENUM ALTER fails on missing base tables → whole run fails (the
  file header documents the requirement; the runner never enforces
  "apply schema.sql when ledger empty").
- **D4.4 [inconsistency]** DDL autocommits in MySQL: the failure paths call
  `connection.rollback()` (:242-247, :117-126) believing partial statements
  are undone — they are not; half-applied migrations silently persist.
- **D4.5 [robustness]** The 'already exists'/'duplicate' skip is a substring
  match on error text (:233-236) — real "Duplicate entry" data errors are
  silently skipped and the run then commits + marks applied.
- **D4.6 [hygiene]** `mark_migration_applied` result discarded (:252) — a
  ledger failure leaves applied-but-unrecorded state (re-executed next run);
  `is_migration_applied` returns False on transient errors (:171-173) →
  re-run on retry; `rollback_migration` is a wired-nowhere stub (:313-317);
  the `main()` trailer retains stale marketing ("Memory savings ~77x",
  :343,393).

---

# E. TMDB fetcher (`tmdb/fetcher.py`)

- **E1.1 [bug]** On-demand TV → `parse_movie_data` reads `title`/
  `original_title`/`release_date` (:228-231) while TV payloads carry
  `name`/`first_air_date` → blank titles/NULL years in `movies` when the
  content pipeline ingests TV (root pair of C3.15/C3.16).
- **E1.2 [bug]** `parse_movie_data` `int(movie_json.get('release_date',
  '0000')[:4])` (:231) raises ValueError on malformed dates (guard catches
  only falsy), poisoning the whole streaming batch (no per-item try).
- **E1.3 [security]** `_make_request` logs the full exception object
  (:57) — `requests` error text includes the URL, which carries `api_key`
  (query param, :45) → **API key leaks into every failure log line**.
- **E1.4 [hygiene/robustness]** `params['api_key'] = self.api_key` mutates the
  caller's dict in place (:45) — stale keys silently reused; `_genre_map()`
  caches a possibly-empty map **for the process lifetime** on a transient
  failure (:241-250); retry budget is 1 with no backoff beyond a flat 0.25 s
  between pages (:28-36, :221,324,336,354,377) — well under TMDB guidance for
  sustained ingestion; `enrich_movie_data` re-fetches details that callers
  then fetch again (:379-381 vs content_pipeline:183-186,319).
- **E1.5 [verified — fixed]** `parse_genres` genre-blank pollution from the
  earlier audits is fixed (ids resolve via the merged movie+TV genre map;
  blanks dropped) — wave-4 W5 holds.

---

# F. Serving & inference (`serving/model_handler.py`, `inference/*`)

- **F1.1 [bug — CRITICAL, verified by execution]** State-dict key drift:
  `model/hybrid_recommender.pt` and `model/hybrid_checkpoint.pt` contain
  `user_embedding.weight` / `movie_embedding.weight` / `fusion.{0,3,6,9}.*`,
  but `training/models.py::HybridRecommender` (:96-119) defines
  `user_emb`/`movie_emb`/`fc.*`. `load_state_dict` **raises**; affected
 sites: `inference/recommend.py:94` (ProductionRecommender hybrid path
  crashes at init and the `__main__` demo dies), `training/train.py:254-255`
  (hybrid resume path), `inference/torchserve_handler.py` indirectly. The NCF
  checkpoint loads cleanly. The hybrid files were produced by a
  renamed-attribute variant and never regenerated.
- **F1.2 [bug]** `inference/torchserve_handler.py:80-81` expects a pickled
  **full nn.Module** at `<model_dir>/model.pth` (then `.eval()`), but every
  save site writes dicts (`{"model_state_dict":…}`, `training/train.py:
  313-342`, `advanced_training.py:350-379`, the Kaggle bare state_dict
  `kaggle_runner:539-541`) → `AttributeError: 'dict' object has no attribute
  'eval'` — the handler cannot load any checkpoint in the repo.
- **F1.3 [bug]** `torchserve_handler.py:61-65` — `torch.device("cuda:" +
  str(gpu_id))` produces `cuda:None` → ValueError when CUDA is visible and
  gpu_id unset (`serving/model_handler.py:113-115` does it correctly).
- **F1.4 [hygiene]** `torchserve_handler.py:352-358` — archiver instructions
  omit `--model-file` (packaged MAR lacks the handler class); its postprocess
  would crash on Deep V3 tuple outputs (:216 vs the tuple `forward`,
  `training/deep_v3.py:140`) which `serving/model_handler.py:199-201`
  handles; the exported genre vocab (20 → 23-dim content vector) cannot
  match the V2 55-dim contract — no `genre_encoder.json` exists and no code
  produces a 52-entry encoder (:115-134, :320-332); torch<2.6/2.6
  `weights_only` semantics inconsistent across load sites.
- **F1.5 [inconsistency]** Two dead serving stacks with conflicting
  contracts: `serving/model_handler.py` (Deep V3 via
  `models/kaggle_t4x2*/...best.pt`) vs `inference/torchserve_handler.py`
  (`model.pth`) — both self-declared unused by the app; the live app's
  `Config.MODEL_DIR='./model'` vs handler default `models/` (plural) vs
  `Config.CHECKPOINT_DIR='./checkpoints'` (which does not exist anywhere) —
  a three-way directory naming trap (config.py:166-167).
- **F1.6 [hygiene]** `model_handler.py` details: `self.manifest`/`map_
  location` set never read (:52,108); `KNOWN_SHAPES` relative-path-only
  matching as an implicit contract (`training/deep_v3.py:150-172`);
  postprocess clips to ML-100k 0.5-5.0 (a different rating scale than TMDB
  0-10 used elsewhere — would silently blend if ever wired via
  `DL_SCORE_WEIGHT=0.15`); confidence bands overlap at boundaries
  (:263-270); the trailing deployment instructions reference a worker-CWD
  dependent `--handler` path (:306-359).
- **F1.7 [verified — fixed]** `model_handler.py` no longer imports
  `advanced_models_v2` with a wrong-architecture fallback; both configured
  checkpoint candidates exist on disk; deep_v3 loader signature matches.
  (Wave-4 item resolved.)
- **F1.8 [hygiene]** `serving.note` — zero-filled dense lookup buffers mean
  served V3 predictions differ from training-time ones (documented;
  faithful serving would need the train-split prior matrices shipped)
  (`model_handler.py:126-133`, `deep_v3.py:144-155`).

---

# G. Training (`training/*`, `kaggle_runner/*`)

- **G1.1 [verified — fixed]** `kaggle_runner/train_kaggle_t4x2.py` is now
  import-safe: full training wrapped in `main()` with `__main__` guard
  (:798-799). But `training/deep_v3.py:3-5` still claims the Kaggle script
  "executes training on import" — stale doc.
- **G1.2 [bug]** `training/advanced_training.py::FocalLoss` retains the
  collapsed weighting that `losses.py` claims was fixed (:80-82: for rating-
  space mse<1 the gradient shrinks ≈0.2×); the *used* implementation is the
  buggy one, the *fixed* module (`training/losses.py`) is EXPERIMENTAL/unused
  (:5-11).
- **G1.3 [bug]** AMP branch drops gradient clipping and never
  `scaler.unscale_` (advanced_training.py:116-126) — GPU trains unclipped
  while CPU clips (:136); deprecated `torch.cuda.amp` API (:117,303).
- **G1.4 [bug/OOM]** `MovieLensDataset` eagerly materializes the full
  (84432×768) plot-embedding tensor **per DataLoader worker** (advanced_
  training.py:45 with num_workers=4, :267-275) — ~4×260 MB amplification;
  `training/train.py` avoids it (numpy view, num_workers=0).
- **G1.5 [bug]** BatchNorm1d with no `drop_last` in every trainer (`train.py:
  224-231`; `advanced_training.py:263-277`; kaggle) — final size-1 batch
  crash whenever rowcount ≡ 1 mod batch_size.
- **G1.6 [bug — dup F1.1 root]** `training/train.py` resume path also
  nonfunctional for hybrid (key drift); scheduler and history are NOT
  persisted on resume (:264-283) — LR-schedule patience resets and prior
  history is overwritten despite the checkpoint storing history.
- **G1.7 [bug]** `kaggle_runner:540-541` saves the **same V3 state dict**
  under both `cinesense_deep_v3_full32m_best.pt` and
  `cinesense_deep_v2_best.pt` — the shipped "v2" artifact is a V3 model
  (mislabeling trap).
- **G1.8 [bug]** Kaggle eval slices `b_true[:100000]` (user-grouped ascending
  → low-ID-user biased "representative" sample) (:614-616); candidate pool is
  an honest head-slice but only disclosed in-file (:772-782).
- **G1.9 [hygiene]** `advanced_models.py::AttentionLayer` self-attends a
  length-1 sequence — architecturally decorative for DeepNCF/AdvancedHybrid
  (:18-22,44,97-99,139,212); `advanced_models_v2.build_model` has zero
  callers with a footgun default `plot_embed_dim=384` vs production 768
  (:112-116, :334-361); `KNOWN_SHAPES` hard-duplicates shapes into a second
  module (G-1.8 cross-ref deep_v3.py:144-147).
- **G1.10 [verified — clean]** Seeds set in every active trainer; seeded
  DataLoader generators; train/eval toggling correct in all trainers; splits
  are per-user chronological everywhere with stable mergesort ordering; the
  tabular priors are built train-only. No `torch.use_deterministic_algorithms`
  / cudnn.determinism anywhere (accepted ambient nondeterminism);
  no `worker_init_fn` for num_workers=4 paths (latent nondeterminism).
- **G1.11 [hygiene]** `distributed_training.py` marked unused and the
  fallback branch is an explicit stub (:5-7, :298-303); `train_model.py`
  (`scripts/`) has `--implicit` store_true default=True (never off) and
  advertises model types whose `--sample` support doesn't exist in the
  advanced path (quick-start UX mismatch) (:30,42-43,103-115).
- **G1.12 [hygiene]** Kaggle download disables TLS verification and loads the
  whole zip into RAM (:113-116); fits Kaggle RAM, not typical laptops.

---

# H. Benchmarks (`benchmarks/*`) & result-artifact truthfulness

- **H1.1 [bug — HIGH]** `run_32m_benchmarks.py` docstring and emitted
  metadata claim "Bootstrap iterations: 500" (:8-9, :500) — **no bootstrap
  code exists in the file**; every CI-like claim in that harness's results is
  false (`run_all_benchmarks.py:134-143` does implement a real paired
  bootstrap).
- **H1.2 [bug — HIGH, verified in stored results]** The benchmark "hybrid"
  preference term is provably constant: `user_pref` contains only
  train-rated items which `recommend_all` excludes from candidates →
  `sigmoid(0)=0.5` always (`recommenders.py:191-207,247`).
  Consequences visible in `benchmark_results.json`: "Content-only" ≡ "No
  ELO" to every decimal (:139-168) and "Equal 33/34/33" ≡ "Pref-heavy
  20/60/20" (:179-250) — ablation rows that are duplicate configurations.
- **H1.3 [HIGH — stale artifacts]** `benchmarks/results/benchmark_report.md`
  and `benchmark_results.json` contain outputs the current code cannot
  produce (Warm/Speedup columns, `warm_p50_ms`, `116,000x` claims vs the
  JSON's own ~101,324 max) — README's "reproduced, not claimed" table quotes
  these stale artifacts. `ml32m_benchmark_report.md` contradicts its own
  companion JSON (0.2285 vs 0.2080) and carries sections the current code
  doesn't emit.
- **H1.4 [MEDIUM]** Three mutually incomparable ML-100K "harnesses":
  70/10/20 (`run_all_benchmarks`) vs 80/10/10 (`run_comprehensive_
  evaluation`) vs 75/10/15 on a 500K reservoir sample (`run_32m_`);
  different popularity definitions (per-user count vs per-interaction count)
  → identical-named metrics on different scales (Popularity NDCG@10 0.0939
  vs 0.0784); different seen-masking (train-only vs train∪val).
- **H1.5 [MEDIUM]** Duplicated metric suites diverge: novelty clamps 1e-10 vs
  1e-6; ILD clips cosine in one and not the other; both share an asymmetric
  zero-norm bug (i-index skips the row, j-index appends 1.0);
  `catalog_coverage` signatures differ (k-sliced vs full); ROC-AUC and
  hand-rolled Spearman ignore ties (`metrics.py` vs
  `comprehensive_metrics.py`, :44-48/:262-296, :91-92/:361-362,
  :113-119/:381-386, :66-76/:303-310, :220-222, :99-111).
- **H1.6 [MEDIUM]** `run_comprehensive_evaluation.py:456` "novelty_entropy
  @10" is mean self-information, not entropy; precision@k divides by constant
  k for short lists (both files); duplicate recommendations double-count.
- **H1.7 [MEDIUM]** `stage_retrieval_benchmark` "ELO" is a ±10 rating tally,
  not ELO; its rerank normalization differs from Part A's → its `final_ndcg`
  is not comparable, yet the same report presents both as "hybrid" (making
  two-stage look ~2.5× better than the actual recommender) (:58-63, :157-160).
- **H1.8 [MEDIUM]** `semantic_benchmark` ground truth is noisy substring
  matching ("seven" matches "Seven Years in Tibet"; "psycho" matches
  "American Psycho") inflating the recall denominator; zero-match queries
  drop from quality/latency lists but stay in `zero_result_rate`/`n_queries`
  denominators — inconsistent bases (:47-50, :113-182).
- **H1.9 [MEDIUM]** Hardcoded "measured" values:
  `diversity_benchmark.py:128-129` (`explanation_coverage=1.0`,
  `attribution_consistency=1.0` never measured — advertised as "100%
  explanation coverage" in reports) and `run_scale_benchmark.py:168`
  (`batch_mget_100_items_latency_ms: 0.08` literal).
- **H1.10 [MEDIUM]** `CineSenseUltraDeepV4` is a 4-term hand-weighted
  heuristic (disclosed in-file, not in the name); two divergent copies of the
  same model classes across the two runners with silently different
  hyperparameters (lr 0.01 vs 0.015, reg 0.05 vs 0.04, clip ranges).
- **H1.11 [LOW]** `metrics.py` "significant = CI-overlap" reported as  Yes
  (p<0.05)"; `data_loader.create_splits` accepts a `seed` it never uses;
  Gini docstring contradicts its own negative-ID drop (:339);
  prediction-scale sniffing silently switches the accuracy threshold
  (:152-156); 32M cohort leader is user-ID-biased (first 520, disclosed
  only in-file); report "Evaluation users: 943" vs per-table 906 users
  (:48/:152); scale-benchmark "recall@10" is ANN-fidelity not recommendation
  accuracy; Part-B refits top-3 rather than reusing results;
  `run_part_d2` rebuilds sets per row; the "cold-start sparse (3-10)" tier
  always yields n=0 (structure guarantee); Part-C peak-memory only traces
  10 recs; stack claims ("MySQL, Redis", "sliding-window LRU caching")
  contradict the in-file "no MySQL/Redis, plain dict".
- **H1.12 [hygiene — dead]** `comprehensive_metrics.py:399-459`
  (`evaluate_full_suite`) has no callers (both runners reimplement);
  `data_loader.py:18-23 GENRE_NAMES` unused (only GENRE_NAMES_32M is used);
  `run_comprehensive_evaluation.py:474` writes results via cwd-relative
  paths (others anchor on `__file__`).
- **H1.13 [README cross-check]** README table row "Offline torch (32M…)
  Raw RMSE 0.77 / ROC-AUC 0.8352" comes from the Kaggle artifact
  (`models/kaggle_t4x2_full32m/metrics.json`) under a *sampled protocol*;
  the parallel local artifact reports RMSE 0.8867 / AUC 0.769 — far apart,
  unannotated; README "300-candidate funnel → top 20, retrieval p95 ~6ms"
  vs `benchmark_results.json` budget-300 p95 = 5.19 ms (and two-stage
  outputs top-**10**, not 20).

---

# I. Tests (`tests/*`, `pytest.ini`)

- **I1.1 [verified]** `pytest --collect-only` collects **89 tests** with zero
  import errors; no test imports a deleted module.
- **I1.2 [MEDIUM — tautological]** `test_recommender_scoring.py:94-100`:
  "weights sum to 1.0" asserts pure arithmetic on hardcoded constants — can
  never fail, tests nothing about the code.
- **I1.3 [MEDIUM]** 23 endpoint tests self-skip on `localhost:5000` down
  (`test_features.py:27-32`, `test_endpoints.py:26-31`,
  `test_ai_comprehensive.py:28-33`) — honest skips, but **zero offline CI
  signal** for the route layer; nothing in pytest.ini separates them for
  opt-in runs (`test_endpoints.py` also defines no `test_*` functions
  historically; the live-server "7/7" README line describes at most one
  module).
- **I1.4 [LOW]** `test_ai_comprehensive.py:173-174` passes on 200/500/503 as
  long as the body contains "error" — a always-500 server would pass;
  checkpoint tests use cwd-relative paths (:229-241) and silently skip
  elsewhere; `test_dependencies.py` catches only ImportError;
  `test_scale_architecture.py` has an unseeded `np.random.randn` (:57) and a
  Barrier(20)+sleep coalescing test that can flake/hang under load (:74-104);
  ThompsonSampling test seeds the **global** RNG stream (:63-70).
- **I1.5 [LOW]** `test_benchmarks.py:100-117` isolation test checks only
  pair-disjointness, never chronological ordering (a temporal-split
  regression would pass); the cache test pokes private `cache.cache`.
- **I1.6 [verified bug, dead feature]** `verify_features.py:157` —
  `.replace('/','\\/')` mutilates every endpoint string; all 9 endpoint
  checks print ✗ but the exit code only counts feature files, so the script
  "verifies" successfully while lying; also `os.chdir` import side effect
  (:15-16).
- **I1.7 [inconsistency]** Test/README/ledger arithmetic disagrees: ledger
  "65 passed / 24 skipped" (=89) vs README "75 tests (60 in default scope)"
  vs ledger's own "83 test functions" — three different totals in the same
  repo state.

---

# J. Frontend (`templates/*`, `static/js/*`, `static/css/*`)

## J-1. Broken/lying features

- **J1.1 [bug]** `templates/detail.html:470` — reviews render
  `r.helpful_count`, the DB column is `helpful_votes` (`db_manager.py:820`,
  `schema.sql:390`) → the counter always shows 0 (grep-verified).
- **J1.2 [bug]** `templates/detail.html:467,488` — `review_title` is
  collected, POSTed, ignored by the backend (`movie_reviews` has no
  `review_title` column, `schema.sql:383-404`) and can never be displayed —
  dead field end-to-end visible to users.
- **J1.3 [bug]** `templates/detail.html:156-163,479-487` — 5-star review UI
  vs backend-validated `1..10` (`api/routes.py:1053`, schema 0-10): scale
  mismatch (wave-4 W6 fixed the >5 crash but not the scale).
- **J1.4 [bug]** `templates/detail.html:249-253,38-48,328-331` +
  `api/routes.py:796` — `/api/movie/<id>` hardcodes `explanation: None`,
  so the "Why We Recommend This" panel can never appear; the only producer
  (`/api/featured`) is never called by the frontend (`main.js:102-104` dead).
- **J1.5 [bug]** `templates/index.html:667-670,716` —
  `loadMoreInSection` dereferences a nonexistent `${sectionName}-view-all`
  element → uncaught TypeError each time the limit-30 or exhausted path
  runs, aborting further section loading.
- **J1.6 [bug]** `templates/index.html:547-555` — the "Action Shows"/
  "Thrillers" sections build `/api/movie/top-rated?order_by=popularity`
  with **no genre filter** (genre filtering lives on `/api/movie/by-genre`,
  never called) — the sections show generic popular rows; only the
  `shouldFilterMovie` label check pretends otherwise.
- **J1.7 [bug]** `templates/index.html:735-738` +
  `static/js/lazy_loading.js:21-31,384-402,137-144` —
  `LazyLoadingManager.initialize()` runs **twice** on the home page (inline
  + DOMContentLoaded) and each `startCacheMonitoring()` starts a
  `setInterval` without clearing — two 30s `/api/cache/stats` pollers;
  global on every page with no stop-on-unload (cumulative leak across
  "navigations" in this MPA).
- **J1.8 [bug]** `static/js/lazy_loading.js:228-381` — the entire
  `CacheMonitorWidget` is instantiated nowhere; it listens for an event
  only `getCacheMonitor()` dispatches, which nothing calls — broken dead
  feature shipped in base.
- **J1.9 [bug]** `templates/index.html:663-707` — scroll-loading has no
  in-flight guard: debounced 200 ms allows overlapping identical fetches
  (offset read before response), out-of-order responses can silently drop a
  batch; on an empty page the offset never advances and no hasMore flag
  exists → refetch loop.
- **J1.10 [bug]** `templates/index.html:582-591 vs 696-702` — the
  media-type and "Hide Compared" filters are applied on first paint but
  bypassed by scroll-loaded rows (`shouldFilterMovie` not called in the
  loadMore path).
- **J1.11 [bug]** `templates/index.html:269-295` — compared-movies fetch
  uses a 1500 ms timeout; the backend routinely exceeds it →
  `comparedMovieIds` stays empty, "Hide Compared" silently no-ops until
  manually retoggled.
- **J1.12 [bug]** `static/js/lazy_loading.js:101-116,388-394` — saved
  localStorage `strategy` applied verbatim without validating against
  `VALID_STRATEGIES` (`routes.py:37`) — corrupt values make every lazy call
  400 instead of falling back to 'mixed'.
- **J1.13 [bug]** `static/js/main.js:51-72` — `API.request` always sends
  `Content-Type: application/json` on GET; `getCacheStats` callers do
  `response.json()` without an ok-check, so 401/500 error bodies get cached
  as healthy-shaped stats and render "0/100 (0%)".
- **J1.14 [bug]** `templates/detail.html:402-426` — watchlist DELETE/POST
  paths lack `res.ok` checks/401 handling; on 401 the UI flips local state
  and appears to succeed; same silent-on-401 in `submitReview`,
  `markHelpful`, and `profile.html:383-392` (401 error payload rendered as
  "Your watchlist is empty").
- **J1.15 [bug]** `templates/index.html:603-605` — when personalization
  filters hide the whole `top-picks` batch, the section stays hidden with
  its skeletons just replaced inside — no user feedback.
- **J1.16 [bug]** `templates/index.html:319-375` — `initializeAISearch`
  binds to `#ai-search` ids that don't exist in the markup (dead code from
  a replaced design).
- **J1.17 [inconsistency]** `templates/chat.html:181-189` — chat JS sends
  `history` and `user_id`; `/api/chat` reads neither — conversational
  context shipped and discarded.
- **J1.18 [inconsistency]** `templates/cache_monitor.html:296-327` —
  `transformCacheStats` multiplies an already-percentage `hit_rate` by 100 →
  "4500%"; health thresholds compare percents against 0-1 → status always
  healthy. (Contrast: `/api/cache/monitor` recomputes correctly.)
- **J1.19 [inconsistency]** `templates/category.html:111-112 via
  index.html:92,138,153` — "View All" for `top-picks`/`top-rated`/`trending`
  passes pseudo-category names into `FIND_IN_SET(genre)` → **all three AI
  "View All" pages show "No movies found"** (only `action`, `thriller`,
  `all` can ever return rows).
- **J1.20 [inconsistency]** `templates/category.html:119-120` — reads
  `movie.release_date` and `movie.id` (neither exists in `movie_details`;
  works only because of the movie_id==tmdb_id ingestion keying); every
  other template uses `movie_id`/`release_year` properly.

## J-2. XSS surfaces (no escaping helper exists anywhere in the codebase)

- **J2.1 [security — highest]** `templates/detail.html:454-473` — review
  HTML built by string concat into `innerHTML` with unescaped
  `r.username`, `r.review_title`, `r.review_text` → **stored XSS** on every
  `/movie/<id>` page.
- **J2.2 [security]** `templates/friends.html:140-156,197,312,384-385` —
  unescaped `f.username` (friend lists), `r.username` (requests),
  `recommended_by`, `p.movie_title/host_name`.
- **J2.3 [security]** `templates/friends.html:130,171` — the app injects
  **error responses** into innerHTML, and `/api/social/friends/add` echoes
  the user-supplied username in the error (`social_routes.py:119`) →
  user-controlled text straight into HTML.
- **J2.4 [security]** Same zero-escaping pattern for movie titles/overviews:
  `index.html` carousel (title in <h1> :465, overview :475),
  `createHorizontalCard` (:614-643), `search.html:279-314`,
  `detail.html:354-383`, `category.html:115-153`, `main.js:156-182`,
  `chat.html:137-159` sidebar, `compare.html:265-287` (cast/directors) —
  stored-XSS-by-ingestion rather than user-authored, same unbounded class.
- **J2.5 [security — lower]** `detail.html:546-553,605-610` — share/watch
  party checkboxes interpolate unescaped friend usernames.
- **J2.6 [verified-correct counterpoint]** `chat.html:120` and `chat.html:187`
  (`textContent`/`| tojson`) are the only safe interpolations — the
  escaping pattern exists, it's just not applied anywhere else.

## J-3. Auth flow, redirect, misc

- **J3.1 [security]** `templates/login.html:64-67` — the open-redirect guard
  rejects leading `//` but not leading **`\`**: `/\evil.com` passes both
  checks and browsers normalize `\`→`/` off-site.
- **J3.2 [bug]** `templates/base.html:130-148` — `logout()` navigates only
  on `response.ok`; non-2xx gives zero feedback (a dead button).
- **J3.3 [verified — clean]** login/signup POST shapes match
  `/api/user/login|signup`; session cookie flow and context-processor state
  consistent; server-side gates on /profile /compare /friends intact.
- **J3.4 [hygiene]** `templates/features.html` doesn't extend `base.html`
  (standalone doc, duplicate head/styles); links to `/search?q=` which
  nothing reads; its demo buttons alert() raw counts; hardcoded marketing
  numbers ("17 AI Features / CUDA GPU Enabled") tie to nothing.
- **J3.5 [hygiene]** `index.html:212-220` — `#infinite-loading-indicator`
  never toggled by any JS; `404.html` hardcodes `/movie/238|155|424`;
  `500.html` uses a `javascript:` href; `base.html:122` loads the Swiper
  bundle on every page while the carousel is hand-rolled; Tailwind Play CDN
  used in production-template context; Chart.js versions differ per page
  (4.4.0 vs 3.9.1); `profile.html` canvas text uses attribute width vs CSS
  width (off-center fallback).
- **J3.6 [hygiene/UX-lies]** `compare.html:175-180` — "Skip This Pair" and
  "Haven't Seen These Movies" are the same handler with no `unseen`
  feedback type in the backend; `detail.html:529-554` "Recommend to friend"
  claims "Recorded ✓" while the backend records nothing for the friend
  (wave-4 W8 truth fix made the copy honest but the feature remains a
  fiction); `detail.html:52` `target="_blank"` without `rel="noopener"`
  and `watch_link` href unsanitized (`javascript:` URLs possible from DB).

---

# K. Scripts, deployment, dependencies, CI

- **K1.1 [MEDIUM]** Six one-off scripts execute DB side effects **at module
  top level with no `__main__` guard** (`scripts/apply_lock_fix.py:9-34`
  (drops/recreates procedures!), `check_trending_cols.py`, `verify_db.py`,
  `verify_candidates.py`, `verify_trending.py`, `scan_sql_cols.py`) —
  importing any of them opens live MySQL sessions.
- **K1.2 [MEDIUM]** DB-mutating scripts without confirmation prompts:
  `apply_lock_fix.py` (DROP PROCEDURE), `update_tv_posters.py:45-54`
  (unconditional UPDATE), `fill_all_tables.py:352-382` (massive synthetic
  fills) — contrast `update_schema.py:216` which prompts.
- **K1.3 [MEDIUM]** `scripts/update_schema.py` — see D4.2 (stale divergent
  third social DDL).
- **K1.4 [hygiene]** `scripts/quick_start.py:13-14` — `os.chdir` at import;
  three parallel "train" entry points (`scripts/train_model.py`,
  `training/train.py`, `scripts/run_pipeline.py` — plus the ai/-pipeline
  trainer); four overlapping "verify the DB" one-offs;
  `fetch_tmdb_data.py`/`fetch_tv_series.py` duplicate genre get-or-insert
  logic; `fill_all_tables.py:87` hard-requires ≥20 movies with no
  documented ordering dependency.
- **K1.5 [MEDIUM]** Procfile/render.yaml have **no migration step**: no
  `release:` command runs `database/run_migration.py`/`schema.sql` — first
  deploy relies entirely on manual bootstrapping + `ensure_social_tables()`
  at runtime; Render free-tier RAM risk from the in-`create_app()` semantic
  index build; build step doesn't provision `model/semantic_cache` artifacts
  the app needs for a fast boot.
- **K1.6 [hygiene]** requirements.txt: `torchserve`/`torch-model-archiver`
  pull heavy pinned-torch deps while unused by the app; `python-multipart`
  (FastAPI-era, unused by Flask); `shap` imported nowhere; Flask 3.0.0 /
  gunicorn 21.2.0 have known CVEs in newer releases (pinned old);
  `matplotlib` in main reqs is diagram-scripts-only; pytest itself is in
  main reqs (labelled dev section but same file).
- **K1.7 [verified — clean]** No hardcoded credentials in scripts (all from
  `.env`); `runtime.txt`/`render.yaml` versions match (3.11.0);
  Procfile/render.yaml consistently use 1 worker × 4 threads with a
  documented rationale matching the heavy `create_app()` work.
- **K1.8 [finding]** **There is no `.github/` directory** — zero CI, zero
  lint gate: nothing prevents the checkpoint-key-drift class (F1.1) or any
  route regression from re-merging; the ledger's audit-era lint discipline
  was manual.

---

# L. Docs / README cross-checks

- **L1.1 [bug — HIGH visibility]** `README.md:8-14` links to
  `docs/overview.md`, `docs/architecture.md`, `docs/methods.md`,
  `docs/database-api.md`, `docs/setup.md`, `docs/results.md`,
  `docs/AUDIT.md` — **all deleted** by the docs-consolidation commit
  (6c3735e; only `PROJECT.md`, `RESULTS.md`, `AUDIT-2026-10-19.md` exist).
  Every doc link in the README is dead.
- **L1.2 [inconsistency]** README:11 "55 `/api` endpoints" → actual 56
  (B4.1); README:33 "75 tests (60 in default scope), 7/7 live-server tests"
  → actual 89 collected / 66 offline / 23 live-server skips (I1.3/I1.7).
- **L1.3 [inconsistency]** README's "reproduced key numbers" table quotes
  stale/uncomparable artifacts (H1.3, H1.13): the ~6 ms p95, the 32M RT
  numbers (Kaggle-sampled protocol vs local 0.8867), and the benchmark NDCG
  numbers that exist identically under three divergent split protocols.
- **L1.4 [verified — correct]** Schema object counts in README (36/24/24/8/11)
  are accurate (D3.9); `RESULTS.md`/`PROJECT.md` consolidation claims were
  not independently re-verified here beyond the above.

---

# M. Architecture / process observations (cross-cutting)

- **M1.** Two of everything: two ANN stacks (FAISS vs HNSW module), two A/B
  engines (different bucket hashes), two serving handlers (different
  checkpoint contracts), three mood-keyword tables, two cache families with
  divergent thread-safety, four trending "score" shapes, four config keys
  for one reranker cap, three pool-size-500 constants, three harnesses for
  the same benchmark claim, three divergent social DDL copies (schema, +
  ensure_social_tables, + scripts/update_schema.py).
- **M2.** The live training-data loop is thinner than documented: feedback
  only reaches the model via the SP + DB warm-start replay
  (`process_user_choice` never fires — C1.1); `ab_test` never reports
  (C3.4); implicit-signal chain unreachable (C3.11); latent encoder never
  fitted (C3.9).
- **M3.** Heavy work on hot paths: CSV scans (B3.2), N+1s (C1.12, C2.40),
  0.25 s sleeps in request path (C3.19), poster download storms (C2.13),
  request-time model downloads (C2.6, C2.11, C2.28).
- **M4.** Silent-fallback layering everywhere means DB outage / model outage
  / empty data produce identical "empty but 200-ish" outputs on trending,
  mood, semantic, visual, chat, and inference paths — staggered degradation
  is by-design, but no response key ever surfaces *which* path fired
  (exceptions listed per-module above).
- **M5.** Verified-clean classes retained from this pass for honesty: no SQL
  injection; no client-supplied user_id trust; order_by double-whitelisted;
  per-user chronological split leakage-free in all active trainers; NCF
  checkpoint loads cleanly; corpus/contract checks of all蓝route fetch URLs
  against backend routes passed except those enumerated in Section J.

---

# N. Suggested fix priority (from this audit only; no code changed)

1. **F1.1** — regenerate hybrid checkpoints under current `training.models`
   (or map legacy keys).
2. **A1.1** — one-line method-name fix for `/api/cache-stats` (or delete it).
3. **J2.1–J2.6** — add a single `esc()` helper and route every injection
   through it; also J1.1/J1.2 (`helpful_count` key, drop or persist
   review_title).
4. **D2.1** — unconditional `close()` for pooled connections.
5. **E1.1/C3.15/C3.16** — TV-shape parsing branch (name/first_air_date) +
   media_type before insert.
6. **C1.1** — wire `/api/feedback` to `process_user_choice` (or delete the
   dead learning layers it implies).
7. **C1.28** — lock the cache manager (or document the single-thread
   assumption); D4.3 fresh-DB migration bootstrap.
8. **B3.3/E1.3/L1.1/J3.1** — leak & link hygiene (str(e) removal, API-key
   logging, README links, redirect `\`).
9. **H1.1–H1.5** — rerun all three harnesses with current code, drop the
   false bootstrap claims, unify metric suites or label them distinct.
10. **C2.30/C2.31** — mood mapping correctness (`scared`/`anger`/`neutral`)
    and the `SEPARATOR ', '` FIND_IN_SET contract (schema or query change —
    decide one canonical delimiter).

---

# O. Wave 5 fix campaign status (2026-10-21)

> This section is the fix-status ledger for the campaign that followed this
> audit. Five area agents (disjoint file ownership) applied fixes for the
> findings enumerated in the fix plan; a final repo-wide verification was run
> by the coordinator: **pyflakes → 0 findings** over `ai/ api/ database/ tmdb/
> training/ inference/ serving/ scripts/ kaggle_runner/ benchmarks/ tests/
> config.py app_integrated.py`, and **pytest → 64 passed / 25 skipped**
> (baseline parity; live-server tests auto-skip offline). Every agent also
> re-verified prior-wave fixes in its area against this document's text.
>
> Scope note: this campaign targeted the findings listed in the fix plan
> (the ID set below, ≈120 items). Findings **not** listed here remain
> documented above and were not addressed in this campaign.
>
> Detailed per-fix reports: `scratch/wave5_fixes_1_frontend.md`,
> `scratch/wave5_fixes_2_backend.md`, `scratch/wave5_fixes_3_ai_db.md`,
> `scratch/wave5_fixes_4_serving_docs.md`, `scratch/wave5_fixes_5_bench_tests.md`.

## Status key

FIXED = applied and verified in this campaign · FIXED (prior wave, verified) =
landed by an earlier pass of this campaign, re-verified against this audit's
text in Wave 5.

## Frontend (agent 1 — `templates/`, `static/`)

| ID | Status | Site | Fix |
|---|---|---|---|
| J1.1 | FIXED | detail.html:486-488 | Reviews render `r.helpful_votes` (real DB column) |
| J1.2 | FIXED | detail.html:154-159,482-484 | Dead `review_title` input/rendering removed (backend has no such column) |
| J1.3 | FIXED | detail.html:504-509 | 5-star UI ×2-mapped to the 1..10 backend scale on POST |
| J1.4 | FIXED | detail.html:326-333 | Explanation panel hidden unless API returns `explanation` (nothing faked) |
| J1.5 | FIXED | index.html:634-641 | `loadMoreInSection` null-guards missing `-view-all` elements |
| J1.6 | FIXED | index.html:247-258,499-507,646-657 | Action/Thrillers sections switched to `/api/movie/by-genre` (offset support verified at routes.py:930-991) |
| J1.7 | FIXED | lazy_loading.js:21-28,149-156 | Idempotent `initialize()`; interval cleared before restart |
| J1.8 | FIXED | lazy_loading.js | `CacheMonitorWidget` class + listener + export deleted (grep-verified dead) |
| J1.9 | FIXED | index.html:626-713 | Per-section in-flight + `exhausted` guards, `finally` reset, no refetch loop |
| J1.10 | FIXED | index.html:676-688 | Scroll-load now applies `shouldFilterMovie` |
| J1.11 | FIXED | index.html:288-292 | compared-movies timeout 1500→8000 ms |
| J1.12 | FIXED | lazy_loading.js:257-270 | Saved strategy validated against whitelist, falls back to `'mixed'` |
| J1.13 | FIXED | main.js:51-82; lazy_loading.js; cache_monitor.html | `response.ok` checked before parsing; error states not cached as data |
| J1.14 | FIXED | detail.html; profile.html | 401/non-OK surfaced ("Please log in") instead of silent state flips |
| J1.16 | FIXED | index.html:325-327 | Dead `initializeAISearch`/`#ai-search` block removed |
| J1.17 | FIXED | chat.html:181-190 | Unused `history`/`user_id` dropped from chat payload (backend reads neither) |
| J1.18 | FIXED | cache_monitor.html | ×100 removed (percent not re-scaled), health thresholds in percent, avg fixed |
| J1.19 | FIXED | category.html:111-123,196-236 | Order-based categories fetch `/api/movie/top-rated?order_by=<server>` instead of pseudo-genre FIND_IN_SET |
| J1.20 | FIXED | category.html:132-135 | `release_year` + `movie_id` used (fields exist); `tmdb_id` last-resort |
| J2.1–J2.5 | FIXED | detail/friends/index/search/category/chat/compare/profile + main.js | Shared `esc()` (base.html + local copies) applied to every enumerated innerHTML surface incl. backend error-echo (friends.html:130,171) |
| J3.1 | FIXED | login.html:63-69 | Redirect guard: single-`/` paths only; backslash rejected |
| J3.2 | FIXED | base.html:152-161 | `logout()` shows error on non-2xx instead of failing silently |
| — | FIXED | detail.html:52 | Play link `rel="noopener noreferrer"` |
| Adjacent | FIXED | category.html:289-297 | Dead `#languageFilter` reference guarded; stale cache-widget container removed |

## Backend app + API (agent 2 — `app_integrated.py`, `api/*`, `config.py`)

| ID | Status | Site | Fix |
|---|---|---|---|
| A1.1 | FIXED | app_integrated.py:665-670 | `/api/cache-stats` calls `app.cache.get_stats()` (no more always-500) |
| A1.2 | FIXED | app_integrated.py:510-516 | Mood-recommendation fallback payload carries `"fallback": true` |
| A1.3 | FIXED | app_integrated.py:375-482 | `/api/chat` body wrapped: JSON 500 instead of HTML error page |
| A1.4 | FIXED | app_integrated.py:375-379 | Dead `load_ai_models('chatbot')` removed |
| A2.1 | FIXED | app_integrated.py:353-366 | Content pipeline skipped in reloader parent (`WERKZEUG_RUN_MAIN` gate) |
| A2.2 | FIXED | app_integrated.py:210-220 | `CORS_ORIGINS` env (comma-separated; default `'*'` preserved), scoped to `/api/*` |
| A2.3 | FIXED | app_integrated.py:841-847 | `HOST` env respected |
| A2.4 | FIXED | app_integrated.py:199-201 | Misleading `SESSION_TYPE='filesystem'` removed with explanatory comment |
| A2.5 | FIXED | app_integrated.py:206-208; config.py:15-18 | `app.json.sort_keys = False`; `DEBUG = FLASK_DEBUG=='1' or FLASK_ENV=='development'` |
| A2.6 | FIXED | app_integrated.py:253-256,273 | Model loads serialized under `_ai_models_lock` |
| A2.8 | FIXED | app_integrated.py:706-713 | Duplicate `logged_in`/`username` render kwargs dropped (context processor supplies them) |
| C3.4 | FIXED | app_integrated.py:258-268,280-298 | Dead `ab_test` feature + load branch removed |
| B1.1 | FIXED | social_routes.py:60-84,810-817 | `_can_edit_list` = owner/editor only; `added_by` re-attribution dropped from ODKU |
| B1.2 | FIXED | social_routes.py:196-206,313-327 | `u.email` removed from friend projections (no consumers verified) |
| B1.3 | FIXED | routes.py:50-56,785-801 | `/api/feedback` anti-replay: 60 s duplicate window → 409, opportunistic purge |
| B1.4 | FIXED | routes.py:44-48,1176-1180 | Helpful-vote cap no longer clears; 429 instead of reopening votes |
| B1.5 | FIXED | routes.py:6,247-254 | Signup `IntegrityError` → 409 "Username or email already exists" |
| B2.2 | FIXED | routes.py:31,80-92 | `MAX_OFFSET` 10000 clamp |
| B2.3 | FIXED | routes.py:95-106,1026-1034,1124-1139 | Non-integer movie ids → 400 (URL-converted ids 404 before handlers) |
| B2.6 | FIXED | routes.py:1071-1078 + db_manager.py:806-829 | `user_rating` persisted for any watchlist status (route passes through; db writes unconditionally) |
| B2.7 | FIXED | routes.py:1146-1160 | Review upsert distinguishes insert (201 "added") vs update (200 "updated") |
| B3.3 | FIXED | routes.py:852-859,1516-1520,1622-1626 | `'details': str(e)` removed from all 500 bodies |
| B3.6 | FIXED | social_routes.py:237-259 | `/friends/accept` IntegrityError → 200 "Already friends" |
| B3.7 | FIXED | social_routes.py:159-174 | Rejected friend requests are sticky → 409 |
| B3.8 | FIXED | social_routes.py:606-625 | Watch-party join excludes declined invites → 409 |
| B3.9 | FIXED | social_routes.py:279-298 | `/friends/remove` 0-affected → 404 |
| C1.1 | FIXED | routes.py:803-852 | `/api/feedback` calls `recommender.process_user_choice(...)` first (records internally on success), `db.record_interaction` only as fallback — **no double write** |

## AI / DB / TMDB / env (agent 3 — `ai/`, `database/`, `tmdb/`, `.env.example`)

| ID | Status | Site | Fix |
|---|---|---|---|
| C1.2 | FIXED | recommender.py:1189-1194 | Comma-separated genre strings split+stripped before encoder loop |
| C1.4 | FIXED | recommender.py:1310-1320 | Substring/word-boundary keyword scores made mutually exclusive (no double-count) |
| C1.5 | FIXED | mood_detector.py:31-37,45-58,75-79 | Genre constants canonicalized (`Music`, `Science Fiction`) — 'sci-fi'/'Musical' dead keys removed |
| C1.7 | FIXED | recommender.py:44-67,323-341 | `_movie_to_embedding` None-tolerant (`_to_float`/`_to_int`) |
| C1.8 | FIXED | recommender.py:50-63,1048-1090 | `_movie_key` tuple fallback (tmdb_id, title) for id-less rows |
| C1.28 | FIXED | cache_manager.py (8 sites) | RLock around all get/put/evict/stats mutations |
| C1.29 | FIXED | cache_manager.py:66-104 | put() refreshes LRU/last_access/access_count on overwrite |
| C1.30 | FIXED | cache_manager.py:340-344 | `total_access_count` counts real accesses (get-hits + overwrites) |
| C1.31 | FIXED | cache_manager.py:377-389; redis_cache.py:407-421 | Flat `movie_max_size`/`vector_max_size` + `size`/`evictions` — payload matches the `/api/cache-stats` handler |
| C1.32 | FIXED | xfetch_cache.py:165-180 | Early-refresh without `compute_fn` returns the fresh value (no longer discarded) |
| C1.33 | FIXED | xfetch_cache.py:83-103 | Singleflight records leader exceptions; waiters re-raise (no None-consumption) |
| C1.34 | FIXED | xfetch_cache.py:76-95 | Waiters honor deadlines; fixed the no-deadline short-circuit regression (validated by `test_singleflight_coalescing`) |
| C1.36 | FIXED | resilience.py:176-200 | Idempotency check+store+exec under ONE lock (at-most-once) |
| C1.37 | FIXED | resilience.py:58-77,100-107 | OPEN-state fallback runs outside the breaker lock |
| C1.38 | FIXED | resilience.py:50,62-66,80-85,105-122 | HALF_OPEN single-probe; failure_count reset at OPEN→HALF_OPEN |
| C1.40 | FIXED | redis_cache.py:48-51,459-466 | `default_ttl` always set; `get_cache_stats` alias for `/api/cache-stats` |
| C1.41 | FIXED | redis_cache.py:492-531 | kwargs warning memoized (once per process) |
| C1.42 | FIXED | redis_cache.py:327-330,447-448 | All JSON set paths use `default=str` (Decimal/datetime survive) |
| C1.43 | FIXED | redis_cache.py:102-110 | Outage flag resets on successful `ping()` |
| C2.29 | FIXED | mood_detector.py:147-158 | Transformers return-shape guarded before `[0]` indexing |
| C2.30 | FIXED | mood_detector.py:165-194 | fear→scared, anger→excited, disgust→sad, neutral→thoughtful; no probability-mass loss |
| C2.31 + D1.9 | FIXED | db_manager.py:372-377,447; schema.sql:962,1184,1196,1660 | `FIND_IN_SET(%s, REPLACE(genres, ', ', ','))` in Python AND stored procedures |
| C2.39 | FIXED | trending_detector.py:68-104 | Cumulative interaction windows converted to per-hour rates; velocity = last-1h rate, recent band acceleration correct (was broken consecutive-bucket math) |
| C3.15 + E1.1 | FIXED | tmdb/fetcher.py:240-256 | TV payloads normalized (name→title, first_air_date→release_date) before parse |
| C3.16 | FIXED | content_pipeline.py:325-334; db_manager.py:151-184 | `media_type('tv')` persisted through `insert_movie` |
| C3.17 | FIXED | content_pipeline.py:33-105 | Duplicate-start rejection; scheduler jobs cancelled on stop/restart |
| C3.18 | FIXED | content_pipeline.py:90-95,293-301 | Scheduler/loop bodies try/except-log-continue (one bad fetch no longer kills the run) |
| C3.22 | FIXED | content_pipeline.py:271-314 | Multi-page discovery walk with 5-page cap and per-page error handling |
| C3.68 | FIXED | 18 files | ALL import-time `logging.basicConfig` removed (grep found 18, more than the audit's 8) |
| D1.8 | FIXED | db_manager.py:86-140 | Cursor patch atomic under lock; partial failure un-applies |
| D2.1 | FIXED | db_manager.py:60-72 | Pooled `close()` unconditional (slot always returned) |
| D2.5 | FIXED | db_manager.py:98-136,993-1003 | `executemany` tracked by commit heuristic + honest `execute_many()` rowcount |
| B2.6 (db) | FIXED | db_manager.py:806-829 | `update_watchlist_status` persists rating for any status |
| D4.3 | FIXED | run_migration.py:306-322,370-377 | Empty DB: schema.sql applied before migrations |
| D4.5 | FIXED | run_migration.py:23-31,287-302 | Skip restricted to genuine already-exists errnos |
| E1.2 | FIXED | tmdb/fetcher.py:228-239,269 | Malformed dates → None year (no ValueError poisoning) |
| E1.3 | FIXED | tmdb/fetcher.py:54-75 | `_redact()` masks api_key in every log |
| E1.4 | FIXED | tmdb/fetcher.py:46-50,220-223,336-340 | params copied (no caller mutation); transient empty genre map not cached |
| A3.4 | FIXED | .env.example:50-63 | `ENABLE_CROSS_ENCODER`, `CROSS_ENCODER_LOCAL_ONLY`, `TMDB_POSTER_BASE_URL` (+ HOST/CORS_ORIGINS/FLASK_DEBUG comments) |
| — | DONE | schema.sql | Splitter validation: **227 statements**, all stored-program bodies balanced |
| — | FIXED | cache_manager.py | Singleflight regression in wave-5 C1.34 code reproduced and fixed (repo's own test passes again) |

## Serving / training / docs / CI (agent 4 — `training/`, `inference/`, `serving/`, `kaggle_runner/`, `scripts/`, README, requirements, `.github/`)

**F1.1 verification (executed this campaign):** both
`model/hybrid_checkpoint.pt` and `model/hybrid_recommender.pt` load and
forward-pass: remap `user_embedding.*→user_emb.*`,
`movie_embedding.*→movie_emb.*`, `fusion.{0,3,6,9}→fc.{0,4,8,12}`;
missing = 12 keys, all BatchNorm (`fc.{1,5,9}.*`); unexpected = `[]`.

| ID | Status | Site | Fix |
|---|---|---|---|
| F1.1 | FIXED + HARDENED | training/models.py:150-330 | `load_hybrid_checkpoint` with unwrap, dim detection, remap, strict=False + **RuntimeError unless missing ⊆ BN params and unexpected == []**; 3 call sites updated to the single-return contract |
| F1.2 | FIXED (prior wave, verified) | torchserve_handler.py:82-136 | Handler loads state dicts and routes hybrids through `load_hybrid_checkpoint` |
| F1.3 | FIXED (prior wave, verified) | torchserve_handler.py:61-68; model_handler.py:113-115 | `gpu_id` None/-1 → CPU, not `cuda:None` ValueError |
| F1.4 | FIXED (prior wave, verified) | torchserve_handler.py:407-419; model_handler.py:306-317 | Archiver instructions include `--model-file` |
| G1.1 | FIXED (prior wave, verified) | deep_v3.py:1-11 | Stale "executes on import" claim removed |
| G1.7 | FIXED (prior wave, verified) | models/README.md:39 | "v2" artifact marked as a V3 duplicate |
| G1.12 | FIXED | kaggle_runner/train_kaggle_t4x2.py:114; benchmarks/data_loader.py:56-63 | `ssl._create_unverified_context()` removed at both sites; TLS failures now fail loudly (set REQUESTS_CA_BUNDLE/SSL_CERT_FILE instead) |
| K1.1 | FIXED (prior wave, verified) | 6 scripts/*.py | `__main__` guards; import opens no connection |
| K1.3 | FIXED (prior wave, verified) | scripts/update_schema.py | DEPRECATED banner; DDL path disabled entirely |
| K1.4 | FIXED (prior wave, verified) | scripts/quick_start.py | `os.chdir` moved into `main()` |
| K1.5 | FIXED (prior wave, verified) | render.yaml:8-15 | `preDeployCommand: python database/run_migration.py` (+ free-tier caveat comment) |
| K1.6 | FIXED | requirements.txt:54-73 | 5 optional deps (torchserve, torch-model-archiver, shap, python-multipart, matplotlib) actually commented out |
| K1.8 | FIXED | .github/workflows/ci.yml (new) | py3.11, pip cache, pyflakes on 7 dirs, `pytest tests -q` |
| L1.1 | FIXED (prior wave, verified) | README.md:6-12 | All doc links point to existing files (glob-verified): docs/PROJECT.md, docs/RESULTS.md, docs/AUDIT-2026-10-19.md, docs/AUDIT-2026-10-19-independent-recheck.md, models/README.md |
| L1.2 | FIXED (prior wave, verified) | README.md:31,38-40 | Honest counts: 69 URL rules (56 /api + 13 pages); 89 tests (66 offline scope; 23 live-server self-skip; totals include optional-dep skips in some environments) |
| L1.3 | FIXED (prior wave, verified) | README.md:23-33 | Every benchmark row protocol-annotated |
| — | FIXED | kaggle_runner/train_kaggle_t4x2.py:787 | Metric key follows the H1.6 rename → `novelty_self_information@10` |
| — | FIXED | ai/trending_detector.py | Unused `numpy` import dropped (holds CI lint green) |

## Benchmarks / tests (agent 5 — `benchmarks/`, `tests/`)

| ID | Status | Site | Fix |
|---|---|---|---|
| H1.1 | FIXED | run_32m_benchmarks.py:52,707-742 | Real paired per-user bootstrap implemented (`paired_bootstrap_ci`, n=500); claims now match code |
| H1.2 | FIXED | recommenders.py:205-283; run_all_benchmarks.py | Constant preference labeled, weight kept; `preference_active` flag recorded per row and disclosed in report footnotes |
| H1.3 | FIXED | benchmark_report.md:3-5; ml32m_benchmark_report.md:3-7 | Stale-artifact banners added; numbers untouched |
| H1.4 + H1.7 | FIXED | runner docstrings ×3 + report banners | Each protocol stated (70/10/20, 80/10/10, 75/10/15-reservoir) with cross-protocol non-comparability disclosure |
| H1.5 | FIXED | metrics.py:91-127; comprehensive_metrics.py:356-397 | Novelty clamp unified 1e-10; ILD zero-norm symmetric (0.0) |
| H1.6 | FIXED | run_comprehensive_evaluation.py; comprehensive_metrics.py | `novelty_entropy@10` → `novelty_self_information@10` everywhere; latent KeyError in the writer fixed; kaggle_runner site updated by the coordinator |
| H1.8 | FIXED | semantic_benchmark.py:23-33,130-204 | Denominators unified to `executed_queries` |
| H1.9 | FIXED | diversity/scale/run_all benchmarks | Fabricated `explanation_coverage`/`attribution_consistency` removed; 0.08 constant renamed to `batch_mget_100_items_target_ms` |
| I1.2 | FIXED | test_recommender_scoring.py:97-107 | Tautology removed: weights read from live `config.Config` |
| I1.3 | FIXED | tests/conftest.py (new) + 3 live modules | `live` marker registered; `pytest -m "not live"` works with zero unknown-mark warnings |
| I1.4 | FIXED | test_scale_architecture.py:56-60 | Seeded `RandomState(0)` replaces unseeded `randn` |
| I1.5 | FIXED | test_benchmarks.py:118-135 | Chronological assertions: max(train ts) < min(val ts) < min(test ts) per user |
| I1.6 | FIXED | verify_features.py | Endpoint check corrected (quoted paths + blueprint url_prefix); failures count into exit code; `os.chdir` inside `main()` |

## Verification summary (coordinator, final state)

- `pyflakes` over all owned dirs + root modules: **0 findings**.
- `pytest tests -q`: **64 passed / 25 skipped**; `pytest -m "not live"`:
  64 passed / 23 deselected (marker filtering functional).
- `tests/verify_features.py`: all endpoint checks ✓, exit 0.
- Legacy hybrid checkpoints: load OK + forward smoke OK for both artifacts
  (missing = BN-only, unexpected = []).
- `database/schema.sql` statement-split validation: 227 statements, balanced
  stored-program bodies.
- Benchmarks were NOT rerun end-to-end (per campaign scope): headline numbers
  in `benchmarks/results/*.md` remain stale-artifact-banded and require a
  regeneration run to become current.

*End of Wave 5 fix-campaign ledger.*
