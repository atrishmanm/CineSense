# CineSense — Full Codebase Audit (500+ issues)

Method: 5 area audits (subagents, file:line verified) + machine scan
(`scripts/gen_audit.py` → `scripts/audit_items.txt`, 192 items).
Sev: [bug] wrong behavior · [inconsistency] same concept differs ·
[robustness] failure hidden/unhandled · [hygiene] dead code/prints.
Status FIXED = repaired this session. Machine-listed lines (bare-except,
empty returns) are NOT repeated here — see `scripts/audit_items.txt`.

Total: **651 findings** = 459 curated (A197 + B78 + C70 + D64 + E50)
+ 192 machine-verified in `scripts/audit_items.txt`.

## A. ai/ (186)

### A(a) exception swallowing (69)
A001 [bug] ai/ab_testing.py:321 — bare `except:`, no log; chi2 errors vanish
A002 [robustness] ai/training_pipeline.py:118 — bare `except` hides I/O/pickle corruption
A003 [hygiene] ai/redis_cache.py:99 — bare `except` hides auth/OOM; narrow to RedisError
A004 [bug] ai/inference_pipeline.py:257 — unknown user silently mapped to user 0
A005 [robustness] ai/inference_pipeline.py:269 — unknown movies dropped, no count
A006 [bug] ai/inference_pipeline.py:464 — returns None, caller crashes on None.recommend
A007 [robustness] ai/trending_detector.py:107 — hides SQL errors, 0.0 filters movie
A008 [robustness] ai/trending_detector.py:165 — log+return [] hides root cause
A009 [robustness] ai/trending_detector.py:255 — same
A010 [robustness] ai/trending_detector.py:308 — same
A011 [robustness] ai/trending_detector.py:370 — same
A012 [robustness] ai/netflix_recommender.py:54 — warn-only, returns []
A013 [robustness] ai/netflix_recommender.py:96 — same
A014 [robustness] ai/netflix_recommender.py:116 — masks missing get_user_ratings as cold user
A015 [robustness] ai/netflix_recommender.py:220 — same
A016 [robustness] ai/netflix_recommender.py:265 — 2.5 default hides model failure
A017 [robustness] ai/recommender.py:105 — DL init failure only warned
A018 [robustness] ai/recommender.py:157 — fallback hides error
A019 [robustness] ai/recommender.py:313 — same
A020 [robustness] ai/recommender.py:358 — same
A021 [bug] ai/recommender.py:444 — per-movie DL crash zeroed inside loop, invisible
A022 [robustness] ai/recommender.py:482 — ranking crash falls back to top movies
A023 [robustness] ai/recommender.py:621 — same pattern
A024 [robustness] ai/recommender.py:724 — double fallback chain hides pipeline crash
A025 [robustness] ai/recommender.py:881 — same
A026 [robustness] ai/recommender.py:964 — raw cache slice returned without scores
A027 [robustness] ai/recommender.py:995 — preference-fetch failure only logged
A028 [robustness] ai/recommender.py:1016 — same
A029 [robustness] ai/recommender.py:1051 — same
A030 [robustness] ai/recommender.py:1183 — same
A031 [robustness] ai/candidate_generator.py:140 — continue hides error
A032 [robustness] ai/candidate_generator.py:169 — same
A033 [robustness] ai/candidate_generator.py:223 — same
A034 [robustness] ai/candidate_generator.py:267 — [] conflates no-history with broken SQL
A035 [robustness] ai/candidate_generator.py:332 — same
A036 [robustness] ai/mood_detector.py:112 — model-load failure only logged
A037 [robustness] ai/mood_detector.py:128 — same
A038 [robustness] ai/mood_detector.py:181 — transformer error hidden by fallback
A039 [robustness] ai/mood_detector.py:296 — DB outage hidden, fallback list served
A040 [robustness] ai/visual_search.py:66 — model=None on any error
A041 [robustness] ai/visual_search.py:80 — same
A042 [robustness] ai/visual_search.py:117 — same
A043 [robustness] ai/visual_search.py:146 — debug-only log, per-poster fails invisible
A044 [robustness] ai/visual_search.py:249 — return [] hides CLIP failure
A045 [robustness] ai/visual_search.py:303 — same
A046 [robustness] ai/visual_search.py:373 — same
A047 [robustness] ai/visual_search.py:412 — same
A048 [robustness] ai/semantic_search.py:243 — cross-encoder outage only warned
A049 [robustness] ai/semantic_search.py:382 — faiss=None, silently slower path
A050 [robustness] ai/semantic_search.py:398 — same
A051 [robustness] ai/semantic_search.py:416 — same
A052 [robustness] ai/semantic_search.py:438 — same
A053 [robustness] ai/conversational_agent.py:148 — DB failure hidden
A054 [robustness] ai/conversational_agent.py:157 — same
A055 [robustness] ai/conversational_agent.py:172 — same
A056 [robustness] ai/content_pipeline.py:85 — log-only
A057 [robustness] ai/content_pipeline.py:114 — same
A058 [robustness] ai/content_pipeline.py:130 — same
A059 [robustness] ai/content_pipeline.py:153 — same
A060 [robustness] ai/content_pipeline.py:190 — same
A061 [robustness] ai/content_pipeline.py:194 — same
A062 [robustness] ai/content_pipeline.py:244 — fetch outage returns []
A063 [robustness] ai/content_pipeline.py:274 — same
A064 [robustness] ai/content_pipeline.py:305 — per-movie parse failure skipped
A065 [robustness] ai/content_pipeline.py:344 — same
A066 [robustness] ai/content_pipeline.py:359 — same
A067 [bug] ai/xfetch_cache.py:57 — Singleflight waiter race can raise KeyError
A068 [bug] ai/resilience.py:69 — same race pattern
A069 [hygiene] ai/_archive/ (6 files) — dead, imported nowhere, still noisy

### A(b) SQL/schema (9)
A070 [bug FIXED] ai/trending_detector.py:269 — `genres.name`, column is `genre_name`
A071 [robustness] ai/trending_detector.py:133 — ONLY_FULL_GROUP_BY risk, GROUP BY movie_id only
A072 [robustness] ai/trending_detector.py:281 — same pattern
A073 [bug] ai/recommender.py:1075 — media_type filter on view without that column
A074 [bug] ai/visual_search.py:111 — same broken filter
A075 [bug] database/db_manager.py:199 — media_type filter, column doesn't exist
A076 [bug FIXED] ai/candidate_generator.py:262 — db.execute_query never existed
A077 [bug] ai/content_pipeline.py:328 — link_movie_director passed tmdb_id as movie_id
A078 [bug] ai/content_pipeline.py:337 — link_movie_actor same

### A(c) failure-masking returns (8)
A079 [inconsistency] ai/netflix_recommender.py:236 — predict 2.5 magic default
A080 [inconsistency] ai/netflix_recommender.py:244 — same
A081 [inconsistency] ai/netflix_recommender.py:267 — same
A082 [robustness] ai/reinforcement.py:44 — select_arm [] on empty arms
A083 [bug] ai/recommender.py:348 — np.random.choice([]) crashes on above
A084 [robustness] ai/recommender.py:339 — get_comparison_pair None,None
A085 [robustness] ai/recommender.py:867 — lazy variant same
A086 [inconsistency] ai/redis_cache.py — get_* None conflates miss with outage
A087 [bug] ai/advanced_ai.py:360 — IndexError on empty genres list

### A(d) hardcoded duplicates of config.py (71)
A088 [inconsistency] ai/cache_manager.py:25 — max_size=100 (MOVIE_CACHE_SIZE)
A089 [inconsistency] ai/cache_manager.py:103 — threshold=0.3 (CACHE_REFILL_THRESHOLD)
A090 [inconsistency] ai/cache_manager.py:151 — max_size=500 (VECTOR_CACHE_SIZE)
A091 [inconsistency] ai/cache_manager.py:209 — 100/500 literals
A092 [inconsistency] ai/cache_manager.py:239 — 0.3 literal
A093 [inconsistency] ai/candidate_generator.py:39 — 300/mixed (CANDIDATE_COUNT)
A094 [inconsistency] ai/candidate_generator.py:58 — funnel 0.4 split
A095 [inconsistency] ai/candidate_generator.py:65 — funnel 0.3 split
A096 [inconsistency] ai/candidate_generator.py:72 — funnel 0.2 split
A097 [inconsistency] ai/candidate_generator.py:80 — funnel 0.1 split
A098 [inconsistency] ai/candidate_generator.py:130 — vote_count_gte=100
A099 [inconsistency] ai/candidate_generator.py:153 — //20 (MOVIES_PER_PAGE)
A100 [inconsistency] ai/candidate_generator.py:155 — max 6 vs MAX_PAGES=10
A101 [inconsistency] ai/candidate_generator.py:297 — count=30 (PAIRWISE_BATCH)
A102 [inconsistency] ai/candidate_generator.py:280 — hardcoded TMDB genre-id list
A103 [inconsistency] ai/recommender.py:37 — k_factor=32, no config
A104 [inconsistency] ai/recommender.py:53 — UCBBandit(c=2.0) literal
A105 [inconsistency] ai/recommender.py:267 — k_factor=32 duplicate
A106 [inconsistency] ai/recommender.py:331 — limit=50
A107 [inconsistency] ai/recommender.py:400 — limit=500
A108 [inconsistency] ai/recommender.py:404 — weight 0.50 literal
A109 [inconsistency] ai/recommender.py:407 — weight 0.30 literal
A110 [inconsistency] ai/recommender.py:410 — weight 0.20 literal
A111 [inconsistency] ai/recommender.py:999 — //20 literal
A112 [inconsistency] ai/recommender.py:845 — threshold=0.3 literal
A113 [inconsistency] ai/reinforcement.py:20 — exploration 0.2 (EXPLORATION_RATE)
A114 [inconsistency] ai/reinforcement.py:31 — top_k=2 (COMPARISON_BATCH)
A115 [inconsistency] ai/reinforcement.py:154 — top_k=2 duplicate
A116 [inconsistency] ai/reinforcement.py:230 — top_k=2 duplicate
A117 [inconsistency] ai/reinforcement.py:334 — top_k=2 duplicate
A118 [inconsistency] ai/reinforcement.py:116 — c=2.0 literal
A119 [inconsistency] ai/reinforcement.py:303 — lr=0.1 (LEARNING_RATE)
A120 [inconsistency] ai/reinforcement.py:334 — epsilon=0.1 vs 0.2, two epsilons
A121 [inconsistency] ai/pairwise_learning.py:20 — k=32 literal
A122 [inconsistency] ai/pairwise_learning.py:20 — initial 1500 literal
A123 [inconsistency] ai/pairwise_learning.py:105 — lr=0.1 literal
A124 [inconsistency] ai/pairwise_learning.py:241 — min_comparisons=5 literal
A125 [inconsistency] ai/embeddings.py:208 — lr=0.1 literal
A126 [inconsistency] ai/embeddings.py:64 — metadata divisors undocumented
A127 [inconsistency] ai/embeddings.py:82 — caps undocumented
A128 [inconsistency] ai/advanced_ai.py:115 — /5.0 literal
A129 [inconsistency] ai/advanced_ai.py:146 — *0.3 literal
A130 [inconsistency] ai/advanced_ai.py:305 — >0.8 threshold
A131 [inconsistency] ai/advanced_ai.py:332 — <5 threshold
A132 [inconsistency] ai/advanced_ai.py:334 — <20 threshold
A133 [inconsistency] ai/advanced_ai.py:366 — year/rating cutoffs
A134 [inconsistency] ai/trending_detector.py:31 — windows [1,6,24,72,168]
A135 [inconsistency] ai/trending_detector.py:37 — 168 default
A136 [inconsistency] ai/trending_detector.py:94 — weights 0.5/0.2/0.3
A137 [inconsistency] ai/trending_detector.py:101 — *10 clamp
A138 [inconsistency] ai/trending_detector.py:112 — limit 20/min 10
A139 [inconsistency] ai/trending_detector.py:329 — 20..60 band
A140 [inconsistency] ai/trending_detector.py:358 — >0.3 threshold
A141 [inconsistency] ai/multi_stage_ranker.py:26 — Platt a/b literals
A142 [inconsistency] ai/multi_stage_ranker.py:46 — 0.50 literal
A143 [inconsistency] ai/multi_stage_ranker.py:47 — 0.30 literal
A144 [inconsistency] ai/multi_stage_ranker.py:48 — 0.20 literal
A145 [inconsistency] ai/multi_stage_ranker.py:78 — -5.0 literal
A146 [inconsistency] ai/multi_stage_ranker.py:81 — (elo-1000)/1000 uncentralized
A147 [inconsistency] ai/multi_stage_ranker.py:91 — <20/0.15/min0.20
A148 [inconsistency] ai/feature_store.py:26 — decay 0.10 vs TEMPORAL 0.7
A149 [inconsistency] ai/feature_store.py:39 — elo 1500 literal
A150 [inconsistency] ai/feature_store.py:40 — year 2000 literal
A151 [inconsistency] ai/feature_store.py:74 — EMA duplicates temporal concept
A152 [inconsistency] ai/inference_pipeline.py:54 — k=100
A153 [inconsistency] ai/inference_pipeline.py:69 — k=50
A154 [inconsistency] ai/inference_pipeline.py:113 — k=500
A155 [inconsistency] ai/inference_pipeline.py:279 — k=10
A156 [inconsistency] ai/inference_pipeline.py:341 — diversity 0.3 (3rd copy)
A157 [inconsistency] ai/inference_pipeline.py:370 — k=10
A158 [inconsistency] ai/ab_testing.py:32 — min_sample 1000
A159 [inconsistency] ai/ab_testing.py:200 — %10000 literal
A160 [inconsistency] ai/ab_testing.py:320 — p<0.05
A161 [inconsistency] ai/ab_testing.py:325 — n>=30
A162 [inconsistency] ai/ab_experimentation.py:74 — p<0.001 vs 0.05 above

### A(e) imports (16)
A163 [inconsistency] ai/candidate_generator.py:18 — second pool, bypasses singleton db
A164 [robustness] ai/candidate_generator.py:352 — module-level DB+TMDB at import
A165 [robustness] ai/recommender.py:1192 — module-level TMDBFetcher at import
A166 [bug] ai/netflix_recommender.py:104 — db.get_user_ratings doesn't exist
A167 [robustness] ai/semantic_search.py:31 — nested fallback can raise, no try
A168 [hygiene] ai/embeddings.py:7 — unused normalize import
A169 [hygiene] ai/data_preprocessor.py:9 — unused train_test_split
A170 [hygiene] ai/conversational_agent.py:8 — unused re
A171 [hygiene] ai/mood_detector.py:9 — unused re
A172 [hygiene] ai/ab_testing.py:8 — unused time/json
A173 [hygiene] ai/reinforcement.py:7 — Config imported, never used
A174 [hygiene] ai/telemetry.py:15 — unused numpy
A175 [hygiene] ai/neumf_scorer.py:18 — unused F import
A176 [robustness] ai/redis_cache.py:6 — unguarded redis import
A177 [robustness] ai/semantic_search.py:6 — unguarded sentence_transformers
A178 [robustness] ai/visual_search.py:6 — unguarded CLIP import

### A(f/g) dead code + naming (8)
A179 [hygiene] ai/visual_search.py:431 — color-search stub always []
A180 [hygiene] ai/recommender.py:988 — genre placeholder try/pass
A181 [inconsistency] ai/embeddings.py:336 — deprecated add_movie duplicates logic
A182 [hygiene] ai/data_preprocessor.py:372 — commented-out line
A183 [hygiene] ai/_archive/ — whole dir dead
A184 [bug] ai/conversational_agent.py:143 — return inside for, first genre only
A185 [bug] ai/mood_detector.py:265 — mutates DB row genres in place
A186 [bug] ai/candidate_generator.py:60 — dedup on id collapses DB movies to None
A187 [inconsistency] ai/recommender.py:192 vs 738 — chosen fallback differs
A188 [inconsistency] cast vs actors naming (embeddings vs DB)
A189 [inconsistency] genre dims 18/20/768, no adapter
A190 [inconsistency] duplicate CandidateGenerator class (inference_pipeline)
A191 [inconsistency] get_top_movies ×3 different semantics
A192 [inconsistency] get_user_embedding ×6 meanings
A193 [inconsistency] 8 recommend-* method names, callers must guess
A194 [inconsistency] TMDB_IMAGE_BASE_URL vs TMDB_POSTER_BASE_URL
A195 [inconsistency] 5 local-only flag conventions
A196 [inconsistency] 15 overlapping score names
A197 [inconsistency] content_pipeline tmdb_id/movie_id confusion

## B. api/ + app (77)

B198 [robustness] api/routes.py:771 — GET watchlist, no try/except
B199 [bug] api/routes.py:809 — PUT watchlist 0 rows still 200
B200 [bug] api/routes.py:818 — DELETE missing entry still 200
B201 [robustness] api/routes.py:825 — GET reviews, no try/except
B202 [bug] api/routes.py:849 — helpful vote always 200, no existence check
B203 [robustness] api/routes.py:860 — genre-stats, no try/except
B204 [robustness] api/routes.py:866 — genre-audit, no try/except
B205 [robustness] api/routes.py:873 — user-profile, no try/except
B206 [robustness] api/routes.py:887 — movie-score, no try/except
B207 [robustness] api/routes.py:787 — get_json outside try (watchlist add)
B208 [robustness] api/routes.py:804 — same (watchlist update)
B209 [robustness] api/routes.py:837 — same (review add)
B210 [bug] api/routes.py:420 — compared-movies failure returns 200 []
B211 [robustness] api/social_routes.py:166 — request.json outside try
B212 [bug] app_integrated.py:533 — visual-search empty returns 200 not 503
B213 [bug] app_integrated.py:586 — text-to-image always 200 after failures
B214 [bug] app_integrated.py:484 — trending except returns 200 []
B215 [bug] app_integrated.py:509 — viral except returns 200 []
B216 [robustness] api/routes.py:104 — signup get_json no None check
B217 [robustness] api/routes.py:143 — login same
B218 [inconsistency] api/routes.py:106 — no email/length/password validation
B219 [bug] api/routes.py:114 — email duplicates allowed
B220 [robustness] api/routes.py:475 — limit uncapped
B221 [robustness] api/routes.py:584 — limit uncapped
B222 [robustness] api/routes.py:692 — limit uncapped
B223 [robustness] api/routes.py:736 — limit uncapped
B224 [robustness] api/routes.py:1074 — limit uncapped
B225 [robustness] api/routes.py:1174 — limit uncapped
B226 [bug] api/routes.py:909 — limit no type coercion, breaks max()
B227 [robustness] api/routes.py:910 — include_tv unchecked
B228 [robustness] api/routes.py:585 — strategy passed raw
B229 [robustness] api/routes.py:734 — genre/media_type raw to DB
B230 [robustness] api/routes.py:1075 — order_by raw (SQL identifier risk)
B231 [robustness] api/routes.py:777 — status_filter unchecked
B232 [robustness] api/routes.py:788 — watchlist fields unvalidated
B233 [robustness] api/routes.py:804 — status/rating un-ranged
B234 [robustness] api/routes.py:839 — rating only None-checked
B235 [bug] app_integrated.py:337 — body user_id spoofs any user
B236 [bug] app_integrated.py:441 — same
B237 [robustness] app_integrated.py:443 — top_k unchecked
B238 [robustness] app_integrated.py:469 — limit unbounded
B239 [robustness] app_integrated.py:523 — top_k unbounded
B240 [robustness] app_integrated.py:557 — top_k unbounded
B241 [robustness] api/social_routes.py:31 — request.json no check
B242 [robustness] api/social_routes.py:125 — same
B243 [robustness] api/social_routes.py:296 — same
B244 [robustness] api/social_routes.py:331 — same
B245 [bug] api/social_routes.py:335 — scheduled_time unvalidated
B246 [bug] api/social_routes.py:336 — invitees unbounded, unchecked IDs
B247 [bug] app_integrated.py:331 — /api/chat no login + spoofable
B248 [bug] app_integrated.py:435 — /api/mood no login + spoofable
B249 [inconsistency] api/routes.py:514 vs 542 — compare guest vs 401
B250 [inconsistency] api/routes.py:460 vs 575 — recommendations guest vs 401
B251 [inconsistency] api/routes.py:413 — compared-movies 200 vs 401 elsewhere
B252 [bug] api/routes.py:849 — helpful vote spammable anon
B253 [inconsistency] app_integrated.py:610 — web /compare redirects, API serves anon
B254 [bug] api/social_routes.py:365 — any user reads any watchparty
B255 [bug] api/social_routes.py:406 — join succeeds with 0 rows
B256 [bug] api/social_routes.py:289 — no friendship verification
B257 [bug] api/social_routes.py:324 — no list-ownership check
B258 [bug] api/social_routes.py:499 — append to any list_id
B259 [bug] api/routes.py:260 — N+1 get_movie_by_id per interaction
B260 [bug] api/routes.py:300 — N+1 per genre movie + writes on read path
B261 [bug] api/routes.py:437 — N+1 per compared id
B262 [robustness] api/routes.py:36 — CSV scan per cache miss
B263 [robustness] api/routes.py:192 — limit=5000 hardcoded
B264 [robustness] api/routes.py:238 — limit=1000
B265 [robustness] api/routes.py:423 — limit=10000
B266 [robustness] app_integrated.py:292 — LIMIT 10000 startup index
B267 [inconsistency] movie list shapes differ across 7 endpoints
B268 [inconsistency] AI search shapes differ (routes vs app)
B269 [inconsistency] chat/mood type+alias shapes differ
B270 [inconsistency] watchlist shapes differ ×3
B271 [inconsistency] error shapes differ ×4, clients can't branch
B272 [inconsistency] /api/cache-stats vs /cache/stats vs /cache/monitor
B273 [inconsistency] POST /search/ai vs GET /movie/search overlap
B274 [bug] api/routes.py:1170 — GET top-rated WRITES to DB
B275 [bug] api/routes.py:1063 — GET search ingests TMDB rows

## C. database/ (70)

C276 [robustness] db_manager.py:205 — f-string SQL, whitelist-only safety
C277 [robustness] db_manager.py:287 — same pattern
C278 [robustness] db_manager.py:833 — generic query() sink, caller SQL unchecked
C279 [robustness] db_manager.py:398 — cursor leak on exception, no finally
C280 [robustness] db_manager.py:526 — same
C281 [robustness] db_manager.py:547 — same
C282 [robustness] db_manager.py:566 — same
C283 [robustness] db_manager.py:589 — same
C284 [robustness] db_manager.py:602 — same
C285 [robustness] db_manager.py:813 — same
C286 [robustness] run_migration.py:62 — cursor leak on execute error
C287 [robustness] run_migration.py:140 — close skipped on fetchone error
C288 [robustness] run_migration.py:159 — close skipped on commit error
C289 [robustness] run_migration.py:193 — close skipped on commit error
C290 [robustness] run_migration.py:320 — close skipped on error
C291 [hygiene] db_manager.py:59 — get_cursor commits on pure SELECTs
C292 [bug] db_manager.py:199 — media_type filter, column doesn't exist
C293 [bug] db_manager.py:281 — same on genre path
C294 [bug] db_manager.py:110 — insert drops media_type, TV flag lost
C295 [inconsistency] db_manager.py:145 — movie_details omits original_title
C296 [inconsistency] ensure_social_tables creates 7 tables absent from schema.sql
C297 [bug] watch_parties status ENUM lacks 'scheduled' (social_routes inserts it)
C298 [bug] watch_party_invites ENUM lacks 'invited' (inserted at :348)
C299 [inconsistency] schema_migrations untracked by schema.sql
C300 [bug] migrations/ empty — migration bookkeeping dead code
C301 [bug] run_migration.py:249 — failed migration marked applied
C302 [inconsistency] two SQL splitters disagree on DELIMITER handling
C303 [bug] schema.sql lacks FK_CHECKS guard, CLI load fails on FK order
C304 [inconsistency] core CREATE TABLE vs extended IF NOT EXISTS
C305 [inconsistency] schema.sql:1277 stale banner (16/13/8/4 vs actual)
C306 [robustness] update_user_interaction_count — 0-row update, no signal
C307 [bug] record_user_interaction — NULL elo corrupts rankings, no rollback
C308 [bug] update_movie_elo — same NULL elo path
C309 [bug] record_user_interaction — chosen not validated in (m1,m2)
C310 [robustness] update_user_preferences — NULL vars insert NULL row
C311 [robustness] get_personalized_recommendations — temp table leak on error
C312 [robustness] get_personalized_recommendations — variable LIMIT
C313 [bug] search_movies_advanced — FOUND_ROWS without SQL_CALC, garbage count
C314 [bug] search_movies_advanced — p_search_type unchecked vs ENUM
C315 [bug] update_cache_stats — (old+new)/2 is not a running average
C316 [bug] update_cache_stats — p_cache_type unchecked vs ENUM
C317 [bug] get_comparison_pair — NULL user falls through, empty leg
C318 [bug] get_mood_recommendations — 'Musical'/'Sci-Fi' never match seed genres
C319 [bug] log_recommendation_feedback — free VARCHAR vs ENUM, no handler
C320 [bug] generate_genre_audit — stale loop vars on missing genre row
C321 [robustness] seed_watchlists — handler swallows ALL errors
C322 [bug] cleanup_temporary_movies — unguarded cascading DELETE
C323 [inconsistency] bulk_elo_reset doesn't bump comparison_count
C324 [robustness] calculate_cache_hit_rate — dup rows throw 1242, no handler
C325 [robustness] user_interactions.movie_1_id unindexed, OR-join full scan
C326 [robustness] user_interactions.movie_2_id same
C327 [robustness] movie_reviews no (user_id,created_at) index
C328 [robustness] recommendation_log no (clicked,watched) index
C329 [robustness] watchlist no (user_id,status,priority) index
C330 [robustness] user_preferences.favorite_genre unindexed despite joins
C331 [robustness] LOWER(..) LIKE '%..%' defeats idx_title + FULLTEXT unused
C332 [hygiene] duplicate idx_user_interactions (180 vs 1267)
C333 [hygiene] PK+idx_user_vector duplicate
C334 [hygiene] PK+idx_movie_vector duplicate
C335 [hygiene] UNIQUE covers idx_movie_keywords
C336 [hygiene] UNIQUE covers idx_stat_date
C337 [inconsistency] movies PK caller-assigned vs AUTO_INCREMENT elsewhere
C338 [inconsistency] tmdb_id vs tmdb_genre_id vs tmdb_person_id
C339 [inconsistency] tmdb_rating vs rating vs user_rating vs avg_rating_preference
C340 [inconsistency] 6 names for event time (timestamp/created_at/...)
C341 [inconsistency] cast vs cast_members across views
C342 [inconsistency] 3 overlapping keyword/embedding concepts
C343 [inconsistency] search_history vs search_logs duality
C344 [inconsistency] cache counters duplicated with different names
C345 [inconsistency] favorite_* store names not *_id FKs

## D. training/ + benchmarks/ + inference/ (64)

D346 [bug] kaggle train:192 — global random split labeled per-user temporal
D347 [bug] kaggle train:195 — only np seed, DataLoader nondeterministic
D348 [bug] training/train.py:178 — global split, users can miss train entirely
D349 [robustness] training/train.py — no seed anywhere
D350 [bug] advanced_training.py:214 — same global-split issue
D351 [robustness] advanced_training.py — no seeds
D352 [robustness] distributed_training.py:89 — no seed set
D353 [inconsistency] distributed val_loader non-distributed vs sharded train
D354 [robustness] sequential_model.py:299 — no seed
D355 [bug] data_loader.py:190 — first-N break makes random subsample dead code
D356 [bug] run_comprehensive:104 — np.random.seed pollutes global RNG
D357 [bug] run_comprehensive:147 — set-order fallback candidates, PYTHONHASHSEED-dependent
D358 [bug] run_all:224 — tie order from set iteration, nondeterministic
D359 [bug] recommenders.py:48 — same tie-order issue
D360 [bug] stage_retrieval:150 — candidates from set, same issue
D361 [bug] diversity_benchmark.py:63 — same
D362 [bug] run_32m:122 — epoch-coupled in-place shuffle
D363 [bug] run_32m:518 — first-3000 slice presented as catalog ranking
D364 [inconsistency] 3 normalizations: (r-1)/4 vs (r-0.5)/4.5, thresholds differ
D365 [bug] comprehensive_metrics gini drops IDs >= catalog_size (32M garbage)
D366 [bug] advanced_training:263 — trains FocalLoss, validates MSE
D367 [bug] distributed:196 — unweighted mean-of-batch-means RMSE
D368 [bug] distributed:184 — DDP-wrapped model in val forward
D369 [bug] losses.py:43 — FocalMSE collapses to ~0 for mse<1
D370 [bug] losses.py:159 — CombinedLoss sums incompatible scales
D371 [inconsistency] novelty denominator differs across 4 files
D372 [bug] run_all:234 — warm latency is dict-get, mislabeled as inference
D373 [inconsistency] kaggle:7 header conflates 87,585 rows with 84,432 vocab
D374 [bug] run_32m:385 — 500+ banner on first-N users + 3000-slice pool
D375 [inconsistency] run_comprehensive:476 hardcoded counts vs computed catalog
D376 [inconsistency] plot dim 384 vs 768 across 4 files
D377 [bug] torchserve_handler:200 — 384-dim fallback vs 768-dim model
D378 [bug] serving/model_handler:61 — fallback shapes match no checkpoint
D379 [inconsistency] run_all:413 header says 4 strategies, evaluates 5
D380 [inconsistency] resume bullets hardcoded, not computed
D381 [bug] inference/recommend.py:212 — default .pt paths don't exist
D382 [bug] kaggle train — no __main__ guard, import runs training
D383 [robustness] run_32m:63 — mkdir side effect on import
D384 [robustness] serving/model_handler:249 — handler instantiated on import
D385 [bug] serving/model_handler:57 — best_model.pth exists nowhere
D386 [bug] torchserve_handler:63 — export omits genre_encoder.json it requires
D387 [bug] model/*.pt gitignored, fresh clone breaks 6 load sites
D388 [robustness] distributed checkpoint_dir missing until rank 0 creates
D389 [inconsistency] DeepNCF ×2 implementations (advanced_models vs v2)
D390 [inconsistency] Attention ×3 variants, no shared module
D391 [inconsistency] FocalLoss same name, different math ×2 files
D392 [inconsistency] MovieLensDataset ×2 (dict vs tuple batches)
D393 [inconsistency] train_epoch/validate copied ×2 files
D394 [inconsistency] BiasedMF ×3 different hyperparams/clipping
D395 [inconsistency] Popularity ×3 implementations
D396 [inconsistency] Content-based ×3 (different weighting/dims)
D397 [inconsistency] multi-stage rankers ×2 + V3/V4 surrogate refusions
D398 [inconsistency] metrics.py vs comprehensive_metrics.py duplicates
D399 [inconsistency] TorchServe handlers ×2, both unwired
D400 [inconsistency] genre vocabs ×3 mismatched order/content
D401 [hygiene] sequential_model train fn, zero callers
D402 [hygiene] losses.py CombinedLoss/BPR/Listwise, zero callers
D403 [hygiene] distributed_training, zero callers
D404 [hygiene] torchserve_handler + model_handler, no references
D405 [hygiene] run_scale_benchmark main never wired into run_all
D406 [hygiene] run_comprehensive never called from run_all
D407 [inconsistency] preprocessing hardcoded data/ paths vs benchmarks/ml100k_data
D408 [hygiene] colab_train_100k.ipynb duplicates train_100k.py
D409 [bug] run_32m:279 MMR gain hardcoded 0.25, untuned

## E. tests/ + frontend + config + docs (50)

E410 [bug] test_features 6 fns — no asserts, 500 still passes
E411 [bug] test_features:148 — main() always prints Server-not-running (None check)
E412 [bug] test_ai_comprehensive 8 fns assert nothing
E413 [bug] test_visual_search — unconditional print True, no request
E414 [bug] test_model_checkpoints — reports missing files, never fails
E415 [bug] verify_features always exits 0, hides warnings
E416 [bug] test_xfetch expiration never tested (no sleep/re-assert)
E417 [bug] singleflight test sequential, never concurrent
E418 [inconsistency] pytest.ini omits test_comprehensive_metrics (5 real tests)
E419 [robustness] PSI tests unseeded, can flake
E420 [robustness] Thompson test stochastic, can flake
E421 [robustness] test_features guard hits network at import (collection stalls)
E422 [robustness] test_ai_comprehensive guard same
E423 [inconsistency] REDIS_* in config, app reads os.getenv directly
E424 [bug] REDIS_PASSWORD read nowhere
E425 [inconsistency] EXPLORATION_RATE set by users, consumed nowhere
E426 [inconsistency] CHECKPOINT_DIR unconsumed
E427 [inconsistency] BATCH sizes unconsumed
E428 [inconsistency] VECTOR_CACHE/REFILL unconsumed (JS hardcodes instead)
E429 [inconsistency] MOVIES_PER_PAGE unconsumed
E430 [inconsistency] PAIRWISE_KNOWN_RATIO unconsumed
E431 [inconsistency] STORE_ONLY/LAZY/EVICTION unconsumed
E432 [inconsistency] USE_CANDIDATE/FINAL_COUNT unconsumed
E433 [inconsistency] .env.example lacks REDIS_*, MODEL_DIR, DL_*, MOOD_*, SEMANTIC_*, VISUAL_*, TMDB_TIMEOUT
E434 [bug] requirements lacks pytest (docs prescribe pytest -q)
E435 [bug] requirements lacks matplotlib (scripts import it)
E436 [inconsistency] PyMySQL shipped, imported nowhere
E437 [bug] main.js:133 — /api/movie/by-genre/<g> 404s, query form is correct
E438 [bug] index.html:669 — offset silently dropped, load-more refetches
E439 [bug] category.html:131 — placeholder.jpg missing from static/
E440 [inconsistency] verify_features NEXT STEPS wrong paths
E441 [inconsistency] results.md test counts (60/76 vs actual 60/75+5)
E442 [inconsistency] architecture.md training/ 9 files vs 10 on disk
E443 [bug] fetch_tmdb_data.py:157 prints python app.py (doesn't exist)
E444 [bug] train_model.py writes ai/models/, nothing loads from there
E445 [bug] quick_start.py checks ai/cache+ai/models, always missing
E446 [bug] watch_kaggle.ps1 no fail-fast on missing CLI
E447 [bug] watch_kaggle.ps1 never exits 1 on kernel failure
E448 [robustness] watch_kaggle.ps1 5h cap exits 0 on timeout
E449 [inconsistency] fill_all_tables hardcodes seed password
E450 [bug] test_server_health returned bool (fixed: asserts now)
E451 [bug] test_features main() None-check (fixed: try/except now)
E452 [bug] test_endpoints collected as fixture error (fixed: main guard)
E453 [inconsistency] tests/verify_features.py:59 mentions sequential_model only
E454 [hygiene] tests/__pycache__ + .pytest_cache committed? (verify gitignored)
E455 [inconsistency] start.ps1 vs start-cinesense.ps1 overlap, unclear which to use
E456 [robustness] Procfile single worker default vs render 2 workers
E457 [inconsistency] gunicorn in requirements but app runs via flask dev server
E458 [bug] MAX_CONTENT_LENGTH set but image size unchecked (routes use it?)
E459 [verified-ok] 500.html has a registered handler (app_integrated.py:696)

## F. Machine-verified list (192)

`scripts/audit_items.txt` (generated by `scripts/gen_audit.py`, re-runnable):
bare-except ×10 · empty-fallback ×70 · sleep ×23 · magic-number ×14 ·
localhost ×17 · unvalidated-input ×38 · select-star ×20. Each entry carries
file:line. Overlaps with curated items above are intentional (machine proof).

[<- Back to README](../README.md)

<!-- MACHINE -->

