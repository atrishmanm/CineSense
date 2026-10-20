"""
Flask API Routes
RESTful API endpoints for the CineSense application
"""

from flask import Blueprint, request, jsonify, session, current_app
import csv
import re
import time
import threading
from pathlib import Path
import mysql.connector
from database.db_manager import db
from config import Config
from ai.recommender import recommender
from ai.cache_manager import cache_manager
from werkzeug.security import generate_password_hash, check_password_hash
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

api = Blueprint('api', __name__, url_prefix='/api')

TMDB_GENRE_CACHE = {}
TMDB_CSV_PATH = Path(__file__).resolve().parent.parent / 'data' / 'tmdb' / 'TMDB_movie_dataset_v11.csv'

# ---- Input validation limits / whitelists ----------------------------------
DEFAULT_LIMIT = 20
MAX_LIMIT = 100                       # hard cap for client-supplied limit params
MAX_OFFSET = 10000                    # wave5 fix (B2.2): cap for pagination offsets
PROFILE_INTERACTION_LIMIT = 5000      # interactions scanned when building taste profile
PREFERENCES_INTERACTION_LIMIT = 1000  # interactions scanned for preference analysis
HISTORY_INTERACTION_LIMIT = 1000      # interactions scanned for the activity timeline
COMPARED_MOVIES_LIMIT = 5000          # interactions scanned for compared-movies list
MAX_MOVIE_ID_CHUNK = 500              # max ids per IN(...) batch query

VALID_ORDER_BY = {'elo_score', 'tmdb_rating', 'popularity'}
VALID_MEDIA_TYPES = {'all', 'movie', 'tv'}
VALID_STRATEGIES = {'mixed', 'genre', 'popularity', 'exploration'}
VALID_SEARCH_TYPES = {'title', 'storyline', 'semantic', 'hybrid'}
VALID_WATCHLIST_STATUSES = {'planned', 'watching', 'completed', 'dropped'}

# In-memory dedupe of (user_id, review_id) helpful votes (audit B202/B252 tail).
# Best-effort, per-process only; capped so it can never grow unbounded.
HELPFUL_VOTE_DEDUPE_MAX = 10000
_helpful_votes_seen = set()
_helpful_votes_lock = threading.Lock()

# Anti-replay window for /api/feedback (audit B1.3): the same (user, pair)
# submission within the window is rejected instead of double-applying the ELO
# delta. In-memory and per-process only; a DB-backed dedupe is future work.
FEEDBACK_REPLAY_WINDOW_SECONDS = 60
FEEDBACK_DEDUPE_MAX_AGE_SECONDS = 300
_recent_feedback = {}
_recent_feedback_lock = threading.Lock()

EMAIL_RE = re.compile(r'^[^@\s]+@[^@\s]+\.[^@\s]+$')


def _json_body():
    """Parse a JSON request body, returning {} when the body is absent or invalid."""
    data = request.get_json(silent=True)
    return data if isinstance(data, dict) else {}


def _parse_limit(value, default=DEFAULT_LIMIT, maximum=MAX_LIMIT):
    """Coerce a client-supplied limit into 1..maximum. Raises ValueError on bad input."""
    if value is None or value == '':
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise ValueError('limit must be an integer')
    if parsed < 1:
        raise ValueError('limit must be a positive integer')
    return min(parsed, maximum)


def _parse_offset(value, default=0):
    """Coerce a client-supplied offset into 0..MAX_OFFSET. Raises ValueError on bad input."""
    if value is None or value == '':
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise ValueError('offset must be an integer')
    if parsed < 0:
        raise ValueError('offset must be a non-negative integer')
    # wave5 fix (B2.2): cap the offset so ?offset=1000000 cannot trigger an
    # unbounded recommendation fetch.
    return min(parsed, MAX_OFFSET)


def _parse_movie_id(value):
    """Validate a client-supplied movie id.

    Returns (parsed_int, error_message); error_message is None when valid.
    wave5 fix (B2.3): non-integer ids used to fall through to SQL and 500.
    """
    if value is None or value == '':
        return None, 'movie_id required'
    try:
        return int(value), None
    except (TypeError, ValueError):
        return None, 'movie_id must be an integer'


def _is_valid_email(email):
    return bool(email) and bool(EMAIL_RE.match(email))


def _fetch_movies_by_ids(movie_ids):
    """Batch-fetch movie rows by id, avoiding an N+1 query per interaction."""
    unique_ids = [mid for mid in dict.fromkeys(movie_ids) if mid]
    if not unique_ids:
        return {}
    movies = {}
    for start in range(0, len(unique_ids), MAX_MOVIE_ID_CHUNK):
        chunk = unique_ids[start:start + MAX_MOVIE_ID_CHUNK]
        placeholders = ','.join(['%s'] * len(chunk))
        rows = db.query(
            f"SELECT * FROM movie_details WHERE movie_id IN ({placeholders})",
            tuple(chunk)
        ) or []
        for row in rows:
            movies[row['movie_id']] = row
    return movies


def get_semantic_search():
    """Return the pre-initialized semantic search engine from app startup"""
    return getattr(current_app, 'semantic_search', None)


def _parse_genre_string(genre_str):
    if not genre_str:
        return []
    return [g.strip() for g in genre_str.split(',') if g.strip()]


def _load_tmdb_genres_for_ids(tmdb_ids):
    missing = [int(t) for t in tmdb_ids if t and int(t) not in TMDB_GENRE_CACHE]
    if not missing:
        return
    if not TMDB_CSV_PATH.exists():
        for tmdb_id in missing:
            TMDB_GENRE_CACHE[tmdb_id] = []
        return

    missing_set = set(str(t) for t in missing)
    with TMDB_CSV_PATH.open('r', encoding='utf-8', newline='') as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row_id = row.get('id')
            if row_id in missing_set:
                genres = _parse_genre_string(row.get('genres'))
                TMDB_GENRE_CACHE[int(row_id)] = genres
                missing_set.remove(row_id)
                if not missing_set:
                    break

    for leftover in missing_set:
        TMDB_GENRE_CACHE[int(leftover)] = []


def _ensure_movie_genres(movie, movie_id):
    existing = _parse_genre_string(movie.get('genres'))
    if existing:
        return existing

    genres = db.get_movie_genres(movie_id)
    if genres:
        return genres

    tmdb_id = movie.get('tmdb_id')
    if tmdb_id:
        _load_tmdb_genres_for_ids([tmdb_id])
        genres = TMDB_GENRE_CACHE.get(int(tmdb_id), [])

    if genres:
        for name in genres:
            # Never persist blank genre names (TMDB id-only parses produce '')
            if not name or not str(name).strip():
                continue
            name = str(name).strip()
            existing_genre = db.get_genre_by_name(name)
            if existing_genre:
                genre_id = existing_genre['genre_id']
            else:
                genre_id = db.insert_genre(name)
            db.link_movie_genre(movie_id, genre_id)

    return genres


def _prime_tmdb_genres_cache(movies):
    tmdb_ids = {
        int(m['tmdb_id']) for m in movies
        if m and m.get('tmdb_id')
    }
    if tmdb_ids:
        _load_tmdb_genres_for_ids(tmdb_ids)


# ============================================================================
# USER ENDPOINTS
# ============================================================================

@api.route('/user/signup', methods=['POST'])
def signup():
    """Create a new user account"""
    try:
        data = _json_body()

        username = (data.get('username') or '').strip()
        email = (data.get('email') or '').strip()
        password = data.get('password') or ''

        if not all([username, email, password]):
            return jsonify({'error': 'Missing required fields'}), 400

        if len(password) < 6:
            return jsonify({'error': 'Password must be at least 6 characters'}), 400

        if len(username) < 3:
            return jsonify({'error': 'Username must be at least 3 characters'}), 400

        if not _is_valid_email(email):
            return jsonify({'error': 'Invalid email address'}), 400

        # Pre-check duplicates so we return 409 instead of a UNIQUE violation 500
        if db.get_user_by_username(username):
            return jsonify({'error': 'Username already exists'}), 409

        existing_email = db.query(
            "SELECT user_id FROM users WHERE email = %s",
            (email,),
            fetch_all=False
        )
        if existing_email:
            return jsonify({'error': 'Email already exists'}), 409

        # Hash password
        password_hash = generate_password_hash(password)

        # Create user
        # wave5 fix (B1.5): the duplicate pre-check above can race with a
        # concurrent signup; rely on the UNIQUE constraint for the true answer
        # instead of returning a 500.
        try:
            user_id = db.create_user(username, email, password_hash)
        except mysql.connector.errors.IntegrityError:
            return jsonify({'error': 'Username or email already exists'}), 409

        # Set session
        session['user_id'] = user_id
        session['username'] = username

        return jsonify({
            'message': 'User created successfully',
            'user_id': user_id,
            'username': username
        }), 201

    except Exception as e:
        logger.error(f"Signup error: {e}")
        return jsonify({'error': 'Internal server error'}), 500


@api.route('/user/login', methods=['POST'])
def login():
    """User login"""
    try:
        data = _json_body()

        username = (data.get('username') or '').strip()
        password = data.get('password') or ''

        if not all([username, password]):
            return jsonify({'error': 'Missing credentials'}), 400
        
        # Get user
        user = db.get_user_by_username(username)
        
        if not user or not check_password_hash(user['password_hash'], password):
            return jsonify({'error': 'Invalid credentials'}), 401
        
        # Set session
        session['user_id'] = user['user_id']
        session['username'] = user['username']
        
        return jsonify({
            'message': 'Login successful',
            'user_id': user['user_id'],
            'username': user['username']
        }), 200
        
    except Exception as e:
        logger.error(f"Login error: {e}")
        return jsonify({'error': 'Internal server error'}), 500


@api.route('/user/logout', methods=['POST'])
def logout():
    """User logout"""
    session.clear()
    return jsonify({'message': 'Logged out successfully'}), 200


@api.route('/user/profile', methods=['GET'])
def get_profile():
    """Get user profile and statistics"""
    try:
        user_id = session.get('user_id')
        
        if not user_id:
            return jsonify({'error': 'Not authenticated'}), 401
        
        # Get user stats
        stats = db.get_user_stats(user_id)

        # Fallback: compute stats from interactions if view is missing or stale
        interactions = db.get_user_interactions(user_id, limit=PROFILE_INTERACTION_LIMIT)
        total_comparisons = len(interactions)
        unique_movie_ids = set()
        for interaction in interactions:
            for key in ('movie_1_id', 'movie_2_id', 'chosen_movie_id'):
                movie_id = interaction.get(key)
                if movie_id:
                    unique_movie_ids.add(movie_id)
        unique_movies_seen = len(unique_movie_ids)

        learning_target = max(1, int(Config.LEARNING_TARGET_COMPARISONS))

        if not stats:
            user = db.get_user_by_id(user_id)
            if not user:
                return jsonify({'error': 'User not found'}), 404
            stats = {
                'user_id': user_id,
                'username': user.get('username'),
                'created_at': user.get('created_at'),
                'total_comparisons': total_comparisons,
                'unique_movies_seen': unique_movies_seen,
                'learning_target': learning_target
            }
        else:
            stats['total_comparisons'] = max(stats.get('total_comparisons', 0), total_comparisons)
            stats['unique_movies_seen'] = max(stats.get('unique_movies_seen', 0), unique_movies_seen)
            stats['learning_target'] = learning_target

        return jsonify(stats), 200
        
    except Exception as e:
        logger.error(f"Profile error: {e}")
        return jsonify({'error': 'Internal server error'}), 500


@api.route('/user/preferences', methods=['GET'])
def get_user_preferences():
    """Get AI-analyzed user preferences and insights"""
    try:
        user_id = session.get('user_id')
        
        if not user_id:
            return jsonify({'error': 'Not authenticated'}), 401
        
        # Get user interactions
        interactions = db.get_user_interactions(user_id, limit=PREFERENCES_INTERACTION_LIMIT)
        
        if not interactions or len(interactions) == 0:
            return jsonify({
                'genre_preferences': [],
                'favorite_directors': [],
                'favorite_actors': [],
                'favorite_movies': [],
                'average_rating': 0,
                'preferred_era': 'N/A',
                'prefers_classics': False
            }), 200
        
        # Analyze genre preferences
        genre_counts = {}
        director_counts = {}
        actor_counts = {}
        chosen_movies = []
        genre_movie_ids = set()
        ratings = []
        years = []

        # Batch-fetch the movies we need instead of one query per interaction (N+1)
        chosen_ids = [i.get('chosen_movie_id') for i in interactions if i.get('chosen_movie_id')]
        movies_by_id = _fetch_movies_by_ids(chosen_ids)

        for interaction in interactions:
            chosen_id = interaction.get('chosen_movie_id')
            if chosen_id:
                movie = movies_by_id.get(chosen_id)
                if movie:
                    chosen_movies.append(movie)
                    genre_movie_ids.add(chosen_id)
                    
                    # Count directors
                    if movie.get('directors'):
                        for director in movie['directors'].split(','):
                            director = director.strip()
                            director_counts[director] = director_counts.get(director, 0) + 1
                    
                    # Count actors
                    if movie.get('cast'):
                        for actor in movie['cast'].split(',')[:5]:  # Top 5 actors
                            actor = actor.strip()
                            actor_counts[actor] = actor_counts.get(actor, 0) + 1
                    
                    # Collect ratings and years
                    if movie.get('tmdb_rating'):
                        ratings.append(float(movie['tmdb_rating']))
                    if movie.get('release_year'):
                        years.append(movie['release_year'])

            # Include both compared movies for genre-only fallback
            movie_1_id = interaction.get('movie_1_id')
            movie_2_id = interaction.get('movie_2_id')
            if movie_1_id:
                genre_movie_ids.add(movie_1_id)
            if movie_2_id:
                genre_movie_ids.add(movie_2_id)

        _prime_tmdb_genres_cache(chosen_movies)

        for movie in chosen_movies:
            for genre in _ensure_movie_genres(movie, movie['movie_id']):
                genre_counts[genre] = genre_counts.get(genre, 0) + 1

        if not genre_counts and genre_movie_ids:
            genre_movies_by_id = _fetch_movies_by_ids(genre_movie_ids)
            genre_movies = list(genre_movies_by_id.values())

            _prime_tmdb_genres_cache(genre_movies)

            for movie in genre_movies:
                for genre in _ensure_movie_genres(movie, movie['movie_id']):
                    genre_counts[genre] = genre_counts.get(genre, 0) + 1
        
        # Sort and format results
        genre_preferences = [
            {'genre': genre, 'count': count}
            for genre, count in sorted(genre_counts.items(), key=lambda x: x[1], reverse=True)
        ]
        
        favorite_directors = [
            {'name': director, 'count': count}
            for director, count in sorted(director_counts.items(), key=lambda x: x[1], reverse=True)
        ]
        
        favorite_actors = [
            {'name': actor, 'count': count}
            for actor, count in sorted(actor_counts.items(), key=lambda x: x[1], reverse=True)
        ]
        
        # Calculate statistics
        avg_rating = sum(ratings) / len(ratings) if ratings else 0
        avg_year = sum(years) / len(years) if years else 2000
        
        # Determine preferred era
        if avg_year < 1980:
            preferred_era = 'Classic Era (pre-1980)'
        elif avg_year < 2000:
            preferred_era = '80s-90s'
        elif avg_year < 2010:
            preferred_era = '2000s'
        else:
            preferred_era = 'Modern (2010+)'
        
        prefers_classics = avg_year < 2000
        
        return jsonify({
            'genre_preferences': genre_preferences[:10],
            'favorite_directors': favorite_directors[:10],
            'favorite_actors': favorite_actors[:10],
            'favorite_movies': chosen_movies[:10],
            'average_rating': avg_rating,
            'preferred_era': preferred_era,
            'prefers_classics': prefers_classics
        }), 200
        
    except Exception as e:
        logger.error(f"Preferences error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'Internal server error'}), 500


@api.route('/user/interactions', methods=['GET'])
def get_user_interactions_api():
    """Get user interaction history with activity timeline"""
    try:
        user_id = session.get('user_id')
        
        if not user_id:
            return jsonify({'error': 'Not authenticated'}), 401
        
        # Get interactions
        interactions = db.get_user_interactions(user_id, limit=HISTORY_INTERACTION_LIMIT)
        
        # Build activity timeline (last 30 days)
        from collections import defaultdict
        from datetime import datetime, timedelta
        
        activity_by_date = defaultdict(int)
        
        for interaction in interactions:
            timestamp = interaction.get('timestamp')
            if timestamp:
                try:
                    date = datetime.fromisoformat(str(timestamp)).date()
                    activity_by_date[str(date)] += 1
                except:
                    pass
        
        # Create timeline for last 30 days
        timeline = []
        today = datetime.now().date()
        for i in range(30, -1, -1):
            date = today - timedelta(days=i)
            date_str = str(date)
            timeline.append({
                'date': date.strftime('%m/%d'),
                'count': activity_by_date.get(date_str, 0)
            })
        
        return jsonify({
            'total_comparisons': len(interactions),
            'activity_timeline': timeline,
            'recent_interactions': interactions[:20]
        }), 200
        
    except Exception as e:
        logger.error(f"Interactions error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'Internal server error'}), 500


@api.route('/user/compared-movies', methods=['GET'])
def get_compared_movies():
    """Get all movies that user has compared"""
    try:
        user_id = session.get('user_id')
        
        if not user_id:
            return jsonify({'movies': []}), 200
        
        # Get all interactions to find compared movies
        interactions = db.get_user_interactions(user_id, limit=COMPARED_MOVIES_LIMIT)
        
        # Collect unique movie IDs from interactions
        compared_ids = set()
        for interaction in interactions:
            if interaction.get('movie_1_id'):
                compared_ids.add(interaction['movie_1_id'])
            if interaction.get('movie_2_id'):
                compared_ids.add(interaction['movie_2_id'])
        
        include_genres = request.args.get('include_genres', '0').lower() in {'1', 'true', 'yes'}

        # Get movie details in a single batched query (N+1 avoidance)
        movies_by_id = _fetch_movies_by_ids(compared_ids)
        movies = list(movies_by_id.values())

        if include_genres and movies:
            _prime_tmdb_genres_cache(movies)
            for movie in movies:
                genres = _ensure_movie_genres(movie, movie['movie_id'])
                if genres:
                    movie['genres'] = ', '.join(genres)
        
        return jsonify({'movies': movies, 'count': len(movies)}), 200
        
    except Exception as e:
        logger.error(f"Compared movies error: {e}")
        return jsonify({'error': 'Failed to get compared movies'}), 500


# ============================================================================
# RECOMMENDATION ENDPOINTS
# ============================================================================

@api.route('/recommendations', methods=['GET'])
def get_recommendations():
    """Get personalized movie recommendations"""
    try:
        user_id = session.get('user_id')

        try:
            limit = _parse_limit(request.args.get('limit'), default=20)
            offset = _parse_offset(request.args.get('offset'))
        except ValueError as exc:
            return jsonify({'error': str(exc)}), 400

        if not user_id:
            # Guest user: return popular movies (offset honored via slice)
            if offset:
                pool = db.get_top_movies(limit=limit + offset, order_by='popularity')
                movies = pool[offset:offset + limit]
            else:
                movies = db.get_top_movies(limit=limit, order_by='popularity')
            return jsonify({
                'movies': movies,
                'personalized': False
            }), 200

        # Get personalized recommendations (fetch limit+offset, slice out offset)
        if offset:
            recommendations = recommender.get_recommendations(
                user_id, n=limit + offset)[offset:offset + limit]
        else:
            recommendations = recommender.get_recommendations(user_id, n=limit)
        
        return jsonify({
            'movies': recommendations,
            'personalized': True
        }), 200
        
    except Exception as e:
        logger.error(f"Recommendations error: {e}")
        return jsonify({'error': 'Internal server error'}), 500


@api.route('/featured', methods=['GET'])
def get_featured():
    """Get featured movie for hero banner"""
    try:
        user_id = session.get('user_id')
        
        featured = recommender.get_featured_movie(user_id)
        
        if not featured:
            return jsonify({'error': 'No featured movie available'}), 404
        
        # Add explanation if user is logged in
        explanation = None
        if user_id:
            explanation = recommender.explain_recommendation(user_id, featured['movie_id'])
        
        return jsonify({
            'movie': featured,
            'explanation': explanation
        }), 200
        
    except Exception as e:
        logger.error(f"Featured movie error: {e}")
        return jsonify({'error': 'Internal server error'}), 500


@api.route('/compare', methods=['GET'])
def get_comparison():
    """Get two movies for pairwise comparison"""
    try:
        user_id = session.get('user_id')
        
        logger.info(f"Getting comparison pair for user_id: {user_id}")
        movie1, movie2 = recommender.get_comparison_pair(user_id)
        
        logger.info(f"Movie1: {movie1}")
        logger.info(f"Movie2: {movie2}")
        
        if not movie1 or not movie2:
            logger.error("Could not generate comparison pair - movies are None")
            return jsonify({'error': 'Could not generate comparison pair'}), 500
        
        return jsonify({
            'movie1': movie1,
            'movie2': movie2
        }), 200
        
    except Exception as e:
        logger.error(f"Comparison error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'Internal server error'}), 500


@api.route('/compare/lazy', methods=['GET'])
def get_comparison_lazy():
    """Get two movies for pairwise comparison using lazy loading (50% known + 50% explore)"""
    try:
        user_id = session.get('user_id')
        
        if not user_id:
            return jsonify({'error': 'Not authenticated'}), 401
        
        logger.info(f"Getting lazy comparison pair for user_id: {user_id}")
        movie1, movie2 = recommender.get_comparison_pair_lazy(user_id)
        
        if not movie1 or not movie2:
            logger.error("Could not generate lazy comparison pair")
            return jsonify({'error': 'Could not generate comparison pair'}), 500
        
        # Get cache stats
        cache_stats = cache_manager.get_stats()
        
        return jsonify({
            'movie1': movie1,
            'movie2': movie2,
            'cache_stats': cache_stats,
            'lazy_loaded': True
        }), 200
        
    except Exception as e:
        logger.error(f"Lazy comparison error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'Internal server error'}), 500


@api.route('/recommendations/lazy', methods=['GET'])
def get_recommendations_lazy():
    """Get personalized recommendations using lazy loading with candidate generation"""
    try:
        user_id = session.get('user_id')
        
        if not user_id:
            return jsonify({'error': 'Not authenticated'}), 401
        
        try:
            limit = _parse_limit(request.args.get('limit'), default=20)
        except ValueError as exc:
            return jsonify({'error': str(exc)}), 400
        strategy = request.args.get('strategy', 'mixed')  # mixed, genre, popularity, exploration

        if strategy not in VALID_STRATEGIES:
            return jsonify({'error': f'Invalid strategy. Allowed: {sorted(VALID_STRATEGIES)}'}), 400

        logger.info(f"Getting lazy recommendations for user_id: {user_id}, limit: {limit}, strategy: {strategy}")
        
        recommendations = recommender.get_recommendations_lazy(
            user_id, 
            n=limit,
            strategy=strategy
        )
        
        # Get cache stats
        cache_stats = cache_manager.get_stats()
        
        return jsonify({
            'movies': recommendations,
            'personalized': True,
            'lazy_loaded': True,
            'strategy': strategy,
            'cache_stats': cache_stats
        }), 200
        
    except Exception as e:
        logger.error(f"Lazy recommendations error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'Internal server error'}), 500


@api.route('/feedback', methods=['POST'])
def submit_feedback():
    """Submit user's pairwise choice"""
    try:
        user_id = session.get('user_id')
        
        if not user_id:
            return jsonify({'error': 'Not authenticated'}), 401
        
        data = _json_body()
        
        movie1_id = data.get('movie1_id')
        movie2_id = data.get('movie2_id')
        chosen_id = data.get('chosen_id')
        
        if not all([movie1_id, movie2_id, chosen_id]):
            logger.error(f"Missing fields: movie1_id={movie1_id}, movie2_id={movie2_id}, chosen_id={chosen_id}")
            return jsonify({'error': 'Missing required fields'}), 400
        
        if chosen_id not in [movie1_id, movie2_id]:
            logger.error(f"Invalid chosen_id: {chosen_id} not in [{movie1_id}, {movie2_id}]")
            return jsonify({'error': 'Invalid chosen movie'}), 400

        session_id = session.get('session_id', str(user_id))

        # wave5 fix (B1.3): reject an identical pair replayed within the window.
        # The timestamp is recorded up-front so concurrent duplicates also lose
        # the race; it is discarded again if the write below fails.
        replay_key = (user_id, movie1_id, movie2_id)
        now = time.time()
        with _recent_feedback_lock:
            seen_at = _recent_feedback.get(replay_key)
            if seen_at is not None and now - seen_at < FEEDBACK_REPLAY_WINDOW_SECONDS:
                return jsonify({'error': 'Already recorded'}), 409
            # Cap dict growth: purge entries older than the max age on insert.
            stale = [k for k, ts in _recent_feedback.items()
                     if now - ts >= FEEDBACK_DEDUPE_MAX_AGE_SECONDS]
            for stale_key in stale:
                _recent_feedback.pop(stale_key, None)
            _recent_feedback[replay_key] = now

        # wave5 fix (C1.1): /api/feedback used to call db.record_interaction
        # directly, so process_user_choice (bandit/embedding/preference
        # updates) never ran for real feedback. Route the write through the AI
        # layer, falling back to the direct DB write if it is unavailable.
        # The rejected id is simply the other movie of the pair.
        updated = False
        if recommender is not None:
            try:
                # NOTE: process_user_choice itself records the interaction row
                # via db.record_interaction internally, so a True result already
                # persisted the choice.
                updated = recommender.process_user_choice(
                    user_id,
                    chosen_id,
                    movie2_id if chosen_id == movie1_id else movie1_id,
                    session_id
                )
            except Exception as exc:
                logger.warning(f"process_user_choice failed; falling back to direct DB write: {exc}")

        if updated:
            logger.info(f"Feedback recorded: user={user_id}, chosen={chosen_id}")
            return jsonify({
                'message': 'Feedback recorded successfully',
                'learning': 'AI model updated based on your preference',
                'success': True
            }), 200

        # Record interaction using stored procedure (handles Elo automatically)
        success = db.record_interaction(
            user_id=user_id,
            movie_1_id=movie1_id,
            movie_2_id=movie2_id,
            chosen_movie_id=chosen_id,
            session_id=session_id
        )
        
        if not success:
            logger.error(f"Database operation failed for user {user_id}")
            with _recent_feedback_lock:
                _recent_feedback.pop(replay_key, None)
            return jsonify({'error': 'Failed to process feedback'}), 500
        
        logger.info(f"Feedback recorded: user={user_id}, chosen={chosen_id}")
        
        return jsonify({
            'message': 'Feedback recorded successfully',
            'learning': 'AI model updated based on your preference',
            'success': True
        }), 200
        
    except Exception as e:
        logger.error(f"Feedback error: {e}")
        import traceback
        traceback.print_exc()
        # wave5 fix (B3.3): do not leak internal exception text to clients.
        return jsonify({'error': 'Internal server error'}), 500


# ============================================================================
# MOVIE ENDPOINTS
# ============================================================================

@api.route('/movie/<int:movie_id>', methods=['GET'])
def get_movie_detail(movie_id):
    """Get detailed information about a movie"""
    try:
        movie = db.get_movie_by_id(movie_id)
        
        if not movie:
            return jsonify({'error': 'Movie not found'}), 404
        
        return jsonify({
            'movie': movie,
            'explanation': None
        }), 200
        
    except Exception as e:
        logger.error(f"Movie detail error: {e}")
        return jsonify({'error': 'Internal server error'}), 500


@api.route('/movie/<int:movie_id>/similar', methods=['GET'])
def get_similar_movies(movie_id):
    """Get movies similar to a given movie using genre overlap and AI embeddings"""
    try:
        try:
            limit = _parse_limit(request.args.get('limit'), default=12)
        except ValueError as exc:
            return jsonify({'error': str(exc)}), 400
        movie = db.get_movie_by_id(movie_id)
        if not movie:
            return jsonify({'error': 'Movie not found'}), 404

        # Try semantic similarity first
        similar = []
        try:
            search_engine = get_semantic_search()
            if search_engine and search_engine.movie_data is not None:
                query = f"{movie.get('title','')} {movie.get('overview','')}"
                results = search_engine.search(query, top_k=limit + 1)
                similar = [r for r in results if r.get('movie_id') != movie_id][:limit]
        except Exception as e:
            logger.warning(f"Semantic similar lookup failed: {e}")

        # Fallback to genre-based similarity
        if not similar:
            similar = db.query("""
                SELECT DISTINCT m.movie_id, m.title, m.poster_path,
                       m.tmdb_rating, m.release_year, m.popularity
                FROM movies m
                JOIN movie_genres mg ON m.movie_id = mg.movie_id
                WHERE mg.genre_id IN (
                    SELECT genre_id FROM movie_genres WHERE movie_id = %s
                )
                AND m.movie_id != %s
                ORDER BY m.popularity DESC
                LIMIT %s
            """, (movie_id, movie_id, limit)) or []

        return jsonify({'movies': similar}), 200

    except Exception as e:
        logger.error(f"Similar movies error: {e}")
        return jsonify({'error': 'Internal server error'}), 500


@api.route('/movie/by-genre', methods=['GET'])
def get_movies_by_genre():
    """Get movies filtered by genre and media type"""
    try:
        genre = (request.args.get('genre', '') or '').strip()
        media_type = request.args.get('media_type', 'all')
        year_from = request.args.get('year_from')
        year_to = request.args.get('year_to')
        sort_by = request.args.get('sort_by', 'rating')
        try:
            limit = _parse_limit(request.args.get('limit'), default=20)
            offset = _parse_offset(request.args.get('offset'), default=0)
        except ValueError as exc:
            return jsonify({'error': str(exc)}), 400

        if media_type not in VALID_MEDIA_TYPES:
            return jsonify({'error': f'Invalid media_type. Allowed: {sorted(VALID_MEDIA_TYPES)}'}), 400

        valid_sorts = {'popularity', 'rating', 'release_date', 'title'}
        if sort_by not in valid_sorts:
            return jsonify({'error': f'Invalid sort_by. Allowed: {sorted(valid_sorts)}'}), 400

        def _parse_year(value):
            if value is None or value == '':
                return None
            year = int(value)
            if not (1888 <= year <= 2100):
                raise ValueError
            return year

        try:
            year_from = _parse_year(year_from)
            year_to = _parse_year(year_to)
        except ValueError:
            return jsonify({'error': 'Invalid year filter (must be between 1888 and 2100)'}), 400

        if len(genre) > 100:
            return jsonify({'error': 'genre is too long'}), 400

        logger.info(
            f"Getting movies by genre: {genre}, media_type: {media_type}, "
            f"years: {year_from}-{year_to}, sort: {sort_by}, offset: {offset}"
        )

        # One code path for all filters (genre may be empty -> all movies)
        movies = db.get_movies_by_genre_and_type(
            genre_name=genre or None,
            media_type=media_type,
            limit=limit,
            offset=offset,
            year_from=year_from,
            year_to=year_to,
            sort_by=sort_by,
        )
        
        return jsonify({
            'success': True,
            'movies': movies,
            'genre': genre,
            'media_type': media_type,
            'count': len(movies)
        }), 200
        
    except Exception as e:
        logger.error(f"Get movies by genre error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'Failed to get movies'}), 500


# ============================================================================
# WATCHLIST ENDPOINTS
# ============================================================================

@api.route('/watchlist', methods=['GET'])
def get_watchlist():
    """Get user's watchlist"""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Login required'}), 401
    status_filter = request.args.get('status')
    if status_filter and status_filter not in VALID_WATCHLIST_STATUSES:
        return jsonify({'error': f'Invalid status. Allowed: {sorted(VALID_WATCHLIST_STATUSES)}'}), 400
    try:
        items = db.get_user_watchlist(user_id, status=status_filter)
        return jsonify({'watchlist': items, 'count': len(items)}), 200
    except Exception as e:
        logger.error(f"Watchlist fetch error: {e}")
        return jsonify({'error': 'Failed to fetch watchlist'}), 500

@api.route('/watchlist', methods=['POST'])
def add_to_watchlist():
    """Add movie to watchlist"""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Login required'}), 401
    data = _json_body()
    raw_movie_id = data.get('movie_id')
    # wave5 fix (B2.3): reject non-integer movie ids with 400 instead of letting
    # them surface as a SQL error / 500.
    movie_id, movie_id_error = _parse_movie_id(raw_movie_id)
    if not raw_movie_id:
        return jsonify({'error': 'movie_id required'}), 400
    if movie_id_error:
        return jsonify({'error': movie_id_error}), 400

    status = data.get('status', 'planned')
    if status not in VALID_WATCHLIST_STATUSES:
        return jsonify({'error': f'Invalid status. Allowed: {sorted(VALID_WATCHLIST_STATUSES)}'}), 400

    try:
        priority = int(data.get('priority', 5))
    except (TypeError, ValueError):
        return jsonify({'error': 'priority must be an integer between 1 and 10'}), 400
    if not 1 <= priority <= 10:
        return jsonify({'error': 'priority must be between 1 and 10'}), 400

    note = data.get('note')
    if note is not None and len(str(note)) > 500:
        return jsonify({'error': 'note must be at most 500 characters'}), 400

    try:
        db.add_to_watchlist(user_id, movie_id, priority, status, note)
        return jsonify({'success': True, 'message': 'Added to watchlist'}), 201
    except Exception as e:
        logger.error(f"Watchlist add error: {e}")
        return jsonify({'error': 'Failed to add to watchlist'}), 500

@api.route('/watchlist/<int:movie_id>', methods=['PUT'])
def update_watchlist(movie_id):
    """Update watchlist entry"""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Login required'}), 401
    data = _json_body()
    status = data.get('status')
    if not status:
        return jsonify({'error': 'status required'}), 400
    if status not in VALID_WATCHLIST_STATUSES:
        return jsonify({'error': f'Invalid status. Allowed: {sorted(VALID_WATCHLIST_STATUSES)}'}), 400

    rating = data.get('rating')
    if rating is not None:
        try:
            rating = float(rating)
        except (TypeError, ValueError):
            return jsonify({'error': 'rating must be a number between 0 and 10'}), 400
        if not 0 <= rating <= 10:
            return jsonify({'error': 'rating must be between 0 and 10'}), 400

    try:
        existing = db.query(
            "SELECT watchlist_id FROM watchlist WHERE user_id = %s AND movie_id = %s",
            (user_id, movie_id),
            fetch_all=False
        )
        if not existing:
            return jsonify({'error': 'Watchlist entry not found'}), 404
        db.update_watchlist_status(user_id, movie_id, status, rating)
    except Exception as e:
        logger.error(f"Watchlist update error: {e}")
        return jsonify({'error': 'Failed to update watchlist'}), 500
    return jsonify({'success': True}), 200

@api.route('/watchlist/<int:movie_id>', methods=['DELETE'])
def remove_from_watchlist(movie_id):
    """Remove from watchlist"""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Login required'}), 401
    try:
        rows = db.remove_from_watchlist(user_id, movie_id)
    except Exception as e:
        logger.error(f"Watchlist delete error: {e}")
        return jsonify({'error': 'Failed to remove from watchlist'}), 500
    if rows <= 0:
        return jsonify({'error': 'Watchlist entry not found'}), 404
    return jsonify({'success': True}), 200


# ============================================================================
# REVIEW ENDPOINTS
# ============================================================================

@api.route('/reviews/<int:movie_id>', methods=['GET'])
def get_reviews(movie_id):
    """Get reviews for a movie"""
    try:
        reviews = db.get_movie_reviews(movie_id)
        return jsonify({'reviews': reviews, 'count': len(reviews)}), 200
    except Exception as e:
        logger.error(f"Reviews fetch error: {e}")
        return jsonify({'error': 'Failed to fetch reviews'}), 500

@api.route('/reviews', methods=['POST'])
def add_review():
    """Add a movie review"""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Login required'}), 401
    data = _json_body()
    raw_movie_id = data.get('movie_id')
    rating = data.get('rating')
    # wave5 fix (B2.3): reject non-integer movie ids with 400 instead of letting
    # them surface as a SQL error / 500.
    movie_id, movie_id_error = _parse_movie_id(raw_movie_id)
    if not raw_movie_id or rating is None:
        return jsonify({'error': 'movie_id and rating required'}), 400
    if movie_id_error:
        return jsonify({'error': movie_id_error}), 400
    try:
        rating = float(rating)
    except (TypeError, ValueError):
        return jsonify({'error': 'rating must be a number between 1 and 10'}), 400
    if not 1 <= rating <= 10:
        return jsonify({'error': 'rating must be between 1 and 10'}), 400
    try:
        # wave5 fix (B2.7): the upsert silently overwrites an existing review;
        # probe first so an overwrite reports 'Review updated' (200) while a
        # first submission reports 'Review added' (201). (MySQL ODKU rowcount
        # 1=insert / 2=update would need a rowcount-returning db_manager
        # method, which is owned elsewhere.)
        existing_review = db.query(
            "SELECT 1 AS ok FROM movie_reviews WHERE user_id = %s AND movie_id = %s",
            (user_id, movie_id),
            fetch_all=False
        )
        db.add_review(user_id, movie_id, rating, data.get('review_text'), data.get('is_spoiler', False))
        if existing_review:
            return jsonify({'success': True, 'message': 'Review updated'}), 200
        return jsonify({'success': True, 'message': 'Review added'}), 201
    except Exception as e:
        logger.error(f"Review add error: {e}")
        return jsonify({'error': 'Failed to add review'}), 500

@api.route('/reviews/<int:review_id>/helpful', methods=['POST'])
def vote_helpful(review_id):
    """Vote a review as helpful (login required; one vote per user per review)"""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Login required'}), 401

    vote_key = (user_id, review_id)
    with _helpful_votes_lock:
        if vote_key in _helpful_votes_seen:
            return jsonify({'error': 'You already voted this review as helpful'}), 409
        # wave5 fix (B1.4): do not wipe the dedupe set at the cap (that reopened
        # every previously-recorded vote); reject new votes instead.
        if len(_helpful_votes_seen) >= HELPFUL_VOTE_DEDUPE_MAX:
            return jsonify({'error': 'Too many votes; limit reached'}), 429
        _helpful_votes_seen.add(vote_key)

    try:
        rows = db.vote_review_helpful(review_id)
    except Exception as e:
        with _helpful_votes_lock:
            _helpful_votes_seen.discard(vote_key)
        logger.error(f"Helpful vote error: {e}")
        return jsonify({'error': 'Failed to record vote'}), 500
    if rows <= 0:
        with _helpful_votes_lock:
            _helpful_votes_seen.discard(vote_key)
        return jsonify({'error': 'Review not found'}), 404
    return jsonify({'success': True}), 200


# ============================================================================
# ANALYTICS ENDPOINTS (uses stored functions, views, cursors)
# ============================================================================

@api.route('/analytics/genre-stats', methods=['GET'])
def genre_stats():
    """Genre comparison statistics using the genre_comparison_stats view"""
    try:
        stats = db.get_genre_comparison_stats()
        return jsonify({'stats': stats}), 200
    except Exception as e:
        logger.error(f"Genre stats error: {e}")
        return jsonify({'error': 'Failed to fetch genre stats'}), 500

@api.route('/analytics/genre-audit', methods=['GET'])
def genre_audit():
    """Genre audit report using cursor-based stored procedure"""
    try:
        report = db.run_genre_audit()
        return jsonify({'report': report}), 200
    except Exception as e:
        logger.error(f"Genre audit error: {e}")
        return jsonify({'error': 'Failed to run genre audit'}), 500

@api.route('/analytics/user-profile', methods=['GET'])
def user_profile_analytics():
    """Get user taste profile + review summary using stored functions"""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Login required'}), 401
    try:
        profile = db.get_user_taste_profile(user_id)
        review_summary = db.get_user_review_summary(user_id)
        advanced_stats = db.get_advanced_user_stats(user_id)
        return jsonify({
            'taste_profile': profile,
            'review_summary': review_summary,
            'stats': advanced_stats
        }), 200
    except Exception as e:
        logger.error(f"User profile analytics error: {e}")
        return jsonify({'error': 'Failed to fetch analytics'}), 500

@api.route('/analytics/movie-score/<int:movie_id>', methods=['GET'])
def movie_score(movie_id):
    """Get weighted recommendation score for a movie"""
    user_id = session.get('user_id')
    try:
        tier = db.get_movie_popularity_tier(movie_id)
        score = db.get_weighted_score(user_id, movie_id) if user_id else None
        return jsonify({'tier': tier, 'weighted_score': score}), 200
    except Exception as e:
        logger.error(f"Movie score error: {e}")
        return jsonify({'error': 'Failed to compute movie score'}), 500


@api.route('/search/ai', methods=['POST'])
def ai_search():
    """
    AI-powered semantic search using sentence-transformer embeddings
    Supports story/content-based search (e.g., "mars movie" finds "The Martian")
    """
    try:
        data = _json_body()
        query = (data.get('query') or '').strip()
        
        if not query:
            return jsonify({'error': 'Query required'}), 400

        try:
            limit = _parse_limit(data.get('limit'), default=20)
        except ValueError as exc:
            return jsonify({'error': str(exc)}), 400

        include_tv = data.get('include_tv', True)
        if isinstance(include_tv, str):
            include_tv = include_tv.strip().lower() not in {'0', 'false', 'no', 'off'}
        else:
            include_tv = bool(include_tv)
        
        logger.info(f"AI search query: {query}, limit: {limit}")
        
        results_by_id = {}
        used_semantic = False
        used_tmdb = False

        def _as_float(value, default=0.0):
            try:
                if value is None:
                    return default
                return float(value)
            except (TypeError, ValueError):
                return default

        def _quality_score(movie):
            rating_component = min(_as_float(movie.get('tmdb_rating')) / 10.0, 1.0) * 0.25
            popularity_component = min(_as_float(movie.get('popularity')) / 100.0, 1.0) * 0.12
            vote_component = min(_as_float(movie.get('vote_count')) / 2000.0, 1.0) * 0.08
            return (rating_component + popularity_component + vote_component) * 100.0

        def _allow_media(movie):
            if include_tv:
                return True
            media_type = (movie.get('media_type') or 'movie').lower()
            return media_type != 'tv'

        def _upsert_result(movie, score, match_type):
            movie_id = movie.get('movie_id')
            if not movie_id or not _allow_media(movie):
                return

            existing = results_by_id.get(movie_id)
            if existing:
                existing['search_score'] = round(max(existing.get('search_score', 0.0), score), 1)
                if match_type not in existing.get('match_type', ''):
                    existing['match_type'] = f"{existing.get('match_type', 'database')}+{match_type}"
                return

            results_by_id[movie_id] = {
                'movie_id': movie_id,
                'title': movie.get('title', ''),
                'overview': movie.get('overview', ''),
                'poster_path': movie.get('poster_path', ''),
                'tmdb_rating': movie.get('tmdb_rating'),
                'release_year': movie.get('release_year'),
                'genres': movie.get('genres', ''),
                'directors': movie.get('directors', movie.get('director', '')),
                'cast': movie.get('cast', ''),
                'media_type': movie.get('media_type', 'movie'),
                'search_score': round(score, 1),
                'match_type': match_type,
            }

        # Step 1: Dataset-backed DB search first (higher priority than cache/embeddings)
        try:
            db_results = db.search_movies_advanced(
                search_query=query,
                search_type='storyline',
                user_id=session.get('user_id'),
                limit=max(limit * 2, 30)
            )
            for movie in db_results:
                relevance = _as_float(movie.get('relevance_score'), 40.0)
                dataset_score = (min(relevance, 100.0) * 0.7) + _quality_score(movie)
                _upsert_result(movie, dataset_score, 'database')
            logger.info(f"Dataset-first DB matches: {len(results_by_id)}")
        except Exception as e:
            logger.warning(f"Advanced DB search failed: {e}")

        # Step 2: Semantic search boosts ranking quality, but does not dominate
        try:
            semantic_search = get_semantic_search()
            if semantic_search and semantic_search.movie_embeddings is not None:
                semantic_results = semantic_search.hybrid_search(query, top_k=max(limit * 2, 20))
                used_semantic = bool(semantic_results)
                for movie in semantic_results:
                    semantic_strength = _as_float(movie.get('final_score', movie.get('relevance_score', 0.0)))
                    semantic_score = (semantic_strength * 45.0) + _quality_score(movie)
                    _upsert_result(movie, semantic_score, 'semantic')
                logger.info(f"Semantic augmentation completed: {len(semantic_results)} candidates")
        except Exception as e:
            logger.warning(f"Semantic search engine failed: {e}")

        # Step 3: Token fallback search for niche/long-tail terms
        if len(results_by_id) < limit:
            try:
                query_tokens = [token for token in query.split() if len(token) > 2]
                for token in query_tokens:
                    token_matches = db.search_movies(token, limit=10)
                    for movie in token_matches:
                        token_relevance = _as_float(movie.get('relevance_score'), 30.0)
                        token_score = (token_relevance * 0.6) + _quality_score(movie)
                        _upsert_result(movie, token_score, 'database')
                    if len(results_by_id) >= limit:
                        break
            except Exception as e:
                logger.warning(f"Token fallback search failed: {e}")

        # Step 4: Online TMDB fetch for missing/new titles, then rank from DB copy
        if len(results_by_id) < max(5, limit // 2):
            try:
                from ai.content_pipeline import pipeline
                fetched = pipeline.fetch_on_demand(search_query=query)
                if fetched:
                    used_tmdb = True
                    refreshed_results = db.search_movies_advanced(
                        search_query=query,
                        search_type='storyline',
                        user_id=session.get('user_id'),
                        limit=max(limit * 2, 30)
                    )
                    for movie in refreshed_results:
                        relevance = _as_float(movie.get('relevance_score'), 35.0)
                        tmdb_score = (relevance * 0.55) + _quality_score(movie)
                        _upsert_result(movie, tmdb_score, 'tmdb')
                    logger.info(f"TMDB supplementation added candidates; total now {len(results_by_id)}")
            except Exception as e:
                logger.warning(f"TMDB fetch failed: {e}")

        results = sorted(
            results_by_id.values(),
            key=lambda item: (
                item.get('search_score', 0.0),
                _as_float(item.get('tmdb_rating')),
                _as_float(item.get('popularity')),
            ),
            reverse=True
        )[:limit]

        source_parts = ['database']
        if used_semantic:
            source_parts.append('semantic')
        if used_tmdb:
            source_parts.append('tmdb')
        
        return jsonify({
            'results': results,
            'query': query,
            'count': len(results),
            'ai_powered': True,
            'source': '+'.join(source_parts),
            'message': 'AI search now prioritizes dataset quality and uses semantic and TMDB as smart augmentation'
        }), 200
        
    except Exception as e:
        logger.error(f"AI search error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'Search failed'}), 500


@api.route('/movie/search', methods=['GET'])
def search_movies():
    """
    Enhanced hybrid search with storyline/semantic support
    1. Try semantic search for natural language queries
    2. Search database with semantic/storyline matching
    3. Fetch from TMDB API if needed (get latest content)
    4. Return comprehensive results
    """
    try:
        query = (request.args.get('q', '') or '').strip()
        try:
            limit = _parse_limit(request.args.get('limit'), default=50)
        except ValueError as exc:
            return jsonify({'error': str(exc)}), 400
        search_type = request.args.get('type', 'hybrid')  # title, storyline, semantic, hybrid
        user_id = session.get('user_id')
        
        if not query:
            return jsonify({'error': 'Search query required'}), 400

        if search_type not in VALID_SEARCH_TYPES:
            return jsonify({'error': f'Invalid search type. Allowed: {sorted(VALID_SEARCH_TYPES)}'}), 400
        
        logger.info(f"🔍 Searching: '{query}' (type: {search_type})")
        
        # Step 1: Try semantic/hybrid search first for natural language queries
        semantic_results = []
        if search_type in ['semantic', 'hybrid'] and len(query.split()) > 2:
            try:
                semantic_search = get_semantic_search()
                if semantic_search:
                    logger.info("🧠 Using semantic search...")
                    if search_type == 'hybrid':
                        semantic_results = semantic_search.hybrid_search(query, top_k=limit)
                    else:
                        semantic_results = semantic_search.search(query, top_k=limit)
                    logger.info(f"✓ Semantic search: {len(semantic_results)} results")
            except Exception as e:
                logger.warning(f"Semantic search failed: {e}")
        
        # Step 2: Try advanced database search with storyline support
        db_movies = []
        try:
            db_movies = db.search_movies_advanced(
                search_query=query,
                search_type='storyline' if search_type == 'hybrid' else search_type,
                user_id=user_id,
                limit=limit
            )
            logger.info(f"📊 Database search returned {len(db_movies)} results")
        except Exception as e:
            logger.warning(f"Advanced search failed, falling back: {e}")
            # Fallback to regular search
            db_movies = db.search_movies(query, limit=limit)
        
        # Step 3: Merge results (semantic + database)
        if semantic_results:
            # Merge: prefer semantic results, add db results that aren't in semantic
            merged_movies = semantic_results.copy()
            
            for db_movie in db_movies:
                if db_movie['movie_id'] not in {m.get('movie_id') for m in merged_movies}:
                    merged_movies.append(db_movie)
            
            db_movies = merged_movies[:limit]
            logger.info(f"🔄 Merged results: {len(db_movies)} movies")
        
        # NOTE: This is a read-only GET. Content ingestion from TMDB is intentionally
        # not performed here (it is handled by the background content pipeline) so a
        # GET never writes to the database.
        
        logger.info(f"📊 Final result: {len(db_movies)} movies")
        
        return jsonify({
            'query': query,
            'search_type': search_type,
            'movies': db_movies,
            'count': len(db_movies),
            'source': 'semantic+database' if semantic_results else 'database',
            'supports_storyline': True,
            'supports_semantic': True,
            'success': True
        }), 200
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        import traceback
        traceback.print_exc()
        # wave5 fix (B3.3): do not leak internal exception text to clients.
        return jsonify({
            'error': 'Internal server error',
            'success': False
        }), 500


@api.route('/movie/top-rated', methods=['GET'])
def get_top_rated():
    """Get top-rated movies with pagination support"""
    try:
        try:
            limit = _parse_limit(request.args.get('limit'), default=20)
            offset = _parse_offset(request.args.get('offset'), default=0)
        except ValueError as exc:
            return jsonify({'error': str(exc)}), 400
        order_by = request.args.get('order_by', 'elo_score')
        media_type = request.args.get('media_type', 'all')

        if order_by not in VALID_ORDER_BY:
            return jsonify({'error': f'Invalid order_by. Allowed: {sorted(VALID_ORDER_BY)}'}), 400
        if media_type not in VALID_MEDIA_TYPES:
            return jsonify({'error': f'Invalid media_type. Allowed: {sorted(VALID_MEDIA_TYPES)}'}), 400

        # Read-only GET: no TMDB ingestion here. New content is ingested by the
        # background content pipeline, never by a client GET.
        movies = db.get_top_movies(limit=limit, order_by=order_by, offset=offset, media_type=media_type)
        
        return jsonify({
            'movies': movies,
            'count': len(movies),
            'offset': offset,
            'has_more': len(movies) == limit,
            'source': 'database'
        }), 200
        
    except Exception as e:
        logger.error(f"Top rated error: {e}")
        return jsonify({'error': 'Internal server error'}), 500


# ============================================================================
# STATS ENDPOINTS
# ============================================================================

@api.route('/stats', methods=['GET'])
def get_stats():
    """Get platform statistics"""
    try:
        stats = {
            'total_movies': db.get_movie_count(),
            'total_users': db.get_user_count()
        }
        
        return jsonify(stats), 200
        
    except Exception as e:
        logger.error(f"Stats error: {e}")
        return jsonify({'error': 'Internal server error'}), 500


@api.route('/cache/stats', methods=['GET'])
def get_cache_stats():
    """Get cache statistics and monitoring data"""
    try:
        # Get cache stats from database
        db_cache_stats = db.get_cache_statistics()
        
        # Format for response
        cache_data = {}
        for stat in db_cache_stats:
            cache_data[stat['cache_type']] = {
                'hits': stat['cache_hits'],
                'misses': stat['cache_misses'],
                'hit_rate': float(stat['hit_rate']) if stat['hit_rate'] else 0,
                'avg_response_time_ms': float(stat['avg_response_time_ms']),
                'memory_usage_kb': stat['memory_usage_kb'],
                'last_updated': str(stat['recorded_at'])
            }
        
        # Get cache manager stats if available
        try:
            cache_manager_stats = cache_manager.get_stats() or {}
        except Exception as exc:
            logger.warning(f"Cache manager stats unavailable: {exc}")
            cache_manager_stats = {
                'movie_cache': {'size': 0, 'max_size': 100, 'hit_rate': 0},
                'vector_cache': {'size': 0, 'max_size': 500, 'hit_rate': 0}
            }
        
        return jsonify({
            'database_stats': cache_data,
            'cache_manager': cache_manager_stats,
            'memory_savings': '77x reduction (54MB → 700KB)',
            'status': 'operational',
            'max_capacity': {
                'movies': 100,
                'vectors': 500
            },
            'success': True
        }), 200
        
    except Exception as e:
        logger.error(f"Cache stats error: {e}")
        import traceback
        traceback.print_exc()
        # wave5 fix (B3.3): do not leak internal exception text to clients.
        return jsonify({
            'error': 'Internal server error',
            'success': False
        }), 500


@api.route('/test/tmdb', methods=['GET'])
def test_tmdb():
    """Test TMDB API - Use this to verify TMDB is working"""
    try:
        query = request.args.get('q', 'mission impossible')
        
        from tmdb.fetcher import TMDBFetcher
        fetcher = TMDBFetcher()
        
        # Test search
        results = fetcher.search_movies(query, page=1)
        
        if results and 'results' in results:
            return jsonify({
                'status': '✅ SUCCESS',
                'query': query,
                'total_results': results.get('total_results', 0),
                'page': results.get('page', 1),
                'movies_found': len(results['results']),
                'titles': [m.get('title', 'N/A') for m in results['results'][:10]],
                'api_working': True
            }), 200
        else:
            return jsonify({
                'status': '❌ FAILED',
                'error': 'No results from TMDB',
                'api_working': False
            }), 500
            
    except Exception as e:
        logger.error(f"TMDB test error: {e}", exc_info=True)
        # Do not leak internal details/tracebacks to the caller
        return jsonify({
            'status': '❌ ERROR',
            'error': 'TMDB test failed',
            'api_working': False
        }), 500


@api.route('/cache/monitor', methods=['GET'])
def monitor_cache():
    """Real-time cache monitoring with detailed metrics"""
    try:
        cache_stats = cache_manager.get_stats() or {}
        movie_cache = cache_stats.get('movie_cache') or {}
        vector_cache = cache_stats.get('vector_cache') or {}

        def _stat(flat_key, nested, nested_key, default=0):
            value = cache_stats.get(flat_key)
            if value is None:
                value = nested.get(nested_key, default)
            return value if value is not None else default

        movie_count = _stat('movie_count', movie_cache, 'size')
        vector_count = _stat('vector_count', vector_cache, 'size')
        movie_max = _stat('movie_max_size', movie_cache, 'max_size', 100) or 100
        vector_max = _stat('vector_max_size', vector_cache, 'max_size', 500) or 500

        # Calculate percentages
        movie_usage_percent = (movie_count / movie_max) * 100 if movie_max else 0
        vector_usage_percent = (vector_count / vector_max) * 100 if vector_max else 0
        
        # Get hit rates
        movie_hits = _stat('movie_hits', movie_cache, 'hits')
        movie_misses = _stat('movie_misses', movie_cache, 'misses')
        vector_hits = _stat('vector_hits', vector_cache, 'hits')
        vector_misses = _stat('vector_misses', vector_cache, 'misses')
        
        movie_total = movie_hits + movie_misses
        vector_total = vector_hits + vector_misses
        
        movie_hit_rate = (movie_hits / movie_total * 100) if movie_total > 0 else 0
        vector_hit_rate = (vector_hits / vector_total * 100) if vector_total > 0 else 0
        
        # Check if refill needed
        try:
            needs_refill = cache_manager.needs_refill()
        except Exception as exc:
            logger.warning(f"needs_refill check failed: {exc}")
            needs_refill = False
        
        return jsonify({
            'timestamp': cache_stats.get('timestamp'),
            'movie_cache': {
                'count': movie_count,
                'max_size': movie_max,
                'usage_percent': round(movie_usage_percent, 2),
                'hits': movie_hits,
                'misses': movie_misses,
                'hit_rate': round(movie_hit_rate, 2),
                'eviction_strategy': 'LRU'
            },
            'vector_cache': {
                'count': vector_count,
                'max_size': vector_max,
                'usage_percent': round(vector_usage_percent, 2),
                'hits': vector_hits,
                'misses': vector_misses,
                'hit_rate': round(vector_hit_rate, 2)
            },
            'status': {
                'needs_refill': needs_refill,
                'health': 'healthy' if movie_hit_rate > 30 else 'low_hit_rate',
                'memory_efficient': True
            },
            'architecture': {
                'sliding_window': True,
                'lazy_loading': True,
                'candidate_generation': True,
                'infinite_content': True
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Cache monitor error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'Internal server error'}), 500


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@api.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@api.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500
