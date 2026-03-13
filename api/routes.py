"""
Flask API Routes
RESTful API endpoints for the CineSense application
"""

from flask import Blueprint, request, jsonify, session, current_app
from database.db_manager import db
from ai.recommender import recommender
from ai.cache_manager import cache_manager
from werkzeug.security import generate_password_hash, check_password_hash
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

api = Blueprint('api', __name__, url_prefix='/api')


def get_semantic_search():
    """Return the pre-initialized semantic search engine from app startup"""
    return getattr(current_app, 'semantic_search', None)


# ============================================================================
# USER ENDPOINTS
# ============================================================================

@api.route('/user/signup', methods=['POST'])
def signup():
    """Create a new user account"""
    try:
        data = request.get_json()
        
        username = data.get('username')
        email = data.get('email')
        password = data.get('password')
        
        if not all([username, email, password]):
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Check if user exists
        existing_user = db.get_user_by_username(username)
        if existing_user:
            return jsonify({'error': 'Username already exists'}), 409
        
        # Hash password
        password_hash = generate_password_hash(password)
        
        # Create user
        user_id = db.create_user(username, email, password_hash)
        
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
        data = request.get_json()
        
        username = data.get('username')
        password = data.get('password')
        
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
        
        if not stats:
            return jsonify({'error': 'User not found'}), 404
        
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
        interactions = db.get_user_interactions(user_id, limit=1000)
        
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
        ratings = []
        years = []
        
        for interaction in interactions:
            chosen_id = interaction.get('chosen_movie_id')
            if chosen_id:
                movie = db.get_movie_by_id(chosen_id)
                if movie:
                    chosen_movies.append(movie)
                    
                    # Count genres
                    if movie.get('genres'):
                        for genre in movie['genres'].split(','):
                            genre = genre.strip()
                            genre_counts[genre] = genre_counts.get(genre, 0) + 1
                    
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
        interactions = db.get_user_interactions(user_id, limit=1000)
        
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
        interactions = db.get_user_interactions(user_id, limit=10000)
        
        # Collect unique movie IDs from interactions
        compared_ids = set()
        for interaction in interactions:
            if interaction.get('movie_1_id'):
                compared_ids.add(interaction['movie_1_id'])
            if interaction.get('movie_2_id'):
                compared_ids.add(interaction['movie_2_id'])
        
        # Get movie details
        movies = []
        for movie_id in compared_ids:
            movie = db.get_movie_by_id(movie_id)
            if movie:
                movies.append(movie)
        
        return jsonify({'movies': movies, 'count': len(movies)}), 200
        
    except Exception as e:
        logger.error(f"Compared movies error: {e}")
        return jsonify({'movies': []}), 200


# ============================================================================
# RECOMMENDATION ENDPOINTS
# ============================================================================

@api.route('/recommendations', methods=['GET'])
def get_recommendations():
    """Get personalized movie recommendations"""
    try:
        user_id = session.get('user_id')
        
        if not user_id:
            # Guest user: return popular movies
            movies = db.get_top_movies(limit=20, order_by='popularity')
            return jsonify({
                'movies': movies,
                'personalized': False
            }), 200
        
        # Get personalized recommendations
        limit = request.args.get('limit', 20, type=int)
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
        
        limit = request.args.get('limit', 20, type=int)
        strategy = request.args.get('strategy', 'mixed')  # mixed, genre, popularity, exploration
        
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
        
        data = request.get_json()
        
        movie1_id = data.get('movie1_id')
        movie2_id = data.get('movie2_id')
        chosen_id = data.get('chosen_id')
        
        if not all([movie1_id, movie2_id, chosen_id]):
            logger.error(f"Missing fields: movie1_id={movie1_id}, movie2_id={movie2_id}, chosen_id={chosen_id}")
            return jsonify({'error': 'Missing required fields'}), 400
        
        if chosen_id not in [movie1_id, movie2_id]:
            logger.error(f"Invalid chosen_id: {chosen_id} not in [{movie1_id}, {movie2_id}]")
            return jsonify({'error': 'Invalid chosen movie'}), 400
        
        # Record interaction using stored procedure (handles Elo automatically)
        session_id = session.get('session_id', str(user_id))
        success = db.record_interaction(
            user_id=user_id,
            movie_1_id=movie1_id,
            movie_2_id=movie2_id,
            chosen_movie_id=chosen_id,
            session_id=session_id
        )
        
        if not success:
            logger.error(f"Database operation failed for user {user_id}")
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
        return jsonify({'error': 'Internal server error', 'details': str(e)}), 500


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
        limit = request.args.get('limit', 12, type=int)
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
        genre = request.args.get('genre', '')
        media_type = request.args.get('media_type', 'all')
        limit = request.args.get('limit', 20, type=int)
        offset = request.args.get('offset', 0, type=int)
        
        logger.info(f"Getting movies by genre: {genre}, media_type: {media_type}, offset: {offset}")
        
        # If no genre specified, get all movies
        if not genre or genre == 'all' or genre == '':
            movies = db.get_top_movies(limit=limit, offset=offset, media_type=media_type)
        else:
            movies = db.get_movies_by_genre_and_type(
                genre_name=genre,
                media_type=media_type,
                limit=limit,
                offset=offset
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
    items = db.get_user_watchlist(user_id, status=status_filter)
    return jsonify({'watchlist': items, 'count': len(items)}), 200

@api.route('/watchlist', methods=['POST'])
def add_to_watchlist():
    """Add movie to watchlist"""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Login required'}), 401
    data = request.get_json()
    movie_id = data.get('movie_id')
    if not movie_id:
        return jsonify({'error': 'movie_id required'}), 400
    try:
        db.add_to_watchlist(user_id, movie_id, data.get('priority', 5), data.get('status', 'planned'), data.get('note'))
        return jsonify({'success': True, 'message': 'Added to watchlist'}), 201
    except Exception as e:
        logger.error(f"Watchlist add error: {e}")
        return jsonify({'error': str(e)}), 500

@api.route('/watchlist/<int:movie_id>', methods=['PUT'])
def update_watchlist(movie_id):
    """Update watchlist entry"""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Login required'}), 401
    data = request.get_json()
    status = data.get('status')
    if not status:
        return jsonify({'error': 'status required'}), 400
    rows = db.update_watchlist_status(user_id, movie_id, status, data.get('rating'))
    return jsonify({'success': rows > 0}), 200

@api.route('/watchlist/<int:movie_id>', methods=['DELETE'])
def remove_from_watchlist(movie_id):
    """Remove from watchlist"""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Login required'}), 401
    rows = db.remove_from_watchlist(user_id, movie_id)
    return jsonify({'success': rows > 0}), 200


# ============================================================================
# REVIEW ENDPOINTS
# ============================================================================

@api.route('/reviews/<int:movie_id>', methods=['GET'])
def get_reviews(movie_id):
    """Get reviews for a movie"""
    reviews = db.get_movie_reviews(movie_id)
    return jsonify({'reviews': reviews, 'count': len(reviews)}), 200

@api.route('/reviews', methods=['POST'])
def add_review():
    """Add a movie review"""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Login required'}), 401
    data = request.get_json()
    movie_id = data.get('movie_id')
    rating = data.get('rating')
    if not movie_id or rating is None:
        return jsonify({'error': 'movie_id and rating required'}), 400
    try:
        db.add_review(user_id, movie_id, rating, data.get('review_text'), data.get('is_spoiler', False))
        return jsonify({'success': True, 'message': 'Review added'}), 201
    except Exception as e:
        logger.error(f"Review add error: {e}")
        return jsonify({'error': str(e)}), 500

@api.route('/reviews/<int:review_id>/helpful', methods=['POST'])
def vote_helpful(review_id):
    """Vote a review as helpful"""
    db.vote_review_helpful(review_id)
    return jsonify({'success': True}), 200


# ============================================================================
# ANALYTICS ENDPOINTS (uses stored functions, views, cursors)
# ============================================================================

@api.route('/analytics/genre-stats', methods=['GET'])
def genre_stats():
    """Genre comparison statistics using the genre_comparison_stats view"""
    stats = db.get_genre_comparison_stats()
    return jsonify({'stats': stats}), 200

@api.route('/analytics/genre-audit', methods=['GET'])
def genre_audit():
    """Genre audit report using cursor-based stored procedure"""
    report = db.run_genre_audit()
    return jsonify({'report': report}), 200

@api.route('/analytics/user-profile', methods=['GET'])
def user_profile_analytics():
    """Get user taste profile + review summary using stored functions"""
    user_id = session.get('user_id')
    if not user_id:
        return jsonify({'error': 'Login required'}), 401
    profile = db.get_user_taste_profile(user_id)
    review_summary = db.get_user_review_summary(user_id)
    advanced_stats = db.get_advanced_user_stats(user_id)
    return jsonify({
        'taste_profile': profile,
        'review_summary': review_summary,
        'stats': advanced_stats
    }), 200

@api.route('/analytics/movie-score/<int:movie_id>', methods=['GET'])
def movie_score(movie_id):
    """Get weighted recommendation score for a movie"""
    user_id = session.get('user_id')
    tier = db.get_movie_popularity_tier(movie_id)
    score = db.get_weighted_score(user_id, movie_id) if user_id else None
    return jsonify({'tier': tier, 'weighted_score': score}), 200


@api.route('/search/ai', methods=['POST'])
def ai_search():
    """
    AI-powered semantic search using sentence-transformer embeddings
    Supports story/content-based search (e.g., "mars movie" finds "The Martian")
    """
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        
        if not query:
            return jsonify({'error': 'Query required'}), 400
        
        limit = data.get('limit', 20)
        include_tv = data.get('include_tv', True)
        
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
        query = request.args.get('q', '')
        limit = request.args.get('limit', 50, type=int)
        search_type = request.args.get('type', 'hybrid')  # title, storyline, semantic, hybrid
        user_id = session.get('user_id')
        
        if not query:
            return jsonify({'error': 'Search query required'}), 400
        
        logger.info(f"🔍 Searching: '{query}' (type: {search_type})")
        
        # Step 1: Try semantic/hybrid search first for natural language queries
        semantic_results = []
        if search_type in ['semantic', 'hybrid'] and len(query.split()) > 2:
            try:
                semantic_search = get_semantic_search()
                if semantic_search:
                    logger.info(f"🧠 Using semantic search...")
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
            # Create lookup for database results
            db_movie_ids = {m['movie_id'] for m in db_movies}
            
            # Merge: prefer semantic results, add db results that aren't in semantic
            merged_movies = semantic_results.copy()
            
            for db_movie in db_movies:
                if db_movie['movie_id'] not in {m.get('movie_id') for m in merged_movies}:
                    merged_movies.append(db_movie)
            
            db_movies = merged_movies[:limit]
            logger.info(f"🔄 Merged results: {len(db_movies)} movies")
        
        # Step 4: If insufficient results, fetch from TMDB
        if len(db_movies) < 5:
            logger.info(f"📡 Fetching from TMDB to supplement results...")
            from ai.content_pipeline import pipeline
            try:
                tmdb_movies = pipeline.fetch_on_demand(search_query=query)
                if tmdb_movies:
                    logger.info(f"✅ TMDB returned {len(tmdb_movies)} movies")
                    # Re-search database with new content
                    db_movies = db.search_movies_advanced(
                        search_query=query,
                        search_type='storyline',
                        user_id=user_id,
                        limit=limit
                    )
            except Exception as e:
                logger.error(f"⚠️ TMDB fetch failed: {e}")
        
        logger.info(f"📊 Final result: {len(db_movies)} movies")
        
        return jsonify({
            'query': query,
            'search_type': search_type,
            'movies': db_movies,
            'count': len(db_movies),
            'source': 'semantic+database+tmdb' if semantic_results else 'database+tmdb',
            'supports_storyline': True,
            'supports_semantic': True,
            'success': True
        }), 200
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'error': 'Internal server error',
            'details': str(e),
            'success': False
        }), 500


@api.route('/movie/top-rated', methods=['GET'])
def get_top_rated():
    """Get top-rated movies with pagination support - fetches from TMDB on-demand"""
    try:
        limit = request.args.get('limit', 20, type=int)
        offset = request.args.get('offset', 0, type=int)
        order_by = request.args.get('order_by', 'elo_score')
        media_type = request.args.get('media_type', 'all')
        
        # First, try to get movies from database
        movies = db.get_top_movies(limit=limit, order_by=order_by, offset=offset, media_type=media_type)
        
        # If database doesn't have enough movies, fetch from TMDB
        if len(movies) < limit:
            from tmdb.fetcher import TMDBFetcher
            fetcher = TMDBFetcher()
            
            # Calculate which TMDB page to fetch
            tmdb_page = (offset // 20) + 1
            
            # Fetch from TMDB based on order_by and media_type
            if media_type == 'tv':
                if order_by == 'popularity':
                    data = fetcher.get_popular_tv_series(page=tmdb_page)
                else:
                    data = fetcher.get_top_rated_tv_series(page=tmdb_page)
            else:  # movie or all
                if order_by == 'popularity':
                    data = fetcher.get_popular_movies(page=tmdb_page)
                elif order_by == 'tmdb_rating':
                    data = fetcher.get_top_rated_movies(page=tmdb_page)
                else:  # elo_score or default
                    data = fetcher.get_popular_movies(page=tmdb_page)
            
            tmdb_movies = data.get('results', []) if data else []
            
            # Store fetched movies in database for future use
            for movie in tmdb_movies:
                try:
                    movie_data = fetcher.parse_movie_data(movie)
                    # Set media_type for TV series
                    if media_type == 'tv' and 'media_type' not in movie_data:
                        movie_data['media_type'] = 'tv'
                    db.insert_movie(movie_data)
                    
                    # Store genres
                    genres = fetcher.parse_genres(movie)
                    for genre in genres:
                        existing_genre = db.get_genre_by_name(genre['name'])
                        if existing_genre:
                            genre_id = existing_genre['genre_id']
                        else:
                            genre_id = db.insert_genre(genre['name'], genre.get('id'))
                        db.link_movie_genre(movie_data['movie_id'], genre_id)
                except Exception as e:
                    logger.warning(f"Error storing movie {movie.get('id')}: {e}")
                    continue
            
            # Re-fetch from database to get the stored movies
            movies = db.get_top_movies(limit=limit, order_by=order_by, offset=offset)
        
        return jsonify({
            'movies': movies,
            'count': len(movies),
            'offset': offset,
            'has_more': len(movies) == limit,
            'source': 'lazy_loaded'
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
            cache_manager_stats = cache_manager.get_stats()
        except:
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
        return jsonify({
            'error': 'Internal server error',
            'details': str(e),
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
        logger.error(f"TMDB test error: {e}")
        import traceback
        return jsonify({
            'status': '❌ ERROR',
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500


@api.route('/cache/monitor', methods=['GET'])
def monitor_cache():
    """Real-time cache monitoring with detailed metrics"""
    try:
        cache_stats = cache_manager.get_stats()
        
        # Calculate percentages
        movie_usage_percent = (cache_stats.get('movie_count', 0) / 100) * 100
        vector_usage_percent = (cache_stats.get('vector_count', 0) / 500) * 100
        
        # Get hit rates
        movie_hits = cache_stats.get('movie_hits', 0)
        movie_misses = cache_stats.get('movie_misses', 0)
        vector_hits = cache_stats.get('vector_hits', 0)
        vector_misses = cache_stats.get('vector_misses', 0)
        
        movie_total = movie_hits + movie_misses
        vector_total = vector_hits + vector_misses
        
        movie_hit_rate = (movie_hits / movie_total * 100) if movie_total > 0 else 0
        vector_hit_rate = (vector_hits / vector_total * 100) if vector_total > 0 else 0
        
        # Check if refill needed
        needs_refill = cache_manager.needs_refill()
        
        return jsonify({
            'timestamp': cache_stats.get('timestamp'),
            'movie_cache': {
                'count': cache_stats.get('movie_count', 0),
                'max_size': 100,
                'usage_percent': round(movie_usage_percent, 2),
                'hits': movie_hits,
                'misses': movie_misses,
                'hit_rate': round(movie_hit_rate, 2),
                'eviction_strategy': 'LRU'
            },
            'vector_cache': {
                'count': cache_stats.get('vector_count', 0),
                'max_size': 500,
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
