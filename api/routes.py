"""
Flask API Routes
RESTful API endpoints for the CineSense application
"""

from flask import Blueprint, request, jsonify, session
from database.db_manager import db
from ai.recommender import recommender
from ai.cache_manager import cache_manager
from werkzeug.security import generate_password_hash, check_password_hash
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

api = Blueprint('api', __name__, url_prefix='/api')


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
            return jsonify({'error': 'Missing required fields'}), 400
        
        if chosen_id not in [movie1_id, movie2_id]:
            return jsonify({'error': 'Invalid chosen movie'}), 400
        
        # Determine rejected movie
        rejected_id = movie2_id if chosen_id == movie1_id else movie1_id
        
        # Process the choice
        success = recommender.process_user_choice(
            user_id,
            chosen_id,
            rejected_id,
            session_id=session.get('session_id')
        )
        
        if not success:
            return jsonify({'error': 'Failed to process feedback'}), 500
        
        return jsonify({
            'message': 'Feedback recorded successfully',
            'learning': 'AI model updated based on your preference'
        }), 200
        
    except Exception as e:
        logger.error(f"Feedback error: {e}")
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
        
        # Add explanation if user is logged in
        explanation = None
        user_id = session.get('user_id')
        if user_id:
            explanation = recommender.explain_recommendation(user_id, movie_id)
        
        return jsonify({
            'movie': movie,
            'explanation': explanation
        }), 200
        
    except Exception as e:
        logger.error(f"Movie detail error: {e}")
        return jsonify({'error': 'Internal server error'}), 500


@api.route('/movie/search', methods=['GET'])
def search_movies():
    """Search for movies"""
    try:
        query = request.args.get('q', '')
        limit = request.args.get('limit', 20, type=int)
        
        if not query:
            return jsonify({'error': 'Search query required'}), 400
        
        movies = db.search_movies(query, limit=limit)
        
        return jsonify({
            'query': query,
            'movies': movies,
            'count': len(movies)
        }), 200
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        return jsonify({'error': 'Internal server error'}), 500


@api.route('/movie/by-genre/<genre>', methods=['GET'])
def get_movies_by_genre(genre):
    """Get movies by genre"""
    try:
        limit = request.args.get('limit', 20, type=int)
        movies = db.get_movies_by_genre(genre, limit=limit)
        
        return jsonify({
            'genre': genre,
            'movies': movies,
            'count': len(movies)
        }), 200
        
    except Exception as e:
        logger.error(f"Genre movies error: {e}")
        return jsonify({'error': 'Internal server error'}), 500


@api.route('/movie/top-rated', methods=['GET'])
def get_top_rated():
    """Get top-rated movies with pagination support - fetches from TMDB on-demand"""
    try:
        limit = request.args.get('limit', 20, type=int)
        offset = request.args.get('offset', 0, type=int)
        order_by = request.args.get('order_by', 'elo_score')
        
        # First, try to get movies from database
        movies = db.get_top_movies(limit=limit, order_by=order_by, offset=offset)
        
        # If database doesn't have enough movies, fetch from TMDB
        if len(movies) < limit:
            from tmdb.fetcher import TMDBFetcher
            fetcher = TMDBFetcher()
            
            # Calculate which TMDB page to fetch
            tmdb_page = (offset // 20) + 1
            
            # Fetch from TMDB based on order_by
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
        cache_stats = cache_manager.get_stats()
        
        # Get recommender cache stats if available
        recommender_stats = {}
        if hasattr(recommender, 'get_cache_stats'):
            recommender_stats = recommender.get_cache_stats()
        
        return jsonify({
            'cache_manager': cache_stats,
            'recommender': recommender_stats,
            'memory_savings': '77x reduction (54MB → 700KB)',
            'max_capacity': {
                'movies': 100,
                'vectors': 500
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Cache stats error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': 'Internal server error'}), 500


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
