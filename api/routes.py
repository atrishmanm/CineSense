"""
Flask API Routes
RESTful API endpoints for the CineSense application
"""

from flask import Blueprint, request, jsonify, session
from database.db_manager import db
from ai.recommender import recommender
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
        
        movie1, movie2 = recommender.get_comparison_pair(user_id)
        
        if not movie1 or not movie2:
            return jsonify({'error': 'Could not generate comparison pair'}), 500
        
        return jsonify({
            'movie1': movie1,
            'movie2': movie2
        }), 200
        
    except Exception as e:
        logger.error(f"Comparison error: {e}")
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
    """Get top-rated movies"""
    try:
        limit = request.args.get('limit', 20, type=int)
        order_by = request.args.get('order_by', 'elo_score')
        
        movies = db.get_top_movies(limit=limit, order_by=order_by)
        
        return jsonify({
            'movies': movies,
            'count': len(movies)
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


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@api.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404


@api.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500
