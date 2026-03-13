"""
CineSense - Fully Integrated Flask Application
AI-Based Movie Recommendation Platform with ALL Features
"""

from flask import Flask, render_template, session, request, jsonify
from flask.json.provider import DefaultJSONProvider
from flask_cors import CORS
from config import Config
from api.routes import api
import secrets
import logging
from decimal import Decimal
import os
import torch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('huggingface_hub').setLevel(logging.WARNING)
logging.getLogger('urllib3').setLevel(logging.WARNING)


def _fallback_chat(message, db_manager):
    """Provide chat responses without AI models by using keyword matching"""
    message_lower = message.lower()
    
    mood_genres = {
        'happy': 'Comedy', 'fun': 'Comedy', 'laugh': 'Comedy',
        'sad': 'Drama', 'emotional': 'Drama',
        'excited': 'Action', 'thrilling': 'Action', 'exciting': 'Action',
        'scary': 'Horror', 'horror': 'Horror', 'creepy': 'Horror',
        'romantic': 'Romance', 'love': 'Romance', 'romance': 'Romance',
        'bored': 'Thriller', 'boring': 'Thriller',
        'think': 'Drama', 'deep': 'Drama',
    }
    
    movies = []
    response = "Here are some movies you might enjoy!"
    
    try:
        # Check for mood/genre keywords
        for keyword, genre in mood_genres.items():
            if keyword in message_lower:
                movies = db_manager.get_movies_by_genre(genre, limit=10)
                response = f"Based on your mood, here are some {genre.lower()} movies:"
                break
        
        if not movies:
            # Check for trending keywords  
            if any(w in message_lower for w in ['trending', 'popular', 'hot', 'viral']):
                movies = db_manager.get_top_movies(limit=10, order_by='popularity')
                response = "Here are the most popular movies right now:"
            elif any(w in message_lower for w in ['recommend', 'suggest', 'watch', 'movie', 'film']):
                movies = db_manager.get_top_movies(limit=10, order_by='tmdb_rating')
                response = "Here are some highly-rated movies I recommend:"
            elif any(w in message_lower for w in ['hi', 'hello', 'hey']):
                response = "Hey there! I'm your movie assistant. Tell me your mood or what kind of movies you like!"
                return {'response': response, 'recommendations': [], 'type': 'chat'}
            elif any(w in message_lower for w in ['thank', 'thanks']):
                response = "You're welcome! Let me know if you need more recommendations."
                return {'response': response, 'recommendations': [], 'type': 'chat'}
            else:
                movies = db_manager.get_top_movies(limit=10, order_by='elo_score')
                response = "Here are some top-rated movies. Tell me your mood for more specific picks!"
    except Exception as e:
        logger.error(f"Fallback chat error: {e}")
        response = "I'd love to help! Tell me your mood or preferred genre."
    
    return {'response': response, 'recommendations': movies, 'type': 'fallback'}


def _fallback_mood(mood_input, db_manager):
    """Provide mood recommendations without AI models"""
    mood_input_lower = mood_input.lower()
    
    mood_genre_map = {
        'happy': (['Comedy', 'Animation', 'Family'], "You're in a great mood! Here are some feel-good movies:"),
        'sad': (['Drama', 'Romance'], "Here are some emotional movies that might resonate:"),
        'excited': (['Action', 'Adventure', 'Sci-Fi'], "Let's keep that energy going!"),
        'scared': (['Horror', 'Thriller'], "Want a thrill? Check these out:"),
        'romantic': (['Romance', 'Drama'], "Feeling the love! Here are some romantic picks:"),
        'bored': (['Action', 'Thriller', 'Sci-Fi'], "Time for something exciting!"),
        'anxious': (['Comedy', 'Animation'], "Here are some calming movies to unwind:"),
        'thoughtful': (['Drama', 'Documentary'], "In a contemplative mood? Try these:"),
    }
    
    # Detect mood from input
    detected_mood = 'thoughtful'
    for mood in mood_genre_map:
        if mood in mood_input_lower:
            detected_mood = mood
            break
    
    genres, explanation = mood_genre_map.get(detected_mood, (['Drama'], "Here are some movies for you:"))
    
    movies = []
    try:
        for genre in genres:
            genre_movies = db_manager.get_movies_by_genre(genre, limit=10)
            if genre_movies:
                movies.extend(genre_movies)
        # Deduplicate
        seen = set()
        unique = []
        for m in movies:
            mid = m.get('movie_id')
            if mid and mid not in seen:
                seen.add(mid)
                unique.append(m)
        movies = unique[:10]
    except Exception as e:
        logger.error(f"Fallback mood error: {e}")
    
    return {
        'mood': detected_mood,
        'explanation': explanation,
        'movies': movies,
        'recommendations': movies,
        'count': len(movies)
    }


def _generate_chat_response(message: str, results: list) -> str:
    """Generate a contextual response label based on query content and result count."""
    m = message.lower()
    n = len(results)
    if any(w in m for w in ['spy', 'espionage', 'secret agent', 'cia', 'raw agent', 'isi']):
        return f'Found {n} spy/espionage titles for you:'
    if any(w in m for w in ['custom', 'customs', 'border officer', 'immigration']):
        return f'Found {n} customs/law-enforcement shows:'
    if any(w in m for w in ['indian', 'bollywood', 'hindi film', 'desi', 'india movies']):
        return f'Here are {n} Indian titles:'
    if any(w in m for w in ['korean', 'k-drama', 'kdrama', 'k drama']):
        return f'Found {n} Korean dramas/films:'
    if any(w in m for w in ['horror', 'scary', 'terrifying', 'ghost', 'spooky']):
        return f'Here are {n} horror picks:'
    if any(w in m for w in ['comedy', 'funny', 'laugh', 'hilarious']):
        return f'Found {n} comedies:'
    if any(w in m for w in ['action', 'fight', 'adventure', 'explosion', 'war']):
        return f'Here are {n} action-packed titles:'
    if any(w in m for w in ['romance', 'romantic', 'love story']):
        return f'Found {n} romantic picks:'
    if any(w in m for w in ['sci-fi', 'science fiction', 'space', 'robot', 'alien', 'future']):
        return f'Found {n} sci-fi titles:'
    if any(w in m for w in ['thriller', 'suspense', 'psychological', 'twist']):
        return f'Here are {n} thrillers:'
    if any(w in m for w in ['documentary', 'true story', 'real life']):
        return f'Found {n} documentary/true-story picks:'
    if n > 0:
        return f'Here are the {n} best matches for "{message}":'
    return "I couldn't find exact matches. Here are some popular picks:"


class DecimalJSONProvider(DefaultJSONProvider):
    """Custom JSON provider to handle Decimal types"""
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        return super().default(obj)


def create_app():
    """Application factory with full feature integration"""
    app = Flask(__name__)
    
    # Configuration
    app.config['SECRET_KEY'] = Config.SECRET_KEY
    app.config['SESSION_TYPE'] = 'filesystem'
    app.config['JSON_SORT_KEYS'] = False
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file upload
    
    # Set custom JSON provider for Decimal handling
    app.json = DecimalJSONProvider(app)
    
    # Enable CORS
    CORS(app)
    
    # ==========================================================================
    # INITIALIZE AI SYSTEMS
    # ==========================================================================
    
    logger.info("Initializing AI systems...")
    
    # Database Manager
    from database.db_manager import DatabaseManager
    db_manager = DatabaseManager()
    app.db_manager = db_manager
    logger.info("✓ Database manager initialized")
    
    # Redis Cache
    try:
        from ai.redis_cache import get_redis_cache
        cache = get_redis_cache(
            host=os.getenv('REDIS_HOST', 'localhost'),
            port=int(os.getenv('REDIS_PORT', 6379)),
            db=int(os.getenv('REDIS_DB', 0))
        )
        if cache and cache.is_available():
            app.cache = cache
            logger.info("✓ Redis cache initialized")
        else:
            app.cache = None
            logger.warning("Redis cache unavailable - running without Redis")
    except Exception as e:
        logger.warning(f"Redis cache not available: {e}")
        app.cache = None
    
    # Device configuration
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    app.device = device
    logger.info(f"✓ Using device: {device}")
    
    app.ai_systems = {}

    def load_ai_models(features=None):
        """Load only the requested AI systems so heavy features do not block unrelated endpoints."""
        requested_features = features or [
            'chatbot',
            'mood_recommender',
            'trending',
            'visual_search',
            'ab_test'
        ]

        if isinstance(requested_features, str):
            requested_features = [requested_features]

        for feature_name in requested_features:
            if feature_name in app.ai_systems:
                continue

            logger.info(f"Loading AI feature: {feature_name}")

            try:
                if feature_name == 'chatbot':
                    from ai.conversational_agent import ConversationalRecommender
                    app.ai_systems['chatbot'] = ConversationalRecommender(db_manager=db_manager)
                    logger.info("✓ Conversational agent loaded")
                elif feature_name == 'mood_recommender':
                    from ai.mood_detector import MoodBasedRecommender
                    app.ai_systems['mood_recommender'] = MoodBasedRecommender(db_manager=db_manager)
                    logger.info("✓ Mood-based recommender loaded")
                elif feature_name == 'trending':
                    from ai.trending_detector import TrendingDetector
                    app.ai_systems['trending'] = TrendingDetector(db_manager=db_manager)
                    logger.info("✓ Trending detector loaded")
                elif feature_name == 'visual_search':
                    from ai.visual_search import VisualMovieSearch
                    app.ai_systems['visual_search'] = VisualMovieSearch(db_manager=db_manager)
                    logger.info("✓ Visual search loaded")
                elif feature_name == 'ab_test':
                    from ai.ab_testing import ABTestingFramework
                    app.ai_systems['ab_test'] = ABTestingFramework(db_manager=db_manager)
                    logger.info("✓ A/B testing framework loaded")
                else:
                    logger.warning(f"Unknown AI feature requested: {feature_name}")
            except Exception as e:
                logger.warning(f"Could not load {feature_name}: {e}")
    
    # Pre-initialize semantic search engine (heavy model load - do it at startup, not during request)
    try:
        from ai.semantic_search import SemanticMovieSearch
        logger.info("Pre-loading semantic search engine...")
        semantic_engine = SemanticMovieSearch()
        
        if not semantic_engine.load_cache():
            logger.info("Building semantic search index from database...")
            movies = db_manager.query("""
                SELECT 
                    m.movie_id, m.title, m.original_title, m.overview,
                    m.poster_path, m.tmdb_rating, m.release_year, m.popularity,
                    GROUP_CONCAT(DISTINCT g.genre_name SEPARATOR ',') as genres,
                    GROUP_CONCAT(DISTINCT d.director_name SEPARATOR ',') as director,
                    GROUP_CONCAT(DISTINCT a.actor_name ORDER BY ma.cast_order SEPARATOR ',') as cast_members,
                    GROUP_CONCAT(DISTINCT mk.keyword SEPARATOR ',') as keywords
                FROM movies m
                LEFT JOIN movie_genres mg ON m.movie_id = mg.movie_id
                LEFT JOIN genres g ON mg.genre_id = g.genre_id
                LEFT JOIN movie_directors md ON m.movie_id = md.movie_id
                LEFT JOIN directors d ON md.director_id = d.director_id
                LEFT JOIN movie_actors ma ON m.movie_id = ma.movie_id
                LEFT JOIN actors a ON ma.actor_id = a.actor_id
                LEFT JOIN movie_keywords mk ON m.movie_id = mk.movie_id
                WHERE m.overview IS NOT NULL AND m.overview != ''
                GROUP BY m.movie_id
                LIMIT 10000
            """, fetch_all=True)
            for movie in movies:
                movie['genres'] = movie['genres'].split(',') if movie['genres'] else []
                movie['cast'] = (movie.get('cast_members', '').split(',') if movie.get('cast_members') else [])[:5]
                movie['keywords'] = (movie['keywords'].split(',') if movie['keywords'] else [])[:20]
            semantic_engine.build_index(movies)
        
        app.semantic_search = semantic_engine
        logger.info("✓ Semantic search engine ready")
    except Exception as e:
        logger.warning(f"Semantic search not available: {e}")
        app.semantic_search = None
    
    # Register blueprints
    app.register_blueprint(api)
    
    # Register social routes
    try:
        from api.social_routes import init_social_routes
        init_social_routes(app, db_manager)
        logger.info("✓ Social features registered")
    except Exception as e:
        logger.warning(f"Could not register social routes: {e}")
    
    # Start content pipeline for continuous content ingestion
    try:
        from ai.content_pipeline import start_content_pipeline
        import threading
        pipeline_thread = threading.Thread(target=start_content_pipeline, daemon=True)
        pipeline_thread.start()
        logger.info("✓ Content pipeline started")
    except Exception as e:
        logger.warning(f"Content pipeline not started: {e}")
    
    # ==========================================================================
    # ADVANCED AI ROUTES
    # ==========================================================================
    
    @app.route('/api/chat', methods=['POST'])
    def chat():
        """Conversational movie recommendations powered by semantic search"""
        load_ai_models('chatbot')

        data = request.json or {}
        user_id = data.get('user_id', session.get('user_id'))
        message = (data.get('message') or '').strip()

        if not message:
            return jsonify({'error': 'Message required'}), 400

        ml = message.lower()

        # --- 1. Pure greetings ---
        greeting_words = ['hi', 'hello', 'hey', 'howdy', 'sup', 'yo']
        if len(message.split()) <= 3 and any(ml.startswith(g) for g in greeting_words):
            return jsonify({
                'response': (
                    "Hey there! I'm your movie assistant.\n\n"
                    "Try asking me:\n"
                    "- \"Indian spy thriller\" - story-based search\n"
                    "- \"I'm feeling happy\" - mood recommendations\n"
                    "- \"Trending movies\" - what's popular now\n"
                    "- \"Custom officer show india\" - find specific content\n\n"
                    "What would you like to watch?"
                ),
                'recommendations': [],
                'type': 'greeting'
            })

        # --- 2. Thanks / Goodbye ---
        if any(w in ml for w in ['thank', 'thanks', 'bye', 'cya', 'goodbye']) and len(message.split()) <= 5:
            return jsonify({
                'response': "You're welcome! Enjoy watching! Come back anytime.",
                'recommendations': [],
                'type': 'chat'
            })

        # --- 3. Trending / Popular ---
        if any(w in ml for w in ['trending', 'popular', 'viral', "what's hot", 'new releases', 'top movies']):
            movies = db_manager.get_top_movies(limit=10, order_by='popularity')
            return jsonify({
                'response': "Here are the most popular movies and shows right now:",
                'recommendations': movies,
                'type': 'trending'
            })

        # --- 4. Short mood-only queries (<=5 words) ---
        mood_keywords = {
            'happy':      ['happy', 'cheerful', 'fun', 'uplifting', 'feel good', 'joy', 'laugh'],
            'sad':        ['sad', 'cry', 'emotional', 'depressed', 'melancholic'],
            'excited':    ['excited', 'pumped', 'adrenaline', 'adventure'],
            'scared':     ['scared', 'scary', 'horror', 'creepy', 'spooky'],
            'romantic':   ['romantic', 'romance', 'love', 'date night'],
            'bored':      ['bored'],
            'thoughtful': ['thoughtful', 'deep', 'philosophical', 'documentary'],
        }
        detected_mood = None
        if len(message.split()) <= 5:
            for mood, kvs in mood_keywords.items():
                if any(kw in ml for kw in kvs):
                    detected_mood = mood
                    break

        if detected_mood:
            try:
                if 'mood_recommender' in app.ai_systems:
                    result = app.ai_systems['mood_recommender'].get_mood_recommendations(
                        user_id=user_id, mood_input=message, top_k=10
                    )
                else:
                    result = _fallback_mood(message, db_manager)
                movies = result.get('movies', [])
                if movies:
                    return jsonify({
                        'response': result.get('explanation', f"Here are some {detected_mood} picks:"),
                        'recommendations': movies,
                        'mood': detected_mood,
                        'type': 'mood'
                    })
            except Exception as e:
                logger.warning(f"Mood handler in chat failed: {e}")

        # --- 5. Semantic search — handles ALL content queries ---
        if app.semantic_search:
            try:
                results = app.semantic_search.hybrid_search(message, top_k=10)
                if results:
                    for r in results:
                        r['search_score'] = round(
                            r.get('final_score', r.get('relevance_score', 0)) * 100, 1
                        )
                    return jsonify({
                        'response': _generate_chat_response(message, results),
                        'recommendations': results,
                        'type': 'semantic'
                    })
            except Exception as e:
                logger.warning(f"Semantic search in chat failed: {e}")

        # --- 6. Keyword fallback ---
        return jsonify(_fallback_chat(message, db_manager))
    
    @app.route('/api/mood-recommendations', methods=['POST'])
    def mood_recommendations():
        """Get recommendations based on mood"""
        load_ai_models('mood_recommender')
        
        data = request.json or {}
        user_id = data.get('user_id', session.get('user_id'))
        mood_input = data.get('mood')
        top_k = data.get('limit', data.get('top_k', 24))
        
        if not mood_input:
            return jsonify({'error': 'Mood input required'}), 400
        
        try:
            if 'mood_recommender' in app.ai_systems:
                mood_rec = app.ai_systems['mood_recommender']
                result = mood_rec.get_mood_recommendations(
                    user_id=user_id,
                    mood_input=mood_input,
                    top_k=top_k
                )
                result['recommendations'] = result.get('movies', [])
                return jsonify(result)
            else:
                return jsonify(_fallback_mood(mood_input, db_manager))
        except Exception as e:
            logger.error(f"Mood recommendation error: {e}")
            return jsonify(_fallback_mood(mood_input, db_manager))
    
    @app.route('/api/trending')
    def get_trending():
        """Get trending movies"""
        load_ai_models('trending')
        
        limit = request.args.get('limit', 20, type=int)
        
        try:
            # Try AI trending detector first
            if 'trending' in app.ai_systems:
                trending = app.ai_systems['trending']
                movies = trending.get_trending_movies(
                    limit=limit,
                    min_interactions=3,
                    time_window_hours=168
                )
                if movies:
                    return jsonify({'trending': movies})
            
            # Fallback: get popular movies from DB as "trending"
            movies = db_manager.get_top_movies(limit=limit, order_by='popularity')
            return jsonify({'trending': movies})
        except Exception as e:
            logger.error(f"Trending error: {e}")
            # Final fallback
            try:
                movies = db_manager.get_top_movies(limit=limit, order_by='popularity')
                return jsonify({'trending': movies})
            except Exception:
                return jsonify({'trending': []})
    
    @app.route('/api/viral')
    def get_viral():
        """Get viral movies"""
        load_ai_models('trending')
        
        try:
            if 'trending' in app.ai_systems:
                trending = app.ai_systems['trending']
                movies = trending.detect_viral_outbreak(
                    threshold_multiplier=3.0
                )
                if movies:
                    return jsonify({'viral': movies})
            
            # Fallback: top popularity movies
            movies = db_manager.get_top_movies(limit=10, order_by='popularity')
            return jsonify({'viral': movies})
        except Exception as e:
            logger.error(f"Viral detection error: {e}")
            return jsonify({'viral': []})
    
    @app.route('/api/visual-search', methods=['POST'])
    def visual_search():
        """Search movies by uploaded image (base64) using CLIP"""
        load_ai_models('visual_search')

        data = request.json or {}
        image_input = data.get('image')
        top_k = data.get('top_k', 10)

        if not image_input:
            return jsonify({'error': 'Image input required (base64)'}), 400

        if 'visual_search' not in app.ai_systems:
            return jsonify({'error': 'Visual search model not loaded', 'results': []}), 503

        try:
            visual = app.ai_systems['visual_search']
            if getattr(visual, 'movie_poster_embeddings', None) is None:
                return jsonify({
                    'results': [],
                    'message': 'Poster image search is warming up. Use a visual description search for now.'
                }), 200

            results = visual.search_by_image(image_input=image_input, top_k=top_k, is_base64=True)
            if not results:
                return jsonify({
                    'results': [],
                    'message': 'No similar posters found. Try a visual description instead.'
                }), 200
            return jsonify({'results': results})
        except Exception as e:
            logger.error(f"Visual search error: {e}")
            return jsonify({'error': str(e), 'results': []}), 500

    @app.route('/api/text-to-image-search', methods=['POST'])
    def text_to_image_search():
        """Search movies by poster description. Uses CLIP if available, else semantic search."""
        load_ai_models('visual_search')

        data = request.json or {}
        description = data.get('description', '')
        top_k = data.get('top_k', 20)

        if not description:
            return jsonify({'error': 'Description required'}), 400

        results = []

        # Try CLIP only if the poster index is already warm; otherwise use semantic fallback immediately.
        if 'visual_search' in app.ai_systems:
            try:
                visual = app.ai_systems['visual_search']
                if getattr(visual, 'movie_poster_embeddings', None) is not None:
                    results = visual.text_to_image_search(text_description=description, top_k=top_k)
                else:
                    logger.info("Visual poster index not ready; using semantic fallback for description search")
            except Exception as e:
                logger.warning(f"CLIP text-to-image failed: {e}")

        # Fallback: semantic search on the text description
        if not results and app.semantic_search:
            try:
                results = app.semantic_search.hybrid_search(description, top_k=top_k)
                for r in results:
                    r['search_score'] = round(
                        r.get('final_score', r.get('relevance_score', 0)) * 100, 1
                    )
            except Exception as e:
                logger.warning(f"Semantic fallback for visual search failed: {e}")

        return jsonify({'results': results})
    
    @app.route('/api/cache-stats')
    def cache_stats():
        """Get cache statistics"""
        if app.cache is None:
            return jsonify({'error': 'Cache not available'}), 503
        
        try:
            stats = app.cache.get_cache_stats()
            return jsonify(stats)
        except Exception as e:
            logger.error(f"Cache stats error: {e}")
            return jsonify({'error': str(e)}), 500
    
    # ==========================================================================
    # WEB ROUTES (Frontend)
    # ==========================================================================
    
    @app.route('/')
    def index():
        """Home page with recommendations"""
        return render_template('index.html')
    
    @app.route('/compare')
    def compare():
        """Pairwise comparison page"""
        if 'user_id' not in session:
            return render_template('login.html', redirect='/compare')
        return render_template('compare.html')
    
    @app.route('/movie/<int:movie_id>')
    def movie_detail(movie_id):
        """Movie detail page"""
        return render_template('detail.html', movie_id=movie_id)
    
    @app.route('/login')
    def login_page():
        """Login page"""
        return render_template('login.html')
    
    @app.route('/signup')
    def signup_page():
        """Signup page"""
        return render_template('signup.html')
    
    @app.route('/profile')
    def profile():
        """User profile page"""
        if 'user_id' not in session:
            return render_template('login.html', redirect='/profile')
        return render_template('profile.html', logged_in=True, username=session.get('username'))
    
    @app.route('/search')
    def search():
        """Search page"""
        return render_template('search.html')
    
    @app.route('/category/<category_name>')
    def category_page(category_name):
        """Category detail page"""
        category_config = {
            'top-picks': {'title': 'Top Picks For You', 'order': 'elo_score'},
            'action': {'title': 'Action Movies', 'order': 'popularity'},
            'thriller': {'title': 'Thrillers', 'order': 'popularity'},
            'top-rated': {'title': 'Top Rated', 'order': 'tmdb_rating'},
            'trending': {'title': 'Trending Now', 'order': 'elo_score'}
        }
        
        config = category_config.get(category_name, {'title': 'Movies', 'order': 'popularity'})
        return render_template('category.html', 
                             category_name=category_name,
                             category_title=config['title'],
                             order_by=config['order'])
    
    @app.route('/monitor')
    def cache_monitor():
        """Cache monitoring dashboard"""
        return render_template('cache_monitor.html')
    
    @app.route('/chat-ui')
    def chat_ui():
        """Chatbot UI page"""
        return render_template('chat.html')
    
    @app.route('/features')
    def features_page():
        """All features showcase page"""
        return render_template('features.html')
    
    @app.route('/friends')
    def friends_page():
        """Social features page"""
        if 'user_id' not in session:
            return render_template('login.html', redirect='/friends')
        return render_template('friends.html')
    
    @app.route('/favicon.ico')
    def favicon():
        """Suppress favicon 404 errors"""
        return '', 204
    
    # ==========================================================================
    # ERROR HANDLERS
    # ==========================================================================
    
    @app.errorhandler(404)
    def not_found(error):
        return render_template('404.html'), 404
    
    @app.errorhandler(500)
    def internal_error(error):
        logger.error(f"Internal error: {error}")
        return render_template('500.html'), 500
    
    # ==========================================================================
    # CONTEXT PROCESSORS
    # ==========================================================================
    
    @app.context_processor
    def inject_user():
        """Make user info available in all templates"""
        return {
            'logged_in': 'user_id' in session,
            'username': session.get('username', 'Guest')
        }
    
    @app.before_request
    def make_session_permanent():
        """Generate session ID for tracking"""
        if 'session_id' not in session:
            session['session_id'] = secrets.token_hex(16)
    
    # Startup logging
    logger.info("=" * 60)
    logger.info("CINESENSE - FULLY INTEGRATED AI PLATFORM")
    logger.info("=" * 60)
    logger.info("✓ Application initialized successfully")
    logger.info(f"✓ Debug mode: {Config.DEBUG}")
    logger.info(f"✓ Device: {device}")
    logger.info(f"✓ Cache: {'Enabled' if app.cache else 'Disabled'}")
    logger.info("=" * 60)
    
    return app


def main():
    """Main entry point"""
    # Fix Windows console encoding for emojis
    import sys
    if sys.platform == 'win32':
        try:
            import io
            sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
            sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
        except Exception:
            pass
    
    app = create_app()
    
    print("\n" + "=" * 70)
    print("🎬 CINESENSE - AI MOVIE RECOMMENDATION PLATFORM")
    print("=" * 70)
    print(f"🌐 Server running on: http://localhost:{Config.PORT}")
    print(f"📚 API documentation: http://localhost:{Config.PORT}/api")
    print(f"💬 Chat interface: http://localhost:{Config.PORT}/chat-ui")
    print(f"👥 Social features: http://localhost:{Config.PORT}/friends")
    print("=" * 70)
    print("\n✨ Features Available:")
    print("   • Conversational AI Chatbot")
    print("   • Mood-Based Recommendations")
    print("   • Trending & Viral Detection")
    print("   • Visual Image Search")
    print("   • Social Features (Friends & Watch Parties)")
    print("   • Redis Caching")
    print("   • A/B Testing Framework")
    print("\n" + "=" * 70)
    print("Press Ctrl+C to stop the server\n")
    
    app.run(
        host='0.0.0.0',
        port=Config.PORT,
        debug=Config.DEBUG,
        use_reloader=False  # Disable auto-reloader to prevent constant restarts
    )


if __name__ == '__main__':
    main()
