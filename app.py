"""
CineSense - Main Flask Application
AI-Based Movie Recommendation Platform
"""

from flask import Flask, render_template, session
from flask.json.provider import DefaultJSONProvider
from flask_cors import CORS
from config import Config
from api.routes import api
import secrets
import logging
from decimal import Decimal

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DecimalJSONProvider(DefaultJSONProvider):
    """Custom JSON provider to handle Decimal types"""
    def default(self, obj):
        if isinstance(obj, Decimal):
            return float(obj)
        return super().default(obj)


def create_app():
    """Application factory"""
    app = Flask(__name__)
    
    # Configuration
    app.config['SECRET_KEY'] = Config.SECRET_KEY
    app.config['SESSION_TYPE'] = 'filesystem'
    app.config['JSON_SORT_KEYS'] = False
    
    # Set custom JSON provider for Decimal handling
    app.json = DecimalJSONProvider(app)
    
    # Enable CORS
    CORS(app)
    
    # Register blueprints
    app.register_blueprint(api)
    
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
    
    @app.route('/monitor')
    def cache_monitor():
        """Cache monitoring dashboard"""
        return render_template('cache_monitor.html')
    
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
    
    # Startup logging (removed before_first_request as it's deprecated in Flask 3.0)
    logger.info("=" * 60)
    logger.info("CINESENSE - AI Movie Recommendation Platform")
    logger.info("=" * 60)
    logger.info("Application initialized successfully")
    logger.info(f"Debug mode: {Config.DEBUG}")
    logger.info("=" * 60)
    
    return app


def main():
    """Main entry point"""
    app = create_app()
    
    print("\n" + "=" * 60)
    print("CINESENSE - AI Movie Recommendation Platform")
    print("=" * 60)
    print(f"Server running on: http://localhost:{Config.PORT}")
    print(f"API documentation: http://localhost:{Config.PORT}/api")
    print("=" * 60)
    print("\nPress Ctrl+C to stop the server\n")
    
    app.run(
        host='0.0.0.0',
        port=Config.PORT,
        debug=Config.DEBUG
    )


if __name__ == '__main__':
    main()
