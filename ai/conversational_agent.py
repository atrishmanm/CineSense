"""
Conversational Movie Discovery Agent
ChatGPT-style interactive movie recommendations
"""

from typing import List, Dict, Optional
import logging
import re

logger = logging.getLogger(__name__)


class ConversationalRecommender:
    """
    Combined system: Chatbot + Movie Recommendations
    """
    
    def __init__(self, movie_recommender=None, db_manager=None):
        """
        Args:
            movie_recommender: Instance of movie recommendation system
            db_manager: Database manager for fetching movies
        """
        self.recommender = movie_recommender
        self.db = db_manager
        self.user_preferences = {}
        self.user_sessions = {}  # session_id -> conversation history
    
    def chat(self, user_id=None, message="", session_id=None):
        """
        Handle a chat message and return response with optional recommendations.
        Used by the /api/chat endpoint.
        
        Returns:
            dict with 'response', 'recommendations', 'mood', etc.
        """
        if not message:
            return {'response': "Please send me a message!", 'recommendations': []}
        
        # Track conversation per session
        if session_id not in self.user_sessions:
            self.user_sessions[session_id] = []
        self.user_sessions[session_id].append(message)
        
        message_lower = message.lower()
        
        # Check for mood-based queries
        mood_keywords = {
            'happy': ['happy', 'cheerful', 'fun', 'uplifting', 'feel good', 'feel-good', 'laugh'],
            'sad': ['sad', 'emotional', 'crying', 'depressed', 'down', 'melancholic'],
            'excited': ['excited', 'exciting', 'thrilling', 'pumped', 'energetic', 'adventure'],
            'scared': ['scary', 'horror', 'terrifying', 'spooky', 'creepy'],
            'romantic': ['romantic', 'romance', 'love', 'date night', 'relationship'],
            'bored': ['bored', 'boring', 'something different', 'entertaining'],
            'anxious': ['anxious', 'stressed', 'relax', 'calming', 'calm'],
            'thoughtful': ['thoughtful', 'deep', 'philosophical', 'meaningful', 'thinking'],
        }
        
        detected_mood = None
        for mood, keywords in mood_keywords.items():
            if any(kw in message_lower for kw in keywords):
                detected_mood = mood
                break
        
        # Check for trending queries
        if any(w in message_lower for w in ['trending', 'popular', 'viral', "what's hot", 'whats hot']):
            movies = self._get_trending_movies()
            return {
                'response': "Here are the trending movies right now!",
                'recommendations': movies,
                'type': 'trending'
            }
        
        # Check if asking for recommendations
        is_rec_query = self._is_recommendation_query(message)
        
        if detected_mood:
            movies = self._get_mood_movies(detected_mood)
            mood_explanations = {
                'happy': "You're in a great mood! Here are some feel-good movies:",
                'sad': "I understand. These movies might resonate with you:",
                'excited': "Let's keep that energy going! Check these out:",
                'scared': "Want a thrill? These movies will keep you on edge:",
                'romantic': "Feeling the love! Here are some romantic picks:",
                'bored': "Time for something exciting! Try these:",
                'anxious': "Let me suggest some calming movies to help you unwind:",
                'thoughtful': "In a contemplative mood? These will engage your mind:",
            }
            return {
                'response': mood_explanations.get(detected_mood, "Here are some movies for you:"),
                'recommendations': movies,
                'mood': detected_mood,
                'type': 'mood'
            }
        
        if is_rec_query:
            movies = self._get_recommendation_movies({}, user_id)
            return {
                'response': "Based on what you told me, here are my recommendations:",
                'recommendations': movies,
                'type': 'recommendation'
            }
        
        # General conversation - return helpful fallback
        return {
            'response': self._get_fallback_response(message),
            'recommendations': [],
            'type': 'chat'
        }
    
    def _get_fallback_response(self, message):
        """Generate a helpful fallback response"""
        message_lower = message.lower()
        if any(w in message_lower for w in ['hi', 'hello', 'hey']):
            return "Hey there! I'm your movie assistant. Tell me what kind of movies you're in the mood for, or ask me for recommendations!"
        if any(w in message_lower for w in ['thank', 'thanks']):
            return "You're welcome! Let me know if you need more recommendations."
        if any(w in message_lower for w in ['help', 'what can you do']):
            return ("I can help you with:\n"
                    "- Movie recommendations based on your mood\n" 
                    "- Trending and popular movies\n"
                    "- Finding specific types of movies\n"
                    "Just tell me what you're looking for!")
        return "I'd love to help you find a great movie! Tell me your mood, or what genre you're interested in."
    
    def _get_mood_movies(self, mood, limit=10):
        """Get movies matching a mood from the database"""
        mood_genre_map = {
            'happy': ['Comedy', 'Animation', 'Family', 'Musical'],
            'sad': ['Drama', 'Romance'],
            'excited': ['Action', 'Adventure', 'Sci-Fi'],
            'scared': ['Horror', 'Thriller', 'Mystery'],
            'romantic': ['Romance', 'Drama', 'Comedy'],
            'bored': ['Action', 'Thriller', 'Sci-Fi', 'Mystery'],
            'anxious': ['Comedy', 'Animation', 'Family', 'Documentary'],
            'thoughtful': ['Drama', 'Documentary', 'Sci-Fi', 'Mystery'],
        }
        genres = mood_genre_map.get(mood, ['Drama', 'Comedy'])
        
        if self.db:
            try:
                for genre in genres:
                    movies = self.db.get_movies_by_genre(genre, limit=limit)
                    if movies:
                        return movies[:limit]
                # Fallback to top movies
                return self.db.get_top_movies(limit=limit)
            except Exception as e:
                logger.error(f"Error fetching mood movies: {e}")
        return []
    
    def _get_trending_movies(self, limit=10):
        """Get trending movies from the database"""
        if self.db:
            try:
                return self.db.get_top_movies(limit=limit, order_by='popularity')
            except Exception as e:
                logger.error(f"Error fetching trending movies: {e}")
        return []
    
    def _get_recommendation_movies(self, prefs, user_id=None, limit=10):
        """Get recommended movies based on extracted preferences"""
        if self.db:
            try:
                # Try genre-based first
                for genre in prefs.get('genres', []):
                    movies = self.db.get_movies_by_genre(genre.capitalize(), limit=limit)
                    if movies:
                        return movies[:limit]
                # Fallback to top movies
                return self.db.get_top_movies(limit=limit)
            except Exception as e:
                logger.error(f"Error fetching recommendation movies: {e}")
        return []
    
    def _is_recommendation_query(self, text: str) -> bool:
        """Check if user is asking for movie recommendations"""
        recommendation_keywords = [
            'recommend', 'suggest', 'movie', 'film', 'watch',
            'looking for', 'want to see', 'show me'
        ]
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in recommendation_keywords)
