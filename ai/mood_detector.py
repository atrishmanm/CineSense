"""
Mood-Based Movie Recommendations
Detects user mood and recommends appropriate movies
"""

from transformers import pipeline
from typing import List, Dict
import logging
import re
import os

logger = logging.getLogger(__name__)


class MoodBasedRecommender:
    """
    Recommend movies based on user's current mood
    """
    
    def __init__(self, db_manager=None):
        """Initialize mood detection system
        
        Args:
            db_manager: Database manager for fetching movies
        """
        self.db = db_manager
        self.sentiment_analyzer = None
        self.emotion_detector = None
        self.local_only = os.getenv('MOOD_MODELS_LOCAL_ONLY', '0').lower() in {'1', 'true', 'yes', 'on'}
        self.models_load_attempted = False
        
        # Map moods to movie attributes
        self.mood_mapping = {
            'happy': {
                'genres': ['Comedy', 'Animation', 'Family', 'Musical', 'Romance'],
                'tone': 'light',
                'themes': ['friendship', 'love', 'triumph', 'celebration'],
                'avoid_genres': ['Horror', 'Thriller']
            },
            'sad': {
                'genres': ['Drama', 'Romance'],
                'tone': 'emotional',
                'themes': ['loss', 'redemption', 'hope', 'coming-of-age'],
                'avoid_genres': ['Horror', 'Comedy']
            },
            'excited': {
                'genres': ['Action', 'Adventure', 'Sci-Fi', 'Fantasy'],
                'tone': 'intense',
                'themes': ['hero-journey', 'adventure', 'epic', 'battle'],
                'avoid_genres': ['Drama', 'Documentary']
            },
            'anxious': {
                'genres': ['Comedy', 'Animation', 'Family', 'Documentary'],
                'tone': 'calming',
                'themes': ['nature', 'animals', 'peace', 'simple-life'],
                'avoid_genres': ['Horror', 'Thriller', 'Action']
            },
            'bored': {
                'genres': ['Action', 'Thriller', 'Mystery', 'Sci-Fi', 'Horror'],
                'tone': 'engaging',
                'themes': ['twist', 'suspense', 'mystery', 'unpredictable'],
                'avoid_genres': ['Documentary', 'Drama']
            },
            'romantic': {
                'genres': ['Romance', 'Drama', 'Comedy'],
                'tone': 'heartwarming',
                'themes': ['love', 'relationships', 'passion', 'dating'],
                'avoid_genres': ['Horror', 'War']
            },
            'scared': {
                'genres': ['Horror', 'Thriller', 'Mystery'],
                'tone': 'intense',
                'themes': ['supernatural', 'psychological', 'suspense'],
                'avoid_genres': []
            },
            'thoughtful': {
                'genres': ['Drama', 'Documentary', 'Sci-Fi', 'Mystery'],
                'tone': 'cerebral',
                'themes': ['philosophy', 'existential', 'social-commentary'],
                'avoid_genres': ['Comedy', 'Action']
            },
            'nostalgic': {
                'genres': ['Drama', 'Romance', 'Comedy', 'Animation'],
                'tone': 'sentimental',
                'themes': ['childhood', 'memories', 'past', 'coming-of-age'],
                'avoid_genres': ['Horror']
            }
        }

    def _load_models(self):
        """Load mood models lazily so startup is fast and network calls are minimized."""
        if self.models_load_attempted:
            return

        self.models_load_attempted = True
        logger.info("Loading mood detection models...")

        try:
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                local_files_only=True
            )
            self.emotion_detector = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                top_k=None,
                local_files_only=True
            )
            logger.info("Mood detection models loaded from local cache")
            return
        except Exception as e:
            if self.local_only:
                logger.warning(f"Mood local-only mode enabled and models unavailable locally: {e}")
                return

        try:
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest"
            )
            self.emotion_detector = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                top_k=None
            )
            logger.info("Mood detection models downloaded and loaded")
        except Exception as e:
            logger.error(f"Failed to load mood models: {e}")
            self.sentiment_analyzer = None
            self.emotion_detector = None
    
    def detect_mood(self, text: str) -> Dict[str, float]:
        """
        Detect user's mood from their message
        
        Args:
            text: User's input text
            
        Returns:
            Dictionary with mood probabilities
        """
        if self.emotion_detector is None:
            self._load_models()

        if self.emotion_detector is None:
            return self._fallback_mood_detection(text)
        
        try:
            # Get emotion predictions
            emotions = self.emotion_detector(text)[0]
            
            # Map emotions to moods
            mood_scores = {}
            
            for emotion in emotions:
                label = emotion['label'].lower()
                score = emotion['score']
                
                # Map emotions to our mood categories
                if label in ['joy', 'happiness']:
                    mood_scores['happy'] = mood_scores.get('happy', 0) + score
                elif label in ['sadness', 'grief']:
                    mood_scores['sad'] = mood_scores.get('sad', 0) + score
                elif label in ['anger', 'frustration']:
                    mood_scores['bored'] = mood_scores.get('bored', 0) + score
                elif label in ['fear', 'anxiety']:
                    mood_scores['anxious'] = mood_scores.get('anxious', 0) + score
                elif label in ['surprise', 'excitement']:
                    mood_scores['excited'] = mood_scores.get('excited', 0) + score
                elif label in ['love', 'affection']:
                    mood_scores['romantic'] = mood_scores.get('romantic', 0) + score
            
            # Normalize scores
            total = sum(mood_scores.values())
            if total > 0:
                mood_scores = {k: v/total for k, v in mood_scores.items()}
            
            return mood_scores
        
        except Exception as e:
            logger.error(f"Mood detection failed: {e}")
            return self._fallback_mood_detection(text)
    
    def _fallback_mood_detection(self, text: str) -> Dict[str, float]:
        """Simple keyword-based mood detection as fallback"""
        text_lower = text.lower()
        
        mood_keywords = {
            'happy': ['happy', 'joy', 'fun', 'cheerful', 'uplifting', 'feel-good', 'laugh'],
            'sad': ['sad', 'depressed', 'down', 'blue', 'melancholic', 'crying'],
            'excited': ['excited', 'pumped', 'energetic', 'thrilled', 'adventure'],
            'anxious': ['anxious', 'stressed', 'worried', 'nervous', 'calm', 'relax'],
            'bored': ['bored', 'boring', 'dull', 'exciting', 'entertaining'],
            'romantic': ['romantic', 'love', 'romance', 'date', 'relationship'],
            'scared': ['scary', 'horror', 'terrifying', 'spooky', 'creepy'],
            'thoughtful': ['thinking', 'philosophical', 'deep', 'meaningful', 'contemplative']
        }
        
        mood_scores = {}
        for mood, keywords in mood_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            if score > 0:
                mood_scores[mood] = score
        
        # Normalize
        total = sum(mood_scores.values())
        if total > 0:
            mood_scores = {k: v/total for k, v in mood_scores.items()}
        else:
            # Default neutral mood
            mood_scores = {'thoughtful': 1.0}
        
        return mood_scores
    
    def get_primary_mood(self, text: str) -> str:
        """
        Get the primary (dominant) mood
        
        Returns:
            Mood name as string
        """
        mood_scores = self.detect_mood(text)
        if not mood_scores:
            return 'thoughtful'
        
        return max(mood_scores, key=mood_scores.get)
    
    def get_mood_recommendations(
        self,
        user_id=None,
        mood_input: str = '',
        top_k: int = 10
    ) -> Dict:
        """
        High-level method: detect mood from text, fetch movies from DB, and rank.
        Used by the /api/mood-recommendations endpoint.
        
        Args:
            user_id: Optional user ID
            mood_input: User's mood text (e.g., "I feel happy" or just "happy")
            top_k: Number of movies to return
            
        Returns:
            Dict with mood info and ranked movie list
        """
        # Detect mood from input
        primary_mood = self.get_primary_mood(mood_input)
        mood_scores = self.detect_mood(mood_input)
        explanation = self.get_mood_explanation(primary_mood)
        
        # Get movies from database
        movies = []
        if self.db:
            try:
                # Fetch movies from preferred genres for this mood
                mood_attrs = self.mood_mapping.get(primary_mood, self.mood_mapping['thoughtful'])
                preferred_genres = mood_attrs['genres']
                
                for genre in preferred_genres:
                    genre_movies = self.db.get_movies_by_genre(genre, limit=50)
                    if genre_movies:
                        # Convert genre string to list for scoring
                        for m in genre_movies:
                            if isinstance(m.get('genres'), str):
                                m['genres'] = [g.strip() for g in m['genres'].split(',') if g.strip()]
                        movies.extend(genre_movies)
                
                # Deduplicate by movie_id
                seen_ids = set()
                unique_movies = []
                for m in movies:
                    mid = m.get('movie_id')
                    if mid and mid not in seen_ids:
                        seen_ids.add(mid)
                        unique_movies.append(m)
                movies = unique_movies
                
                # If still not enough, get top movies
                if len(movies) < top_k:
                    top_movies = self.db.get_top_movies(limit=100)
                    for m in top_movies:
                        if isinstance(m.get('genres'), str):
                            m['genres'] = [g.strip() for g in m['genres'].split(',') if g.strip()]
                    movies.extend(top_movies)
                    # Deduplicate again
                    seen_ids2 = set()
                    unique2 = []
                    for m in movies:
                        mid = m.get('movie_id')
                        if mid and mid not in seen_ids2:
                            seen_ids2.add(mid)
                            unique2.append(m)
                    movies = unique2
                    
            except Exception as e:
                logger.error(f"Error fetching movies for mood: {e}")
        
        # Rank movies by mood
        ranked_movies = self.recommend_by_mood(primary_mood, movies, top_k=top_k)
        
        return {
            'mood': primary_mood,
            'mood_scores': mood_scores,
            'explanation': explanation,
            'movies': ranked_movies,
            'count': len(ranked_movies)
        }
    
    def recommend_by_mood(
        self, 
        mood: str, 
        movies: List[Dict],
        top_k: int = 20
    ) -> List[Dict]:
        """
        Filter and rank movies based on mood
        
        Args:
            mood: Mood name (e.g., 'happy', 'sad', 'excited')
            movies: List of movie dictionaries
            top_k: Number of movies to return
            
        Returns:
            Ranked list of movies matching the mood
        """
        if mood not in self.mood_mapping:
            logger.warning(f"Unknown mood: {mood}, using default")
            mood = 'thoughtful'
        
        mood_attrs = self.mood_mapping[mood]
        preferred_genres = set(mood_attrs['genres'])
        avoid_genres = set(mood_attrs.get('avoid_genres', []))
        
        # Score movies by mood match
        scored_movies = []
        fallback_movies = []
        for movie in movies:
            raw_genres = movie.get('genres') or []
            if isinstance(raw_genres, str):
                movie_genres = {genre.strip() for genre in raw_genres.split(',') if genre.strip()}
            else:
                movie_genres = {genre for genre in raw_genres if genre}
            
            # Calculate genre match score
            genre_match = len(movie_genres & preferred_genres)
            genre_avoid = len(movie_genres & avoid_genres)
            
            # Penalize movies with genres to avoid
            if genre_avoid > 0:
                mood_score = max(0, genre_match - genre_avoid * 2)
            else:
                mood_score = genre_match
            
            # Boost if primary genre matches
            if movie_genres:
                primary_genre = next(iter(movie_genres))
                if primary_genre in preferred_genres:
                    mood_score += 2
            
            # Add mood score to movie
            movie_copy = movie.copy()
            movie_copy['mood_score'] = mood_score
            movie_copy['mood'] = mood
            fallback_movies.append(movie_copy)
            
            if mood_score > 0:
                scored_movies.append(movie_copy)
        
        # Sort by mood score (then by rating if available)
        sort_key = lambda x: (
            x.get('mood_score', 0),
            x.get('tmdb_rating', x.get('vote_average', 0)) or 0,
            x.get('popularity', 0) or 0
        )
        scored_movies.sort(key=sort_key, reverse=True)
        
        if scored_movies:
            return scored_movies[:top_k]

        fallback_movies.sort(key=sort_key, reverse=True)
        return fallback_movies[:top_k]
    
    def get_mood_explanation(self, mood: str) -> str:
        """
        Get human-readable explanation for mood-based recommendations
        
        Returns:
            Explanation string
        """
        if mood not in self.mood_mapping:
            return "I'll recommend some thoughtful, engaging movies."
        
        attrs = self.mood_mapping[mood]
        genres = ', '.join(attrs['genres'][:3])
        
        explanations = {
            'happy': f"You seem happy! I'll recommend uplifting {genres} movies.",
            'sad': f"I understand. Here are some {genres} movies that might resonate with you.",
            'excited': f"Let's keep that energy up! I'll suggest exciting {genres} movies.",
            'anxious': f"Let me suggest some calming {genres} movies to help you relax.",
            'bored': f"Time for something exciting! Check out these {genres} movies.",
            'romantic': f"Feeling the love! Here are some beautiful {genres} movies.",
            'scared': f"Want something thrilling? These {genres} movies will deliver.",
            'thoughtful': f"In a contemplative mood? These {genres} movies will engage your mind.",
            'nostalgic': f"Let's revisit the past with these {genres} movies."
        }
        
        return explanations.get(mood, "Here are some movies you might enjoy.")


# Example usage and testing
if __name__ == '__main__':
    recommender = MoodBasedRecommender()
    
    # Test mood detection
    test_messages = [
        "I'm feeling really happy today!",
        "I'm so sad and need something emotional",
        "I'm bored and need excitement",
        "I feel anxious and need to relax",
        "I want to watch something romantic"
    ]
    
    print("Testing Mood Detection:")
    print("=" * 60)
    
    for msg in test_messages:
        moods = recommender.detect_mood(msg)
        primary = recommender.get_primary_mood(msg)
        explanation = recommender.get_mood_explanation(primary)
        
        print(f"\nMessage: {msg}")
        print(f"Detected moods: {moods}")
        print(f"Primary mood: {primary}")
        print(f"Explanation: {explanation}")
    
    # Test with sample movies
    sample_movies = [
        {
            'title': 'The Dark Knight',
            'genres': ['Action', 'Crime', 'Drama'],
            'vote_average': 9.0
        },
        {
            'title': 'Toy Story',
            'genres': ['Animation', 'Comedy', 'Family'],
            'vote_average': 8.3
        },
        {
            'title': 'The Notebook',
            'genres': ['Romance', 'Drama'],
            'vote_average': 7.8
        }
    ]
    
    print("\n\nTesting Movie Recommendations:")
    print("=" * 60)
    
    mood = 'happy'
    recommendations = recommender.recommend_by_mood(mood, sample_movies, top_k=5)
    
    print(f"\nRecommendations for '{mood}' mood:")
    for i, movie in enumerate(recommendations, 1):
        print(f"{i}. {movie['title']} (Mood Score: {movie['mood_score']})")
