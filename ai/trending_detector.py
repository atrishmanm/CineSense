"""
Trending & Virality Detection
Detects which movies are trending using velocity and acceleration metrics
"""

import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class TrendingDetector:
    """
    Detect viral movies using interaction velocity and acceleration
    
    Concept:
    - Velocity: Rate of new interactions (first derivative)
    - Acceleration: Change in velocity (second derivative)
    - Trending score combines both with recency weighting
    """
    
    def __init__(self, db_manager):
        """
        Args:
            db_manager: Database manager instance
        """
        self.db = db_manager
        self.time_windows = [1, 6, 24, 72, 168]  # hours: 1h, 6h, 1d, 3d, 7d
    
    def calculate_trending_score(
        self,
        movie_id: int,
        lookback_hours: int = 168
    ) -> float:
        """
        Calculate trending score for a movie
        
        Higher score = more trending
        
        Args:
            movie_id: Movie ID
            lookback_hours: How far back to analyze (default: 7 days)
            
        Returns:
            Trending score (0-100)
        """
        try:
            # Get interaction counts for different time windows
            now = datetime.now()
            interaction_counts = []
            
            for window_hours in self.time_windows:
                start_time = now - timedelta(hours=window_hours)
                
                count = self.db.query("""
                    SELECT COUNT(*) as count
                    FROM user_interactions
                    WHERE chosen_movie_id = %s
                      AND timestamp >= %s
                """, (movie_id, start_time))
                
                interaction_counts.append(count[0]['count'] if count else 0)
            
            # Calculate velocity (change in interactions over time)
            velocities = []
            for i in range(len(interaction_counts) - 1):
                # Interactions per hour
                time_diff = self.time_windows[i+1] - self.time_windows[i]
                count_diff = interaction_counts[i+1] - interaction_counts[i]
                velocity = count_diff / time_diff if time_diff > 0 else 0
                velocities.append(velocity)
            
            # Calculate acceleration (change in velocity)
            accelerations = []
            for i in range(len(velocities) - 1):
                acceleration = velocities[i] - velocities[i+1]
                accelerations.append(acceleration)
            
            # Trending score formula
            # Weight recent activity more heavily
            if len(velocities) > 0:
                recent_velocity = velocities[0]  # Most recent
                avg_velocity = np.mean(velocities)
                
                # Recent acceleration
                recent_acceleration = accelerations[0] if accelerations else 0
                
                # Combined score
                # High positive acceleration = going viral
                # High constant velocity = sustained popularity
                trending_score = (
                    recent_velocity * 0.5 +      # Recent activity weight
                    avg_velocity * 0.2 +          # Sustained activity weight
                    recent_acceleration * 0.3     # Growth rate weight
                )
                
                # Normalize to 0-100 scale (heuristic)
                trending_score = min(max(trending_score * 10, 0), 100)
                
                return float(trending_score)
            
            return 0.0
        
        except Exception as e:
            logger.error(f"Error calculating trending score for movie {movie_id}: {e}")
            return 0.0
    
    def get_trending_movies(
        self,
        limit: int = 20,
        min_interactions: int = 10,
        time_window_hours: int = 168
    ) -> List[Dict]:
        """
        Get currently trending movies
        
        Args:
            limit: Number of movies to return
            min_interactions: Minimum interactions to be considered
            time_window_hours: Time window for analysis
            
        Returns:
            List of trending movies with scores
        """
        try:
            # Get movies with recent activity
            start_time = datetime.now() - timedelta(hours=time_window_hours)
            
            recent_movies = self.db.query("""
                SELECT 
                    m.movie_id,
                    m.title,
                    m.poster_path,
                    m.release_year,
                    m.vote_average,
                    COUNT(ui.interaction_id) as interaction_count
                FROM movies m
                JOIN user_interactions ui ON m.movie_id = ui.chosen_movie_id
                WHERE ui.timestamp >= %s
                GROUP BY m.movie_id
                HAVING interaction_count >= %s
                ORDER BY interaction_count DESC
                LIMIT 100
            """, (start_time, min_interactions))
            
            # Calculate trending score for each
            trending_movies = []
            for movie in recent_movies:
                score = self.calculate_trending_score(movie['movie_id'])
                
                movie['trending_score'] = round(score, 2)
                movie['trend_indicator'] = self._get_trend_indicator(score)
                
                if score > 0:
                    trending_movies.append(movie)
            
            # Sort by trending score
            trending_movies.sort(key=lambda x: x['trending_score'], reverse=True)
            
            return trending_movies[:limit]
        
        except Exception as e:
            logger.error(f"Error getting trending movies: {e}")
            return []
    
    def _get_trend_indicator(self, score: float) -> str:
        """
        Get emoji/text indicator for trend level
        """
        if score >= 80:
            return "ðŸ”¥ VIRAL"
        elif score >= 60:
            return "ðŸ“ˆ HOT"
        elif score >= 40:
            return "â¬†ï¸ RISING"
        elif score >= 20:
            return "âž¡ï¸ STEADY"
        else:
            return "ðŸ“‰ COOLING"
    
    def detect_viral_outbreak(
        self,
        threshold_multiplier: float = 3.0
    ) -> List[Dict]:
        """
        Detect movies experiencing viral growth
        (interactions suddenly spike above historical average)
        
        Args:
            threshold_multiplier: How many times above average = viral
            
        Returns:
            List of movies going viral
        """
        try:
            # Get all movies with recent activity
            now = datetime.now()
            last_24h = now - timedelta(hours=24)
            last_week = now - timedelta(days=7)
            
            movies = self.db.query("""
                SELECT DISTINCT m.movie_id, m.title
                FROM movies m
                JOIN user_interactions ui ON m.movie_id = ui.chosen_movie_id
                WHERE ui.timestamp >= %s
            """, (last_week,))
            
            viral_movies = []
            
            for movie in movies:
                movie_id = movie['movie_id']
                
                # Count last 24h
                recent_count = self.db.query("""
                    SELECT COUNT(*) as count
                    FROM user_interactions
                    WHERE chosen_movie_id = %s
                      AND timestamp >= %s
                """, (movie_id, last_24h))
                
                recent = recent_count[0]['count'] if recent_count else 0
                
                # Count previous 6 days (daily average)
                historical_count = self.db.query("""
                    SELECT COUNT(*) as count
                    FROM user_interactions
                    WHERE chosen_movie_id = %s
                      AND timestamp BETWEEN %s AND %s
                """, (movie_id, last_week, last_24h))
                
                historical = historical_count[0]['count'] if historical_count else 0
                avg_daily = historical / 6.0 if historical > 0 else 0
                
                # Check if viral (recent >> historical average)
                if avg_daily > 0 and recent > avg_daily * threshold_multiplier:
                    growth_rate = (recent / avg_daily - 1) * 100
                    
                    viral_movies.append({
                        'movie_id': movie_id,
                        'title': movie['title'],
                        'recent_interactions': recent,
                        'avg_daily': round(avg_daily, 1),
                        'growth_rate': round(growth_rate, 1),
                        'viral_score': round(recent / avg_daily, 2)
                    })
            
            # Sort by growth rate
            viral_movies.sort(key=lambda x: x['growth_rate'], reverse=True)
            
            return viral_movies
        
        except Exception as e:
            logger.error(f"Error detecting viral outbreak: {e}")
            return []
    
    def get_trending_by_genre(
        self,
        genre: str,
        limit: int = 10
    ) -> List[Dict]:
        """
        Get trending movies in a specific genre
        """
        try:
            # Get genre ID
            genre_data = self.db.query("""
                SELECT genre_id FROM genres WHERE name = %s
            """, (genre,))
            
            if not genre_data:
                return []
            
            genre_id = genre_data[0]['genre_id']
            
            # Get movies in this genre with recent activity
            start_time = datetime.now() - timedelta(days=7)
            
            movies = self.db.query("""
                SELECT DISTINCT
                    m.movie_id,
                    m.title,
                    m.poster_path,
                    m.release_year
                FROM movies m
                JOIN movie_genres mg ON m.movie_id = mg.movie_id
                JOIN user_interactions ui ON m.movie_id = ui.chosen_movie_id
                WHERE mg.genre_id = %s
                  AND ui.timestamp >= %s
                GROUP BY m.movie_id
                LIMIT 50
            """, (genre_id, start_time))
            
            # Score each movie
            trending = []
            for movie in movies:
                score = self.calculate_trending_score(movie['movie_id'])
                if score > 0:
                    movie['trending_score'] = score
                    trending.append(movie)
            
            trending.sort(key=lambda x: x['trending_score'], reverse=True)
            
            return trending[:limit]
        
        except Exception as e:
            logger.error(f"Error getting trending by genre: {e}")
            return []
    
    def predict_next_viral(
        self,
        top_k: int = 10
    ) -> List[Dict]:
        """
        Predict movies likely to go viral soon
        Based on early positive acceleration
        """
        try:
            # Look at movies with growing but not yet peaked activity
            movies = self.get_trending_movies(limit=100, min_interactions=5)
            
            predictions = []
            for movie in movies:
                score = movie['trending_score']
                
                # Look for movies with moderate score but high acceleration
                if 20 <= score <= 60:  # Growing, not yet peaked
                    movie_id = movie['movie_id']
                    
                    # Check if acceleration is positive
                    # (simplified: compare last 24h to previous 24h)
                    now = datetime.now()
                    last_24h = now - timedelta(hours=24)
                    prev_24h = now - timedelta(hours=48)
                    
                    recent = self.db.query("""
                        SELECT COUNT(*) as count
                        FROM user_interactions
                        WHERE chosen_movie_id = %s
                          AND timestamp >= %s
                    """, (movie_id, last_24h))
                    
                    previous = self.db.query("""
                        SELECT COUNT(*) as count
                        FROM user_interactions
                        WHERE chosen_movie_id = %s
                          AND timestamp BETWEEN %s AND %s
                    """, (movie_id, prev_24h, last_24h))
                    
                    recent_count = recent[0]['count'] if recent else 0
                    prev_count = previous[0]['count'] if previous else 0
                    
                    if prev_count > 0:
                        growth = (recent_count - prev_count) / prev_count
                        
                        if growth > 0.3:  # 30% growth
                            movie['predicted_viral_probability'] = min(growth * 100, 100)
                            predictions.append(movie)
            
            # Sort by prediction
            predictions.sort(
                key=lambda x: x.get('predicted_viral_probability', 0),
                reverse=True
            )
            
            return predictions[:top_k]
        
        except Exception as e:
            logger.error(f"Error predicting viral movies: {e}")
            return []


# Example usage
if __name__ == '__main__':
    from database.db_manager import db
    
    detector = TrendingDetector(db)
    
    print("Trending Detection System")
    print("=" * 60)
    
    # Get trending movies
    trending = detector.get_trending_movies(limit=10)
    
    print("\nTop 10 Trending Movies:")
    print("-" * 60)
    for i, movie in enumerate(trending, 1):
        print(f"{i}. {movie['title']}")
        print(f"   Score: {movie['trending_score']} {movie['trend_indicator']}")
        print(f"   Interactions: {movie['interaction_count']}")
        print()
    
    # Detect viral outbreaks
    viral = detector.detect_viral_outbreak()
    
    if viral:
        print("\nViral Outbreaks Detected:")
        print("-" * 60)
        for movie in viral[:5]:
            print(f"ðŸ”¥ {movie['title']}")
            print(f"   Growth: {movie['growth_rate']}%")
            print(f"   Recent: {movie['recent_interactions']} (avg: {movie['avg_daily']})")
            print()
