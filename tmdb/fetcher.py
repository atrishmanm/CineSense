"""
TMDB API Integration
Fetches movie data from The Movie Database API
"""

import requests
import time
from config import Config
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TMDBFetcher:
    """Handles all TMDB API interactions"""
    
    def __init__(self):
        self.api_key = Config.TMDB_API_KEY
        self.base_url = Config.TMDB_BASE_URL
        self.image_base_url = Config.TMDB_IMAGE_BASE_URL
        self.session = requests.Session()
    
    def _make_request(self, endpoint, params=None):
        """Make API request with error handling"""
        if params is None:
            params = {}
        params['api_key'] = self.api_key
        
        url = f"{self.base_url}/{endpoint}"
        
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"TMDB API request failed: {e}")
            return None
    
    def get_popular_movies(self, page=1):
        """Fetch popular movies"""
        endpoint = "movie/popular"
        params = {'page': page, 'language': 'en-US'}
        return self._make_request(endpoint, params)
    
    def get_top_rated_movies(self, page=1):
        """Fetch top rated movies"""
        endpoint = "movie/top_rated"
        params = {'page': page, 'language': 'en-US'}
        return self._make_request(endpoint, params)
    
    def get_popular_tv_series(self, page=1):
        """Fetch popular TV series"""
        endpoint = "tv/popular"
        params = {'page': page, 'language': 'en-US'}
        return self._make_request(endpoint, params)
    
    def get_top_rated_tv_series(self, page=1):
        """Fetch top rated TV series"""
        endpoint = "tv/top_rated"
        params = {'page': page, 'language': 'en-US'}
        return self._make_request(endpoint, params)
    
    def get_tv_details(self, tv_id):
        """Get detailed information about a TV series"""
        endpoint = f"tv/{tv_id}"
        params = {'language': 'en-US', 'append_to_response': 'credits,videos'}
        return self._make_request(endpoint, params)
    
    def get_now_playing(self, page=1):
        """Fetch now playing movies"""
        endpoint = "movie/now_playing"
        params = {'page': page, 'language': 'en-US'}
        return self._make_request(endpoint, params)
    
    def get_movie_details(self, movie_id):
        """Get detailed information about a movie"""
        endpoint = f"movie/{movie_id}"
        params = {'language': 'en-US', 'append_to_response': 'credits,videos'}
        return self._make_request(endpoint, params)
    
    def discover_movies(self, **kwargs):
        """
        Discover movies with filters - INFINITE STREAM support
        
        Args:
            page: Page number (1-500)
            sort_by: Sort method (popularity.desc, vote_average.desc, etc.)
            with_genres: Genre IDs (comma-separated or list)
            year: Release year
            vote_average.gte: Minimum rating
            vote_count.gte: Minimum vote count
            with_cast: Cast member IDs
            with_crew: Crew member IDs (directors)
            primary_release_date.gte: Minimum release date
            primary_release_date.lte: Maximum release date
        
        Returns:
            dict: {results: [...], page: int, total_pages: int, total_results: int}
        """
        endpoint = "discover/movie"
        params = {
            'language': 'en-US',
            'sort_by': kwargs.get('sort_by', 'popularity.desc'),
            'include_adult': False,
            'include_video': False,
            'page': kwargs.get('page', 1)
        }
        
        # Add optional filters
        if 'with_genres' in kwargs:
            genres = kwargs['with_genres']
            params['with_genres'] = ','.join(map(str, genres)) if isinstance(genres, list) else genres
        if 'year' in kwargs:
            params['year'] = kwargs['year']
        if 'vote_average.gte' in kwargs:
            params['vote_average.gte'] = kwargs['vote_average.gte']
        if 'vote_count.gte' in kwargs:
            params['vote_count.gte'] = kwargs['vote_count.gte']
        if 'with_cast' in kwargs:
            params['with_cast'] = kwargs['with_cast']
        if 'with_crew' in kwargs:
            params['with_crew'] = kwargs['with_crew']
        if 'primary_release_date.gte' in kwargs:
            params['primary_release_date.gte'] = kwargs['primary_release_date.gte']
        if 'primary_release_date.lte' in kwargs:
            params['primary_release_date.lte'] = kwargs['primary_release_date.lte']
        
        return self._make_request(endpoint, params)
    
    def search_movies(self, query, page=1):
        """Search for movies"""
        endpoint = "search/movie"
        params = {
            'query': query,
            'page': page,
            'language': 'en-US',
            'include_adult': False
        }
        return self._make_request(endpoint, params)
    
    def get_genres(self):
        """Get list of all genres"""
        endpoint = "genre/movie/list"
        params = {'language': 'en-US'}
        return self._make_request(endpoint, params)
    
    def get_poster_url(self, poster_path, size='w500'):
        """Get full poster URL"""
        if not poster_path:
            return None
        return f"{self.image_base_url}{size}{poster_path}"
    
    def get_backdrop_url(self, backdrop_path, size='w1280'):
        """Get full backdrop URL"""
        if not backdrop_path:
            return None
        return f"{self.image_base_url}{size}{backdrop_path}"
    
    def get_similar_movies(self, movie_id, page=1):
        """Get movies similar to a given movie"""
        endpoint = f"movie/{movie_id}/similar"
        params = {'page': page, 'language': 'en-US'}
        return self._make_request(endpoint, params)
    
    def get_recommendations(self, movie_id, page=1):
        """Get recommended movies based on a movie"""
        endpoint = f"movie/{movie_id}/recommendations"
        params = {'page': page, 'language': 'en-US'}
        return self._make_request(endpoint, params)
    
    def stream_movies(self, filters=None, max_pages=10):
        """
        LAZY LOADING: Stream movies page by page
        
        This is a generator - fetches on demand, never loads all at once
        
        Args:
            filters: dict of discovery filters
            max_pages: Maximum pages to fetch (default 10 = 200 movies)
        
        Yields:
            dict: Movie data one at a time
        """
        filters = filters or {}
        
        for page in range(1, max_pages + 1):
            filters['page'] = page
            data = self.discover_movies(**filters)
            
            if not data or 'results' not in data:
                break
            
            for movie in data['results']:
                yield movie
            
            # Stop if we've reached the last page
            if page >= data.get('total_pages', 0):
                break
            
            time.sleep(0.25)  # Rate limiting
    
    def parse_movie_data(self, movie_json):
        """Parse TMDB movie JSON into database format"""
        return {
            'movie_id': movie_json['id'],
            'tmdb_id': movie_json['id'],
            'title': movie_json.get('title', ''),
            'original_title': movie_json.get('original_title', ''),
            'overview': movie_json.get('overview', ''),
            'release_year': int(movie_json.get('release_date', '0000')[:4]) if movie_json.get('release_date') else None,
            'runtime': movie_json.get('runtime'),
            'poster_path': movie_json.get('poster_path'),
            'backdrop_path': movie_json.get('backdrop_path'),
            'tmdb_rating': movie_json.get('vote_average'),
            'vote_count': movie_json.get('vote_count'),
            'popularity': movie_json.get('popularity'),
            'watch_link': None  # Will be populated later
        }
    
    def parse_genres(self, movie_json):
        """Extract genres from movie JSON"""
        if 'genres' in movie_json:
            return movie_json['genres']
        elif 'genre_ids' in movie_json:
            # Map genre IDs to names (you'd need to fetch genre list first)
            return [{'id': gid, 'name': ''} for gid in movie_json['genre_ids']]
        return []
    
    def parse_credits(self, movie_json):
        """Extract cast and crew from movie JSON"""
        credits = movie_json.get('credits', {})
        
        # Extract directors
        directors = []
        if 'crew' in credits:
            directors = [
                {
                    'id': person['id'],
                    'name': person['name'],
                    'popularity': person.get('popularity', 0)
                }
                for person in credits['crew']
                if person.get('job') == 'Director'
            ]
        
        # Extract cast (top 10)
        cast = []
        if 'cast' in credits:
            cast = [
                {
                    'id': person['id'],
                    'name': person['name'],
                    'character': person.get('character'),
                    'order': person.get('order', 0),
                    'popularity': person.get('popularity', 0)
                }
                for person in credits['cast'][:10]
            ]
        
        return directors, cast
    
    def fetch_diverse_movies(self, target_count=5000, min_vote_count=100, min_rating=6.0):
        """
        Fetch a diverse set of movies for the database
        Combines popular, top-rated, and genre-specific movies
        """
        movies = []
        movie_ids = set()
        
        logger.info(f"Fetching {target_count} diverse movies from TMDB...")
        
        # Strategy 1: Popular movies (50%)
        logger.info("Fetching popular movies...")
        for page in range(1, int(target_count * 0.5 / 20) + 1):
            data = self.get_popular_movies(page=page)
            if data and 'results' in data:
                for movie in data['results']:
                    if movie['id'] not in movie_ids and len(movies) < target_count * 0.5:
                        if movie.get('vote_count', 0) >= min_vote_count:
                            movie_ids.add(movie['id'])
                            movies.append(movie)
            time.sleep(0.25)  # Rate limiting
        
        # Strategy 2: Top rated movies (30%)
        logger.info("Fetching top rated movies...")
        for page in range(1, int(target_count * 0.3 / 20) + 1):
            data = self.get_top_rated_movies(page=page)
            if data and 'results' in data:
                for movie in data['results']:
                    if movie['id'] not in movie_ids and len(movies) < target_count * 0.8:
                        if movie.get('vote_count', 0) >= min_vote_count:
                            movie_ids.add(movie['id'])
                            movies.append(movie)
            time.sleep(0.25)
        
        # Strategy 3: Genre diversity (20%)
        logger.info("Fetching genre-diverse movies...")
        genres_data = self.get_genres()
        if genres_data and 'genres' in genres_data:
            for genre in genres_data['genres']:
                data = self.discover_movies(
                    with_genres=genre['id'],
                    sort_by='vote_average.desc',
                    page=1
                )
                if data and 'results' in data:
                    for movie in data['results'][:5]:  # Top 5 per genre
                        if movie['id'] not in movie_ids and len(movies) < target_count:
                            if movie.get('vote_count', 0) >= min_vote_count:
                                movie_ids.add(movie['id'])
                                movies.append(movie)
                time.sleep(0.25)
        
        logger.info(f"Fetched {len(movies)} movies (basic info)")
        return movies
    
    def enrich_movie_data(self, movie_list):
        """
        Fetch detailed information for each movie
        Including credits, runtime, etc.
        """
        enriched_movies = []
        
        logger.info(f"Enriching {len(movie_list)} movies with detailed data...")
        
        for idx, movie in enumerate(movie_list):
            try:
                detailed = self.get_movie_details(movie['id'])
                if detailed:
                    enriched_movies.append(detailed)
                    
                    if (idx + 1) % 50 == 0:
                        logger.info(f"Enriched {idx + 1}/{len(movie_list)} movies")
                
                time.sleep(0.3)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Error enriching movie {movie.get('title', movie['id'])}: {e}")
                continue
        
        logger.info(f"Successfully enriched {len(enriched_movies)} movies")
        return enriched_movies


if __name__ == "__main__":
    # Test TMDB connection
    fetcher = TMDBFetcher()
    
    if not fetcher.api_key:
        print("❌ TMDB API key not found in .env file")
        exit(1)
    
    print("Testing TMDB API connection...")
    data = fetcher.get_popular_movies(page=1)
    
    if data and 'results' in data:
        print(f"✓ Connected to TMDB API successfully!")
        print(f"  Fetched {len(data['results'])} popular movies")
        print(f"\n  Sample movie: {data['results'][0]['title']}")
    else:
        print("❌ Failed to connect to TMDB API")
