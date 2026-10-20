"""
Hybrid Content Ingestion Pipeline for CineSense
Combines incremental updates, scheduled refreshes, and inventory-aware fetching
for continuous availability of new content without manual intervention
"""

import schedule
import time
import threading
from datetime import datetime, timedelta
from database.db_manager import db
from tmdb.fetcher import TMDBFetcher
import logging

# wave5 fix (C3.68): module-level logging.basicConfig hijacked the root
# logger on import; modules now only obtain their own namespaced logger.
logger = logging.getLogger(__name__)


class ContentPipeline:
    """
    Hybrid content ingestion pipeline that ensures continuous content availability
    through multiple strategies:
    1. Incremental Updates - Fetch trending/new releases daily
    2. Scheduled Refreshes - Update existing content metadata weekly
    3. Inventory-Aware Fetching - Fetch missing content on-demand
    """
    
    def __init__(self):
        self.fetcher = TMDBFetcher()
        self.is_running = False
        self.scheduler_thread = None
        # wave5 fix (C3.17): job handles so stop()/start() can cancel the
        # jobs registered on the (global) schedule object instead of letting
        # duplicates pile up.
        self._scheduled_jobs = []
        
    def start(self):
        """Start the content pipeline background scheduler"""
        if self.is_running and (
            self.scheduler_thread is None or self.scheduler_thread.is_alive()
        ):
            # wave5 fix (C3.17): duplicate start (manual call + startup hook)
            # used to schedule every job twice and double-fetch; reject it.
            logger.warning("Content pipeline already running; duplicate start rejected")
            return
        
        self.is_running = True
        
        # Schedule tasks
        # wave5 fix (C3.17): cancel any jobs left over from a previous start
        # before re-registering, so stop()/start() does not duplicate them.
        for job in self._scheduled_jobs:
            try:
                schedule.cancel_job(job)
            except Exception:
                pass
        self._scheduled_jobs = [
            schedule.every().day.at("02:00").do(self.incremental_update),
            schedule.every(7).days.at("03:00").do(self.scheduled_refresh),
            schedule.every(6).hours.do(self.fetch_trending),
        ]
        
        # Run initial fetch
        threading.Thread(target=self.initial_fetch, daemon=True).start()
        
        # Start scheduler thread
        self.scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        self.scheduler_thread.start()
        
        logger.info("✅ Content pipeline started successfully")
    
    def stop(self):
        """Stop the content pipeline"""
        self.is_running = False
        # wave5 fix (C3.17): cancel registered jobs so they do not fire after
        # stop(), and signal/join the scheduler thread (which previously
        # stayed alive up to its next 60s tick).
        for job in self._scheduled_jobs:
            try:
                schedule.cancel_job(job)
            except Exception:
                pass
        self._scheduled_jobs = []
        if (
            self.scheduler_thread
            and self.scheduler_thread.is_alive()
            and self.scheduler_thread is not threading.current_thread()
        ):
            self.scheduler_thread.join(timeout=5)
        self.scheduler_thread = None
        logger.info("Content pipeline stopped")
    
    def _run_scheduler(self):
        """Run the scheduler loop"""
        while self.is_running:
            # wave5 fix (C3.18): a single escaping exception used to kill the
            # scheduler thread while is_running stayed True (jobs never ran
            # again). Log and continue instead.
            try:
                schedule.run_pending()
            except Exception as e:
                logger.error(f"Scheduler job error (continuing): {e}")
            time.sleep(60)  # Check every minute
    
    # ========================================================================
    # 1. INCREMENTAL UPDATES
    # ========================================================================
    
    def incremental_update(self):
        """Fetch new releases and trending content daily"""
        logger.info("🔄 Running incremental update...")
        
        try:
            # Fetch new releases from last 30 days
            self._fetch_recent_releases()
            
            # Fetch trending content
            self.fetch_trending()
            
            # Fetch popular content
            self._fetch_popular_content()
            
            logger.info("✅ Incremental update completed")
        except Exception as e:
            logger.error(f"❌ Incremental update failed: {e}")
    
    def _fetch_recent_releases(self):
        """Fetch movies released in the last 30 days"""
        logger.info("Fetching recent releases...")
        
        # Calculate date range
        today = datetime.now()
        thirty_days_ago = today - timedelta(days=30)
        
        for page in range(1, 6):  # Fetch 5 pages (100 movies)
            try:
                data = self.fetcher.discover_movies(
                    page=page,
                    sort_by='primary_release_date.desc',
                    **{
                        'primary_release_date.gte': thirty_days_ago.strftime('%Y-%m-%d'),
                        'primary_release_date.lte': today.strftime('%Y-%m-%d'),
                        'vote_count.gte': 10
                    }
                )
                
                if data and 'results' in data:
                    self._process_and_store_movies(data['results'])
                    logger.info(f"Fetched page {page} of recent releases")
                
                time.sleep(0.25)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Error fetching recent releases page {page}: {e}")
    
    def fetch_trending(self):
        """Fetch trending movies"""
        logger.info("Fetching trending content...")
        
        try:
            # Fetch now playing movies
            for page in range(1, 4):
                data = self.fetcher.get_now_playing(page=page)
                if data and 'results' in data:
                    self._process_and_store_movies(data['results'])
                time.sleep(0.25)
            
            logger.info("✅ Trending content fetched")
        except Exception as e:
            logger.error(f"Error fetching trending content: {e}")
    
    def _fetch_popular_content(self):
        """Fetch popular movies and TV shows"""
        logger.info("Fetching popular content...")
        
        try:
            # Popular movies
            for page in range(1, 6):
                data = self.fetcher.get_popular_movies(page=page)
                if data and 'results' in data:
                    self._process_and_store_movies(data['results'])
                time.sleep(0.25)
            
            # Popular TV series
            for page in range(1, 4):
                data = self.fetcher.get_popular_tv_series(page=page)
                if data and 'results' in data:
                    self._process_and_store_movies(data['results'], media_type='tv')
                time.sleep(0.25)
            
            logger.info("✅ Popular content fetched")
        except Exception as e:
            logger.error(f"Error fetching popular content: {e}")
    
    # ========================================================================
    # 2. SCHEDULED REFRESHES
    # ========================================================================
    
    def scheduled_refresh(self):
        """Update metadata for existing movies weekly"""
        logger.info("🔄 Running scheduled refresh...")
        
        try:
            # Get movies that haven't been updated in the last 7 days
            query = """
                SELECT movie_id, tmdb_id 
                FROM movies 
                WHERE created_at < DATE_SUB(NOW(), INTERVAL 7 DAY)
                ORDER BY popularity DESC
                LIMIT 500
            """
            
            with db.get_cursor() as cursor:
                cursor.execute(query)
                movies = cursor.fetchall()
            
            logger.info(f"Refreshing {len(movies)} movies...")
            
            for movie in movies:
                try:
                    # Fetch updated details
                    details = self.fetcher.get_movie_details(movie['tmdb_id'])
                    if details:
                        movie_data = self.fetcher.parse_movie_data(details)
                        db.insert_movie(movie_data)
                    
                    time.sleep(0.25)  # Rate limiting
                    
                except Exception as e:
                    logger.error(f"Error refreshing movie {movie['tmdb_id']}: {e}")
            
            logger.info("✅ Scheduled refresh completed")
        except Exception as e:
            logger.error(f"❌ Scheduled refresh failed: {e}")
    
    # ========================================================================
    # 3. INVENTORY-AWARE FETCHING
    # ========================================================================
    
    def fetch_on_demand(self, search_query=None, genre=None, year=None):
        """
        Fetch content on-demand based on search queries or filters
        This is called when user searches for content not in inventory
        Fetches multiple pages for better coverage
        """
        logger.info(f"🔍 On-demand fetch: query='{search_query}', genre={genre}, year={year}")
        
        try:
            if search_query:
                # Search TMDB directly - fetch multiple pages for franchises
                all_movies = []
                max_pages = 3  # Fetch up to 3 pages (60 movies)
                
                for page in range(1, max_pages + 1):
                    results = self.fetcher.search_movies(search_query, page=page)
                    if results and 'results' in results and len(results['results']) > 0:
                        movies = self._process_and_store_movies(results['results'])
                        all_movies.extend(movies)
                        logger.info(f"📥 Page {page}: Found {len(movies)} movies")
                        time.sleep(0.25)  # Rate limiting
                    else:
                        break  # No more results
                
                logger.info(f"✅ Total stored {len(all_movies)} movies for: {search_query}")
                return all_movies
            
            elif genre or year:
                # Discover with filters.
                # wave5 fix (C3.22): this branch used to fetch ONLY page 1
                # (and return [] on any error). Now it walks total_pages
                # with a small per-run cap (5 pages) and a per-page
                # try/except so one bad fetch cannot kill the whole fetch.
                max_pages = 5
                all_movies = []
                
                params = {'page': 1}
                if genre:
                    params['with_genres'] = genre
                if year:
                    params['year'] = year
                
                try:
                    results = self.fetcher.discover_movies(**params)
                except Exception as e:
                    logger.error(f"Error discovering content (page 1): {e}")
                    results = None
                
                if results and 'results' in results:
                    movies = self._process_and_store_movies(results['results'])
                    all_movies.extend(movies)
                    
                    total_pages = min(results.get('total_pages', 1), max_pages)
                    
                    for page in range(2, total_pages + 1):
                        try:
                            page_params = dict(params)
                            page_params['page'] = page
                            page_results = self.fetcher.discover_movies(**page_params)
                            if page_results and 'results' in page_results:
                                page_movies = self._process_and_store_movies(
                                    page_results['results']
                                )
                                all_movies.extend(page_movies)
                                logger.info(f"📥 Discover page {page}: stored {len(page_movies)} movies")
                            time.sleep(0.25)  # Rate limiting
                        except Exception as e:
                            logger.error(f"Error fetching discover page {page}: {e}")
                            continue
                
                if all_movies:
                    logger.info(f"✅ Found and stored {len(all_movies)} movies")
                    return all_movies
            
            return []
            
        except Exception as e:
            logger.error(f"❌ On-demand fetch failed: {e}")
            return []
    
    def ensure_content_availability(self, min_movies=1000):
        """Ensure minimum number of movies are available in database"""
        try:
            # Check current inventory
            query = "SELECT COUNT(*) as count FROM movies"
            with db.get_cursor() as cursor:
                cursor.execute(query)
                result = cursor.fetchone()
                current_count = result['count'] if result else 0
            
            if current_count < min_movies:
                logger.info(f"📦 Inventory low ({current_count}/{min_movies}), fetching more content...")
                
                # Fetch more content to reach minimum
                pages_needed = (min_movies - current_count) // 20 + 1
                
                for page in range(1, min(pages_needed + 1, 26)):  # Max 500 movies per fetch
                    # wave5 fix (C3.18): one bad page must not abort the
                    # bulk replenishment loop; log and keep going.
                    try:
                        data = self.fetcher.get_popular_movies(page=page)
                        if data and 'results' in data:
                            self._process_and_store_movies(data['results'])
                        time.sleep(0.25)
                    except Exception as e:
                        logger.error(f"Error fetching popular movies page {page}: {e}")
                        continue
                
                logger.info("✅ Inventory replenished")
            else:
                logger.info(f"✅ Inventory sufficient ({current_count} movies)")
                
        except Exception as e:
            logger.error(f"Error checking inventory: {e}")
    
    # ========================================================================
    # HELPER METHODS
    # ========================================================================
    
    def _process_and_store_movies(self, movies, media_type='movie'):
        """Process and store movies in database"""
        stored_movies = []
        
        for movie in movies:
            try:
                movie_data = self.fetcher.parse_movie_data(movie)
                if media_type == 'tv':
                    movie_data['media_type'] = 'tv'
                
                db.insert_movie(movie_data)
                
                # Store genres
                genres = self.fetcher.parse_genres(movie)
                for genre in genres:
                    genre_id = db.insert_genre(genre['name'], genre.get('id'))
                    db.link_movie_genre(movie_data['movie_id'], genre_id)
                
                # Fetch and store cast/crew if needed
                if movie.get('id'):
                    self._store_credits(movie['id'], movie_data['movie_id'])
                
                stored_movies.append(movie_data)
                
            except Exception as e:
                logger.error(f"Error processing movie: {e}")
        
        return stored_movies
    
    def _store_credits(self, tmdb_id, movie_id=None):
        """Fetch and store movie credits (cast and crew)

        ``tmdb_id`` is used to fetch details from TMDB, while ``movie_id`` is the
        internal movies PK used for the link rows. They are normally identical
        (movies.movie_id == tmdb id), but keeping them separate avoids passing a
        TMDB id where a movie_id is required.
        """
        try:
            details = self.fetcher.get_movie_details(tmdb_id)
            if not details:
                return
            
            credits = details.get('credits', {})
            db_movie_id = movie_id if movie_id is not None else tmdb_id
            
            # Store directors
            crew = credits.get('crew', [])
            directors = [c for c in crew if c.get('job') == 'Director']
            
            for director in directors[:3]:  # Store up to 3 directors
                director_id = db.insert_director(
                    director['name'],
                    director.get('id')
                )
                db.link_movie_director(db_movie_id, director_id)
            
            # Store actors
            cast = credits.get('cast', [])
            for actor in cast[:10]:  # Store top 10 actors
                actor_id = db.insert_actor(
                    actor['name'],
                    actor.get('id')
                )
                db.link_movie_actor(
                    db_movie_id,
                    actor_id,
                    actor.get('order', 0),
                    actor.get('character')
                )
            
        except Exception as e:
            logger.error(f"Error storing credits for {tmdb_id}: {e}")
    
    def initial_fetch(self):
        """Initial content fetch when pipeline starts"""
        logger.info("🚀 Running initial content fetch...")
        
        try:
            # Ensure minimum inventory
            self.ensure_content_availability(min_movies=500)
            
            # Fetch trending
            self.fetch_trending()
            
            logger.info("✅ Initial fetch completed")
        except Exception as e:
            logger.error(f"❌ Initial fetch failed: {e}")


# Global pipeline instance
pipeline = ContentPipeline()


def start_content_pipeline():
    """Start the content pipeline"""
    pipeline.start()


def stop_content_pipeline():
    """Stop the content pipeline"""
    pipeline.stop()


if __name__ == "__main__":
    # Test the pipeline
    start_content_pipeline()
    logger.info("Content pipeline is running. Press Ctrl+C to stop.")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        stop_content_pipeline()
        logger.info("Pipeline stopped.")
