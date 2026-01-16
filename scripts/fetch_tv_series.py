"""
TMDB TV Series Fetching Script
Downloads and stores TV series data in the database
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tmdb.fetcher import TMDBFetcher
from database.db_manager import db
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def fetch_and_store_tv_series(target_count=500):
    """Main function to fetch and store TV series"""
    
    fetcher = TMDBFetcher()
    
    # Check API key
    if not fetcher.api_key:
        logger.error("TMDB API key not found. Please set TMDB_API_KEY in .env file")
        return False
    
    try:
        # Step 1: Fetch popular and top-rated TV series
        logger.info("=" * 60)
        logger.info("STEP 1: Fetching TV series from TMDB")
        logger.info("=" * 60)
        
        tv_series_list = []
        pages_to_fetch = min(25, (target_count // 20) + 1)  # 20 series per page
        
        # Fetch popular TV series
        logger.info("Fetching popular TV series...")
        for page in range(1, pages_to_fetch + 1):
            try:
                response = fetcher.get_popular_tv_series(page=page)
                if response and 'results' in response:
                    popular = response['results']
                    tv_series_list.extend(popular)
                    logger.info(f"  Page {page}: +{len(popular)} series")
                time.sleep(0.3)  # Rate limiting
            except Exception as e:
                logger.error(f"Error fetching popular TV page {page}: {e}")
        
        # Fetch top-rated TV series
        logger.info("Fetching top-rated TV series...")
        for page in range(1, pages_to_fetch + 1):
            try:
                response = fetcher.get_top_rated_tv_series(page=page)
                if response and 'results' in response:
                    top_rated = response['results']
                    tv_series_list.extend(top_rated)
                    logger.info(f"  Page {page}: +{len(top_rated)} series")
                time.sleep(0.3)  # Rate limiting
            except Exception as e:
                logger.error(f"Error fetching top-rated TV page {page}: {e}")
        
        # Remove duplicates by ID
        seen_ids = set()
        unique_series = []
        for series in tv_series_list:
            series_id = series.get('id')
            if series_id and series_id not in seen_ids:
                seen_ids.add(series_id)
                unique_series.append(series)
        
        logger.info(f"✓ Fetched {len(unique_series)} unique TV series")
        
        if not unique_series:
            logger.error("No TV series fetched from TMDB")
            return False
        
        # Step 2: Enrich with detailed data
        logger.info("\n" + "=" * 60)
        logger.info("STEP 2: Enriching TV series with detailed information")
        logger.info("=" * 60)
        
        enriched_series = []
        for idx, series in enumerate(unique_series[:target_count]):
            try:
                tv_id = series.get('id')
                logger.info(f"  [{idx+1}/{min(target_count, len(unique_series))}] Fetching details for: {series.get('name', 'Unknown')}")
                
                detailed_series = fetcher.get_tv_details(tv_id)
                if detailed_series:
                    enriched_series.append(detailed_series)
                
                time.sleep(0.3)  # Rate limiting
                
            except Exception as e:
                logger.error(f"Error enriching TV series {series.get('name')}: {e}")
        
        logger.info(f"✓ Enriched {len(enriched_series)} TV series")
        
        # Step 3: Store in database
        logger.info("\n" + "=" * 60)
        logger.info("STEP 3: Storing TV series in database")
        logger.info("=" * 60)
        
        stored_count = 0
        
        for idx, series in enumerate(enriched_series):
            try:
                # Parse TV series data into movie format
                tmdb_id = series.get('id')
                series_data = {
                    'movie_id': tmdb_id,  # Use TMDB ID as movie_id
                    'tmdb_id': tmdb_id,
                    'title': series.get('name'),
                    'overview': series.get('overview', ''),
                    'release_date': series.get('first_air_date'),
                    'poster_path': series.get('poster_path'),
                    'backdrop_path': series.get('backdrop_path'),
                    'tmdb_rating': float(series.get('vote_average', 0)),
                    'vote_count': series.get('vote_count', 0),
                    'popularity': float(series.get('popularity', 0)),
                    'language': series.get('original_language', 'en'),
                    'media_type': 'tv',  # Important: mark as TV series
                    'runtime': series.get('episode_run_time', [45])[0] if series.get('episode_run_time') else 45,
                    'budget': 0,
                    'revenue': 0,
                    'tagline': series.get('tagline', ''),
                    'status': series.get('status', 'Released')
                }
                
                # Insert TV series
                db.insert_movie(series_data)
                
                # Process genres
                genres = series.get('genres', [])
                for genre in genres:
                    # Get or create genre
                    existing_genre = db.get_genre_by_name(genre['name'])
                    if existing_genre:
                        genre_id = existing_genre['genre_id']
                    else:
                        genre_id = db.insert_genre(genre['name'], genre.get('id'))
                    
                    # Link genre to TV series
                    db.link_movie_genre(tmdb_id, genre_id)
                
                stored_count += 1
                
                if (idx + 1) % 10 == 0:
                    logger.info(f"  Progress: {stored_count} TV series stored")
                    
            except Exception as e:
                logger.error(f"Error storing TV series {series.get('name')}: {e}")
                import traceback
                traceback.print_exc()
        
        # Final statistics
        logger.info("\n" + "=" * 60)
        logger.info("COMPLETION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"✓ Successfully stored {stored_count} TV series in database")
        logger.info(f"✓ Database now contains {db.get_movie_count()} total movies/series")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"Fatal error during TV series fetching: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Fetch TV series from TMDB and store in database')
    parser.add_argument('--count', type=int, default=500, help='Target number of TV series to fetch')
    parser.add_argument('--test', action='store_true', help='Test mode: only fetch 20 TV series')
    
    args = parser.parse_args()
    
    target = 20 if args.test else args.count
    
    print(f"\n{'='*60}")
    print("CINESENSE - TMDB TV SERIES FETCHER")
    print(f"{'='*60}")
    print(f"Target: {target} TV series")
    print(f"Mode: {'TEST' if args.test else 'FULL'}")
    print(f"{'='*60}\n")
    
    success = fetch_and_store_tv_series(target_count=target)
    
    if success:
        print("\n✓ TV series fetching completed successfully!")
        print("TV shows should now appear on your homepage!")
    else:
        print("\n✗ TV series fetching failed. Please check the logs.")
        sys.exit(1)
