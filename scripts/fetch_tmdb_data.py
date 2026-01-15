"""
TMDB Data Fetching Script
Downloads and stores movie data in the database
"""

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tmdb.fetcher import TMDBFetcher
from database.db_manager import db
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def fetch_and_store_movies(target_count=5000):
    """Main function to fetch and store movies"""
    
    fetcher = TMDBFetcher()
    
    # Check API key
    if not fetcher.api_key:
        logger.error("TMDB API key not found. Please set TMDB_API_KEY in .env file")
        return False
    
    try:
        # Step 1: Fetch basic movie list
        logger.info("=" * 60)
        logger.info("STEP 1: Fetching diverse movie list from TMDB")
        logger.info("=" * 60)
        
        movies_basic = fetcher.fetch_diverse_movies(target_count=target_count)
        
        if not movies_basic:
            logger.error("No movies fetched from TMDB")
            return False
        
        logger.info(f"✓ Fetched {len(movies_basic)} movies")
        
        # Step 2: Enrich with detailed data
        logger.info("\n" + "=" * 60)
        logger.info("STEP 2: Enriching movies with detailed information")
        logger.info("=" * 60)
        
        movies_detailed = fetcher.enrich_movie_data(movies_basic)
        
        if not movies_detailed:
            logger.error("No movies enriched")
            return False
        
        logger.info(f"✓ Enriched {len(movies_detailed)} movies")
        
        # Step 3: Store in database
        logger.info("\n" + "=" * 60)
        logger.info("STEP 3: Storing movies in database")
        logger.info("=" * 60)
        
        stored_count = 0
        
        for idx, movie in enumerate(movies_detailed):
            try:
                # Parse movie data
                movie_data = fetcher.parse_movie_data(movie)
                
                # Insert movie
                db.insert_movie(movie_data)
                
                # Process genres
                genres = fetcher.parse_genres(movie)
                for genre in genres:
                    # Get or create genre
                    existing_genre = db.get_genre_by_name(genre['name'])
                    if existing_genre:
                        genre_id = existing_genre['genre_id']
                    else:
                        genre_id = db.insert_genre(genre['name'], genre.get('id'))
                    
                    # Link movie to genre
                    db.link_movie_genre(movie_data['movie_id'], genre_id)
                
                # Process credits (directors and cast)
                directors, cast = fetcher.parse_credits(movie)
                
                # Store directors
                for director in directors:
                    director_id = db.insert_director(
                        director['name'],
                        director['id'],
                        director.get('popularity')
                    )
                    db.link_movie_director(movie_data['movie_id'], director_id)
                
                # Store cast
                for actor in cast:
                    actor_id = db.insert_actor(
                        actor['name'],
                        actor['id'],
                        actor.get('popularity')
                    )
                    db.link_movie_actor(
                        movie_data['movie_id'],
                        actor_id,
                        actor.get('order', 0),
                        actor.get('character')
                    )
                
                stored_count += 1
                
                if (idx + 1) % 100 == 0:
                    logger.info(f"Stored {idx + 1}/{len(movies_detailed)} movies")
                
            except Exception as e:
                logger.error(f"Error storing movie {movie.get('title', movie.get('id'))}: {e}")
                continue
        
        # Final statistics
        logger.info("\n" + "=" * 60)
        logger.info("COMPLETION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"✓ Successfully stored {stored_count} movies in database")
        logger.info(f"✓ Database now contains {db.get_movie_count()} total movies")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"Fatal error during data fetching: {e}")
        return False


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Fetch movies from TMDB and store in database')
    parser.add_argument('--count', type=int, default=3000, help='Target number of movies to fetch')
    parser.add_argument('--test', action='store_true', help='Test mode: only fetch 50 movies')
    
    args = parser.parse_args()
    
    target = 50 if args.test else args.count
    
    print(f"\n{'='*60}")
    print("CINESENSE - TMDB DATA FETCHER")
    print(f"{'='*60}")
    print(f"Target: {target} movies")
    print(f"Mode: {'TEST' if args.test else 'FULL'}")
    print(f"{'='*60}\n")
    
    success = fetch_and_store_movies(target_count=target)
    
    if success:
        print("\n✓ Data fetching completed successfully!")
        print("You can now run the application: python app.py")
    else:
        print("\n✗ Data fetching failed. Please check the logs.")
        sys.exit(1)
