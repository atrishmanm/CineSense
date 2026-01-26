"""
Diagnostic script to test CineSense search functionality
Run this to identify why search isn't working
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from tmdb.fetcher import TMDBFetcher
from database.db_manager import db
from config import Config

print("=" * 60)
print("CINESENSE SEARCH DIAGNOSTIC TEST")
print("=" * 60)

# Test 1: Check TMDB API Key
print("\n1️⃣ Testing TMDB API Key...")
print(f"   API Key configured: {bool(Config.TMDB_API_KEY)}")
print(f"   API Key length: {len(Config.TMDB_API_KEY)} characters")
if Config.TMDB_API_KEY:
    print(f"   API Key preview: {Config.TMDB_API_KEY[:10]}...")
else:
    print("   ❌ ERROR: No TMDB API key found!")
    sys.exit(1)

# Test 2: Test TMDB API Connection
print("\n2️⃣ Testing TMDB API Connection...")
try:
    fetcher = TMDBFetcher()
    results = fetcher.search_movies("mission impossible", page=1)
    
    if results and 'results' in results:
        print(f"   ✅ SUCCESS: TMDB API is working!")
        print(f"   Total results: {results.get('total_results', 0)}")
        print(f"   Movies found on page 1: {len(results['results'])}")
        print(f"\n   First 5 titles:")
        for i, movie in enumerate(results['results'][:5], 1):
            print(f"      {i}. {movie.get('title', 'N/A')} ({movie.get('release_date', 'N/A')[:4]})")
    else:
        print("   ❌ ERROR: TMDB returned no results")
        sys.exit(1)
except Exception as e:
    print(f"   ❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Test Database Connection
print("\n3️⃣ Testing Database Connection...")
try:
    with db.get_connection() as conn:
        print(f"   ✅ Database connected: {Config.DB_CONFIG['database']}")
except Exception as e:
    print(f"   ❌ ERROR: Database connection failed: {e}")
    sys.exit(1)

# Test 4: Check current movies in database
print("\n4️⃣ Checking current database content...")
try:
    query = "SELECT COUNT(*) as count FROM movies"
    with db.get_cursor() as cursor:
        cursor.execute(query)
        result = cursor.fetchone()
        movie_count = result['count'] if result else 0
    
    print(f"   Total movies in database: {movie_count}")
    
    # Check for Mission Impossible movies
    query = """
        SELECT title, release_year, tmdb_rating 
        FROM movies 
        WHERE LOWER(title) LIKE '%mission%impossible%'
        ORDER BY release_year
    """
    with db.get_cursor() as cursor:
        cursor.execute(query)
        mi_movies = cursor.fetchall()
    
    print(f"\n   Mission Impossible movies in database: {len(mi_movies)}")
    if mi_movies:
        for movie in mi_movies:
            print(f"      - {movie['title']} ({movie['release_year']}) - ⭐ {movie['tmdb_rating']}")
    else:
        print("      ⚠️ No Mission Impossible movies found in database")
        
except Exception as e:
    print(f"   ❌ ERROR: {e}")

# Test 5: Test search function
print("\n5️⃣ Testing search function...")
try:
    results = db.search_movies("mission impossible", limit=10)
    print(f"   Search returned: {len(results)} movies")
    
    if results:
        print(f"\n   Top 5 results:")
        for i, movie in enumerate(results[:5], 1):
            title = movie.get('title', 'N/A')
            year = movie.get('release_year', 'N/A')
            rating = movie.get('tmdb_rating', 'N/A')
            print(f"      {i}. {title} ({year}) - ⭐ {rating}")
    else:
        print("      ⚠️ No results from search")
        
except Exception as e:
    print(f"   ❌ ERROR: {e}")
    import traceback
    traceback.print_exc()

# Test 6: Test content pipeline
print("\n6️⃣ Testing content pipeline fetch...")
try:
    from ai.content_pipeline import pipeline
    
    print("   Fetching 'mission impossible' from TMDB...")
    movies = pipeline.fetch_on_demand(search_query="mission impossible")
    
    print(f"   ✅ Fetched and stored: {len(movies) if movies else 0} movies")
    
    # Search again
    results = db.search_movies("mission impossible", limit=10)
    print(f"   After fetch, search returned: {len(results)} movies")
    
except Exception as e:
    print(f"   ❌ ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 60)
print("DIAGNOSTIC COMPLETE")
print("=" * 60)

print("\n📋 SUMMARY:")
print("   1. TMDB API: Check output above")
print("   2. Database: Check output above")
print("   3. Search: Check output above")
print("\n💡 If TMDB is working but search isn't, the issue is likely:")
print("   - Database not storing movies properly")
print("   - Search query not matching stored titles")
print("   - Movies stored with different titles/formats")
print("\n🔧 NEXT STEPS:")
print("   1. Open: http://localhost:5000/api/test/tmdb?q=mission%20impossible")
print("   2. Check the JSON response")
print("   3. If it shows movies, TMDB is working")
print("   4. Then try search again")
print("=" * 60)
