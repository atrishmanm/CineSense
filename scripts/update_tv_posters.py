"""
Quick script to update TV series poster URLs to poster paths in the database
"""
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db_manager import DatabaseManager

def update_tv_posters():
    """Update TV series poster URLs to poster paths"""
    db = DatabaseManager()
    
    # Get all TV series
    query = "SELECT movie_id, poster_path, backdrop_path FROM movies WHERE media_type = 'tv'"
    
    with db.get_cursor(dictionary=True) as cursor:
        cursor.execute(query)
        tv_series = cursor.fetchall()
    
    print(f"Found {len(tv_series)} TV series to check")
    
    updated = 0
    for series in tv_series:
        poster_path = series.get('poster_path')
        backdrop_path = series.get('backdrop_path')
        
        # Check if paths contain full URLs (need fixing)
        needs_update = False
        new_poster = poster_path
        new_backdrop = backdrop_path
        
        if poster_path and 'image.tmdb.org' in poster_path:
            # Extract just the path part
            new_poster = poster_path.split('/t/p/')[-1].replace('w500', '').replace('original', '')
            needs_update = True
            
        if backdrop_path and 'image.tmdb.org' in backdrop_path:
            # Extract just the path part
            new_backdrop = backdrop_path.split('/t/p/')[-1].replace('w500', '').replace('original', '')
            needs_update = True
        
        if needs_update:
            update_query = """
                UPDATE movies 
                SET poster_path = %s, backdrop_path = %s 
                WHERE movie_id = %s
            """
            with db.get_cursor() as cursor:
                cursor.execute(update_query, (new_poster, new_backdrop, series['movie_id']))
            updated += 1
            print(f"Updated series {series['movie_id']}: {poster_path} -> {new_poster}")
    
    print(f"\n✓ Updated {updated} TV series poster paths")
    return True

if __name__ == '__main__':
    try:
        success = update_tv_posters()
        if success:
            print("\n✓ TV series poster paths updated successfully!")
        else:
            print("\n✗ Update failed")
            sys.exit(1)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
