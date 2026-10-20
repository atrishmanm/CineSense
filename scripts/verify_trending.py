"""Verify trending_detector queries run on the live DB after the column fix.

Audit K1.1: DB access moved under the __main__ guard; importing this module
opens no connection.
"""
import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

import mysql.connector


def main():
    DB = os.getenv('DB_NAME', 'cinesense')
    c = mysql.connector.connect(
        host=os.getenv('DB_HOST', 'localhost'),
        port=int(os.getenv('DB_PORT', '3306')),
        user=os.getenv('DB_USER', 'root'),
        password=os.getenv('DB_PASSWORD', ''),
        database=DB,
    )
    cur = c.cursor(dictionary=True)
    cur.execute(
        "SELECT COLUMN_NAME FROM information_schema.COLUMNS "
        "WHERE TABLE_SCHEMA=%s AND TABLE_NAME='user_interactions'",
        (DB,))
    print('interactions cols:', sorted(r['COLUMN_NAME'] for r in cur.fetchall()))

    start = datetime.now() - timedelta(hours=168)
    cur.execute(
        """SELECT m.movie_id, m.title, m.poster_path, m.release_year, m.tmdb_rating,
                  COUNT(ui.interaction_id) as interaction_count
           FROM movies m
           JOIN user_interactions ui ON m.movie_id = ui.chosen_movie_id
           WHERE ui.timestamp >= %s
           GROUP BY m.movie_id
           HAVING interaction_count >= %s
           ORDER BY interaction_count DESC LIMIT 100""",
        (start, 10))
    rows = cur.fetchall()
    print('trending query ok, rows:', len(rows))


if __name__ == '__main__':
    main()
