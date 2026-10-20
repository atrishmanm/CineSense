"""Run the fixed candidate_generator queries against the live DB.

Audit K1.1: DB access moved under the __main__ guard; importing this module
opens no connection.
"""
import os
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
    cur.execute("SELECT user_id FROM user_interactions LIMIT 1")
    row = cur.fetchone()
    uid = row['user_id'] if row else 1

    cur.execute(
        """SELECT g.genre_id, COUNT(*) as interaction_count
           FROM user_interactions ui
           JOIN movie_genres mg ON ui.chosen_movie_id = mg.movie_id
           JOIN genres g ON mg.genre_id = g.genre_id
           WHERE ui.user_id = %s AND ui.chosen_movie_id IS NOT NULL
           GROUP BY g.genre_id ORDER BY interaction_count DESC LIMIT 5""",
        (uid,))
    print('fav genres ok:', cur.fetchall())

    cur.execute(
        """SELECT m.movie_id, m.title, m.tmdb_id
           FROM movies m
           JOIN user_interactions ui ON m.movie_id = ui.chosen_movie_id
           WHERE ui.user_id = %s AND ui.chosen_movie_id IS NOT NULL
           ORDER BY ui.timestamp DESC LIMIT %s""",
        (uid, 15))
    print('known movies ok, rows:', len(cur.fetchall()))


if __name__ == '__main__':
    main()
