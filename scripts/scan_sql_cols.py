"""Scan all SQL in py files for table.column refs; report ones missing live."""
import os
import re
from dotenv import load_dotenv
load_dotenv()
import mysql.connector

DB = os.getenv('DB_NAME', 'cinesense')
c = mysql.connector.connect(
    host=os.getenv('DB_HOST', 'localhost'),
    port=int(os.getenv('DB_PORT', '3306')),
    user=os.getenv('DB_USER', 'root'),
    password=os.getenv('DB_PASSWORD', ''),
    database=DB,
)
cur = c.cursor()
cur.execute("SELECT TABLE_NAME, COLUMN_NAME FROM information_schema.COLUMNS "
            "WHERE TABLE_SCHEMA=%s", (DB,))
live = {}
for t, col in cur.fetchall():
    live.setdefault(t.lower(), set()).add(col.lower())

# alias -> table guesses per file are too complex; check common aliases
ALIAS_TABLES = {
    'm': 'movies', 'u': 'users', 'ui': 'user_interactions',
    'g': 'genres', 'mg': 'movie_genres', 'mov': 'movies',
}
roots = ['ai', 'api', 'database', 'scripts', 'inference', 'tmdb']
problems = []
for root in roots:
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if not fn.endswith('.py'):
                continue
            p = os.path.join(dp, fn)
            try:
                src = open(p, encoding='utf-8').read()
            except UnicodeDecodeError:
                src = open(p, encoding='utf-8', errors='replace').read()
            for alias, col in re.findall(r'\b([a-z]{1,3})\.([A-Za-z_][A-Za-z0-9_]*)', src):
                a = alias.lower()
                if a in ALIAS_TABLES and col.lower() not in ('id',):
                    table = ALIAS_TABLES[a]
                    if table in live and col.lower() not in live[table]:
                        problems.append(f"{p}: {alias}.{col} (no such col in {table})")
for line in sorted(set(problems)):
    print(line)
print('done,', len(set(problems)), 'suspects')
