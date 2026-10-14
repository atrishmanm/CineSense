"""Print movies-table columns + scan trending_detector queries for bad columns."""
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
cur.execute(
    "SELECT COLUMN_NAME FROM information_schema.COLUMNS "
    "WHERE TABLE_SCHEMA=%s AND TABLE_NAME='movies' ORDER BY ORDINAL_POSITION",
    (DB,))
cols = [r[0] for r in cur.fetchall()]
print('movies columns:', cols)

src = open('ai/trending_detector.py', encoding='utf-8').read()
used = sorted(set(re.findall(r'\bm\.(\w+)', src)) | set(re.findall(r'\bmovies\.(\w+)', src)))
print('m.* cols used:', used)
print('MISSING:', [u for u in used if u not in cols])
