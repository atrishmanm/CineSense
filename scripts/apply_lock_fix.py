"""Apply the three rewritten concurrency procedures to the live DB and verify.

Audit K1.1: DB side effects now run only under the __main__ guard; importing
this module opens no connection and mutates nothing.
"""
import os
import re
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
    sql = open('database/schema.sql', encoding='utf-8').read()
    names = ['read_movie_elo_shared_lock',
             'generate_elo_report_table_read_lock',
             'bulk_elo_reset_table_write_lock']
    cur = c.cursor()
    for name in names:
        m = re.search(r'CREATE PROCEDURE %s\(.*?^END //$' % name,
                      sql, re.DOTALL | re.MULTILINE)
        assert m, name
        body = m.group(0)
        body = re.sub(r'\s*//\s*$', '', body)
        cur.execute(f'DROP PROCEDURE IF EXISTS {name}')
        cur.execute(body)
        print('created:', name)

    cur.execute("SELECT COUNT(*) FROM information_schema.ROUTINES "
                "WHERE ROUTINE_SCHEMA=%s AND ROUTINE_TYPE='PROCEDURE'", (DB,))
    print('live procedures:', cur.fetchone()[0])
    cur.callproc('generate_elo_report_table_read_lock')
    for rs in cur.stored_results():
        rows = rs.fetchall()
        print('report rows:', len(rows), '| top:', rows[0] if rows else None)


if __name__ == '__main__':
    main()
