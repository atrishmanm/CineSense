"""Verify live DB object counts against docs/PROJECT.md claims.

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
    cur = c.cursor()
    cur.execute("SELECT COUNT(*) FROM information_schema.TABLES WHERE TABLE_SCHEMA=%s", (DB,))
    print('tables:', cur.fetchone()[0])
    cur.execute("SELECT COUNT(*) FROM information_schema.VIEWS WHERE TABLE_SCHEMA=%s", (DB,))
    print('views:', cur.fetchone()[0])
    cur.execute("SELECT ROUTINE_TYPE, COUNT(*) FROM information_schema.ROUTINES "
                "WHERE ROUTINE_SCHEMA=%s GROUP BY ROUTINE_TYPE", (DB,))
    print('routines:', cur.fetchall())
    cur.execute("SELECT COUNT(*) FROM information_schema.TRIGGERS WHERE TRIGGER_SCHEMA=%s", (DB,))
    print('triggers:', cur.fetchone()[0])
    cur.execute("SELECT COUNT(*) FROM information_schema.KEY_COLUMN_USAGE "
                "WHERE TABLE_SCHEMA=%s AND REFERENCED_TABLE_NAME IS NOT NULL", (DB,))
    print('foreign_keys:', cur.fetchone()[0])
    cur.execute("SELECT ROUTINE_NAME FROM information_schema.ROUTINES WHERE ROUTINE_SCHEMA=%s",
                (DB,))
    print('procedure_names:', sorted(r[0] for r in cur.fetchall()))


if __name__ == '__main__':
    main()
