"""
Populate CineSense with synthetic data and ensure every base table has at least N rows.

What this script does:
1) Calls application APIs over HTTP to create realistic user activity.
2) Fills remaining low-row base tables using metadata-driven synthetic inserts.

Usage:
    python scripts/fill_all_tables.py --base-url http://127.0.0.1:5000 --min-rows 10
"""

from __future__ import annotations

import argparse
import random
import string
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from database.db_manager import db


def _rand_text(prefix: str, length: int = 24) -> str:
    suffix = "".join(random.choices(string.ascii_lowercase + string.digits, k=max(6, length - len(prefix))))
    return (prefix + suffix)[:length]


def _api_post(session: requests.Session, base_url: str, path: str, payload: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
    try:
        resp = session.post(f"{base_url}{path}", json=payload, timeout=20)
        data = {}
        try:
            data = resp.json()
        except Exception:
            data = {"raw": resp.text[:500]}
        return resp.status_code, data
    except Exception as exc:
        return 0, {"error": str(exc)}


def _api_get(session: requests.Session, base_url: str, path: str) -> Tuple[int, Dict[str, Any]]:
    try:
        resp = session.get(f"{base_url}{path}", timeout=20)
        data = {}
        try:
            data = resp.json()
        except Exception:
            data = {"raw": resp.text[:500]}
        return resp.status_code, data
    except Exception as exc:
        return 0, {"error": str(exc)}


def _ensure_server_alive(base_url: str) -> None:
    probe_urls = [
        f"{base_url}/",
        f"{base_url}/api/trending",
    ]
    for _ in range(20):
        for url in probe_urls:
            try:
                r = requests.get(url, timeout=8)
                if r.status_code < 500:
                    return
            except Exception:
                pass
        time.sleep(1.5)
    raise RuntimeError(
        f"Server is not reachable at {base_url}. Start app_integrated.py first and retry."
    )


def _seed_via_api(base_url: str, min_rows: int) -> None:
    print("[API] Seeding synthetic users and activity through HTTP endpoints...")
    _ensure_server_alive(base_url)

    movie_rows = db.query("SELECT movie_id FROM movies ORDER BY popularity DESC LIMIT 200") or []
    movie_ids = [int(r["movie_id"]) for r in movie_rows]
    if len(movie_ids) < 20:
        raise RuntimeError("Need at least 20 movies in DB before API seeding.")

    sessions: List[requests.Session] = []
    users: List[Dict[str, Any]] = []

    # Create/login >= min_rows users
    for i in range(min_rows):
        s = requests.Session()
        uname = f"synth_user_{i+1:02d}"
        email = f"{uname}@cinesense.local"
        password = "Passw0rd!123"

        code, data = _api_post(
            s,
            base_url,
            "/api/user/signup",
            {"username": uname, "email": email, "password": password},
        )
        if code == 201:
            uid = data.get("user_id")
        else:
            code2, data2 = _api_post(
                s,
                base_url,
                "/api/user/login",
                {"username": uname, "password": password},
            )
            if code2 != 200:
                print(f"[API][WARN] Could not signup/login {uname}: {code} {data} / {code2} {data2}")
                continue
            uid = data2.get("user_id")

        sessions.append(s)
        users.append({"user_id": uid, "username": uname})

    if len(sessions) < min_rows:
        raise RuntimeError(f"Could only authenticate {len(sessions)} users via API.")

    # Watchlist + reviews
    for i, s in enumerate(sessions):
        movie_id = movie_ids[i % len(movie_ids)]
        _api_post(s, base_url, "/api/watchlist", {"movie_id": movie_id, "priority": 5, "status": "planned"})
        _api_post(
            s,
            base_url,
            "/api/reviews",
            {
                "movie_id": movie_id,
                "rating": round(6.0 + (i % 5) * 0.8, 1),
                "review_text": f"Synthetic review #{i+1}",
                "is_spoiler": False,
            },
        )

    # Interactions / feedback
    s0 = sessions[0]
    for i in range(max(min_rows, 10)):
        m1 = movie_ids[(2 * i) % len(movie_ids)]
        m2 = movie_ids[(2 * i + 1) % len(movie_ids)]
        chosen = m1 if i % 2 == 0 else m2
        _api_post(s0, base_url, "/api/feedback", {"movie1_id": m1, "movie2_id": m2, "chosen_id": chosen})

    # Friend requests + accepts
    sender = sessions[0]
    for i in range(1, min_rows):
        _api_post(sender, base_url, "/api/social/friends/add", {"friend_username": users[i]["username"]})

    for i in range(1, min_rows):
        recv = sessions[i]
        code, data = _api_get(recv, base_url, "/api/social/friends/requests")
        if code == 200:
            for req in data.get("requests", []):
                _api_post(recv, base_url, "/api/social/friends/accept", {"request_id": req["request_id"]})

    # Watch parties
    now = datetime.utcnow()
    for i in range(min_rows):
        host = sessions[i % len(sessions)]
        movie_id = movie_ids[(i + 15) % len(movie_ids)]
        invitee_idx = (i + 1) % len(users)
        scheduled = (now + timedelta(days=i + 1)).strftime("%Y-%m-%d %H:%M:%S")
        _api_post(
            host,
            base_url,
            "/api/social/watchparty/create",
            {"movie_id": movie_id, "scheduled_time": scheduled, "invitees": [users[invitee_idx]["user_id"]]},
        )

    # Collaborative lists + list movies
    for i in range(min_rows):
        owner = sessions[i % len(sessions)]
        code, data = _api_post(
            owner,
            base_url,
            "/api/social/lists/create",
            {
                "name": f"Synthetic List {i+1}",
                "description": "Autogenerated list",
                "is_public": bool(i % 2),
            },
        )
        if code in (200, 201) and data.get("list_id"):
            list_id = data["list_id"]
            movie_id = movie_ids[(i + 30) % len(movie_ids)]
            _api_post(owner, base_url, f"/api/social/lists/{list_id}/add", {"movie_id": movie_id})

    print("[API] Done.")


def _get_base_tables() -> List[str]:
    rows = db.query("SHOW FULL TABLES WHERE Table_type = 'BASE TABLE'") or []
    tables: List[str] = []
    for r in rows:
        tables.append(list(r.values())[0])
    return tables


def _row_count(table: str) -> int:
    return int(db.query(f"SELECT COUNT(*) AS c FROM {table}", fetch_all=False)["c"])


def _get_columns(table: str) -> List[Dict[str, Any]]:
    return db.query(
        """
        SELECT
            COLUMN_NAME,
            DATA_TYPE,
            COLUMN_TYPE,
            IS_NULLABLE,
            COLUMN_DEFAULT,
            EXTRA,
            CHARACTER_MAXIMUM_LENGTH
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = %s
        ORDER BY ORDINAL_POSITION
        """,
        (table,),
    ) or []


def _get_foreign_keys(table: str) -> Dict[str, Tuple[str, str]]:
    rows = db.query(
        """
        SELECT COLUMN_NAME, REFERENCED_TABLE_NAME, REFERENCED_COLUMN_NAME
        FROM INFORMATION_SCHEMA.KEY_COLUMN_USAGE
        WHERE TABLE_SCHEMA = DATABASE()
          AND TABLE_NAME = %s
          AND REFERENCED_TABLE_NAME IS NOT NULL
        """,
        (table,),
    ) or []
    return {
        r["COLUMN_NAME"]: (r["REFERENCED_TABLE_NAME"], r["REFERENCED_COLUMN_NAME"])
        for r in rows
    }


def _fetch_ref_values(table: str, column: str) -> List[Any]:
    rows = db.query(f"SELECT {column} AS v FROM {table} LIMIT 500") or []
    return [r["v"] for r in rows if r.get("v") is not None]


def _enum_first(column_type: str) -> str:
    # enum('a','b') -> a
    if not column_type.startswith("enum("):
        return "value"
    inside = column_type[len("enum("):-1]
    parts = [p.strip().strip("'") for p in inside.split(",") if p.strip()]
    return parts[0] if parts else "value"


def _synthetic_value(col: Dict[str, Any], row_idx: int) -> Any:
    dt = (col["DATA_TYPE"] or "").lower()
    ct = (col.get("COLUMN_TYPE") or "").lower()
    max_len = col.get("CHARACTER_MAXIMUM_LENGTH") or 255

    if dt in {"int", "bigint", "smallint", "mediumint", "tinyint"}:
        return row_idx + 1
    if dt in {"decimal", "float", "double", "real"}:
        return round(1.5 + (row_idx % 9) * 0.7, 3)
    if dt == "enum":
        return _enum_first(ct)
    if dt in {"varchar", "char"}:
        return _rand_text(f"s_{col['COLUMN_NAME']}_", min(max_len, 64))
    if dt in {"text", "mediumtext", "longtext"}:
        return f"Synthetic {col['COLUMN_NAME']} #{row_idx + 1}"
    if dt == "date":
        return datetime.utcnow().date().isoformat()
    if dt in {"datetime", "timestamp"}:
        return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    if dt == "time":
        return "12:00:00"
    if dt == "year":
        return 2020 + (row_idx % 6)
    if dt == "json":
        return "{}"
    return _rand_text("val_", 32)


def _try_fill_table(table: str, min_rows: int, fk_cache: Dict[Tuple[str, str], List[Any]]) -> int:
    current = _row_count(table)
    if current >= min_rows:
        return 0

    cols = _get_columns(table)
    fks = _get_foreign_keys(table)

    insertable = [c for c in cols if "auto_increment" not in (c.get("EXTRA") or "").lower()]
    if not insertable:
        return 0

    inserted = 0
    attempts = 0
    needed = min_rows - current

    while inserted < needed and attempts < needed * 20:
        attempts += 1
        row_idx = current + inserted + attempts
        col_names: List[str] = []
        values: List[Any] = []
        unresolved_fk = False

        for c in insertable:
            name = c["COLUMN_NAME"]
            nullable = c.get("IS_NULLABLE") == "YES"
            default = c.get("COLUMN_DEFAULT")
            dt = (c.get("DATA_TYPE") or "").lower()

            # Let DB generate timestamps with CURRENT_TIMESTAMP defaults unless needed.
            if dt in {"timestamp", "datetime"} and default and "current_timestamp" in str(default).lower():
                continue

            if name in fks:
                ref_table, ref_col = fks[name]
                key = (ref_table, ref_col)
                if key not in fk_cache:
                    fk_cache[key] = _fetch_ref_values(ref_table, ref_col)
                ref_vals = fk_cache.get(key, [])
                if not ref_vals:
                    if nullable:
                        continue
                    unresolved_fk = True
                    break
                val = random.choice(ref_vals)
            elif default is not None and random.random() < 0.4:
                continue
            else:
                val = _synthetic_value(c, row_idx)

            if val is None and not nullable:
                unresolved_fk = True
                break

            col_names.append(name)
            values.append(val)

        if unresolved_fk or not col_names:
            continue

        placeholders = ", ".join(["%s"] * len(values))
        sql = f"INSERT IGNORE INTO {table} ({', '.join(col_names)}) VALUES ({placeholders})"
        try:
            affected = db.execute(sql, tuple(values))
            if affected:
                inserted += 1
        except Exception:
            # best effort: skip malformed synthetic row and continue
            continue

    return inserted


def _top_up_all_tables(min_rows: int) -> None:
    print("[DB] Top-up pass for base tables...")
    tables = _get_base_tables()
    fk_cache: Dict[Tuple[str, str], List[Any]] = {}

    # Multiple passes help when FK chains depend on previously seeded tables.
    for _ in range(4):
        progress = 0
        for t in tables:
            before = _row_count(t)
            if before >= min_rows:
                continue
            added = _try_fill_table(t, min_rows, fk_cache)
            if added > 0:
                progress += added
        if progress == 0:
            break

    print("[DB] Top-up pass complete.")


def _print_counts() -> None:
    print("\n=== BASE TABLE COUNTS ===")
    for t in sorted(_get_base_tables()):
        print(f"{t}\t{_row_count(t)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fill CineSense tables with synthetic data")
    parser.add_argument("--base-url", default="http://127.0.0.1:5000", help="Base URL of running CineSense app")
    parser.add_argument("--min-rows", type=int, default=10, help="Minimum rows per base table")
    args = parser.parse_args()

    random.seed(42)

    print("Starting synthetic fill process...")
    _seed_via_api(args.base_url, args.min_rows)
    _top_up_all_tables(args.min_rows)
    _print_counts()
    print("\nDone.")


if __name__ == "__main__":
    main()
