"""
DEPRECATED — Database Schema Update for New Features
====================================================

Superseded — canonical social table DDL lives in ``database/schema.sql``;
this script is deprecated for schema creation and must NOT be used to create
tables (audit K1.3 / D4.2, ``docs/AUDIT-2026-10-19-independent-recheck.md``).

The DDL that used to live here (the three divergent social-DDL copies) drifted
away from the canonical schema:

* ``friend_requests`` used ``user_id``/``friend_id`` instead of the canonical
  ``from_user_id``/``to_user_id`` (+ ``responded_at``/``message`` columns);
* ``watch_parties`` was missing the ``'pending'``/``'active'`` ENUM values;
* ``watch_party_invites`` was missing the ``'pending'`` status value;
* it also created ``movie_lists``/``list_movies`` (canonical: ``media_lists``
  shape in schema.sql) and an ``ab_experiments`` table that exists nowhere else.

Schema creation now happens exclusively via ``database/schema.sql`` (executed
by ``python database/run_migration.py``, which also bootstraps the schema on
an empty ledger). Tables created with the stale DDL from this script may be
incompatible with the application's queries (``api/social_routes.py`` reads
the canonical columns).
"""

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DEPRECATION_NOTICE = (
    "\n" + "=" * 60 + "\n"
    "scripts/update_schema.py is DEPRECATED for schema creation.\n"
    "Superseded: canonical social-table DDL lives in database/schema.sql.\n"
    "Create/update the schema with:\n"
    "    python database/run_migration.py\n"
    "or:\n"
    "    mysql -u root -p cinesense < database/schema.sql\n"
    "\n"
    "The previous DDL in this script diverged from the canonical schema\n"
    "(friend_requests column names, watch_parties/watch_party_invites ENUM\n"
    "values, media-lists and ab_experiments tables) and would break the\n"
    "application's queries — do not execute it.\n"
    + "=" * 60
)


def main():
    """Print the deprecation notice; performs no DB work."""
    print(DEPRECATION_NOTICE)
    logger.info("No-op: this script no longer creates or alters tables.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
