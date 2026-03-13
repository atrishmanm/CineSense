"""
Legacy wrapper for applying additional non-core schema updates.

Core schema lives in database/schema.sql.
This wrapper only invokes scripts/update_schema.py for optional social tables.
"""

import logging
import sys

from scripts.update_schema import update_schema

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def apply_todo_schema_updates():
    """Apply optional social-feature schema updates using the maintained script."""
    logger.info("=" * 60)
    logger.info("Applying optional social schema updates")
    logger.info("=" * 60)

    try:
        success_count, error_count = update_schema()
        logger.info("Social schema update summary: success=%s errors=%s", success_count, error_count)
        return error_count == 0
    except Exception as exc:
        logger.error("Social schema update failed: %s", exc)
        return False


if __name__ == '__main__':
    logger.info("Starting optional schema migration...")
    success = apply_todo_schema_updates()
    if success:
        logger.info("Schema migration completed successfully")
        sys.exit(0)

    logger.error("Schema migration completed with errors")
    sys.exit(1)
