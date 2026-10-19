-- Migration: 2026_10_19_social_enum_gaps.sql
-- Closes the social-ENUM gaps from audit items C297/C298 so databases created
-- from an older schema (or an older db_manager.ensure_social_tables() copy)
-- match the authoritative DDL in database/schema.sql:
--   * watch_parties.status        gains 'scheduled' and defaults to it
--   * watch_party_invites.status  gains 'invited'   and defaults to it
--
-- Picked up by database/run_migration.py, which lists the .sql files in this
-- directory in sorted order, executes them, and records each file name in
-- schema_migrations (audit C300) only after a successful commit (audit C301).
--
-- Idempotent: on a schema already built from the current database/schema.sql
-- these are no-op rewrites, not errors.
-- NOTE: run_migration.py applies files from migrations/ INSTEAD of schema.sql
-- whenever this directory holds at least one file -- a fresh database must be
-- bootstrapped from database/schema.sql first (or the directory emptied).

ALTER TABLE watch_parties
    MODIFY COLUMN status
    ENUM('scheduled', 'pending', 'active', 'live', 'completed', 'cancelled')
    DEFAULT 'scheduled';

ALTER TABLE watch_party_invites
    MODIFY COLUMN status
    ENUM('invited', 'pending', 'accepted', 'declined')
    DEFAULT 'invited';
