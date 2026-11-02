"""
Applies the numbered schema migrations in backend/database/migrations/.

SQLite's PRAGMA user_version holds the number of the last migration applied.
Each pending migration runs in its own transaction with that number, so it
either applies completely or not at all. Migrations only go forward: to undo
one, restore the backup taken just before it.
"""

import importlib
import logging
import pkgutil
import re
import sqlite3
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

from backend.database import connection, migrations

logger = logging.getLogger("Migrations")


class Migration(NamedTuple):
    version: int
    name: str
    upgrade: Callable[[sqlite3.Connection], None]


class DatabaseNewerThanCodeError(RuntimeError):
    """The database was migrated by a newer version of the code."""


def load_migrations() -> list[Migration]:
    """Finds migrations/mNNNN_name.py files. The numbers must run 1, 2, 3... with no gaps."""
    found = []
    for module_info in pkgutil.iter_modules(migrations.__path__):
        match = re.fullmatch(r"m(\d{4})_\w+", module_info.name)
        if not match:
            raise ValueError(f"Badly named migration: {module_info.name}")
        module = importlib.import_module(f"{migrations.__name__}.{module_info.name}")
        found.append(Migration(int(match.group(1)), module_info.name, module.upgrade))

    found.sort(key=lambda migration: migration.version)
    versions = [migration.version for migration in found]
    if versions != list(range(1, len(found) + 1)):
        raise ValueError(f"Migration numbers must run 1, 2, 3... with no gaps: {versions}")
    return found


def _user_version(conn: sqlite3.Connection) -> int:
    return conn.execute("PRAGMA user_version").fetchone()[0]


def _has_tables(conn: sqlite3.Connection) -> bool:
    return (
        conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' LIMIT 1").fetchone()
        is not None
    )


def _backup(conn: sqlite3.Connection, version: int) -> Path:
    """Copies the database next to itself. The backup API also captures what's still in the WAL."""
    path = connection.DB_PATH.with_name(f"{connection.DB_PATH.name}.bak-v{version}")
    target = sqlite3.connect(path)
    try:
        conn.backup(target)
    finally:
        target.close()
    return path


def run_migrations(pending: list[Migration] | None = None) -> int:
    """Brings the database up to date and returns its version."""
    all_migrations = load_migrations() if pending is None else pending
    latest = all_migrations[-1].version if all_migrations else 0

    conn = connection.get_connection()
    conn.isolation_level = None  # transactions are explicit below
    try:
        current = _user_version(conn)
        if current > latest:
            raise DatabaseNewerThanCodeError(
                f"The database is at version {current}, but this code only knows migrations "
                f"up to {latest}. Update the code, or restore an older backup."
            )
        if current == latest:
            return current

        if _has_tables(conn):
            logger.info("MIGRATION_BACKUP: %s", _backup(conn, current))

        for migration in all_migrations:
            if migration.version <= current:
                continue
            # Takes the write lock first, then re-reads the version, in case
            # another process applied this migration in the meantime.
            conn.execute("BEGIN IMMEDIATE")
            try:
                if _user_version(conn) < migration.version:
                    migration.upgrade(conn)
                    conn.execute(f"PRAGMA user_version = {migration.version:d}")
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                logger.exception("MIGRATION_FAILED: %s", migration.name)
                raise
            logger.info("MIGRATION_APPLIED: %s", migration.name)
        return latest
    finally:
        conn.close()
