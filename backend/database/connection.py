"""The connection to the operational SQLite database."""

import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_DIR = PROJECT_ROOT / "db"
DB_PATH = DB_DIR / "company.db"


def get_connection() -> sqlite3.Connection:
    """
    Opens a SQLite connection to the operational company database.

    WAL mode lets readers proceed without blocking on a concurrent writer,
    and busy_timeout makes a writer that finds the database locked retry for
    up to 5s instead of raising "database is locked" immediately — this
    project has multiple processes (Streamlit UI, FastAPI inventory service)
    hitting the same file, so brief write contention is expected.
    """

    DB_DIR.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    return conn


def _inserted_id(cursor: sqlite3.Cursor) -> int:
    """Row id of the INSERT just run on `cursor` (sqlite3 types lastrowid as Optional)."""
    row_id = cursor.lastrowid
    if row_id is None:
        raise RuntimeError("INSERT did not produce a row id")
    return row_id
