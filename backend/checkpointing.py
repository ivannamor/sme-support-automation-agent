"""
Conversation checkpointer shared by every DoorAssistantAgent.

One SqliteSaver per process, in its own file (db/checkpoints.db):
- shared by all agents, so switching the active model (which builds new
  agents) keeps ongoing conversations and pending HITL confirmations;
- on disk, so conversation state no longer grows in RAM for the life of
  the process.

Checkpoints are the LLM's working memory, not an archive: threads idle for
longer than CHECKPOINT_RETENTION_DAYS are deleted by prune_stale_threads()
(run at app startup). The human-readable record of conversations is the
conversation_log table in the main database.
"""

import datetime
import logging
import sqlite3
import threading
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

from backend.secrets_provider import get_secret

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT_DB_PATH = PROJECT_ROOT / "db" / "checkpoints.db"
DEFAULT_RETENTION_DAYS = 7

logger = logging.getLogger("Checkpointing")

_saver: SqliteSaver | None = None
_saver_lock = threading.Lock()


def _connect() -> sqlite3.Connection:
    CHECKPOINT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Streamlit serves sessions from several threads; SqliteSaver serializes
    # its own use of the connection with a lock.
    conn = sqlite3.connect(CHECKPOINT_DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS thread_activity (
            thread_id      TEXT PRIMARY KEY,
            last_active_at TEXT NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def get_checkpointer() -> SqliteSaver:
    """The process-wide checkpointer, created on first use."""
    global _saver
    with _saver_lock:
        if _saver is None:
            _saver = SqliteSaver(_connect())
        return _saver


def _now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat()


def touch_thread(thread_id: str) -> None:
    """Records activity on a thread, so prune_stale_threads() keeps it."""
    conn = _connect()
    try:
        conn.execute(
            """
            INSERT INTO thread_activity (thread_id, last_active_at) VALUES (?, ?)
            ON CONFLICT(thread_id) DO UPDATE SET last_active_at = excluded.last_active_at
            """,
            (thread_id, _now()),
        )
        conn.commit()
    finally:
        conn.close()


def prune_stale_threads(retention_days: int | None = None) -> int:
    """
    Deletes the checkpoints of threads idle for more than `retention_days`
    (default: CHECKPOINT_RETENTION_DAYS, or 7). Returns how many were deleted.
    """
    if retention_days is None:
        retention_days = int(get_secret("CHECKPOINT_RETENTION_DAYS", str(DEFAULT_RETENTION_DAYS)))
    cutoff = (
        datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=retention_days)
    ).isoformat()

    conn = _connect()
    try:
        stale = [
            row[0]
            for row in conn.execute(
                "SELECT thread_id FROM thread_activity WHERE last_active_at < ?", (cutoff,)
            )
        ]
    finally:
        conn.close()
    if not stale:
        return 0

    saver = get_checkpointer()
    for thread_id in stale:
        saver.delete_thread(thread_id)

    conn = _connect()
    try:
        conn.executemany("DELETE FROM thread_activity WHERE thread_id = ?", [(t,) for t in stale])
        conn.commit()
    finally:
        conn.close()

    logger.info(f"CHECKPOINTS_PRUNED: {len(stale)} thread(s) idle for over {retention_days} days")
    return len(stale)
