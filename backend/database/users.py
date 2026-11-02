"""Staff accounts and failed login counts."""

import sqlite3
from typing import Any

from backend.database.connection import get_connection


def create_user(username: str, password_hash: str, role: str) -> bool:
    """Inserts a new user. Returns False if username already exists."""
    conn = get_connection()
    try:
        conn.execute(
            "INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
            (username, password_hash, role),
        )
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()


def get_user(username: str) -> dict[str, Any] | None:
    """Returns user row dict or None."""
    conn = get_connection()
    row = conn.execute(
        "SELECT id, username, password_hash, role FROM users WHERE username = ?",
        (username,),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def update_user_password(username: str, password_hash: str) -> bool:
    """Returns False if the user doesn't exist."""
    conn = get_connection()
    try:
        cursor = conn.execute(
            "UPDATE users SET password_hash = ? WHERE username = ?",
            (password_hash, username),
        )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def register_login_attempt(key: str, now: float, window_seconds: int, max_failures: int) -> int:
    """
    Counts a login attempt and returns the count. Atomic, so parallel requests
    each get their own number. A lockout runs from the last failure that counted:
    attempts made while locked don't extend it, and a stale count starts over.
    """
    conn = get_connection()
    try:
        conn.execute("DELETE FROM login_attempts WHERE last_failure <= ?", (now - window_seconds,))
        failures = conn.execute(
            """
            INSERT INTO login_attempts (username, failures, last_failure)
            VALUES (:key, 1, :now)
            ON CONFLICT(username) DO UPDATE SET
                failures = failures + 1,
                last_failure = CASE WHEN failures >= :max THEN last_failure ELSE :now END
            RETURNING failures
            """,
            {"key": key, "now": now, "max": max_failures},
        ).fetchone()[0]
        conn.commit()
        return failures
    finally:
        conn.close()


def clear_login_attempts(key: str) -> None:
    conn = get_connection()
    try:
        conn.execute("DELETE FROM login_attempts WHERE username = ?", (key,))
        conn.commit()
    finally:
        conn.close()
