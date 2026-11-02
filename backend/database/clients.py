"""The client directory."""

from typing import Any

from backend.database.connection import get_connection


def upsert_client(full_name: str, email: str, phone: str = "", address: str = "") -> int:
    """
    Inserts a new client or updates an existing one (matched by the
    (full_name, email) pair together — same trust model as the rest of this
    app, which has no client authentication). Matching on email alone would
    silently merge two differently named people who happen to share an
    inbox (or the same tester submitting several demo requests under one
    email) into a single overwritten row. Returns the client's id.
    """
    conn = get_connection()
    conn.execute(
        """
        INSERT INTO clients (full_name, email, phone, address)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(full_name, email) DO UPDATE SET
            phone = excluded.phone,
            address = excluded.address,
            updated_at = datetime('now')
        """,
        (full_name, email, phone, address),
    )
    client_id = conn.execute(
        "SELECT id FROM clients WHERE full_name = ? AND email = ?", (full_name, email)
    ).fetchone()["id"]
    conn.commit()
    conn.close()
    return client_id


def get_clients() -> list[dict[str, Any]]:
    """Returns the full client directory, most recently updated first."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT id, full_name, email, phone, address, created_at, updated_at "
        "FROM clients ORDER BY updated_at DESC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
