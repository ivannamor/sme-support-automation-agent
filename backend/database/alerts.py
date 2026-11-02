"""Alerts sent by installers to the manager."""

from typing import Any

from backend.database.connection import _inserted_id, get_connection
from backend.database.tickets import _log_ticket_status


def create_alert(
    sent_by: str,
    raw_message: str,
    structured_message: str,
    job_reference: str = "",
    booking_id: int | None = None,
) -> int:
    conn = get_connection()
    cursor = conn.execute(
        """
        INSERT INTO installer_alerts (sent_by, raw_message, structured_message, job_reference, booking_id)
        VALUES (?, ?, ?, ?, ?)
        """,
        (sent_by, raw_message, structured_message, job_reference, booking_id or None),
    )
    alert_id = _inserted_id(cursor)
    # Set linked ticket to BLOCKED.
    if booking_id:
        conn.execute(
            "UPDATE pending_bookings SET ticket_status = 'blocked' WHERE id = ?",
            (booking_id,),
        )
        _log_ticket_status(conn, booking_id, "blocked", sent_by, note=structured_message[:200])
    conn.commit()
    conn.close()
    return alert_id


def get_alerts(unread_only: bool = False) -> list[dict[str, Any]]:
    conn = get_connection()
    query = (
        "SELECT id, sent_by, raw_message, structured_message, job_reference, "
        "booking_id, is_read, created_at "
        "FROM installer_alerts"
        + (" WHERE is_read = 0" if unread_only else "")
        + " ORDER BY created_at DESC"
    )
    rows = conn.execute(query).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_alerts_for_booking(booking_id: int) -> list[dict[str, Any]]:
    """Returns all alerts linked to a specific booking, newest first."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT id, sent_by, structured_message, created_at "
        "FROM installer_alerts WHERE booking_id = ? ORDER BY created_at DESC",
        (booking_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def mark_alerts_read(alert_ids: list[int]) -> None:
    if not alert_ids:
        return
    conn = get_connection()
    placeholders = ",".join("?" * len(alert_ids))
    conn.execute(
        f"UPDATE installer_alerts SET is_read = 1 WHERE id IN ({placeholders})",
        alert_ids,
    )
    conn.commit()
    conn.close()
