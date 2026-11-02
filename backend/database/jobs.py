"""Job completion reports submitted by installers, pending manager approval."""

from typing import Any

from backend.database.connection import _inserted_id, get_connection
from backend.database.tickets import _log_ticket_status


def create_job_completion(
    sent_by: str,
    structured_report: str,
    raw_notes: str = "",
    client_email: str = "",
    booking_id: int | None = None,
) -> int:
    conn = get_connection()
    cursor = conn.execute(
        """
        INSERT INTO job_completions
            (sent_by, booking_id, structured_report, raw_notes, client_email)
        VALUES (?, ?, ?, ?, ?)
        """,
        (sent_by, booking_id, structured_report, raw_notes, client_email),
    )
    job_id = _inserted_id(cursor)
    # Advance ticket to IN REVIEW.
    if booking_id:
        conn.execute(
            "UPDATE pending_bookings SET ticket_status = 'in_review' WHERE id = ?",
            (booking_id,),
        )
        _log_ticket_status(conn, booking_id, "in_review", sent_by)
    conn.commit()
    conn.close()
    return job_id


def get_job_completions(status: str = "pending") -> list[dict[str, Any]]:
    conn = get_connection()
    rows = conn.execute(
        """
        SELECT id, sent_by, booking_id, structured_report, raw_notes,
               client_email, status, created_at, resolved_at, resolved_by
        FROM job_completions
        WHERE status = ?
        ORDER BY created_at DESC
        """,
        (status,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def resolve_job_completion(job_id: int, status: str, resolved_by: str) -> bool:
    conn = get_connection()
    # Fetch the linked booking_id before updating.
    row = conn.execute(
        "SELECT booking_id FROM job_completions WHERE id = ? AND status = 'pending'",
        (job_id,),
    ).fetchone()
    cursor = conn.execute(
        """
        UPDATE job_completions
        SET status = ?, resolved_at = datetime('now'), resolved_by = ?
        WHERE id = ? AND status = 'pending'
        """,
        (status, resolved_by, job_id),
    )
    updated = cursor.rowcount > 0
    if updated and row and row["booking_id"]:
        booking_id = row["booking_id"]
        ticket_status = "closed" if status == "approved" else "in_progress"
        conn.execute(
            "UPDATE pending_bookings SET ticket_status = ? WHERE id = ?",
            (ticket_status, booking_id),
        )
        _log_ticket_status(conn, booking_id, ticket_status, resolved_by)
    conn.commit()
    conn.close()
    return updated
