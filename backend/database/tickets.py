"""Tickets: pending bookings, job claims and the survey-to-estimate lifecycle."""

from typing import Any

from backend.database.connection import _inserted_id, get_connection


def create_pending_booking(
    model: str,
    date_str: str,
    time_str: str,
    requested_by: str = "client",
    notes: str = "",
    client_email: str = "",
    client_phone: str = "",
    client_name: str = "",
    client_address: str = "",
    quantity: int = 1,
    ticket_phase: str = "installation",
    client_id: int | None = None,
    client_lat: float | None = None,
    client_lon: float | None = None,
    ticket_type: str = "window",
) -> int:
    """Inserts a pending booking and returns its new id."""
    conn = get_connection()
    cursor = conn.execute(
        """
        INSERT INTO pending_bookings
            (requested_by, model, date_str, time_str, notes,
             client_email, client_phone, client_name, client_address, quantity,
             ticket_phase, client_id, survey_date_str, survey_time_str,
             client_lat, client_lon, ticket_type)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            requested_by,
            model,
            date_str,
            time_str,
            notes,
            client_email,
            client_phone,
            client_name,
            client_address,
            quantity,
            ticket_phase,
            client_id,
            date_str if ticket_phase == "survey" else None,
            time_str if ticket_phase == "survey" else None,
            client_lat,
            client_lon,
            ticket_type,
        ),
    )
    booking_id = _inserted_id(cursor)
    _log_ticket_status(conn, booking_id, "new", requested_by)
    conn.commit()
    conn.close()
    return booking_id


_BOOKING_COLUMNS = """
    id, requested_by, model, date_str, time_str, notes,
    client_email, client_phone, client_name, client_address, quantity,
    assigned_to, started_at,
    ticket_status, ticket_phase, rejection_reason,
    height_cm, width_cm, depth_cm, glass_type, estimated_price,
    client_estimate_response, survey_notes, survey_date_str, survey_time_str,
    estimate_resolved_at, estimate_resolved_by, client_id,
    client_lat, client_lon, ticket_type,
    status, created_at, resolved_at, resolved_by
"""


def get_pending_bookings(status: str = "pending") -> list[dict[str, Any]]:
    """Returns booking rows filtered by status."""
    conn = get_connection()
    rows = conn.execute(
        f"""
        SELECT {_BOOKING_COLUMNS}
        FROM pending_bookings
        WHERE status = ?
        ORDER BY created_at DESC
        """,
        (status,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_approved_bookings(assigned_to: str | None = None) -> list[dict[str, Any]]:
    """
    Returns approved bookings that are not yet closed/cancelled — those are
    finished/dead tickets and must not show up as active/claimable/blocking.
    - assigned_to=None  → all open approved bookings
    - assigned_to=''    → only unassigned (available) bookings
    - assigned_to='x'   → only bookings claimed by installer 'x'
    """
    _COLS = f"SELECT {_BOOKING_COLUMNS} FROM pending_bookings"
    conn = get_connection()
    if assigned_to is None:
        rows = conn.execute(
            _COLS + " WHERE status = 'approved' AND ticket_status NOT IN ('closed', 'cancelled') "
            "ORDER BY date_str, time_str"
        ).fetchall()
    elif assigned_to == "":
        rows = conn.execute(
            _COLS + " WHERE status = 'approved' AND ticket_status NOT IN ('closed', 'cancelled') "
            "AND assigned_to IS NULL ORDER BY date_str, time_str"
        ).fetchall()
    else:
        rows = conn.execute(
            _COLS + " WHERE status = 'approved' AND ticket_status NOT IN ('closed', 'cancelled') "
            "AND assigned_to = ? ORDER BY date_str, time_str",
            (assigned_to,),
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def claim_booking(booking_id: int, installer_username: str) -> bool:
    """
    Assigns an approved booking to an installer and sets ticket to IN PROGRESS.
    Returns False if the installer already has an active booking (single-ticket rule).
    """
    conn = get_connection()
    existing = conn.execute(
        "SELECT id FROM pending_bookings "
        "WHERE status='approved' AND ticket_status NOT IN ('closed', 'cancelled') AND assigned_to=? LIMIT 1",
        (installer_username,),
    ).fetchone()
    if existing:
        conn.close()
        return False
    cursor = conn.execute(
        """
        UPDATE pending_bookings
        SET assigned_to = ?, started_at = datetime('now'), ticket_status = 'in_progress'
        WHERE id = ? AND status = 'approved' AND assigned_to IS NULL
        """,
        (installer_username, booking_id),
    )
    updated = cursor.rowcount > 0
    if updated:
        _log_ticket_status(conn, booking_id, "in_progress", installer_username)
    conn.commit()
    conn.close()
    return updated


def release_booking(booking_id: int, installer_username: str) -> bool:
    """Releases a booking back to the available pool and sets ticket to OPEN."""
    conn = get_connection()
    cursor = conn.execute(
        """
        UPDATE pending_bookings
        SET assigned_to = NULL, started_at = NULL, ticket_status = 'open'
        WHERE id = ? AND status = 'approved' AND assigned_to = ?
        """,
        (booking_id, installer_username),
    )
    updated = cursor.rowcount > 0
    if updated:
        _log_ticket_status(
            conn, booking_id, "open", installer_username, note="released by installer"
        )
    conn.commit()
    conn.close()
    return updated


def resolve_booking(
    booking_id: int, status: str, resolved_by: str, rejection_reason: str = ""
) -> bool:
    """Approves or rejects a pending booking. On approval sets ticket to OPEN."""
    ticket_status = "open" if status == "approved" else "rejected"
    conn = get_connection()
    cursor = conn.execute(
        """
        UPDATE pending_bookings
        SET status = ?, ticket_status = ?, rejection_reason = ?,
            resolved_at = datetime('now'), resolved_by = ?
        WHERE id = ? AND status = 'pending'
        """,
        (status, ticket_status, rejection_reason, resolved_by, booking_id),
    )
    updated = cursor.rowcount > 0
    if updated:
        _log_ticket_status(conn, booking_id, ticket_status, resolved_by, note=rejection_reason)
    conn.commit()
    conn.close()
    return updated


def decline_booking(booking_id: int, installer_username: str, reason: str) -> int:
    """
    Records that an installer declined an available (unclaimed) booking and
    notifies the manager via the alerts feed. Unlike an on-site alert, this
    does not change ticket_status — the job stays available for other
    installers to claim.
    """
    conn = get_connection()
    row = conn.execute(
        "SELECT model, date_str, time_str FROM pending_bookings WHERE id = ?",
        (booking_id,),
    ).fetchone()
    detail = f" ({row['model']} on {row['date_str']} at {row['time_str']})" if row else ""
    code = ticket_code(booking_id)
    structured = (
        f"Installer {installer_username} declined installation {code}{detail}.\n\nReason: {reason}"
    )
    cursor = conn.execute(
        """
        INSERT INTO installer_alerts (sent_by, raw_message, structured_message, job_reference)
        VALUES (?, ?, ?, ?)
        """,
        (installer_username, reason, structured, f"Declined booking {code}"),
    )
    alert_id = _inserted_id(cursor)
    conn.commit()
    conn.close()
    return alert_id


def get_installer_active_booking(installer_username: str) -> dict[str, Any] | None:
    """Returns the single non-closed booking currently claimed by this installer, or None."""
    conn = get_connection()
    row = conn.execute(
        """
        SELECT id, model, date_str, time_str
        FROM pending_bookings
        WHERE status = 'approved' AND ticket_status NOT IN ('closed', 'cancelled') AND assigned_to = ?
        LIMIT 1
        """,
        (installer_username,),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def save_survey_measurements(
    booking_id: int,
    height_cm: float,
    width_cm: float,
    depth_cm: float,
    glass_type: str,
    survey_notes: str = "",
    changed_by: str = "",
) -> bool:
    """
    Records the installer's site-survey measurements and advances the ticket
    to IN REVIEW for the manager to price. Only valid on a survey-phase
    ticket the installer is actively working (claimed, in progress).
    """
    conn = get_connection()
    cursor = conn.execute(
        """
        UPDATE pending_bookings
        SET height_cm = ?, width_cm = ?, depth_cm = ?, glass_type = ?,
            survey_notes = ?, ticket_status = 'in_review'
        WHERE id = ? AND ticket_phase = 'survey' AND ticket_status = 'in_progress'
        """,
        (height_cm, width_cm, depth_cm, glass_type, survey_notes, booking_id),
    )
    updated = cursor.rowcount > 0
    if updated:
        _log_ticket_status(conn, booking_id, "in_review", changed_by)
    conn.commit()
    conn.close()
    return updated


def resolve_survey_estimate(booking_id: int, estimated_price: float, resolved_by: str) -> bool:
    """
    Records the manager's computed price estimate and moves the ticket to
    AWAITING CLIENT — the client must accept/reject before installation work
    can begin. Only valid on a survey report currently in review.
    """
    conn = get_connection()
    cursor = conn.execute(
        """
        UPDATE pending_bookings
        SET estimated_price = ?, estimate_resolved_at = datetime('now'),
            estimate_resolved_by = ?, ticket_status = 'awaiting_client'
        WHERE id = ? AND ticket_phase = 'survey' AND ticket_status = 'in_review'
        """,
        (estimated_price, resolved_by, booking_id),
    )
    updated = cursor.rowcount > 0
    if updated:
        _log_ticket_status(conn, booking_id, "awaiting_client", resolved_by)
    conn.commit()
    conn.close()
    return updated


def reject_survey_report(booking_id: int, resolved_by: str) -> bool:
    """
    Rejects a submitted survey report (e.g. bad/incomplete measurements) and
    sends the ticket back to IN PROGRESS so the installer can redo it —
    mirrors resolve_job_completion's rejection behavior for job reports.
    """
    conn = get_connection()
    cursor = conn.execute(
        """
        UPDATE pending_bookings
        SET ticket_status = 'in_progress', estimate_resolved_at = datetime('now'),
            estimate_resolved_by = ?
        WHERE id = ? AND ticket_phase = 'survey' AND ticket_status = 'in_review'
        """,
        (resolved_by, booking_id),
    )
    updated = cursor.rowcount > 0
    if updated:
        _log_ticket_status(
            conn, booking_id, "in_progress", resolved_by, note="survey report rejected"
        )
    conn.commit()
    conn.close()
    return updated


def resolve_client_estimate_response(
    booking_id: int,
    response: str,
    new_date_str: str | None = None,
    new_time_str: str | None = None,
    changed_by: str = "",
    *,
    client_email: str,
) -> bool:
    """
    Records the client's accept/reject decision on a priced estimate.

    Accept: preserves the original survey date/time in survey_date_str/time_str,
    overwrites date_str/time_str with the new installation appointment, and
    flips the ticket into the installation phase, re-opened for an installer
    to claim. Reject: marks the ticket cancelled (distinct from 'rejected',
    which means the manager rejected the initial survey request — this is
    the client declining a priced estimate, a different point in the flow).

    client_email must match the ticket's own email (case-insensitive) — the
    ownership check is enforced here too, not only by the calling tool.
    """
    if response == "accept":
        conn = get_connection()
        cursor = conn.execute(
            """
            UPDATE pending_bookings
            SET client_estimate_response = 'accepted',
                survey_date_str = date_str, survey_time_str = time_str,
                date_str = ?, time_str = ?,
                ticket_phase = 'installation', ticket_status = 'open'
            WHERE id = ? AND ticket_status = 'awaiting_client'
              AND lower(client_email) = lower(?) AND client_email != ''
            """,
            (new_date_str, new_time_str, booking_id, client_email),
        )
        new_status = "open"
    else:
        conn = get_connection()
        cursor = conn.execute(
            """
            UPDATE pending_bookings
            SET client_estimate_response = 'rejected', ticket_status = 'cancelled'
            WHERE id = ? AND ticket_status = 'awaiting_client'
              AND lower(client_email) = lower(?) AND client_email != ''
            """,
            (booking_id, client_email),
        )
        new_status = "cancelled"
    updated = cursor.rowcount > 0
    if updated:
        _log_ticket_status(conn, booking_id, new_status, changed_by or "client")
    conn.commit()
    conn.close()
    return updated


def find_tickets_by_email(
    client_email: str, ticket_status: str | None = None
) -> list[dict[str, Any]]:
    """
    Finds tickets by client email (case-insensitive), optionally filtered by
    ticket_status. Callers must pass a *verified* email — this does no
    ownership check of its own.
    """
    if not client_email.strip():
        return []
    conn = get_connection()
    if ticket_status:
        rows = conn.execute(
            f"SELECT {_BOOKING_COLUMNS} FROM pending_bookings "
            "WHERE lower(client_email) = lower(?) AND ticket_status = ? ORDER BY created_at DESC",
            (client_email.strip(), ticket_status),
        ).fetchall()
    else:
        rows = conn.execute(
            f"SELECT {_BOOKING_COLUMNS} FROM pending_bookings "
            "WHERE lower(client_email) = lower(?) ORDER BY created_at DESC",
            (client_email.strip(),),
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_booking(booking_id: int) -> dict[str, Any] | None:
    """Returns one ticket by id, or None."""
    conn = get_connection()
    row = conn.execute(
        f"SELECT {_BOOKING_COLUMNS} FROM pending_bookings WHERE id = ?", (booking_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def ticket_code(ticket_id: int) -> str:
    """Canonical ticket reference shown everywhere — UI, chat, emails, alerts."""
    return f"TKT-{ticket_id:05d}"


def get_all_tickets() -> list[dict[str, Any]]:
    """Returns all non-rejected bookings ordered by ticket priority for the manager board."""
    conn = get_connection()
    rows = conn.execute(
        f"""
        SELECT {_BOOKING_COLUMNS}
        FROM pending_bookings
        WHERE status != 'rejected'
        ORDER BY
            CASE ticket_status
                WHEN 'blocked'         THEN 1
                WHEN 'in_review'       THEN 2
                WHEN 'new'             THEN 3
                WHEN 'in_progress'     THEN 4
                WHEN 'awaiting_client' THEN 5
                WHEN 'open'            THEN 6
                WHEN 'closed'          THEN 7
                WHEN 'cancelled'       THEN 8
                ELSE 9
            END,
            date_str, time_str
        """
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def _log_ticket_status(
    conn, booking_id: int, ticket_status: str, changed_by: str, note: str = ""
) -> None:
    """
    Appends one row to the ticket audit trail. Takes an existing connection so
    the log write commits atomically with whatever status change triggered it
    — callers pass their own open `conn`, they must still call conn.commit().
    """
    conn.execute(
        "INSERT INTO ticket_log (booking_id, ticket_status, changed_by, note) VALUES (?, ?, ?, ?)",
        (booking_id, ticket_status, changed_by or "unknown", note),
    )


def get_ticket_log(booking_id: int) -> list[dict[str, Any]]:
    """Returns the full status-change history for a ticket, oldest first."""
    conn = get_connection()
    rows = conn.execute(
        "SELECT ticket_status, changed_by, note, created_at FROM ticket_log "
        "WHERE booking_id = ? ORDER BY created_at, id",
        (booking_id,),
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def update_ticket_coordinates(booking_id: int, client_lat: float, client_lon: float) -> None:
    """
    Persists geocoded coordinates for a ticket created before the ticket-map
    feature existed (client_lat/client_lon were null at request time) — used
    to backfill the map lazily instead of requiring a one-off migration script.
    """
    conn = get_connection()
    conn.execute(
        "UPDATE pending_bookings SET client_lat = ?, client_lon = ? WHERE id = ?",
        (client_lat, client_lon, booking_id),
    )
    conn.commit()
    conn.close()


_EDITABLE_TICKET_FIELDS = {
    "client_name",
    "client_email",
    "client_phone",
    "client_address",
    "date_str",
    "time_str",
    "quantity",
    "notes",
    "ticket_type",
}


def update_ticket_details(booking_id: int, changed_by: str, **fields) -> bool:
    """
    Lets the manager correct booking details after the fact (e.g. after a
    phone call with the client) — full name, contact info, address, schedule,
    quantity, or notes. Deliberately does NOT allow changing ticket_status,
    model, or price here: those already have dedicated tools/workflows
    (approve/reject, survey pricing, job completion) that carry their own
    business rules, and letting a free-form edit bypass them would break the
    ticket lifecycle's invariants.

    Only fields actually passed (non-None) are updated. Returns False if no
    editable field was given or the booking_id doesn't exist.
    """
    updates = {k: v for k, v in fields.items() if k in _EDITABLE_TICKET_FIELDS and v is not None}
    if not updates:
        return False

    conn = get_connection()
    set_clause = ", ".join(f"{col} = ?" for col in updates)
    cursor = conn.execute(
        f"UPDATE pending_bookings SET {set_clause} WHERE id = ?",
        (*updates.values(), booking_id),
    )
    updated = cursor.rowcount > 0
    if updated:
        row = conn.execute(
            "SELECT ticket_status FROM pending_bookings WHERE id = ?", (booking_id,)
        ).fetchone()
        note = "edited: " + ", ".join(sorted(updates))
        _log_ticket_status(conn, booking_id, row["ticket_status"], changed_by, note=note)
    conn.commit()
    conn.close()
    return updated


def get_booking_client_email(booking_id: int) -> str:
    """Returns the client_email stored on a booking, or empty string."""
    conn = get_connection()
    row = conn.execute(
        "SELECT client_email FROM pending_bookings WHERE id = ?",
        (booking_id,),
    ).fetchone()
    conn.close()
    return row["client_email"] if row else ""
