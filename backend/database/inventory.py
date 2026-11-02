"""Inventory lookups and installer reservation requests."""

from typing import Any

from backend.database.connection import _inserted_id, get_connection


def get_inventory() -> list[dict[str, Any]]:
    """
    Returns the full inventory table as a list of dictionaries.
    """

    conn = get_connection()
    cursor = conn.cursor()

    rows = cursor.execute(
        """
        SELECT id, model, category, stock, price, backorder_count, installation_hours
        FROM inventory
        ORDER BY model
        """
    ).fetchall()

    conn.close()

    return [dict(row) for row in rows]


def get_product(model: str) -> dict[str, Any] | None:
    """
    Returns one inventory item by exact model name.
    """

    conn = get_connection()
    cursor = conn.cursor()

    row = cursor.execute(
        """
        SELECT id, model, category, stock, price, backorder_count, installation_hours
        FROM inventory
        WHERE model = ?
        """,
        (model,),
    ).fetchone()

    conn.close()

    return dict(row) if row else None


def create_pending_reservation(
    model: str,
    quantity: int,
    requested_by: str,
    reason: str = "",
) -> int:
    """Inserts a pending reservation and returns its new id."""
    conn = get_connection()
    cursor = conn.execute(
        """
        INSERT INTO pending_reservations (requested_by, model, quantity, reason)
        VALUES (?, ?, ?, ?)
        """,
        (requested_by, model, quantity, reason),
    )
    reservation_id = _inserted_id(cursor)
    conn.commit()
    conn.close()
    return reservation_id


def get_pending_reservations(
    status: str = "pending",
    requested_by: str | None = None,
) -> list[dict[str, Any]]:
    conn = get_connection()
    if requested_by:
        rows = conn.execute(
            """
            SELECT id, requested_by, model, quantity, reason,
                   status, created_at, resolved_at, resolved_by
            FROM pending_reservations
            WHERE status = ? AND requested_by = ?
            ORDER BY created_at DESC
            """,
            (status, requested_by),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT id, requested_by, model, quantity, reason,
                   status, created_at, resolved_at, resolved_by
            FROM pending_reservations
            WHERE status = ?
            ORDER BY created_at DESC
            """,
            (status,),
        ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def resolve_reservation(reservation_id: int, status: str, resolved_by: str) -> bool:
    conn = get_connection()
    cursor = conn.execute(
        """
        UPDATE pending_reservations
        SET status = ?, resolved_at = datetime('now'), resolved_by = ?
        WHERE id = ? AND status = 'pending'
        """,
        (status, resolved_by, reservation_id),
    )
    updated = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return updated


def reopen_reservation(reservation_id: int) -> bool:
    """
    Puts an approved reservation back to 'pending' — used when approval
    claimed the row but the stock deduction then failed.
    """
    conn = get_connection()
    cursor = conn.execute(
        """
        UPDATE pending_reservations
        SET status = 'pending', resolved_at = NULL, resolved_by = NULL
        WHERE id = ? AND status = 'approved'
        """,
        (reservation_id,),
    )
    updated = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return updated
