"""Usage log, conversation archive, analytics and the operational data reset."""

import datetime
import json

from backend.database.connection import get_connection
from backend.secrets_provider import get_secret

DEFAULT_CONVERSATION_RETENTION_DAYS = 365


def log_usage(
    role: str,
    model: str,
    input_tokens: int,
    output_tokens: int,
    cost_usd: float,
) -> None:
    conn = get_connection()
    conn.execute(
        """
        INSERT INTO usage_log (role, model, input_tokens, output_tokens, cost_usd)
        VALUES (?, ?, ?, ?, ?)
        """,
        (role, model, input_tokens, output_tokens, cost_usd),
    )
    conn.commit()
    conn.close()


def log_conversation_turn(
    thread_id: str,
    role: str,
    user_identity: str,
    user_message: str,
    assistant_message: str,
    tools_used: list[str],
    model: str,
    hitl_decision: str | None = None,
) -> None:
    """
    Archives one assistant turn. `user_message` is empty for a HITL resume
    turn, where `hitl_decision` ('confirmed' / 'rejected') says what the user
    chose instead.
    """
    conn = get_connection()
    conn.execute(
        """
        INSERT INTO conversation_log
            (thread_id, role, user_identity, user_message, assistant_message,
             tools_used, model, hitl_decision)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            thread_id,
            role,
            user_identity,
            user_message,
            assistant_message,
            json.dumps(tools_used),
            model,
            hitl_decision,
        ),
    )
    conn.commit()
    conn.close()


def prune_conversation_log(retention_days: int | None = None) -> int:
    """
    Deletes archived turns older than `retention_days` (default:
    CONVERSATION_RETENTION_DAYS, or 365). Returns how many rows were deleted.
    """
    if retention_days is None:
        retention_days = int(
            get_secret("CONVERSATION_RETENTION_DAYS", str(DEFAULT_CONVERSATION_RETENTION_DAYS))
        )
    conn = get_connection()
    cursor = conn.execute(
        "DELETE FROM conversation_log WHERE created_at < datetime('now', ?)",
        (f"-{retention_days} days",),
    )
    conn.commit()
    conn.close()
    return cursor.rowcount


def get_usage_stats(period: str = "day") -> dict:
    """
    Returns aggregated usage for the given period ('day', 'week', 'month').
    """
    if period == "day":
        since = "datetime('now', 'start of day')"
    elif period == "week":
        since = "datetime('now', '-6 days', 'start of day')"
    else:
        since = "datetime('now', 'start of month')"

    conn = get_connection()
    rows = conn.execute(
        f"""
        SELECT role,
               SUM(input_tokens)  AS input_tokens,
               SUM(output_tokens) AS output_tokens,
               SUM(cost_usd)      AS cost_usd,
               COUNT(*)           AS requests
        FROM usage_log
        WHERE created_at >= {since}
        GROUP BY role
        """,
    ).fetchall()
    conn.close()

    by_role = {r["role"]: dict(r) for r in rows}
    totals = {
        "input_tokens": sum(r["input_tokens"] for r in by_role.values()),
        "output_tokens": sum(r["output_tokens"] for r in by_role.values()),
        "cost_usd": sum(r["cost_usd"] for r in by_role.values()),
        "requests": sum(r["requests"] for r in by_role.values()),
    }
    return {"totals": totals, "by_role": by_role}


def get_business_analytics(period: str = "month") -> dict:
    """
    Business metrics (as opposed to get_usage_stats' LLM cost/token metrics)
    derived from tickets that reached 'closed' status within the period.

    pending_bookings has no closed_at column, so ticket_log's 'closed' rows
    are the actual source of truth for when a ticket closed — joined with
    pending_bookings for revenue (estimated_price) and inventory for labor
    hours (quantity × installation_hours).

    "week"/"month" are rolling windows (7 and 30 days including today), not
    calendar-week/calendar-month-to-date, so the chart always has a full,
    consistent number of days to plot rather than shrinking to almost
    nothing right after a calendar boundary.
    """
    if period == "day":
        window_days = 1
    elif period == "week":
        window_days = 7
    else:
        window_days = 30

    since = f"datetime('now', '-{window_days - 1} days', 'start of day')"

    conn = get_connection()
    rows = conn.execute(
        f"""
        SELECT tl.created_at AS closed_at, pb.quantity, pb.estimated_price,
               inv.installation_hours
        FROM ticket_log tl
        JOIN pending_bookings pb ON pb.id = tl.booking_id
        LEFT JOIN inventory inv ON inv.model = pb.model
        WHERE tl.ticket_status = 'closed' AND tl.created_at >= {since}
        """
    ).fetchall()
    conn.close()

    closed_count = len(rows)
    total_revenue = sum(r["estimated_price"] or 0 for r in rows)
    total_hours = sum((r["quantity"] or 1) * (r["installation_hours"] or 0) for r in rows)

    by_day: dict[str, dict[str, float]] = {}
    for r in rows:
        day = r["closed_at"][:10]
        entry = by_day.setdefault(day, {"closed": 0, "revenue": 0.0})
        entry["closed"] += 1
        entry["revenue"] += r["estimated_price"] or 0

    # Backfill every day in the window with zeros so the chart always shows
    # a full, continuous 1/7/30-day axis instead of only the sparse days
    # that happened to have a closed ticket.
    today = datetime.date.today()
    for offset in range(window_days):
        day = (today - datetime.timedelta(days=offset)).isoformat()
        by_day.setdefault(day, {"closed": 0, "revenue": 0.0})

    return {
        "closed_count": closed_count,
        "total_revenue": total_revenue,
        "total_hours": total_hours,
        "by_day": dict(sorted(by_day.items())),
    }


def reset_operational_data() -> None:
    """
    Hard-deletes all operational data: tickets, alerts, job completions,
    and pending QA entries. Inventory, users, and costs are preserved.
    """
    conn = get_connection()
    conn.execute("DELETE FROM pending_bookings")
    conn.execute("DELETE FROM pending_reservations")
    conn.execute("DELETE FROM installer_alerts")
    conn.execute("DELETE FROM job_completions")
    conn.execute("DELETE FROM qa_pending")
    conn.execute("DELETE FROM client_verification_codes")
    conn.commit()
    conn.close()
