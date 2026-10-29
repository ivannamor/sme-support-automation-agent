import datetime
import json
import sqlite3
from pathlib import Path
from typing import Any

from backend.company_config import COMPANY_CONTEXT

# ============================================================
# Paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_DIR = PROJECT_ROOT / "db"
DB_PATH = DB_DIR / "company.db"

# This is the reviewed section map used by the catalogue RAG ingestion.
CATALOGUE_SECTIONS_PATH = PROJECT_ROOT / "sections" / "catalogue" / "catalogue_sections.json"


# ============================================================
# Product loading
# ============================================================


def load_catalogue_products(
    sections_path: Path = CATALOGUE_SECTIONS_PATH,
) -> list[str]:
    """
    Loads all product_name values from the reviewed catalogue sections JSON.

    Only sections with doc_type == "product" are used.
    This keeps the SQL inventory aligned with the catalogue RAG metadata.
    """

    if not sections_path.exists():
        raise FileNotFoundError(
            f"Catalogue sections file not found: {sections_path}\n"
            "Run/fix the catalogue ingestion first, or check the sections path."
        )

    with open(sections_path, encoding="utf-8") as file:
        sections = json.load(file)

    products: list[str] = []

    for section in sections:
        if section.get("doc_type") != "product":
            continue

        product_name = section.get("product_name")

        if not product_name:
            continue

        product_name = str(product_name).strip()

        if product_name.lower() == "null":
            continue

        products.append(product_name)

    # Remove duplicates while preserving order.
    return list(dict.fromkeys(products))


def infer_category(model: str) -> str:
    """
    Assigns a simple inventory category from the product name.
    This is mock business data, but stable enough for testing tools.
    """

    model_upper = model.upper()

    if model_upper.startswith(tuple(COMPANY_CONTEXT["product_code_prefixes"])):
        return "product"

    if model_upper.startswith("ABC"):
        return "product"

    if model_upper.startswith("XYZ"):
        return "accessory"

    return "unknown"


def generate_mock_price(model: str) -> float:
    """
    Generates deterministic mock prices.
    Same model -> same price every run.
    """

    model_upper = model.upper()

    if model_upper.startswith("ABC100"):
        base_price = 280
    elif model_upper.startswith("ABC200"):
        base_price = 240
    elif model_upper.startswith("ABC300"):
        base_price = 190
    elif model_upper.startswith(tuple(COMPANY_CONTEXT["product_code_prefixes"])):
        base_price = 140
    elif model_upper.startswith("XYZ"):
        base_price = 45
    else:
        base_price = 100

    variation = sum(ord(char) for char in model) % 90
    return round(base_price + variation, 2)


def generate_mock_stock(model: str) -> int:
    """
    Generates deterministic mock stock.
    Same model -> same stock every run.
    """

    return sum(ord(char) for char in model) % 18


def build_inventory_seed_items() -> list[tuple]:
    """
    Builds seed rows for the inventory table from catalogue products.
    """

    products = load_catalogue_products()

    items = []

    for product in products:
        items.append(
            (
                product,
                infer_category(product),
                generate_mock_stock(product),
                generate_mock_price(product),
                0,
            )
        )

    return items


# ============================================================
# Database initialization
# ============================================================


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


def init_db(reset: bool = False) -> None:
    """
    Creates and seeds the door_company SQLite database.

    Parameters
    ----------
    reset:
        If True, drops the inventory table and recreates it.
        Use only when you want to reset test data.

    Default behavior preserves existing stock changes made by the assistant.
    New catalogue products are inserted if missing.
    """

    conn = get_connection()
    cursor = conn.cursor()

    if reset:
        cursor.execute("DROP TABLE IF EXISTS inventory")

    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS inventory (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model TEXT UNIQUE NOT NULL,
            category TEXT NOT NULL,
            stock INTEGER NOT NULL DEFAULT 0,
            price REAL NOT NULL DEFAULT 0.0,
            backorder_count INTEGER NOT NULL DEFAULT 0
        )
        """
    )

    # Installer/manager accounts. Clients have no account (role = 'client' by convention).
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('installer', 'manager'))
        )
        """
    )

    # Inventory reservation requests submitted by installers, pending manager approval.
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS pending_reservations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            requested_by TEXT NOT NULL,
            model TEXT NOT NULL,
            quantity INTEGER NOT NULL DEFAULT 1,
            reason TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'pending'
                CHECK(status IN ('pending', 'approved', 'rejected')),
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            resolved_at TEXT,
            resolved_by TEXT
        )
        """
    )

    # Shared app configuration (e.g. the active LLM model chosen by the manager).
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS app_config (
            key   TEXT PRIMARY KEY,
            value TEXT NOT NULL
        )
        """
    )

    # Persistent usage log — all roles, all sessions.
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS usage_log (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            role           TEXT NOT NULL DEFAULT '',
            model          TEXT NOT NULL DEFAULT '',
            input_tokens   INTEGER NOT NULL DEFAULT 0,
            output_tokens  INTEGER NOT NULL DEFAULT 0,
            cost_usd       REAL NOT NULL DEFAULT 0.0,
            created_at     TEXT NOT NULL DEFAULT (datetime('now'))
        )
        """
    )

    # Q&A pairs awaiting manager approval before being written to the semantic cache.
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS qa_pending (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            question TEXT NOT NULL,
            answer TEXT NOT NULL,
            source_role TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            status TEXT NOT NULL DEFAULT 'pending'
                CHECK(status IN ('pending', 'approved', 'rejected'))
        )
        """
    )

    # Cost variables used by the price estimate tool.
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS installation_costs (
            key TEXT PRIMARY KEY,
            value REAL NOT NULL,
            description TEXT NOT NULL DEFAULT ''
        )
        """
    )
    cursor.executemany(
        "INSERT OR IGNORE INTO installation_costs (key, value, description) VALUES (?, ?, ?)",
        [
            ("hourly_rate", 45.00, "Installer hourly labour rate (€/hr)"),
            ("travel_cost_per_km", 0.30, "Vehicle travel cost (€/km)"),
            ("generic_install_cost", 50.00, "Fixed overhead per installation job (€)"),
            ("vat_rate", 0.22, "VAT rate applied to the subtotal (e.g. 0.22 = 22%)"),
            ("glass_price_standard", 90.00, "Standard low-E double glazing (€/m²)"),
            ("glass_price_triple", 140.00, "Triple glazing (€/m²)"),
            ("glass_price_security", 220.00, "Security/anti-break-in laminated glazing (€/m²)"),
            ("survey_hours_per_unit", 0.5, "Site survey time per window/door unit (hours)"),
        ],
    )

    # Client directory, populated when a site survey is requested. Keyed on
    # (full_name, email) together, not email alone — a shared inbox (or, in
    # practice, the same tester re-running requests under different names
    # with one email) must not silently overwrite an earlier, differently
    # named client's row.
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS clients (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name TEXT NOT NULL DEFAULT '',
            email TEXT NOT NULL DEFAULT '',
            phone TEXT NOT NULL DEFAULT '',
            address TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at TEXT NOT NULL DEFAULT (datetime('now')),
            UNIQUE(full_name, email)
        )
        """
    )

    # Migration: older databases enforced UNIQUE(email) alone. Rebuild onto
    # the composite key above, preserving every existing row.
    existing_clients_sql = cursor.execute(
        "SELECT sql FROM sqlite_master WHERE type='table' AND name='clients'"
    ).fetchone()
    if existing_clients_sql and "email TEXT UNIQUE" in existing_clients_sql[0]:
        cursor.execute("ALTER TABLE clients RENAME TO clients_old_email_unique")
        cursor.execute(
            """
            CREATE TABLE clients (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                full_name TEXT NOT NULL DEFAULT '',
                email TEXT NOT NULL DEFAULT '',
                phone TEXT NOT NULL DEFAULT '',
                address TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now')),
                UNIQUE(full_name, email)
            )
            """
        )
        cursor.execute(
            """
            INSERT INTO clients (id, full_name, email, phone, address, created_at, updated_at)
            SELECT id, full_name, email, phone, address, created_at, updated_at FROM clients_old_email_unique
            """
        )
        cursor.execute("DROP TABLE clients_old_email_unique")

    # Booking requests submitted by clients, pending manager approval.
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS pending_bookings (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            requested_by TEXT NOT NULL DEFAULT 'client',
            model TEXT NOT NULL,
            date_str TEXT NOT NULL,
            time_str TEXT NOT NULL,
            notes TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'pending'
                CHECK(status IN ('pending', 'approved', 'rejected')),
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            resolved_at TEXT,
            resolved_by TEXT
        )
        """
    )

    # Structured alerts sent from installers to the manager.
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS installer_alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sent_by TEXT NOT NULL,
            raw_message TEXT NOT NULL,
            structured_message TEXT NOT NULL,
            job_reference TEXT NOT NULL DEFAULT '',
            is_read INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
        """
    )

    # One-time email verification codes for client sessions (backend/client_auth.py).
    # One active code per email; only a hash of the code is stored.
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS client_verification_codes (
            email TEXT PRIMARY KEY,
            code_hash TEXT NOT NULL,
            expires_at REAL NOT NULL,
            attempts INTEGER NOT NULL DEFAULT 0,
            created_at REAL NOT NULL
        )
        """
    )

    # Job completion reports submitted by installers, pending manager approval.
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS job_completions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            sent_by TEXT NOT NULL,
            booking_id INTEGER,
            structured_report TEXT NOT NULL,
            raw_notes TEXT NOT NULL DEFAULT '',
            client_email TEXT NOT NULL DEFAULT '',
            status TEXT NOT NULL DEFAULT 'pending'
                CHECK(status IN ('pending', 'approved', 'rejected')),
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            resolved_at TEXT,
            resolved_by TEXT
        )
        """
    )

    # Audit trail of ticket_status transitions — who changed a ticket to what,
    # and when. Populated by every function that mutates ticket_status.
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS ticket_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            booking_id INTEGER NOT NULL,
            ticket_status TEXT NOT NULL,
            changed_by TEXT NOT NULL DEFAULT '',
            note TEXT NOT NULL DEFAULT '',
            created_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
        """
    )

    # Add installation_hours to inventory if it doesn't exist yet.
    existing_cols = [row[1] for row in cursor.execute("PRAGMA table_info(inventory)").fetchall()]
    if "installation_hours" not in existing_cols:
        cursor.execute(
            "ALTER TABLE inventory ADD COLUMN installation_hours REAL NOT NULL DEFAULT 1.5"
        )

    # Add columns to pending_bookings if they don't exist yet.
    booking_cols = [
        row[1] for row in cursor.execute("PRAGMA table_info(pending_bookings)").fetchall()
    ]
    if "client_email" not in booking_cols:
        cursor.execute(
            "ALTER TABLE pending_bookings ADD COLUMN client_email TEXT NOT NULL DEFAULT ''"
        )
    if "client_phone" not in booking_cols:
        cursor.execute(
            "ALTER TABLE pending_bookings ADD COLUMN client_phone TEXT NOT NULL DEFAULT ''"
        )
    if "quantity" not in booking_cols:
        cursor.execute(
            "ALTER TABLE pending_bookings ADD COLUMN quantity INTEGER NOT NULL DEFAULT 1"
        )
    if "assigned_to" not in booking_cols:
        cursor.execute("ALTER TABLE pending_bookings ADD COLUMN assigned_to TEXT")
    if "started_at" not in booking_cols:
        cursor.execute("ALTER TABLE pending_bookings ADD COLUMN started_at TEXT")
    if "ticket_status" not in booking_cols:
        cursor.execute("ALTER TABLE pending_bookings ADD COLUMN ticket_status TEXT DEFAULT 'new'")
    if "client_name" not in booking_cols:
        cursor.execute("ALTER TABLE pending_bookings ADD COLUMN client_name TEXT DEFAULT ''")
    if "client_address" not in booking_cols:
        cursor.execute("ALTER TABLE pending_bookings ADD COLUMN client_address TEXT DEFAULT ''")
    if "rejection_reason" not in booking_cols:
        cursor.execute("ALTER TABLE pending_bookings ADD COLUMN rejection_reason TEXT DEFAULT ''")

    # Site-survey → priced-estimate → installation workflow columns.
    # ticket_phase defaults to 'installation' (not 'survey') so every ticket
    # created before this migration — which all went straight to installation
    # under the old flow — keeps its correct historical phase.
    if "ticket_phase" not in booking_cols:
        cursor.execute(
            "ALTER TABLE pending_bookings ADD COLUMN ticket_phase TEXT NOT NULL DEFAULT 'installation'"
        )
    if "client_id" not in booking_cols:
        cursor.execute("ALTER TABLE pending_bookings ADD COLUMN client_id INTEGER")
    if "height_cm" not in booking_cols:
        cursor.execute("ALTER TABLE pending_bookings ADD COLUMN height_cm REAL")
    if "width_cm" not in booking_cols:
        cursor.execute("ALTER TABLE pending_bookings ADD COLUMN width_cm REAL")
    if "depth_cm" not in booking_cols:
        cursor.execute("ALTER TABLE pending_bookings ADD COLUMN depth_cm REAL")
    if "glass_type" not in booking_cols:
        cursor.execute("ALTER TABLE pending_bookings ADD COLUMN glass_type TEXT")
    if "estimated_price" not in booking_cols:
        cursor.execute("ALTER TABLE pending_bookings ADD COLUMN estimated_price REAL")
    if "client_estimate_response" not in booking_cols:
        cursor.execute("ALTER TABLE pending_bookings ADD COLUMN client_estimate_response TEXT")
    if "survey_notes" not in booking_cols:
        cursor.execute("ALTER TABLE pending_bookings ADD COLUMN survey_notes TEXT DEFAULT ''")
    if "survey_date_str" not in booking_cols:
        cursor.execute("ALTER TABLE pending_bookings ADD COLUMN survey_date_str TEXT")
    if "survey_time_str" not in booking_cols:
        cursor.execute("ALTER TABLE pending_bookings ADD COLUMN survey_time_str TEXT")
    if "estimate_resolved_at" not in booking_cols:
        cursor.execute("ALTER TABLE pending_bookings ADD COLUMN estimate_resolved_at TEXT")
    if "estimate_resolved_by" not in booking_cols:
        cursor.execute("ALTER TABLE pending_bookings ADD COLUMN estimate_resolved_by TEXT")
    # Geocoded once at survey-request time (see request_site_survey_tool) so the
    # ticket map doesn't need to re-geocode the address on every dashboard render.
    if "client_lat" not in booking_cols:
        cursor.execute("ALTER TABLE pending_bookings ADD COLUMN client_lat REAL")
    if "client_lon" not in booking_cols:
        cursor.execute("ALTER TABLE pending_bookings ADD COLUMN client_lon REAL")
    # Product type — defaults to 'window' since every model in the catalogue
    # today is a window (see inventory table); ready for 'door' once real
    # door products are onboarded. Set at request time from what the client
    # actually asked for (see request_site_survey_tool), editable afterwards
    # by the manager like any other ticket field.
    if "ticket_type" not in booking_cols:
        cursor.execute(
            "ALTER TABLE pending_bookings ADD COLUMN ticket_type TEXT NOT NULL DEFAULT 'window' "
            "CHECK(ticket_type IN ('door', 'window'))"
        )

    # Repair/backfill: restore any (client_name, client_email) pair seen on a
    # real ticket but missing from the clients directory — e.g. a client
    # whose row was overwritten by the old email-only-unique constraint
    # migrated away from above. Each ticket keeps its own client_name/email
    # snapshot from request time, so this data is still intact in
    # pending_bookings even though the directory lost it. Picks each pair's
    # most recent address. Runs on every startup — cheap, and INSERT OR
    # IGNORE against the (full_name, email) unique constraint makes it a
    # no-op once the directory is already complete.
    cursor.execute(
        """
        INSERT OR IGNORE INTO clients (full_name, email, phone, address)
        SELECT client_name, client_email, client_phone, client_address
        FROM (
            SELECT client_name, client_email, client_phone, client_address,
                   ROW_NUMBER() OVER (
                       PARTITION BY client_name, client_email
                       ORDER BY created_at DESC
                   ) AS rn
            FROM pending_bookings
            WHERE client_name != '' AND client_email != ''
        )
        WHERE rn = 1
        """
    )

    # Add booking_id to installer_alerts so alerts can be linked to a ticket.
    alert_cols = [
        row[1] for row in cursor.execute("PRAGMA table_info(installer_alerts)").fetchall()
    ]
    if "booking_id" not in alert_cols:
        cursor.execute("ALTER TABLE installer_alerts ADD COLUMN booking_id INTEGER")

    items = build_inventory_seed_items()

    # INSERT OR IGNORE preserves runtime changes to stock/price/backorders.
    cursor.executemany(
        """
        INSERT OR IGNORE INTO inventory
            (model, category, stock, price, backorder_count)
        VALUES (?, ?, ?, ?, ?)
        """,
        items,
    )

    conn.commit()
    conn.close()

    print(f"Database initialized: {DB_PATH}")
    print(f"Inventory products available: {len(items)}")


# ============================================================
# Inventory helpers for tools.py / assistant.py
# ============================================================


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


# ============================================================
# App config helpers (shared settings across roles/sessions)
# ============================================================


def get_config(key: str, default: str = "") -> str:
    conn = get_connection()
    row = conn.execute("SELECT value FROM app_config WHERE key = ?", (key,)).fetchone()
    conn.close()
    return row["value"] if row else default


def set_config(key: str, value: str) -> None:
    conn = get_connection()
    conn.execute(
        "INSERT INTO app_config (key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )
    conn.commit()
    conn.close()


# ============================================================
# Usage log helpers
# ============================================================


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


# ============================================================
# Job completion helpers
# ============================================================


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


def get_booking_client_email(booking_id: int) -> str:
    """Returns the client_email stored on a booking, or empty string."""
    conn = get_connection()
    row = conn.execute(
        "SELECT client_email FROM pending_bookings WHERE id = ?",
        (booking_id,),
    ).fetchone()
    conn.close()
    return row["client_email"] if row else ""


# ============================================================
# User helpers
# ============================================================


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


# ============================================================
# Client directory helpers
# ============================================================


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


# ============================================================
# Pending booking helpers
# ============================================================


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


# ============================================================
# Site-survey → priced-estimate → installation lifecycle helpers
# ============================================================


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


# ============================================================
# Installation cost helpers
# ============================================================

VALID_COST_KEYS = {
    "hourly_rate",
    "travel_cost_per_km",
    "generic_install_cost",
    "vat_rate",
    "glass_price_standard",
    "glass_price_triple",
    "glass_price_security",
    "survey_hours_per_unit",
}

# The 3 glass types installers can record during a site survey, and their
# corresponding installation_costs key for €/m² pricing.
GLASS_TYPES = {
    "standard": "glass_price_standard",
    "triple": "glass_price_triple",
    "security": "glass_price_security",
}


def get_costs() -> dict[str, float]:
    """Returns all cost variables as a key→value dict."""
    conn = get_connection()
    rows = conn.execute("SELECT key, value FROM installation_costs").fetchall()
    conn.close()
    return {row["key"]: row["value"] for row in rows}


def update_cost(key: str, value: float) -> bool:
    """Updates one cost variable. Returns False if the key is unknown."""
    if key not in VALID_COST_KEYS:
        return False
    conn = get_connection()
    conn.execute(
        "UPDATE installation_costs SET value = ? WHERE key = ?",
        (value, key),
    )
    conn.commit()
    conn.close()
    return True


# ============================================================
# Installer alert helpers
# ============================================================


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


# ============================================================
# Pending reservation helpers
# ============================================================


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


# ============================================================
# Main execution
# ============================================================

if __name__ == "__main__":
    init_db(reset=False)

    print("\nSample inventory rows:")
    for item in get_inventory()[:10]:
        print(item)
