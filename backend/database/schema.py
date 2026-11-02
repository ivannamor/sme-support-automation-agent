"""Creates and seeds the database schema."""

from backend.database import connection
from backend.database.connection import get_connection
from backend.database.seed import build_inventory_seed_items


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

    # Failed staff logins per username (backend/auth.py). Never holds passwords.
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS login_attempts (
            username TEXT PRIMARY KEY,
            failures INTEGER NOT NULL,
            last_failure REAL NOT NULL
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

    # Archive of assistant conversations, one row per turn — a record for
    # people (never fed back to the LLM). Holds personal data, so rows older
    # than CONVERSATION_RETENTION_DAYS are deleted (prune_conversation_log).
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS conversation_log (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            thread_id         TEXT NOT NULL,
            role              TEXT NOT NULL,
            user_identity     TEXT NOT NULL,
            user_message      TEXT NOT NULL DEFAULT '',
            assistant_message TEXT NOT NULL DEFAULT '',
            tools_used        TEXT NOT NULL DEFAULT '[]',
            model             TEXT NOT NULL DEFAULT '',
            hitl_decision     TEXT,
            created_at        TEXT NOT NULL DEFAULT (datetime('now'))
        )
        """
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_conversation_log_created_at "
        "ON conversation_log (created_at)"
    )
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_conversation_log_thread_id ON conversation_log (thread_id)"
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

    print(f"Database initialized: {connection.DB_PATH}")
    print(f"Inventory products available: {len(items)}")
