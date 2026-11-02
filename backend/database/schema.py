"""Brings the database up to date, then refreshes the data it starts with."""

from backend.database import connection
from backend.database.connection import get_connection
from backend.database.migrate import run_migrations
from backend.database.seed import build_inventory_seed_items


def init_db(reset: bool = False) -> None:
    """
    Migrates the door_company SQLite database and seeds it.

    Parameters
    ----------
    reset:
        If True, empties the inventory table and reseeds it.
        Use only when you want to reset test data.

    Default behavior preserves existing stock changes made by the assistant.
    New catalogue products are inserted if missing.
    """

    run_migrations()

    conn = get_connection()
    cursor = conn.cursor()

    if reset:
        cursor.execute("DELETE FROM inventory")
        cursor.execute("DELETE FROM sqlite_sequence WHERE name = 'inventory'")

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
