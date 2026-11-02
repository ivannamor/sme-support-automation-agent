"""Shared app settings and installation costs."""

from backend.database.connection import get_connection


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
