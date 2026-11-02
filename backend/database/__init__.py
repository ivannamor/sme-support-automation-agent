"""
The operational SQLite database. Import from here: `from backend.database import get_product`.

DB_PATH is deliberately not re-exported. Tests and the eval sandbox replace it on
backend.database.connection; a stale patch on this package must fail, or it would
silently reach the real database.
"""

from backend.database.activity import (
    DEFAULT_CONVERSATION_RETENTION_DAYS,
    get_business_analytics,
    get_usage_stats,
    log_conversation_turn,
    log_usage,
    prune_conversation_log,
    reset_operational_data,
)
from backend.database.alerts import (
    create_alert,
    get_alerts,
    get_alerts_for_booking,
    mark_alerts_read,
)
from backend.database.clients import get_clients, upsert_client
from backend.database.connection import get_connection
from backend.database.inventory import (
    create_pending_reservation,
    get_inventory,
    get_pending_reservations,
    get_product,
    reopen_reservation,
    resolve_reservation,
)
from backend.database.jobs import create_job_completion, get_job_completions, resolve_job_completion
from backend.database.schema import init_db
from backend.database.seed import (
    build_inventory_seed_items,
    generate_mock_price,
    generate_mock_stock,
    infer_category,
    load_catalogue_products,
)
from backend.database.settings import (
    GLASS_TYPES,
    VALID_COST_KEYS,
    get_config,
    get_costs,
    set_config,
    update_cost,
)
from backend.database.tickets import (
    claim_booking,
    create_pending_booking,
    decline_booking,
    find_tickets_by_email,
    get_all_tickets,
    get_approved_bookings,
    get_booking,
    get_booking_client_email,
    get_installer_active_booking,
    get_pending_bookings,
    get_ticket_log,
    reject_survey_report,
    release_booking,
    resolve_booking,
    resolve_client_estimate_response,
    resolve_survey_estimate,
    save_survey_measurements,
    ticket_code,
    update_ticket_coordinates,
    update_ticket_details,
)
from backend.database.users import (
    clear_login_attempts,
    create_user,
    get_user,
    register_login_attempt,
    update_user_password,
)

__all__ = [
    "DEFAULT_CONVERSATION_RETENTION_DAYS",
    "GLASS_TYPES",
    "VALID_COST_KEYS",
    "build_inventory_seed_items",
    "claim_booking",
    "clear_login_attempts",
    "create_alert",
    "create_job_completion",
    "create_pending_booking",
    "create_pending_reservation",
    "create_user",
    "decline_booking",
    "find_tickets_by_email",
    "generate_mock_price",
    "generate_mock_stock",
    "get_alerts",
    "get_alerts_for_booking",
    "get_all_tickets",
    "get_approved_bookings",
    "get_booking",
    "get_booking_client_email",
    "get_business_analytics",
    "get_clients",
    "get_config",
    "get_connection",
    "get_costs",
    "get_installer_active_booking",
    "get_inventory",
    "get_job_completions",
    "get_pending_bookings",
    "get_pending_reservations",
    "get_product",
    "get_ticket_log",
    "get_usage_stats",
    "get_user",
    "infer_category",
    "init_db",
    "load_catalogue_products",
    "log_conversation_turn",
    "log_usage",
    "mark_alerts_read",
    "prune_conversation_log",
    "register_login_attempt",
    "reject_survey_report",
    "release_booking",
    "reopen_reservation",
    "reset_operational_data",
    "resolve_booking",
    "resolve_client_estimate_response",
    "resolve_job_completion",
    "resolve_reservation",
    "resolve_survey_estimate",
    "save_survey_measurements",
    "set_config",
    "ticket_code",
    "update_cost",
    "update_ticket_coordinates",
    "update_ticket_details",
    "update_user_password",
    "upsert_client",
]
