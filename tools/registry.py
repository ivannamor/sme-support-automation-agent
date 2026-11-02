"""The list of every tool."""

from tools.client import (
    find_my_tickets_by_email_tool,
    request_site_survey_tool,
    respond_to_estimate_tool,
)
from tools.emails import send_estimate_to_client_tool
from tools.installer import (
    claim_installation_tool,
    complete_job_tool,
    list_my_installations_tool,
    release_installation_tool,
    send_alert_to_manager_tool,
    submit_survey_measurements_tool,
)
from tools.inventory import (
    approve_reservation_tool,
    inventory_api_tool,
    list_pending_reservations_tool,
    reject_reservation_tool,
    request_inventory_reservation_tool,
)
from tools.knowledge import (
    general_door_knowledge_tool,
    search_installation_manuals_tool,
    search_technical_manuals,
)
from tools.manager import (
    approve_booking_tool,
    approve_job_tool,
    approve_qa_tool,
    approve_survey_estimate_tool,
    list_pending_bookings_tool,
    list_pending_jobs_tool,
    list_pending_qa_tool,
    list_pending_survey_reports_tool,
    reject_booking_tool,
    reject_job_tool,
    reject_qa_tool,
    reject_survey_estimate_tool,
)
from tools.pricing import price_estimate_tool, update_installation_costs_tool
from tools.scheduling import book_installation_tool, check_calendar_availability_tool
from tools.tickets import edit_ticket_tool, get_my_ticket_status_tool, get_ticket_status_tool
from tools.travel import calculate_travel_time_tool

TOOLS = [
    search_technical_manuals,
    search_installation_manuals_tool,
    inventory_api_tool,
    check_calendar_availability_tool,
    book_installation_tool,
    calculate_travel_time_tool,
    request_site_survey_tool,
    respond_to_estimate_tool,
    find_my_tickets_by_email_tool,
    request_inventory_reservation_tool,
    list_pending_reservations_tool,
    approve_reservation_tool,
    reject_reservation_tool,
    send_alert_to_manager_tool,
    price_estimate_tool,
    update_installation_costs_tool,
    complete_job_tool,
    submit_survey_measurements_tool,
    send_estimate_to_client_tool,
    general_door_knowledge_tool,
    # Installer assignment tools
    list_my_installations_tool,
    claim_installation_tool,
    release_installation_tool,
    get_my_ticket_status_tool,
    # Manager ticket query
    get_ticket_status_tool,
    edit_ticket_tool,
    # Manager approval tools
    list_pending_bookings_tool,
    list_pending_qa_tool,
    list_pending_jobs_tool,
    list_pending_survey_reports_tool,
    approve_booking_tool,
    reject_booking_tool,
    approve_qa_tool,
    reject_qa_tool,
    approve_job_tool,
    reject_job_tool,
    approve_survey_estimate_tool,
    reject_survey_estimate_tool,
]
