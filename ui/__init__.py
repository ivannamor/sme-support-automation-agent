"""Shared Streamlit UI used across the client, installer and manager pages."""

from ui.booking_form import render_booking_form
from ui.briefing import render_installer_job_briefing
from ui.chat import (
    handle_user_turn,
    render_chat_autoscroll,
    render_chat_history,
    render_hitl_and_input,
)
from ui.client_portal import render_client_installation_status, render_client_pending_estimates
from ui.i18n import t
from ui.layout import inject_global_css, render_column_resize_handle, render_page_header
from ui.session import get_agent, init_chat_session, require_auth
from ui.sidebar import render_sidebar
from ui.ticket_map import render_ticket_map
from ui.tickets import render_stat_tiles, render_ticket_card, render_ticket_summary

__all__ = [
    "get_agent",
    "handle_user_turn",
    "init_chat_session",
    "inject_global_css",
    "render_booking_form",
    "render_chat_autoscroll",
    "render_chat_history",
    "render_client_installation_status",
    "render_client_pending_estimates",
    "render_column_resize_handle",
    "render_hitl_and_input",
    "render_installer_job_briefing",
    "render_page_header",
    "render_sidebar",
    "render_stat_tiles",
    "render_ticket_card",
    "render_ticket_map",
    "render_ticket_summary",
    "require_auth",
    "t",
]
