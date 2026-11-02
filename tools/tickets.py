"""Ticket lookups and corrections."""

import datetime
import re
from typing import Literal

from langchain.tools import tool
from pydantic import BaseModel, Field

from backend.database import get_all_tickets, ticket_code, update_ticket_details
from backend.log_context import username_var
from tools.common import logger


def _parse_ticket_reference(ref: str) -> int | None:
    """Extracts the numeric booking id from a reference like 'TKT-00007' or a bare '7'."""
    match = re.search(r"(\d+)", ref)
    return int(match.group(1)) if match else None


def _format_ticket_details(ticket: dict) -> str:
    lines = [
        f"{ticket_code(ticket['id'])} — {ticket.get('quantity', 1)}x {ticket['model']} "
        f"({ticket.get('ticket_type', 'window')})",
        f"Phase: {ticket.get('ticket_phase', 'installation')} | Status: {ticket.get('ticket_status', 'new')}",
        f"Client: {ticket.get('client_name') or ticket.get('requested_by', 'unknown')}",
    ]
    if ticket.get("client_email"):
        lines.append(f"Email: {ticket['client_email']}")
    if ticket.get("client_address"):
        lines.append(f"Address: {ticket['client_address']}")
    lines.append(f"Scheduled: {ticket['date_str']} {ticket['time_str']}")
    if ticket.get("assigned_to"):
        lines.append(f"Assigned installer: {ticket['assigned_to']}")
    if ticket.get("estimated_price"):
        lines.append(f"Estimated price: €{ticket['estimated_price']:.2f}")
    if ticket.get("notes"):
        lines.append(f"Notes: {ticket['notes']}")
    return "\n".join(lines)


def _summarize_tickets(
    tickets: list,
    *,
    ticket_reference: str | None,
    status_filter: str | None,
    model_filter: str | None = None,
) -> str:
    """
    Shared logic for looking up ticket status/details or counts, over whatever
    ticket list the caller has already scoped (all tickets for the manager,
    only assigned tickets for an installer).
    """
    scope_note = ""
    if model_filter:
        needle = model_filter.strip().upper()
        tickets = [t for t in tickets if needle in t.get("model", "").upper()]
        scope_note = f" for model '{model_filter}'"

    if ticket_reference:
        booking_id = _parse_ticket_reference(ticket_reference)
        match = next((t for t in tickets if t["id"] == booking_id), None)
        if not match:
            return f"No ticket found matching '{ticket_reference}'{scope_note} in this scope."
        return _format_ticket_details(match)

    if status_filter:
        key = status_filter.strip().lower().replace(" ", "_")
        matching = [t for t in tickets if t.get("ticket_status") == key]
        if not matching:
            return f"No tickets with status '{status_filter}'{scope_note}."
        lines = [f"{len(matching)} ticket(s) with status '{status_filter}'{scope_note}:"]
        for t in matching[:30]:
            lines.append(
                f"  {ticket_code(t['id'])} — {t.get('quantity', 1)}x {t['model']} — "
                f"{t.get('client_name') or t.get('requested_by', 'unknown')} — {t['date_str']}"
            )
        if len(matching) > 30:
            lines.append(f"  ... and {len(matching) - 30} more.")
        return "\n".join(lines)

    if not tickets:
        return f"No tickets{scope_note} in this scope."
    counts: dict = {}
    for t in tickets:
        key = t.get("ticket_status", "new")
        counts[key] = counts.get(key, 0) + 1
    lines = [f"Ticket counts by status{scope_note} ({len(tickets)} total):"]
    for status, count in sorted(counts.items(), key=lambda kv: -kv[1]):
        lines.append(f"  {status}: {count}")
    return "\n".join(lines)


class TicketQueryInput(BaseModel):
    ticket_reference: str | None = Field(
        default=None,
        description="Specific ticket code like 'TKT-00007', if asking about one particular ticket.",
    )
    status_filter: str | None = Field(
        default=None,
        description="Filter by ticket status (e.g. 'new', 'closed', 'in_progress'), if asking about a category/count.",
    )
    model_filter: str | None = Field(
        default=None,
        description="Filter by hardware model/product (e.g. 'IRIS'), if the question is about a specific product. "
        "Combine freely with status_filter, e.g. 'open IRIS tickets'.",
    )


@tool(args_schema=TicketQueryInput)
def get_ticket_status_tool(
    ticket_reference: str | None = None,
    status_filter: str | None = None,
    model_filter: str | None = None,
) -> str:
    """
    Looks up ticket status/details across ALL tickets in the system — use this
    for questions like "how many closed tickets are there?", "how many open
    IRIS tickets do I have?", or "what's the status of TKT-00007?". Manager-only.
    """
    return _summarize_tickets(
        get_all_tickets(),
        ticket_reference=ticket_reference,
        status_filter=status_filter,
        model_filter=model_filter,
    )


@tool(args_schema=TicketQueryInput)
def get_my_ticket_status_tool(
    ticket_reference: str | None = None,
    status_filter: str | None = None,
    model_filter: str | None = None,
) -> str:
    """
    Looks up ticket status/details, scoped ONLY to tickets assigned to this
    installer (any phase, any status, not just currently active ones) — use
    this for questions like "how many tickets do I have?" or "what's the
    status of TKT-00007?". Installer-only — never shows other installers' tickets.
    """
    installer = username_var.get()
    my_tickets = [t for t in get_all_tickets() if t.get("assigned_to") == installer]
    return _summarize_tickets(
        my_tickets,
        ticket_reference=ticket_reference,
        status_filter=status_filter,
        model_filter=model_filter,
    )


class EditTicketInput(BaseModel):
    ticket_reference: str = Field(description="Ticket code to edit, e.g. 'TKT-00007'.")
    client_name: str | None = Field(default=None, description="Corrected full name, if changing.")
    client_email: str | None = Field(default=None, description="Corrected email, if changing.")
    client_phone: str | None = Field(
        default=None, description="Corrected phone number, if changing."
    )
    client_address: str | None = Field(
        default=None, description="Corrected site address, if changing."
    )
    date_str: str | None = Field(
        default=None, description="Corrected date (YYYY-MM-DD), if rescheduling."
    )
    time_str: str | None = Field(
        default=None, description="Corrected time (HH:MM), if rescheduling."
    )
    quantity: int | None = Field(default=None, description="Corrected quantity, if changing.")
    notes: str | None = Field(default=None, description="Corrected/updated notes, if changing.")
    ticket_type: Literal["door", "window"] | None = Field(
        default=None, description="Corrected product type (door or window), if it was tagged wrong."
    )


@tool(args_schema=EditTicketInput)
def edit_ticket_tool(
    ticket_reference: str,
    client_name: str | None = None,
    client_email: str | None = None,
    client_phone: str | None = None,
    client_address: str | None = None,
    date_str: str | None = None,
    time_str: str | None = None,
    quantity: int | None = None,
    notes: str | None = None,
    ticket_type: str | None = None,
) -> str:
    """
    Corrects booking details on an existing ticket — e.g. after a phone call
    where the client gave a new phone number or asked to reschedule. Only
    pass the fields that are actually changing; everything else is left as-is.

    Does NOT change ticket status, hardware model, or price — those go through
    their own dedicated tools (approve/reject, survey pricing, job completion).
    Manager-only.
    """
    booking_id = _parse_ticket_reference(ticket_reference)
    if booking_id is None:
        return f"Could not parse a ticket ID from '{ticket_reference}'."

    if date_str:
        try:
            datetime.datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            return "Edit Ticket Error: date_str must be in YYYY-MM-DD format."
    if time_str:
        try:
            datetime.datetime.strptime(time_str, "%H:%M")
        except ValueError:
            return "Edit Ticket Error: time_str must be in HH:MM format."
    if quantity is not None and quantity < 1:
        return "Edit Ticket Error: quantity must be at least 1."

    updated = update_ticket_details(
        booking_id,
        username_var.get(),
        client_name=client_name,
        client_email=client_email,
        client_phone=client_phone,
        client_address=client_address,
        date_str=date_str,
        time_str=time_str,
        quantity=quantity,
        notes=notes,
        ticket_type=ticket_type,
    )
    if not updated:
        return f"Could not update {ticket_code(booking_id)} — it may not exist, or no fields were provided."
    logger.info(f"TICKET_EDITED: id={booking_id} by={username_var.get()}")
    return f"{ticket_code(booking_id)} updated successfully."
