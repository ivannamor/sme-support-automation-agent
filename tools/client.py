"""Client-facing tools: survey requests, their tickets and estimate responses."""

import datetime
from typing import Literal
from zoneinfo import ZoneInfo

from langchain.tools import tool
from pydantic import BaseModel, Field

from backend.database import (
    create_pending_booking,
    find_tickets_by_email,
    get_booking,
    get_inventory,
    get_product,
    resolve_client_estimate_response,
    ticket_code,
    upsert_client,
)
from backend.llm_provider import get_llm
from backend.log_context import username_var, verified_client_email_var
from tools.common import logger
from tools.scheduling import APPOINTMENT_DURATION_HOURS, book_installation_tool
from tools.travel import _compute_travel, _geocode_location, _standardize_address


class BookingFormExtraction(BaseModel):
    """Schema-only helper for parsing free-text/voice input into booking form fields."""

    client_name: str | None = Field(
        default=None, description="Full name of the client, if mentioned."
    )
    client_address: str | None = Field(
        default=None, description="Installation address (street and city), if mentioned."
    )
    model: str | None = Field(default=None, description="Product model name/code, if mentioned.")
    quantity: int | None = Field(default=None, description="Number of units, if mentioned.")
    date_str: str | None = Field(
        default=None, description="Requested date resolved to YYYY-MM-DD, if mentioned."
    )
    time_str: str | None = Field(
        default=None, description="Requested time in 24-hour HH:MM, if mentioned."
    )
    client_email: str | None = Field(
        default=None, description="Client email address, if mentioned."
    )
    client_phone: str | None = Field(default=None, description="Client phone number, if mentioned.")


def extract_booking_fields(raw_text: str) -> dict:
    """
    Parses a free-text (or transcribed voice) message into booking form fields
    using LLM structured output. Only fields actually mentioned are returned —
    the rest are omitted so the UI can leave them untouched for manual entry.

    The model list is passed in explicitly so the extraction is grounded in
    real catalogue codes (e.g. normalizing "Magis windows" to "MAGIS 40")
    instead of echoing back the client's own loose wording — without this,
    the caller's exact-match lookup against the real model list silently
    fails and the form falls back to whatever model happens to sort first
    alphabetically, which reads as the assistant "getting the model wrong".
    """
    today = datetime.date.today().strftime("%Y-%m-%d")
    model_options = sorted({item["model"] for item in get_inventory()})
    llm = get_llm(temperature=0, max_tokens=300).with_structured_output(BookingFormExtraction)

    result = llm.invoke(
        f"Current date: {today}\n"
        f"Valid product model codes: {', '.join(model_options)}\n\n"
        "Extract installation booking details from the free-text message below. "
        "Only fill fields that are actually mentioned or clearly implied; leave the rest null — "
        "do not guess or invent values. Resolve relative dates (e.g. 'domani', 'next Tuesday') "
        "against the current date above and always output dates as YYYY-MM-DD. If a product is "
        "mentioned, set 'model' to the matching code from the valid list above (not the client's "
        "own wording) — leave it null if no code in the list clearly matches.\n\n"
        f"Message: {raw_text}"
    )

    return {key: value for key, value in result.model_dump().items() if value not in (None, "")}


def compute_appointment_duration(
    quantity: int, hours_per_unit: float, travel_duration_hours: float
) -> float:
    """
    Shared appointment-duration formula for both survey and installation
    calendar events: time on the job scales with how many units there are,
    plus round-trip travel (the installer has to drive back too).
    """
    return quantity * hours_per_unit + 2 * travel_duration_hours


class SiteSurveyRequestInput(BaseModel):
    model: str = Field(
        description=(
            "A single specific hardware model code, e.g. 'IRIS' or 'MAGIS 40' — never a series/"
            "collection name ('Vero Legno', 'Legno-Alluminio', 'PVC'). If the client only named a "
            "series, ask which specific model in that series they want before calling this tool."
        )
    )
    quantity: int = Field(description="Number of windows/doors to survey.")
    date_str: str = Field(description="Requested site survey date in YYYY-MM-DD format.")
    time_str: str = Field(description="Requested site survey time in HH:MM format (24-hour).")
    client_name: str = Field(description="Full name of the client.")
    client_email: str = Field(
        description="Client email address — used to find this ticket later and send the estimate."
    )
    client_phone: str = Field(description="Client phone number for the installer.")
    client_address: str = Field(
        description="Full site address (street, city) where the survey will take place."
    )
    notes: str = Field(default="", description="Optional notes about the request.")
    product_type: Literal["door", "window"] = Field(
        default="window",
        description=(
            "Whether this request is for a door or a window. Infer this from what the client "
            "actually said (e.g. 'porta'/'portoncino'/'entrance door' means door, 'finestra'/"
            "'window' means window). Every model currently in the catalogue is a window, so "
            "default to 'window' unless the client's own wording clearly points to a door."
        ),
    )


@tool(args_schema=SiteSurveyRequestInput)
def request_site_survey_tool(
    model: str,
    quantity: int,
    date_str: str,
    time_str: str,
    client_name: str,
    client_email: str,
    client_phone: str,
    client_address: str,
    notes: str = "",
    product_type: str = "window",
) -> str:
    """
    Submits a site-survey request on behalf of a client for manager approval.

    Use this when a client wants to move forward with a product — this books
    a SURVEY visit, not an installation. No price estimate is given at this
    stage: a real quote can only be produced after an installer measures the
    windows/doors on site and picks a glass type. Do NOT use
    book_installation_tool or price_estimate_tool directly for client requests.
    """
    model = model.upper().strip()

    if quantity < 1:
        return "Site Survey Request Error: quantity must be at least 1."

    try:
        survey_dt = datetime.datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H:%M")
    except ValueError:
        return "Site Survey Request Error: invalid date or time format. Use YYYY-MM-DD and HH:MM."

    timezone = ZoneInfo("Europe/Rome")
    if survey_dt.replace(tzinfo=timezone) < datetime.datetime.now(timezone):
        return (
            f"Site Survey Request Error: {date_str} {time_str} is in the past. "
            "Please choose a future date and time for the survey visit."
        )

    # Standardize once, here, so the client directory and every ticket for
    # this address share one consistent form instead of accumulating as many
    # spellings as there are ways to type the same address. Best-effort: if
    # this fails, the client's own wording is kept rather than blocking the
    # request over it.
    client_address = _standardize_address(client_address)

    try:
        client_id = upsert_client(
            full_name=client_name,
            email=client_email,
            phone=client_phone,
            address=client_address,
        )

        # Geocode once, here, so the ticket map (rendered on every dashboard
        # load) never needs to re-geocode the address — Nominatim's rate
        # limit doesn't tolerate that. Best-effort: a geocoding hiccup must
        # not block the survey request itself, it just means no map pin.
        client_lat, client_lon = None, None
        try:
            client_lat, client_lon, _ = _geocode_location(client_address)
        except Exception as exc:
            logger.warning(f"SURVEY_GEOCODE_FAILED: {exc}")

        booking_id = create_pending_booking(
            model=model,
            date_str=date_str,
            time_str=time_str,
            requested_by=username_var.get(),
            notes=notes,
            client_email=client_email,
            client_phone=client_phone,
            client_name=client_name,
            client_address=client_address,
            quantity=quantity,
            ticket_phase="survey",
            client_id=client_id,
            client_lat=client_lat,
            client_lon=client_lon,
            ticket_type=product_type,
        )
        logger.info(f"SURVEY_REQUEST_CREATED: id={booking_id} model={model} {date_str} {time_str}")
        return (
            f"Your site survey request for **{quantity}× {model}** on **{date_str}** at **{time_str}** "
            f"has been submitted (reference **{ticket_code(booking_id)}**). "
            "A manager will review and confirm it shortly. After the survey, we'll send you a "
            "priced estimate based on the exact measurements."
        )
    except Exception as exc:
        logger.error(f"SURVEY_REQUEST_FAILED: {exc}", exc_info=True)
        return "Site Survey Request Error: could not save your request. Please try again."


# ── client response to a priced estimate ─────────────────────────────────────

# Ticket ownership comes from the session's verified email (one-time code,
# backend/client_auth.py) — never from an LLM-supplied argument, which the
# user can steer by prompting.
_CLIENT_NOT_VERIFIED = (
    "Client email not verified. To see or respond to their tickets, the client "
    "must first verify their email in the 'Verify your email' box inside the "
    "estimate or installation-status panel at the top of this page (it sends a "
    "one-time code). Do not ask them to type their email in the chat instead."
)


@tool
def find_my_tickets_by_email_tool() -> str:
    """
    Finds the client's own tickets that are awaiting their response to a
    priced estimate, using the email the client verified in this session.
    Use this when a client wants to accept/reject an estimate but doesn't have
    their ticket code handy.
    """
    client_email = verified_client_email_var.get().strip().lower()
    if not client_email:
        return _CLIENT_NOT_VERIFIED
    tickets = find_tickets_by_email(client_email, ticket_status="awaiting_client")
    if not tickets:
        return f"No tickets awaiting a response were found for {client_email}."
    lines = [
        f"{ticket_code(t['id'])} — {t.get('quantity', 1)}× {t['model']}, estimate: €{t.get('estimated_price', 0):.2f}"
        for t in tickets
    ]
    return "\n".join(lines)


class RespondToEstimateInput(BaseModel):
    booking_id: int = Field(description="ID of the ticket with the priced estimate to respond to.")
    response: str = Field(description="Either 'accept' or 'reject'.")
    new_date_str: str = Field(
        default="", description="Preferred installation date (YYYY-MM-DD) — required if accepting."
    )
    new_time_str: str = Field(
        default="", description="Preferred installation time (HH:MM) — required if accepting."
    )


@tool(args_schema=RespondToEstimateInput)
def respond_to_estimate_tool(
    booking_id: int,
    response: str,
    new_date_str: str = "",
    new_time_str: str = "",
) -> str:
    """
    Records the client's accept/reject decision on a priced estimate.

    On accept, also schedules the installation appointment — you MUST have
    collected a preferred installation date/time first. On reject, the ticket
    is cancelled.
    """
    response = response.lower().strip()
    if response not in ("accept", "reject"):
        return "Response Error: response must be 'accept' or 'reject'."

    client_email = verified_client_email_var.get().strip().lower()
    if not client_email:
        return _CLIENT_NOT_VERIFIED

    booking = get_booking(booking_id)
    # Someone else's ticket gets the same answer as a missing one, so ticket
    # ids can't be probed for existence.
    if (
        not booking
        or booking.get("status") != "approved"
        or booking.get("ticket_status") != "awaiting_client"
        or (booking.get("client_email") or "").strip().lower() != client_email
    ):
        return f"{ticket_code(booking_id)} is not currently awaiting a response from you."

    client_actor = client_email

    if response == "reject":
        resolve_client_estimate_response(
            booking_id,
            "reject",
            changed_by=client_actor,
            client_email=client_email,
        )
        logger.info(f"ESTIMATE_REJECTED_BY_CLIENT: id={booking_id}")
        return f"Estimate for {ticket_code(booking_id)} has been declined and the ticket is now cancelled."

    if not new_date_str or not new_time_str:
        return "Response Error: a preferred installation date and time are required to accept the estimate."

    quantity = booking.get("quantity", 1)
    try:
        travel = _compute_travel(booking.get("client_address", ""))
        product = get_product(booking["model"])
        duration_hours = compute_appointment_duration(
            quantity,
            product.get("installation_hours", 1.5) if product else 1.5,
            travel["duration_hours"],
        )
    except Exception as exc:
        logger.warning(f"INSTALL_DURATION_TRAVEL_FAILED: {exc} — falling back to default duration")
        duration_hours = APPOINTMENT_DURATION_HOURS

    result = book_installation_tool.invoke(
        {
            "model": booking["model"],
            "date_str": new_date_str,
            "time_str": new_time_str,
            "purpose": "Installation",
            "duration_hours": duration_hours,
        }
    )
    if "CONFIRMED" not in result and "scheduled" not in result.lower():
        return f"Could not schedule the installation: {result}"

    resolve_client_estimate_response(
        booking_id,
        "accept",
        new_date_str,
        new_time_str,
        changed_by=client_actor,
        client_email=client_email,
    )
    logger.info(
        f"ESTIMATE_ACCEPTED_BY_CLIENT: id={booking_id} install={new_date_str} {new_time_str}"
    )
    return (
        f"Estimate for {ticket_code(booking_id)} accepted — installation scheduled for "
        f"{new_date_str} at {new_time_str}. {result}"
    )
