"""Emails sent to clients about their bookings and estimates."""

from langchain.tools import tool
from pydantic import BaseModel, Field

from backend.company_config import COMPANY_CONTEXT
from backend.llm_provider import get_llm
from tools.common import logger

_ESTIMATE_EMAIL_TEMPLATE = f"""You are writing a professional email to a client on behalf of {COMPANY_CONTEXT["company_name"]}.

Write a concise email that:
1. Greets the client warmly
2. Presents the attached price estimate for their installation
3. Tells them exactly how to respond: log back into the client portal and open
   the "Check a pending estimate" section using the same email address as this
   request, then accept or reject the estimate there (choosing an installation
   date/time if accepting) — this takes a couple of clicks, no need to write
   anything in chat
4. Signs off professionally

Estimate details:
{{estimate_text}}
"""

_CONFIRMATION_EMAIL_TEMPLATE = f"""You are writing a professional booking confirmation email to a client on behalf of {COMPANY_CONTEXT["company_name"]}.

Write a warm, professional email that:
1. Confirms the installation booking with the details below
2. Briefly explains what to expect (a certified installer will arrive at the scheduled time)
3. Gives 2-3 short, generic pre-installation recommendations (e.g. ensure clear
   access to the installation area, have someone present to let the installer
   in, remove any obstructions near the door)
4. Provides a contact point for questions or rescheduling
5. Signs off professionally

Booking details:
- Client: {{client_name}}
- Model: {{model}} × {{quantity}}
- Date: {{date_str}} at {{time_str}}
- Address: {{client_address}}
"""


def _send_booking_confirmation_email(booking: dict) -> str:
    """
    Sends a booking-confirmation email to the client after a manager approval.
    Returns a short human-readable status note to append to the tool result.
    """
    client_email = booking.get("client_email", "")
    if not client_email or "@" not in client_email:
        return " No client email on file — confirmation email not sent."

    try:
        llm = get_llm(temperature=0.3, max_tokens=400)
        body = llm.invoke(
            _CONFIRMATION_EMAIL_TEMPLATE.format(
                client_name=booking.get("client_name") or booking.get("requested_by", ""),
                model=booking["model"],
                quantity=booking.get("quantity", 1),
                date_str=booking["date_str"],
                time_str=booking["time_str"],
                client_address=booking.get("client_address", ""),
            )
        ).content.strip()
    except Exception as exc:
        logger.warning(f"BOOKING_CONFIRMATION_EMAIL_GENERATION_FAILED: {exc}")
        return f" Confirmation email generation failed: {exc}"

    from backend.email_utils import send_email

    sent = send_email(
        to=client_email,
        subject="Your Installation Booking is Confirmed",
        body=body,
    )
    return (
        f" Confirmation email sent to {client_email}."
        if sent
        else f" Confirmation email delivery failed for {client_email} (check SMTP config)."
    )


_REJECTION_EMAIL_TEMPLATE = f"""You are writing a professional email to a client on behalf of {COMPANY_CONTEXT["company_name"]}.

Their installation booking request has been declined. Write a warm, professional email that:
1. Apologises for the inconvenience
2. Clearly explains the reason for the rejection (provided below)
3. Proposes rescheduling or an alternative solution
4. Provides a contact point for next steps
5. Signs off professionally

Rejection reason: {{rejection_reason}}

Booking details:
- Client: {{client_name}}
- Model: {{model}} × {{quantity}}
- Requested date: {{date_str}} at {{time_str}}
"""

_THANKYOU_EMAIL_TEMPLATE = f"""You are writing a professional post-installation thank-you email on behalf of {COMPANY_CONTEXT["company_name"]}.

Write a warm, professional email that:
1. Thanks the client for choosing the company
2. Confirms the installation has been successfully completed (use the details below)
3. Reassures them you are available for any inconvenience or follow-up questions
4. Briefly mentions availability for future maintenance, upgrades, or special offers
5. Signs off warmly

Job completion details:
{{structured_report}}
"""


def _send_rejection_email(booking: dict, reason: str) -> str:
    """
    Sends a rejection email for a declined site-survey/booking request.
    Consolidated here so app/pages/manager.py can import instead of
    maintaining its own duplicate copy of this logic.
    """
    client_email = booking.get("client_email", "")
    if not client_email or "@" not in client_email:
        return " No client email on file — rejection email not sent."

    try:
        llm = get_llm(temperature=0.3, max_tokens=400)
        body = llm.invoke(
            _REJECTION_EMAIL_TEMPLATE.format(
                rejection_reason=reason,
                client_name=booking.get("client_name") or booking.get("requested_by", ""),
                model=booking["model"],
                quantity=booking.get("quantity", 1),
                date_str=booking["date_str"],
                time_str=booking["time_str"],
            )
        ).content.strip()
    except Exception as exc:
        logger.warning(f"REJECTION_EMAIL_GENERATION_FAILED: {exc}")
        return f" Rejection email generation failed: {exc}"

    from backend.email_utils import send_email

    sent = send_email(to=client_email, subject="Update on Your Installation Request", body=body)
    return (
        f" Rejection email sent to {client_email}."
        if sent
        else f" Rejection email delivery failed for {client_email} (check SMTP config)."
    )


def _send_price_estimate_email(client_email: str, estimate_text: str) -> str:
    """
    Drafts and sends a priced-estimate email. Consolidated here so both
    send_estimate_to_client_tool and approve_survey_estimate_tool share one
    implementation instead of duplicating the LLM-drafting + send logic.
    """
    if not client_email or "@" not in client_email:
        return " No client email on file — estimate email not sent."

    try:
        llm = get_llm(temperature=0.3, max_tokens=400)
        body = llm.invoke(
            _ESTIMATE_EMAIL_TEMPLATE.format(estimate_text=estimate_text)
        ).content.strip()
    except Exception as exc:
        logger.warning(f"ESTIMATE_EMAIL_GENERATION_FAILED: {exc} — using raw estimate")
        body = f"Please find your installation price estimate below:\n\n{estimate_text}"

    from backend.email_utils import send_email

    sent = send_email(
        to=client_email,
        subject=f"Your {COMPANY_CONTEXT['company_name']} Installation Estimate",
        body=body,
    )
    return (
        f" Estimate email sent to {client_email}."
        if sent
        else f" Estimate email delivery failed for {client_email} (check SMTP config)."
    )


class SendEstimateInput(BaseModel):
    client_email: str = Field(description="Client email address to send the estimate to.")
    estimate_text: str = Field(description="The full price estimate text from price_estimate_tool.")


@tool(args_schema=SendEstimateInput)
def send_estimate_to_client_tool(client_email: str, estimate_text: str) -> str:
    """
    Sends a price estimate to a client by email.
    The email body is generated by the AI from the estimate text.
    Manager-only.
    """
    if not client_email or "@" not in client_email:
        return "Email Error: invalid or missing client email address."

    email_note = _send_price_estimate_email(client_email, estimate_text)
    if "sent to" in email_note and "failed" not in email_note:
        return f"Price estimate email sent successfully to {client_email}."
    return (
        "Email could not be sent (SMTP not configured or delivery failed). "
        "Check SMTP credentials in .env."
    )
