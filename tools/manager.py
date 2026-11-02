"""Manager approval tools: bookings, survey reports, the QA cache and job reports."""

from langchain.tools import tool
from pydantic import BaseModel, Field

from backend.database import (
    get_approved_bookings,
    get_costs,
    get_job_completions,
    get_pending_bookings,
    get_product,
    reject_survey_report,
    resolve_booking,
    resolve_job_completion,
    resolve_survey_estimate,
    ticket_code,
)
from backend.llm_provider import get_llm
from backend.log_context import username_var
from backend.qa_cache import approve as qa_approve
from backend.qa_cache import get_pending as qa_get_pending
from backend.qa_cache import reject as qa_reject
from tools.client import compute_appointment_duration
from tools.common import logger
from tools.emails import (
    _THANKYOU_EMAIL_TEMPLATE,
    _send_booking_confirmation_email,
    _send_price_estimate_email,
    _send_rejection_email,
)
from tools.pricing import _compute_survey_estimate
from tools.scheduling import APPOINTMENT_DURATION_HOURS, book_installation_tool
from tools.travel import _compute_travel

# ── list tools (read-only) ────────────────────────────────────────────────────


@tool
def list_pending_bookings_tool() -> str:
    """
    Lists all pending booking requests waiting for manager approval.
    Use this before approve_booking_tool or reject_booking_tool to see the IDs.
    Manager-only.
    """
    pending = get_pending_bookings(status="pending")
    if not pending:
        return "No pending booking requests."
    lines = []
    for b in pending:
        product = get_product(b["model"])
        stock = product["stock"] if product else "unknown"
        entry = (
            f"{ticket_code(b['id'])} — {b.get('quantity', 1)}× {b['model']} on {b['date_str']} at {b['time_str']}"
            f"\n  Requested by: {b['requested_by']} | Stock: {stock}"
        )
        if b.get("client_email"):
            entry += f"\n  Email: {b['client_email']}"
        if b.get("client_phone"):
            entry += f"  Phone: {b['client_phone']}"
        if b.get("notes"):
            entry += f"\n  Notes: {b['notes']}"
        lines.append(entry)
    return "\n\n".join(lines)


@tool
def list_pending_qa_tool() -> str:
    """
    Lists all Q&A pairs pending review for the knowledge cache.
    Use this before approve_qa_tool or reject_qa_tool to see the IDs.
    Manager-only.
    """
    pending = qa_get_pending()
    if not pending:
        return "No Q&A pairs pending cache review."
    lines = []
    for item in pending:
        q = item["question"]
        a = item["answer"]
        lines.append(
            f"#{item['id']} [{item['source_role']}] — Q: {q[:120]}{'...' if len(q) > 120 else ''}\n"
            f"  A: {a[:200]}{'...' if len(a) > 200 else ''}"
        )
    return "\n\n".join(lines)


@tool
def list_pending_jobs_tool() -> str:
    """
    Lists all pending job completion reports waiting for manager approval.
    Use this before approve_job_tool or reject_job_tool to see the IDs.
    Manager-only.
    """
    pending = get_job_completions(status="pending")
    if not pending:
        return "No pending job completion reports."
    lines = []
    for job in pending:
        entry = f"#{job['id']} — by {job['sent_by']} at {job['created_at'][:16]}"
        if job.get("booking_id"):
            entry += f" (booking {ticket_code(job['booking_id'])})"
        if job.get("client_email"):
            entry += f"\n  Client email: {job['client_email']}"
        report_preview = job["structured_report"][:200]
        entry += f"\n  {report_preview}{'...' if len(job['structured_report']) > 200 else ''}"
        lines.append(entry)
    return "\n\n".join(lines)


# ── booking approval/rejection ─────────────────────────────────────────────────


class ApproveBookingInput(BaseModel):
    booking_id: int = Field(description="ID of the pending booking request to approve.")


@tool(args_schema=ApproveBookingInput)
def approve_booking_tool(booking_id: int) -> str:
    """
    Approves a pending site-survey request: creates the calendar event for the
    survey visit, marks it approved, and emails the client a confirmation.
    Always call list_pending_bookings_tool first to confirm the booking ID.
    Manager-only.
    """
    pending = get_pending_bookings(status="pending")
    booking = next((b for b in pending if b["id"] == booking_id), None)
    if not booking:
        return f"Booking {ticket_code(booking_id)} not found among pending requests."

    quantity = booking.get("quantity", 1)
    try:
        travel = _compute_travel(booking.get("client_address", ""))
        duration_hours = compute_appointment_duration(
            quantity, get_costs().get("survey_hours_per_unit", 0.5), travel["duration_hours"]
        )
    except Exception as exc:
        logger.warning(f"SURVEY_DURATION_TRAVEL_FAILED: {exc} — falling back to default duration")
        duration_hours = APPOINTMENT_DURATION_HOURS

    result = book_installation_tool.invoke(
        {
            "model": booking["model"],
            "date_str": booking["date_str"],
            "time_str": booking["time_str"],
            "purpose": "Survey",
            "duration_hours": duration_hours,
        }
    )

    username = username_var.get()
    if "CONFIRMED" in result or "scheduled" in result.lower():
        resolve_booking(booking_id, "approved", username)
        logger.info(f"BOOKING_APPROVED_VIA_CHAT: id={booking_id} by={username}")
        email_note = _send_booking_confirmation_email(booking)
        return f"Booking {ticket_code(booking_id)} approved and added to calendar. {result}{email_note}"
    else:
        return f"Calendar booking failed for {ticket_code(booking_id)}: {result}"


class RejectBookingInput(BaseModel):
    booking_id: int = Field(description="ID of the pending booking request to reject.")
    reason: str = Field(
        default="", description="Optional reason to include in the rejection email to the client."
    )


@tool(args_schema=RejectBookingInput)
def reject_booking_tool(booking_id: int, reason: str = "") -> str:
    """
    Rejects a pending site-survey request and emails the client the reason.
    Always call list_pending_bookings_tool first to confirm the booking ID.
    Manager-only.
    """
    pending = get_pending_bookings(status="pending")
    booking = next((b for b in pending if b["id"] == booking_id), None)
    if not booking:
        return f"Booking {ticket_code(booking_id)} not found among pending requests."

    username = username_var.get()
    resolve_booking(booking_id, "rejected", username, reason)
    logger.info(f"BOOKING_REJECTED_VIA_CHAT: id={booking_id} by={username}")
    email_note = (
        _send_rejection_email(booking, reason) if reason else " No reason given — email not sent."
    )
    return f"Booking {ticket_code(booking_id)} ({booking['model']} on {booking['date_str']}) has been rejected.{email_note}"


@tool
def list_pending_survey_reports_tool() -> str:
    """
    Lists survey reports (measurements submitted by an installer) awaiting a
    priced estimate. Use this before approve_survey_estimate_tool or
    reject_survey_estimate_tool to see the IDs. Manager-only.
    """
    reports = [
        b
        for b in get_approved_bookings(assigned_to=None)
        if b.get("ticket_phase") == "survey" and b.get("ticket_status") == "in_review"
    ]
    if not reports:
        return "No survey reports awaiting a priced estimate."
    lines = []
    for b in reports:
        lines.append(
            f"{ticket_code(b['id'])} — {b.get('quantity', 1)}× {b['model']}, "
            f"{b.get('height_cm')}×{b.get('width_cm')}×{b.get('depth_cm')}cm, glass: {b.get('glass_type')}"
            + (f"\n  Notes: {b['survey_notes']}" if b.get("survey_notes") else "")
        )
    return "\n\n".join(lines)


class ApproveSurveyEstimateInput(BaseModel):
    booking_id: int = Field(description="ID of the survey report to price and approve.")


@tool(args_schema=ApproveSurveyEstimateInput)
def approve_survey_estimate_tool(booking_id: int) -> str:
    """
    Computes a priced estimate from the survey measurements, saves it, and
    emails it to the client for their acceptance. Use
    list_pending_survey_reports_tool first to confirm the booking ID.
    Manager-only.
    """
    reports = [
        b
        for b in get_approved_bookings(assigned_to=None)
        if b.get("ticket_phase") == "survey" and b.get("ticket_status") == "in_review"
    ]
    booking = next((b for b in reports if b["id"] == booking_id), None)
    if not booking:
        return f"Survey report {ticket_code(booking_id)} not found among reports awaiting pricing."

    try:
        estimate = _compute_survey_estimate(booking)
    except ValueError as exc:
        return f"Estimate Error: {exc}"
    except Exception as exc:
        logger.error(f"SURVEY_ESTIMATE_CALC_FAILED: {exc}", exc_info=True)
        return "Estimate Error: could not compute travel/pricing data. Please try again."

    username = username_var.get()
    updated = resolve_survey_estimate(booking_id, estimate["total"], username)
    if not updated:
        return f"Could not save the estimate for {ticket_code(booking_id)} — it may have already been resolved."

    logger.info(
        f"SURVEY_ESTIMATE_APPROVED: id={booking_id} total={estimate['total']:.2f} by={username}"
    )
    email_note = _send_price_estimate_email(booking.get("client_email", ""), estimate["breakdown"])
    return (
        f"Estimate for {ticket_code(booking_id)} computed and saved (€{estimate['total']:.2f}).{email_note}\n\n"
        f"{estimate['breakdown']}"
    )


class RejectSurveyEstimateInput(BaseModel):
    booking_id: int = Field(description="ID of the survey report to reject.")


@tool(args_schema=RejectSurveyEstimateInput)
def reject_survey_estimate_tool(booking_id: int) -> str:
    """
    Rejects a survey report (e.g. incomplete/incorrect measurements) and sends
    the ticket back to the installer to redo. Manager-only.
    """
    username = username_var.get()
    updated = reject_survey_report(booking_id, username)
    if not updated:
        return f"Could not reject {ticket_code(booking_id)} — it may not be a survey report awaiting pricing."
    logger.info(f"SURVEY_REPORT_REJECTED: id={booking_id} by={username}")
    return f"Survey report {ticket_code(booking_id)} rejected — ticket sent back to the installer."


# ── QA cache approval/rejection (no HITL — low risk) ─────────────────────────


class ApproveQAInput(BaseModel):
    pending_id: int = Field(
        description="ID of the pending Q&A entry to add to the knowledge cache."
    )


@tool(args_schema=ApproveQAInput)
def approve_qa_tool(pending_id: int) -> str:
    """
    Adds a pending Q&A pair to the knowledge cache.
    Use list_pending_qa_tool first to see available IDs.
    Manager-only.
    """
    pending = qa_get_pending()
    item = next((p for p in pending if p["id"] == pending_id), None)
    if not item:
        return f"Q&A entry #{pending_id} not found in the pending review list."

    username = username_var.get()
    result = qa_approve(pending_id, approved_by=username)
    if result:
        logger.info(f"QA_APPROVED_VIA_CHAT: id={pending_id} by={username}")
        return f"Q&A entry #{pending_id} added to the knowledge cache."
    return f"Q&A entry #{pending_id} skipped — a near-identical entry already exists in the cache."


class RejectQAInput(BaseModel):
    pending_id: int = Field(description="ID of the pending Q&A entry to reject.")


@tool(args_schema=RejectQAInput)
def reject_qa_tool(pending_id: int) -> str:
    """
    Rejects a pending Q&A pair — it will not be added to the knowledge cache.
    Use list_pending_qa_tool first to see available IDs.
    Manager-only.
    """
    pending = qa_get_pending()
    item = next((p for p in pending if p["id"] == pending_id), None)
    if not item:
        return f"Q&A entry #{pending_id} not found in the pending review list."

    qa_reject(pending_id)
    logger.info(f"QA_REJECTED_VIA_CHAT: id={pending_id} by={username_var.get()}")
    return f"Q&A entry #{pending_id} rejected and removed from the review queue."


# ── job completion approval/rejection ──────────────────────────────────────────


class ApproveJobInput(BaseModel):
    job_id: int = Field(description="ID of the job completion report to approve.")
    client_email: str = Field(
        default="",
        description="Client email for the thank-you message. Leave empty if already on file.",
    )


@tool(args_schema=ApproveJobInput)
def approve_job_tool(job_id: int, client_email: str = "") -> str:
    """
    Approves a job completion report and sends a thank-you email to the client.
    Use list_pending_jobs_tool first to see available IDs.
    Manager-only.
    """
    from backend.email_utils import send_email

    pending = get_job_completions(status="pending")
    job = next((j for j in pending if j["id"] == job_id), None)
    if not job:
        return f"Job completion report #{job_id} not found among pending reports."

    username = username_var.get()
    resolve_job_completion(job_id, "approved", username)

    email_to = client_email or job.get("client_email", "")
    if email_to and "@" in email_to:
        try:
            llm = get_llm(temperature=0.3, max_tokens=400)
            body = llm.invoke(
                _THANKYOU_EMAIL_TEMPLATE.format(structured_report=job["structured_report"])
            ).content.strip()
            sent = send_email(
                to=email_to,
                subject="Thank You — Your Installation is Complete",
                body=body,
            )
            email_note = (
                f" Thank-you email sent to {email_to}."
                if sent
                else f" Email delivery failed for {email_to} (check SMTP config)."
            )
        except Exception as exc:
            email_note = f" Email generation failed: {exc}"
    else:
        email_note = " No client email — thank-you email not sent."

    logger.info(f"JOB_APPROVED_VIA_CHAT: id={job_id} by={username}")
    return f"Job completion report #{job_id} approved.{email_note}"


class RejectJobInput(BaseModel):
    job_id: int = Field(description="ID of the job completion report to reject.")


@tool(args_schema=RejectJobInput)
def reject_job_tool(job_id: int) -> str:
    """
    Rejects a job completion report.
    Use list_pending_jobs_tool first to see available IDs.
    Manager-only.
    """
    pending = get_job_completions(status="pending")
    job = next((j for j in pending if j["id"] == job_id), None)
    if not job:
        return f"Job completion report #{job_id} not found among pending reports."

    username = username_var.get()
    resolve_job_completion(job_id, "rejected", username)
    logger.info(f"JOB_REJECTED_VIA_CHAT: id={job_id} by={username}")
    return f"Job completion report #{job_id} (by {job['sent_by']}) has been rejected."
