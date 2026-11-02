"""The client's estimates and installation status, behind email verification."""

import datetime

import streamlit as st

from backend.database import ticket_code
from ui.i18n import t
from ui.session import _k, bind_request_context
from ui.tickets import _TICKET_STATUS_COLOR
from utils.escaping import html_md_escape, md_escape, md_no_images

_VERIFY_ERROR_KEYS = {
    "cooldown": "verify_cooldown",
    "invalid_email": "verify_invalid_email",
    "email_unavailable": "verify_email_unavailable",
}


def _render_client_email_verification(key_prefix: str) -> str:
    """
    Returns the session's verified client email — or, if there is none yet,
    renders the one-time-code form and returns "". Client sessions are
    anonymous (login.py's "Continue as Client" collects nothing), so this is
    what proves ownership before any existing ticket is shown or changed.
    The result lives in st.session_state["verified_client_email"], so
    verifying in one panel unlocks the others and the chat tools too.
    """
    from backend import client_auth

    verified = st.session_state.get("verified_client_email", "")
    if verified:
        col_info, col_change = st.columns([3, 2])
        col_info.caption(t("verified_as").format(email=verified))
        if col_change.button(t("verify_change_email"), key=_k(f"{key_prefix}_verify_change")):
            for key in ("verified_client_email", "verify_pending_email", "verify_demo_code"):
                st.session_state.pop(key, None)
            bind_request_context()
            st.rerun()
        return verified

    st.caption(t("verify_email_hint"))
    email = st.text_input(
        t("booking_form_email"),
        key=_k(f"{key_prefix}_verify_email"),
        placeholder="you@example.com",
    )
    if st.button(t("verify_send_code"), key=_k(f"{key_prefix}_verify_send")):
        result = client_auth.request_code(email)
        if result.status in ("sent", "demo"):
            st.session_state["verify_pending_email"] = client_auth.normalize_email(email)
            st.session_state["verify_demo_code"] = result.demo_code
        else:
            st.error(t(_VERIFY_ERROR_KEYS[result.status]))

    pending_email = st.session_state.get("verify_pending_email")
    if pending_email:
        demo_code = st.session_state.get("verify_demo_code")
        if demo_code:
            st.warning(t("verify_demo_code").format(code=demo_code))
        else:
            st.info(t("verify_code_sent").format(email=pending_email))
        code = st.text_input(
            t("verify_code_label"), key=_k(f"{key_prefix}_verify_code"), max_chars=6
        )
        if st.button(t("verify_button"), key=_k(f"{key_prefix}_verify_submit"), type="primary"):
            if client_auth.verify_code(pending_email, code):
                st.session_state["verified_client_email"] = pending_email
                st.session_state.pop("verify_pending_email", None)
                st.session_state.pop("verify_demo_code", None)
                bind_request_context()
                st.rerun()
            else:
                st.error(t("verify_failed"))
    return ""


def render_client_pending_estimates() -> None:
    """
    Lets a client accept/reject a priced estimate with a couple of clicks,
    instead of having to return to the chat and describe what they want in
    free text. Requires a verified email (_render_client_email_verification)
    — the same ownership rule find_my_tickets_by_email_tool and
    respond_to_estimate_tool enforce.

    Calls respond_to_estimate_tool directly (bypassing the LLM agent),
    the same direct-tool-call pattern the manager dashboard uses for
    Approve/Reject actions.
    """
    with st.expander(t("pending_estimate_title")):
        email = _render_client_email_verification("estimate")
        if not email:
            return

        from backend.database import find_tickets_by_email

        tickets = find_tickets_by_email(email, ticket_status="awaiting_client")
        if not tickets:
            st.caption(t("no_pending_estimates"))
            return

        from tools import respond_to_estimate_tool

        for ticket in tickets:
            with st.container(border=True):
                st.markdown(
                    f"**{ticket_code(ticket['id'])}** — {ticket.get('quantity', 1)}× {md_escape(ticket['model'])} "
                    f"— **€{ticket.get('estimated_price', 0):.2f}**"
                )
                st.caption(t("preferred_install_datetime_hint"))
                col_date, col_time = st.columns(2)
                date_val = col_date.date_input(
                    t("booking_form_date"),
                    min_value=datetime.date.today(),
                    key=_k(f"est_date_{ticket['id']}"),
                )
                time_val = col_time.time_input(
                    t("booking_form_time"), key=_k(f"est_time_{ticket['id']}")
                )

                col_accept, col_reject = st.columns(2)
                with col_accept:
                    if st.button(
                        t("accept_estimate_button"),
                        key=_k(f"est_accept_{ticket['id']}"),
                        type="primary",
                        use_container_width=True,
                    ):
                        result = respond_to_estimate_tool.invoke(
                            {
                                "booking_id": ticket["id"],
                                "response": "accept",
                                "new_date_str": date_val.strftime("%Y-%m-%d"),
                                "new_time_str": time_val.strftime("%H:%M"),
                            }
                        )
                        if "accepted —" in result:
                            st.success(result)
                            st.rerun()
                        else:
                            st.error(result)
                with col_reject:
                    if st.button(
                        t("reject_button"),
                        key=_k(f"est_reject_{ticket['id']}"),
                        use_container_width=True,
                    ):
                        result = respond_to_estimate_tool.invoke(
                            {"booking_id": ticket["id"], "response": "reject"}
                        )
                        if "has been declined" in result:
                            st.warning(result)
                            st.rerun()
                        else:
                            st.error(result)


def _install_outcome_badge(approved: bool) -> str:
    color = _TICKET_STATUS_COLOR["closed"] if approved else _TICKET_STATUS_COLOR["blocked"]
    label = t("install_outcome_approved") if approved else t("install_outcome_rejected")
    return (
        f'<span style="border:1px solid {color};color:{color};padding:1px 7px;'
        f"border-radius:3px;font-size:0.72rem;font-weight:700;"
        f'text-transform:uppercase;letter-spacing:0.08em;">{label}</span>'
    )


def render_client_installation_status() -> None:
    """
    Client-facing counterpart to render_client_pending_estimates — same
    email-lookup, bordered-card style, sitting right next to it — but for
    the OTHER end of the ticket lifecycle: the installer's completion
    report once the manager has actually resolved it (approve_job_tool /
    reject_job_tool). An approved report closes the ticket and already
    triggers a thank-you email (see approve_job_tool), but email delivery
    depends on SMTP config and the client's inbox; this makes the same
    outcome checkable directly in the portal regardless. Rejected reports
    are shown too (worded as "still being finished up", not "rejected" —
    the report was sent back to the installer for rework, not the
    installation itself being refused) since the client asked to see both
    outcomes, not just the positive one.
    """
    with st.expander(t("installation_status_title")):
        email = _render_client_email_verification("install_status")
        if not email:
            return

        from backend.database import find_tickets_by_email, get_job_completions

        tickets = find_tickets_by_email(email)
        ticket_by_id = {tk["id"]: tk for tk in tickets}
        if not ticket_by_id:
            st.caption(t("no_installation_updates"))
            return

        resolved_jobs = [
            job
            for job in get_job_completions(status="approved")
            + get_job_completions(status="rejected")
            if job.get("booking_id") in ticket_by_id
        ]
        if not resolved_jobs:
            st.caption(t("no_installation_updates"))
            return

        resolved_jobs.sort(key=lambda job: job.get("resolved_at") or "", reverse=True)
        for job in resolved_jobs:
            ticket = ticket_by_id[job["booking_id"]]
            approved = job["status"] == "approved"
            with st.container(border=True):
                st.markdown(
                    f"**{ticket_code(ticket['id'])}** — {ticket.get('quantity', 1)}× {html_md_escape(ticket['model'])} "
                    f"&nbsp; {_install_outcome_badge(approved)}",
                    unsafe_allow_html=True,
                )
                st.caption(
                    t("install_outcome_approved_note")
                    if approved
                    else t("install_outcome_rejected_note")
                )
                if job.get("structured_report"):
                    st.markdown(md_no_images(job["structured_report"]))
