"""Installer page — assigned jobs panel on the right, technical chat on the left."""

import sys
from pathlib import Path

import streamlit as st

# TODO PR 11: drop with the app/ restructure (imports ui_shared as top-level).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_DIR      = Path(__file__).resolve().parents[1]
for p in (str(PROJECT_ROOT), str(APP_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from ui_shared import (
    get_agent,
    handle_user_turn,
    init_chat_session,
    render_chat_autoscroll,
    render_chat_history,
    render_column_resize_handle,
    render_hitl_and_input,
    render_installer_job_briefing,
    render_page_header,
    render_sidebar,
    render_ticket_card,
    render_ticket_map,
    render_ticket_summary,
    require_auth,
    t,
    ticket_code,
)

from backend.database import (
    claim_booking,
    decline_booking,
    get_approved_bookings,
    get_installer_active_booking,
    release_booking,
)

st.set_page_config(page_title="Installer Portal", page_icon=None, layout="wide")

require_auth(allowed_roles=["installer"])
init_chat_session("installer")

render_page_header(t("installer_title"))

selected_model = render_sidebar("installer")
agent          = get_agent(selected_model, "installer")

chat_col, panel_col = st.columns([3, 2])
render_column_resize_handle()

# ═══════════════════════════════════════════════════════════════════════════════
# RIGHT PANEL — JOBS
# ═══════════════════════════════════════════════════════════════════════════════
with panel_col:
    with st.container(key="dashboard_zone", height="stretch"):
        installer = st.session_state.get("username", "installer")
        my_jobs = get_approved_bookings(assigned_to=installer)

        render_ticket_summary(my_jobs)

        tab_mine, tab_available, tab_map = st.tabs([t("tab_my_jobs"), t("tab_available"), t("map_tab_label")])

        # ── my active jobs ────────────────────────────────────────────────────────
        with tab_mine:
            if not my_jobs:
                st.info(t("no_jobs"))
            else:
                for job in my_jobs:
                    phase = job.get("ticket_phase", "installation")
                    with st.container(border=True):
                        render_ticket_card(job, show_stock=False)
                        render_installer_job_briefing(job)

                        col_release, col_complete = st.columns(2)
                        with col_release:
                            if st.button(t("release_button"), key=f"release_{job['id']}", use_container_width=True):
                                if release_booking(job["id"], installer):
                                    st.success(t("released_msg").format(code=ticket_code(job['id'])))
                                else:
                                    st.error(t("could_not_release"))
                                st.rerun()
                        with col_complete:
                            complete_label = t("report_measurements_button") if phase == "survey" else t("close_report_button")
                            if st.button(complete_label, key=f"complete_{job['id']}", use_container_width=True):
                                st.session_state[f"close_job_{job['id']}"] = True

                        if not st.session_state.get(f"close_job_{job['id']}"):
                            continue

                        if phase == "survey":
                            with st.form(key=f"survey_form_{job['id']}"):
                                col_h, col_w, col_d = st.columns(3)
                                height_cm = col_h.number_input(t("height_cm_label"), min_value=0.0, step=1.0, key=f"h_{job['id']}")
                                width_cm  = col_w.number_input(t("width_cm_label"),  min_value=0.0, step=1.0, key=f"w_{job['id']}")
                                depth_cm  = col_d.number_input(t("depth_cm_label"),  min_value=0.0, step=1.0, key=f"d_{job['id']}")
                                glass_type = st.selectbox(
                                    t("glass_type_caption").rstrip(":"),
                                    ["standard", "triple", "security"],
                                    format_func=lambda g: {
                                        "standard": t("glass_standard"),
                                        "triple": t("glass_triple"),
                                        "security": t("glass_security"),
                                    }[g],
                                    key=f"glass_{job['id']}",
                                )
                                survey_notes = st.text_area(
                                    t("notes_optional_label"),
                                    placeholder=t("notes_optional_placeholder"),
                                    key=f"survey_notes_{job['id']}",
                                )
                                submit = st.form_submit_button(t("submit_measurements_button"), use_container_width=True)
                                if submit:
                                    if height_cm > 0 and width_cm > 0 and depth_cm > 0:
                                        msg = (
                                            f"I measured {ticket_code(job['id'])}: "
                                            f"height {height_cm}cm, width {width_cm}cm, depth {depth_cm}cm, "
                                            f"glass type {glass_type}."
                                            + (f" Notes: {survey_notes}" if survey_notes else "")
                                        )
                                        st.session_state[f"close_job_{job['id']}"] = False
                                        handle_user_turn(msg, agent, selected_model, is_synthetic=True)
                                    else:
                                        st.warning(t("height_width_depth_error"))
                        else:
                            with st.form(key=f"close_form_{job['id']}"):
                                notes = st.text_area(
                                    t("completion_notes_label"),
                                    placeholder=t("completion_notes_placeholder"),
                                    key=f"close_notes_{job['id']}",
                                )
                                submit = st.form_submit_button(t("submit_report_button"), use_container_width=True)
                                if submit:
                                    if notes.strip():
                                        # Inject the report as a chat message so the agent handles it
                                        # through complete_job_tool (with HITL if needed)
                                        client_email = job.get("client_email", "")
                                        msg = (
                                            f"I have completed installation {ticket_code(job['id'])} "
                                            f"({job['model']} on {job['date_str']}). "
                                            f"Booking reference: {ticket_code(job['id'])}. "
                                            f"Client email: {client_email or 'not provided'}. "
                                            f"Notes: {notes}"
                                        )
                                        st.session_state[f"close_job_{job['id']}"] = False
                                        handle_user_turn(msg, agent, selected_model, is_synthetic=True)
                                    else:
                                        st.warning(t("please_add_notes"))

        # ── available jobs ────────────────────────────────────────────────────────
        with tab_available:
            available    = get_approved_bookings(assigned_to="")
            active_job   = get_installer_active_booking(installer)
            already_busy = active_job is not None

            if not available:
                st.info(t("no_jobs_available"))
            else:
                if already_busy:
                    st.warning(
                        t("already_active_job_warning").format(
                            code=ticket_code(active_job['id']), model=active_job['model'], date=active_job['date_str']
                        )
                    )
                for job in available:
                    with st.container(border=True):
                        render_ticket_card(job, show_stock=False)

                        col_claim, col_reject = st.columns(2)
                        with col_claim:
                            if st.button(
                                t("claim_button"),
                                key=f"claim_{job['id']}",
                                use_container_width=True,
                                disabled=already_busy,
                            ):
                                if claim_booking(job["id"], installer):
                                    st.success(t("claimed_msg").format(code=ticket_code(job['id'])))
                                else:
                                    st.warning(t("could_not_claim"))
                                st.rerun()
                        with col_reject:
                            if st.button(t("reject_button"), key=f"reject_{job['id']}", use_container_width=True):
                                st.session_state[f"reject_open_{job['id']}"] = True

                        if st.session_state.get(f"reject_open_{job['id']}"):
                            with st.form(key=f"reject_form_{job['id']}"):
                                reason = st.text_area(
                                    t("reason_for_declining"),
                                    placeholder=t("reason_declining_placeholder"),
                                    key=f"reject_reason_{job['id']}",
                                )
                                submitted = st.form_submit_button(t("submit_button"), use_container_width=True)
                                if submitted:
                                    if reason.strip():
                                        decline_booking(job["id"], installer, reason.strip())
                                        st.info(t("declined_msg").format(code=ticket_code(job['id'])))
                                        st.session_state[f"reject_open_{job['id']}"] = False
                                        st.rerun()
                                    else:
                                        st.warning(t("please_explain_declining"))

        # ── map ───────────────────────────────────────────────────────────────────
        with tab_map:
            render_ticket_map(my_jobs + available)

        st.divider()
        st.caption(t("installer_sub"))

# ═══════════════════════════════════════════════════════════════════════════════
# LEFT PANEL — CHAT
# ═══════════════════════════════════════════════════════════════════════════════
with chat_col, st.container(key="chat_zone", height="stretch"):
    st.subheader(t("assistant_label"))
    render_chat_history(agent)

    user_prompt = render_hitl_and_input(
        agent, selected_model,
        prompt_hint=t("installer_sub"),
    )

    if user_prompt:
        handle_user_turn(user_prompt, agent, selected_model)

    render_chat_autoscroll()
