"""
Manager page — ticket dashboard (tickets/clients/alerts/cache/analytics) + chat.
"""

import datetime
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# TODO PR 11: drop with the app/ restructure (imports ui_shared as top-level).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_DIR = Path(__file__).resolve().parents[1]
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
    render_page_header,
    render_sidebar,
    render_stat_tiles,
    render_ticket_card,
    render_ticket_map,
    render_ticket_summary,
    require_auth,
    t,
    ticket_code,
)

from backend.database import (
    get_alerts,
    get_alerts_for_booking,
    get_all_tickets,
    get_business_analytics,
    get_clients,
    get_job_completions,
    get_usage_stats,
    mark_alerts_read,
    reset_operational_data,
    resolve_job_completion,
    update_ticket_details,
)
from backend.email_utils import send_email
from backend.llm_provider import get_llm
from backend.qa_cache import approve as qa_approve
from backend.qa_cache import get_pending as qa_get_pending
from backend.qa_cache import reject as qa_reject
from tools import (
    _THANKYOU_EMAIL_TEMPLATE,
    approve_booking_tool,
    approve_survey_estimate_tool,
    reject_booking_tool,
    reject_survey_estimate_tool,
)

st.set_page_config(page_title="Manager Portal", page_icon=None, layout="wide")

require_auth(allowed_roles=["manager"])
init_chat_session("manager")

render_page_header(t("manager_title"))


@st.dialog(t("reset_dialog_title"))
def _confirm_reset():
    st.warning(t("reset_dialog_warning"))
    col_yes, col_no = st.columns(2)
    with col_yes:
        if st.button(t("reset_confirm_button"), type="primary", use_container_width=True):
            reset_operational_data()
            st.success(t("reset_success"))
            st.rerun()
    with col_no:
        if st.button(t("cancel"), use_container_width=True):
            st.rerun()


chat_col, panel_col = st.columns([3, 2])
render_column_resize_handle()

# ═══════════════════════════════════════════════════════════════════════════════
# RIGHT PANEL — DASHBOARD
# ═══════════════════════════════════════════════════════════════════════════════
with panel_col:
    with st.container(key="dashboard_zone", height="stretch"):
        render_ticket_summary(get_all_tickets())

        tab_tickets, tab_map, tab_clients, tab_alerts, tab_qa, tab_analytics, tab_llm_usage = (
            st.tabs(
                [
                    t("tab_tickets"),
                    t("map_tab_label"),
                    t("tab_clients"),
                    t("tab_alerts"),
                    t("tab_qa"),
                    t("tab_analytics"),
                    t("tab_llm_usage"),
                ]
            )
        )

        _STATUS_FILTER_OPTIONS = [
            "All",
            "new",
            "open",
            "in_progress",
            "awaiting_client",
            "blocked",
            "in_review",
            "closed",
            "cancelled",
        ]

        # ── tickets ───────────────────────────────────────────────────────────────
        with tab_tickets:
            filter_col, type_col, batch_col = st.columns([2, 1, 1])
            with filter_col:
                status_filter = st.selectbox(
                    t("ticket_filter_label"),
                    _STATUS_FILTER_OPTIONS,
                    format_func=lambda s: t("filter_all") if s == "All" else t(f"status_{s}"),
                    key="ticket_filter",
                    label_visibility="collapsed",
                )
            with type_col:
                type_filter = st.selectbox(
                    t("ticket_type_filter_label"),
                    ["All", "window", "door"],
                    format_func=lambda s: t("filter_all") if s == "All" else t(f"ticket_type_{s}"),
                    key="ticket_type_filter",
                    label_visibility="collapsed",
                )

            tickets = get_all_tickets()
            if status_filter != "All":
                tickets = [tk for tk in tickets if tk.get("ticket_status") == status_filter]
            if type_filter != "All":
                tickets = [tk for tk in tickets if tk.get("ticket_type", "window") == type_filter]

            # Collect all NEW ticket IDs for batch approval (survey requests only).
            new_tickets = [tk for tk in tickets if tk.get("ticket_status") == "new"]
            with batch_col:
                if new_tickets and st.button(
                    t("approve_all_new").format(count=len(new_tickets)), use_container_width=True
                ):
                    approved, failed = 0, []
                    for tk in new_tickets:
                        if not st.session_state.get(f"sel_{tk['id']}", True):
                            continue
                        result = approve_booking_tool.invoke({"booking_id": tk["id"]})
                        if "approved and added to calendar" in result:
                            approved += 1
                        else:
                            failed.append(f"{ticket_code(tk['id'])}: {result}")
                    if approved:
                        st.success(t("approved_n_tickets").format(count=approved))
                    for err in failed:
                        st.error(err)
                    if approved:
                        st.rerun()

            # Pre-fetch pending job completions indexed by booking_id (installation phase).
            pending_jobs = {
                j["booking_id"]: j
                for j in get_job_completions(status="pending")
                if j.get("booking_id")
            }

            if not tickets:
                st.info(
                    t("no_tickets_yet")
                    if status_filter == "All"
                    else t("no_status_tickets").format(status=t(f"status_{status_filter}"))
                )
            else:
                table_rows = [
                    {
                        "Code": ticket_code(tk["id"]),
                        "Product": tk["model"],
                        "Qty": tk.get("quantity", 1),
                        "Client": tk.get("client_name") or tk.get("requested_by", "—"),
                        "Date": tk["date_str"],
                        "Time": tk["time_str"],
                        "Phase": t(f"phase_{tk.get('ticket_phase', 'installation')}"),
                        "Status": t(f"status_{tk.get('ticket_status', 'new')}").upper(),
                    }
                    for tk in tickets
                ]
                # Cell-selection mode instead of row-selection: glide-data-grid
                # only renders the row-selection checkbox gutter for row modes —
                # cell mode lets clicking any cell (in practice, the Code column)
                # select that row directly, with just a highlight, no checkbox column.
                event = st.dataframe(
                    pd.DataFrame(table_rows),
                    hide_index=True,
                    use_container_width=True,
                    on_select="rerun",
                    selection_mode="single-cell",
                    key=f"tickets_table_{status_filter}",
                )
                selected_cells = event.selection.cells if event and event.selection else []
                selected_rows = [row for row, _col in selected_cells]
                # Defensive bound-check: the table can shrink between reruns (e.g.
                # a ticket gets approved/closed) while a stale selection index from
                # a previous render is still held in the widget's state.
                selected_rows = [r for r in selected_rows if r < len(tickets)]

                if not selected_rows:
                    st.caption(t("select_ticket_hint"))
                else:
                    ticket = tickets[selected_rows[0]]
                    ts = ticket.get("ticket_status", "new")
                    phase = ticket.get("ticket_phase", "installation")
                    tcode = ticket_code(ticket["id"])
                    email = ticket.get("client_email", "")

                    with st.container(border=True):
                        render_ticket_card(ticket, show_stock=True)

                        # ── status-specific UI ─────────────────────────────────
                        if ts == "new":
                            sel_key = f"sel_{ticket['id']}"
                            if sel_key not in st.session_state:
                                st.session_state[sel_key] = True
                            st.checkbox(t("include_in_batch"), key=sel_key)

                            col_a, col_r = st.columns(2)
                            with col_a:
                                if st.button(
                                    t("approve_button"),
                                    key=f"t_approve_{ticket['id']}",
                                    use_container_width=True,
                                ):
                                    result = approve_booking_tool.invoke(
                                        {"booking_id": ticket["id"]}
                                    )
                                    if "approved and added to calendar" in result:
                                        st.success(result)
                                        st.rerun()
                                    else:
                                        st.error(result)
                            with col_r:
                                if st.button(
                                    t("reject_button"),
                                    key=f"t_reject_{ticket['id']}",
                                    use_container_width=True,
                                ):
                                    st.session_state[f"reject_open_{ticket['id']}"] = True

                            if st.session_state.get(f"reject_open_{ticket['id']}"):
                                with st.form(key=f"reject_form_{ticket['id']}"):
                                    reason = st.text_area(
                                        t("reason_for_rejection"),
                                        placeholder=t("reason_rejection_placeholder"),
                                        key=f"reject_reason_{ticket['id']}",
                                    )
                                    submitted = st.form_submit_button(
                                        t("confirm_rejection_button"), use_container_width=True
                                    )
                                    if submitted:
                                        if reason.strip():
                                            result = reject_booking_tool.invoke(
                                                {
                                                    "booking_id": ticket["id"],
                                                    "reason": reason.strip(),
                                                }
                                            )
                                            st.warning(result)
                                            st.session_state[f"reject_open_{ticket['id']}"] = False
                                            st.rerun()
                                        else:
                                            st.warning(t("please_provide_reason"))

                        elif ts == "blocked":
                            alerts = get_alerts_for_booking(ticket["id"])
                            if alerts:
                                with st.expander(t("view_alert_expander")):
                                    a = alerts[0]
                                    st.caption(f"{a['sent_by']} — {a['created_at'][:16]}")
                                    st.markdown(a["structured_message"])

                        elif ts == "in_review" and phase == "survey":
                            with st.expander(t("view_survey_measurements")):
                                st.markdown(
                                    f"**{t('dimensions_label')}** {ticket.get('height_cm')}×{ticket.get('width_cm')}×"
                                    f"{ticket.get('depth_cm')} cm  \n"
                                    f"**{t('glass_type_caption')}** {ticket.get('glass_type')}"
                                )
                                if ticket.get("survey_notes"):
                                    st.caption(
                                        t("notes_inline").format(notes=ticket["survey_notes"])
                                    )

                            col_a, col_r = st.columns(2)
                            with col_a:
                                if st.button(
                                    t("approve_and_price"),
                                    key=f"sr_approve_{ticket['id']}",
                                    use_container_width=True,
                                ):
                                    result = approve_survey_estimate_tool.invoke(
                                        {"booking_id": ticket["id"]}
                                    )
                                    if "computed and saved" in result:
                                        st.success(result)
                                    else:
                                        st.error(result)
                                    st.rerun()
                            with col_r:
                                if st.button(
                                    t("reject_button"),
                                    key=f"sr_reject_{ticket['id']}",
                                    use_container_width=True,
                                ):
                                    result = reject_survey_estimate_tool.invoke(
                                        {"booking_id": ticket["id"]}
                                    )
                                    st.warning(result)
                                    st.rerun()

                        elif ts == "in_review" and phase == "installation":
                            job = pending_jobs.get(ticket["id"])
                            if job:
                                with st.expander(t("view_completion_report")):
                                    st.markdown(job["structured_report"])

                                email_key = f"job_email_{job['id']}"
                                if email_key not in st.session_state:
                                    st.session_state[email_key] = job.get("client_email") or email
                                client_email_val = st.text_input(
                                    t("client_email_thankyou_label"),
                                    value=st.session_state[email_key],
                                    key=f"job_email_input_{job['id']}",
                                )
                                st.session_state[email_key] = client_email_val

                                col_a, col_r = st.columns(2)
                                with col_a:
                                    if st.button(
                                        t("approve_and_email"),
                                        key=f"jr_approve_{job['id']}",
                                        use_container_width=True,
                                    ):
                                        username = st.session_state.get("username", "manager")
                                        resolve_job_completion(job["id"], "approved", username)
                                        if client_email_val and "@" in client_email_val:
                                            try:
                                                llm = get_llm(temperature=0.3, max_tokens=400)
                                                body = llm.invoke(
                                                    _THANKYOU_EMAIL_TEMPLATE.format(
                                                        structured_report=job["structured_report"]
                                                    )
                                                ).content.strip()
                                                sent = send_email(
                                                    to=client_email_val,
                                                    subject="Thank You — Your Installation is Complete",
                                                    body=body,
                                                )
                                                email_status = (
                                                    t("email_status_sent")
                                                    if sent
                                                    else t("email_status_failed")
                                                )
                                                msg = t("email_sent_msg").format(
                                                    status=email_status, email=client_email_val
                                                )
                                            except Exception as exc:
                                                msg = t("email_error_msg").format(error=exc)
                                        else:
                                            msg = t("approved_no_email_msg")
                                        st.success(msg)
                                        st.rerun()
                                with col_r:
                                    if st.button(
                                        t("reject_button"),
                                        key=f"jr_reject_{job['id']}",
                                        use_container_width=True,
                                    ):
                                        username = st.session_state.get("username", "manager")
                                        resolve_job_completion(job["id"], "rejected", username)
                                        st.warning(t("report_rejected_msg"))
                                        st.rerun()

                        elif ts == "awaiting_client":
                            st.caption(
                                t("estimate_sent_caption").format(
                                    price=f"{ticket.get('estimated_price', 0):.2f}"
                                )
                            )

                        # ── edit ticket details (e.g. after a call with the client) ──
                        with st.expander(t("edit_ticket_expander")):
                            with st.form(key=f"edit_form_{ticket['id']}"):
                                e_name = st.text_input(
                                    t("client_name_label"), value=ticket.get("client_name", "")
                                )
                                e_email = st.text_input(
                                    t("client_email_label"), value=ticket.get("client_email", "")
                                )
                                e_phone = st.text_input(
                                    t("client_phone_label"), value=ticket.get("client_phone", "")
                                )
                                e_address = st.text_input(
                                    t("client_address_label"),
                                    value=ticket.get("client_address", ""),
                                )
                                col_d, col_t, col_q = st.columns(3)
                                e_date = col_d.text_input(
                                    t("booking_form_date"), value=ticket.get("date_str", "")
                                )
                                e_time = col_t.text_input(
                                    t("booking_form_time"), value=ticket.get("time_str", "")
                                )
                                e_qty = col_q.number_input(
                                    t("booking_form_quantity"),
                                    min_value=1,
                                    step=1,
                                    value=int(ticket.get("quantity", 1)),
                                )
                                e_notes = st.text_area(
                                    t("notes_field_label"), value=ticket.get("notes", "")
                                )
                                save = st.form_submit_button(
                                    t("save_changes_button"), use_container_width=True
                                )

                                if save:
                                    try:
                                        datetime.datetime.strptime(e_date, "%Y-%m-%d")
                                        datetime.datetime.strptime(e_time, "%H:%M")
                                    except ValueError:
                                        st.error(t("date_time_format_error"))
                                    else:
                                        username = st.session_state.get("username", "manager")
                                        updated = update_ticket_details(
                                            ticket["id"],
                                            username,
                                            client_name=e_name,
                                            client_email=e_email,
                                            client_phone=e_phone,
                                            client_address=e_address,
                                            date_str=e_date,
                                            time_str=e_time,
                                            quantity=int(e_qty),
                                            notes=e_notes,
                                        )
                                        if updated:
                                            st.success(
                                                t("ticket_updated_msg").format(
                                                    code=ticket_code(ticket["id"])
                                                )
                                            )
                                            st.rerun()
                                        else:
                                            st.error(t("could_not_update_ticket"))

        # ── map ───────────────────────────────────────────────────────────────────
        with tab_map:
            render_ticket_map(get_all_tickets())

        # ── clients ───────────────────────────────────────────────────────────────
        with tab_clients:
            search = st.text_input(
                t("search_label"),
                key="clients_search",
                placeholder=t("search_by_name_placeholder"),
                label_visibility="collapsed",
            )
            clients = sorted(get_clients(), key=lambda c: c["full_name"].lower())
            if search.strip():
                clients = [c for c in clients if search.strip().lower() in c["full_name"].lower()]

            if not clients:
                st.info(t("no_clients_yet") if not search.strip() else t("no_clients_match"))
            else:
                for c in clients:
                    with st.expander(c["full_name"]):
                        st.markdown(f"**{t('email_label_bold')}** {c['email']}")
                        if c.get("phone"):
                            st.caption(t("phone_label_inline").format(phone=c["phone"]))
                        if c.get("address"):
                            st.caption(t("address_inline").format(address=c["address"]))
                        st.caption(t("client_since_label").format(date=c["created_at"][:10]))

        # ── alerts ────────────────────────────────────────────────────────────────
        with tab_alerts:
            unread_alerts = get_alerts(unread_only=True)
            all_alerts = get_alerts(unread_only=False)

            if not all_alerts:
                st.info(t("no_alerts_yet"))
            else:
                if unread_alerts:
                    for alert in unread_alerts:
                        with st.container(border=True):
                            header = f"**{alert['sent_by']}** — {alert['created_at'][:16]}"
                            if alert.get("booking_id"):
                                header += f" | {t('ticket_ref_inline').format(code=ticket_code(alert['booking_id']))}"
                            st.markdown(header)
                            if alert["job_reference"]:
                                st.caption(t("ref_label_inline").format(ref=alert["job_reference"]))
                            st.markdown(alert["structured_message"])
                    if st.button(t("mark_all_read_button"), use_container_width=True):
                        mark_alerts_read([a["id"] for a in unread_alerts])
                        st.rerun()

                with st.expander(t("all_alerts_expander").format(count=len(all_alerts))):
                    for alert in all_alerts:
                        read_label = t("read_label") if alert["is_read"] else t("status_new")
                        cap = f"#{alert['id']} [{read_label}] {alert['sent_by']} — {alert['created_at'][:16]}"
                        if alert.get("booking_id"):
                            cap += f" | {ticket_code(alert['booking_id'])}"
                        st.caption(cap)
                        with st.container(border=True):
                            st.markdown(alert["structured_message"])

        # ── QA cache pending ──────────────────────────────────────────────────────
        with tab_qa:
            qa_pending = qa_get_pending()
            if not qa_pending:
                st.info(t("no_qa_pending"))
            else:
                for item in qa_pending:
                    with st.container(border=True):
                        st.markdown(
                            f"**{t('qa_question_label').format(role=item['source_role'])}** {item['question']}"
                        )
                        st.caption(t("submitted_label").format(date=item["created_at"][:10]))
                        with st.expander(t("view_answer_expander")):
                            st.markdown(item["answer"])
                        col_add, col_skip, _ = st.columns([2, 2, 4])
                        with col_add:
                            if st.button(
                                t("add_to_cache_button"),
                                key=f"qa_approve_{item['id']}",
                                use_container_width=True,
                            ):
                                username = st.session_state.get("username", "manager")
                                result = qa_approve(item["id"], approved_by=username)
                                st.success(t("added_msg") if result else t("skipped_duplicate_msg"))
                                st.rerun()
                        with col_skip:
                            if st.button(
                                t("skip_button"),
                                key=f"qa_reject_{item['id']}",
                                use_container_width=True,
                            ):
                                qa_reject(item["id"])
                                st.rerun()

        def _period_format(p):
            return {"day": t("period_day"), "week": t("period_week"), "month": t("period_month")}[p]

        # ── business analytics (tickets/revenue/hours) ──────────────────────────────
        with tab_analytics:
            biz_period = st.radio(
                t("period_label"),
                ["day", "week", "month"],
                format_func=_period_format,
                horizontal=True,
                key="biz_period",
            )
            biz = get_business_analytics(biz_period)
            render_stat_tiles(
                [
                    {"label": t("closed_tickets_label"), "value": f"{biz['closed_count']:,}"},
                    {"label": t("total_revenue_label"), "value": f"€{biz['total_revenue']:.2f}"},
                    {"label": t("hours_worked_label"), "value": f"{biz['total_hours']:.1f}"},
                ]
            )
            # by_day is always fully populated (one entry per day in the window,
            # zero-filled) so the chart shows a consistent 1/7/30-day axis.
            chart_df = pd.DataFrame(
                [
                    {"Date": day, t("total_revenue_label"): v["revenue"]}
                    for day, v in biz["by_day"].items()
                ]
            ).set_index("Date")
            st.bar_chart(chart_df)

        # ── LLM usage analytics ──────────────────────────────────────────────────
        with tab_llm_usage:
            period = st.radio(
                t("period_label"),
                ["day", "week", "month"],
                format_func=_period_format,
                horizontal=True,
                key="usage_period",
            )
            stats = get_usage_stats(period)
            totals = stats["totals"]
            by_role = stats["by_role"]
            render_stat_tiles(
                [
                    {"label": t("cost_label"), "value": f"${totals['cost_usd']:.4f}"},
                    {"label": t("requests_label"), "value": f"{totals['requests']:,}"},
                    {
                        "label": t("tokens_label"),
                        "value": f"{totals['input_tokens'] + totals['output_tokens']:,}",
                    },
                ]
            )
            if by_role:
                for r, data in by_role.items():
                    st.caption(
                        f"**{r.capitalize()}**: {data['requests']} req · "
                        f"${data['cost_usd']:.4f} · "
                        f"{data['input_tokens'] + data['output_tokens']:,} tok"
                    )
            else:
                st.caption(t("no_usage_period"))

            st.divider()
            if st.button(t("reset_all_button"), use_container_width=True):
                _confirm_reset()

# ═══════════════════════════════════════════════════════════════════════════════
# LEFT PANEL — CHAT
# ═══════════════════════════════════════════════════════════════════════════════
with chat_col, st.container(key="chat_zone", height="stretch"):
    st.subheader(t("assistant_label"))
    selected_model = render_sidebar("manager")
    agent = get_agent(selected_model, "manager")

    render_chat_history(agent)

    user_prompt = render_hitl_and_input(
        agent,
        selected_model,
        prompt_hint=t("manager_sub"),
    )

    if user_prompt:
        handle_user_turn(user_prompt, agent, selected_model)

    render_chat_autoscroll()
