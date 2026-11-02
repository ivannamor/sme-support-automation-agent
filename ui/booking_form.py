"""The structured site-survey request form."""

import contextlib
import datetime

import streamlit as st

from backend.database import get_inventory
from ui.chat import handle_user_turn
from ui.i18n import _month_name, t
from ui.session import _get, _k

_BOOKING_FORM_FIELD_KEYS = [
    "booking_form_client_name",
    "booking_form_client_address",
    "booking_form_model",
    "booking_form_quantity",
    "booking_form_day",
    "booking_form_month",
    "booking_form_hour",
    "booking_form_minute",
    "booking_form_client_email",
    "booking_form_client_phone",
]


def _clear_booking_form_fields() -> None:
    """
    Clears every booking-form field from session state so the form starts
    blank the next time it opens. Without this, a value the client never
    re-entered (most dangerously the email — the natural key upsert_client
    matches on) silently carries over from a previous, unrelated booking in
    the same browser session and overwrites that earlier client's row
    instead of creating a new one.
    """
    for field in _BOOKING_FORM_FIELD_KEYS:
        st.session_state.pop(_k(field), None)


def render_booking_form(agent, selected_model: str) -> None:
    """
    Compact structured booking request, shown inline in the conversation only
    after the client has expressed booking intent (see _detect_booking_intent
    in handle_user_turn, which sets the show flag and reruns) — there is no
    standalone "open form" button anywhere else on the page.

    Fields are auto-filled once, the first time the form opens, from whatever
    has already been said in the conversation (extract_booking_fields run over
    the chat transcript) — there is no separate free-text/voice box, since the
    standard chat already covers that; anything not mentioned is simply left
    for manual entry, and anything auto-filled can still be edited manually
    before sending.

    Submission feeds one consolidated message into the normal agent turn
    (handle_user_turn) — the agent still runs its own HITL confirmation
    exactly as the conversational booking path does.
    """
    show_key = _k("show_booking_form")
    if not st.session_state.get(show_key):
        return

    model_options = sorted({item["model"] for item in get_inventory()})

    autofill_key = _k("booking_form_autofilled")
    if not st.session_state.get(autofill_key):
        st.session_state[autofill_key] = True
        transcript = "\n".join(
            f"{m['role']}: {m['content']}" for m in _get("messages", []) if m.get("content")
        )
        if transcript.strip():
            from tools import extract_booking_fields

            try:
                extracted = extract_booking_fields(transcript)
            except Exception:
                extracted = {}

            if extracted.get("client_name"):
                st.session_state[_k("booking_form_client_name")] = str(extracted["client_name"])
            if extracted.get("client_address"):
                st.session_state[_k("booking_form_client_address")] = str(
                    extracted["client_address"]
                )
            if extracted.get("model"):
                candidate = str(extracted["model"]).upper().strip()
                if candidate in model_options:
                    st.session_state[_k("booking_form_model")] = candidate
            if extracted.get("quantity"):
                with contextlib.suppress(TypeError, ValueError):
                    st.session_state[_k("booking_form_quantity")] = max(
                        1, int(extracted["quantity"])
                    )
            if extracted.get("date_str"):
                try:
                    d = datetime.datetime.strptime(str(extracted["date_str"]), "%Y-%m-%d").date()
                    st.session_state[_k("booking_form_day")] = d.day
                    st.session_state[_k("booking_form_month")] = d.month
                except ValueError:
                    pass
            if extracted.get("time_str"):
                try:
                    tm = datetime.datetime.strptime(str(extracted["time_str"]), "%H:%M").time()
                    st.session_state[_k("booking_form_hour")] = tm.hour
                    st.session_state[_k("booking_form_minute")] = min(
                        [0, 15, 30, 45], key=lambda m: abs(m - tm.minute)
                    )
                except ValueError:
                    pass
            if extracted.get("client_email"):
                st.session_state[_k("booking_form_client_email")] = str(extracted["client_email"])
            if extracted.get("client_phone"):
                st.session_state[_k("booking_form_client_phone")] = str(extracted["client_phone"])
            st.rerun()

    today = datetime.date.today()

    with st.container(border=True):
        st.markdown(f"**{t('booking_form_title')}**")

        client_name = st.text_input(t("booking_form_name"), key=_k("booking_form_client_name"))
        client_address = st.text_input(
            t("booking_form_address"), key=_k("booking_form_client_address")
        )

        col_model, col_qty = st.columns([3, 1])
        with col_model:
            if model_options:
                model = st.selectbox(
                    t("booking_form_model"), model_options, key=_k("booking_form_model")
                )
            else:
                model = st.text_input(t("booking_form_model"), key=_k("booking_form_model"))
        with col_qty:
            quantity = st.number_input(
                t("booking_form_quantity"), min_value=1, step=1, key=_k("booking_form_quantity")
            )

        st.caption(t("booking_form_date"))
        col_day, col_month, col_hour, col_minute = st.columns(4)
        with col_day:
            day = st.selectbox(
                t("booking_form_day"), list(range(1, 32)), key=_k("booking_form_day")
            )
        with col_month:
            month = st.selectbox(
                t("booking_form_month"),
                list(range(1, 13)),
                format_func=_month_name,
                key=_k("booking_form_month"),
            )
        with col_hour:
            hour = st.selectbox(
                t("booking_form_hour"), list(range(0, 24)), key=_k("booking_form_hour")
            )
        with col_minute:
            minute = st.selectbox(
                t("booking_form_minute"),
                [0, 15, 30, 45],
                format_func=lambda m: f"{m:02d}",
                key=_k("booking_form_minute"),
            )

        if st.button(f"⏱ {t('booking_form_asap_button')}", key=_k("booking_form_asap")):
            tomorrow = today + datetime.timedelta(days=1)
            st.session_state[_k("booking_form_day")] = tomorrow.day
            st.session_state[_k("booking_form_month")] = tomorrow.month
            st.session_state[_k("booking_form_hour")] = 9
            st.session_state[_k("booking_form_minute")] = 0
            st.rerun()

        col_email, col_phone = st.columns(2)
        with col_email:
            client_email = st.text_input(
                t("booking_form_email"), key=_k("booking_form_client_email")
            )
        with col_phone:
            client_phone = st.text_input(
                t("booking_form_phone"), key=_k("booking_form_client_phone")
            )

        col_submit, col_cancel = st.columns([3, 1])
        with col_submit:
            submitted = st.button(
                t("booking_form_submit"),
                type="primary",
                key=_k("booking_form_submit"),
                use_container_width=True,
            )
        with col_cancel:
            if st.button(t("cancel"), key=_k("booking_form_close"), use_container_width=True):
                st.session_state[show_key] = False
                st.session_state[autofill_key] = False
                _clear_booking_form_fields()
                st.rerun()

        if submitted:
            required = {
                t("booking_form_name"): client_name,
                t("booking_form_address"): client_address,
                t("booking_form_email"): client_email,
                t("booking_form_phone"): client_phone,
            }
            missing = [label for label, value in required.items() if not value.strip()]

            if missing:
                st.error(f"{t('booking_form_missing')}: {', '.join(missing)}")
                return

            try:
                year = (
                    today.year if datetime.date(today.year, month, day) >= today else today.year + 1
                )
                survey_date = datetime.date(year, month, day)
            except ValueError:
                st.error(t("booking_form_invalid_date"))
                return

            date_str = survey_date.strftime("%Y-%m-%d")
            time_str = f"{hour:02d}:{minute:02d}"

            synthetic_message = t("booking_form_synthetic_message").format(
                name=client_name,
                address=client_address,
                model=model,
                quantity=quantity,
                date=date_str,
                time=time_str,
                email=client_email,
                phone=client_phone,
            )
            st.session_state[show_key] = False
            st.session_state[autofill_key] = False
            _clear_booking_form_fields()
            handle_user_turn(synthetic_message, agent, selected_model, is_synthetic=True)
