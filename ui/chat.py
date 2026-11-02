"""Chat rendering and the handling of each turn."""

import contextlib
import re
import time
from urllib.parse import quote as _url_quote

import streamlit as st
import streamlit.components.v1 as components

from backend.assistant import TurnResult
from backend.qa_cache import lookup as _cache_lookup
from backend.qa_cache import submit_for_review as qa_submit
from ui.i18n import t
from ui.intents import _detect_booking_intent, _detect_hitl_intent
from ui.session import _get, _k, _set, bind_request_context

_CACHE_ELIGIBLE_TOOLS = {"search_technical_manuals", "general_door_knowledge_tool"}
_CACHE_INELIGIBLE_TOOLS = {
    "inventory_api_tool",
    "book_installation_tool",
    "check_calendar_availability_tool",
    "price_estimate_tool",
    "request_site_survey_tool",
    "calculate_travel_time_tool",
    "complete_job_tool",
    "send_alert_to_manager_tool",
    "send_estimate_to_client_tool",
    "update_installation_costs_tool",
    "request_inventory_reservation_tool",
    "submit_survey_measurements_tool",
    "respond_to_estimate_tool",
    "find_my_tickets_by_email_tool",
    "approve_survey_estimate_tool",
    "reject_survey_estimate_tool",
}


def _is_cache_eligible(tool_calls: list) -> bool:
    if not tool_calls:
        return False
    names = {tc.get("tool") for tc in tool_calls}
    return bool(names & _CACHE_ELIGIBLE_TOOLS) and not bool(names & _CACHE_INELIGIBLE_TOOLS)


def render_chat_autoscroll() -> None:
    """
    Jumps the chat pane to its latest message on every rerun (new user
    message, new assistant reply, HITL confirmation, etc.) — same
    reach-into-the-parent-document trick as render_column_resize_handle,
    since .st-key-chat_zone's scroll position isn't anything Streamlit
    itself exposes. Without this, a reply that's taller than the pane
    leaves the view sitting wherever it was (usually the top, from before
    the message was sent) instead of showing the message that just arrived.

    Skipped while the booking form is open: the form is often taller than
    the pane, and jumping to its bottom hides its own title and the
    name/address fields above the fold instead of showing what the client
    actually needs to see and fill in first.
    """
    if _get("show_booking_form"):
        return
    components.html(
        """
        <script>
        (function() {
            const doc = window.parent.document;
            const zone = doc.querySelector('.st-key-chat_zone');
            if (zone) {
                zone.scrollTop = zone.scrollHeight;
            }
        })();
        </script>
        """,
        height=0,
    )


def _avatar_svg(bg_hex: str, fg_hex: str, glyph: str) -> str:
    """Small inline circular avatar (data URI) — a minimal, on-brand
    alternative to Streamlit's default chat icons."""
    svg = (
        '<svg xmlns="http://www.w3.org/2000/svg" width="32" height="32" viewBox="0 0 32 32">'
        f'<circle cx="16" cy="16" r="16" fill="{bg_hex}"/>'
        f'<text x="16" y="21" font-family="Inter, sans-serif" font-size="12" font-weight="600" '
        f'fill="{fg_hex}" text-anchor="middle">{glyph}</text>'
        "</svg>"
    )
    return "data:image/svg+xml;utf8," + _url_quote(svg)


_ASSISTANT_AVATAR = _avatar_svg("#14532D", "#F0F0EE", "AI")


def _initials(name: str) -> str:
    parts = [p for p in re.split(r"[\s_.-]+", name) if p]
    if not parts:
        return "U"
    if len(parts) == 1:
        return parts[0][:2].upper()
    return (parts[0][0] + parts[1][0]).upper()


def _chat_avatar(role: str) -> str:
    if role == "assistant":
        return _ASSISTANT_AVATAR
    username = st.session_state.get("username", "U")
    return _avatar_svg("#3F3F3F", "#F0F0EE", _initials(username))


def render_chat_history(agent, height: int = 560):
    """
    Renders chat messages. No fixed-height scroll container of its own —
    the chat column itself (app/ui_shared.py's `.st-key-chat_zone`-equivalent,
    the plain st.column) is the one scrollable region for the whole left
    pane (header/HITL banner/messages/input together), via CSS in
    assets/theme.css, so there's exactly one scrollbar for chat, not a nested
    scrollbar-within-a-scrollbar. `height` is accepted but unused, kept so
    existing call sites don't need updating.
    """
    audio_store = _get("audio_store", {})
    autoplay_idx = _get("autoplay_idx", -1)
    voice_mode = _get("voice_mode", False)

    with st.container(border=False):
        for msg_idx, message in enumerate(_get("messages", [])):
            with st.chat_message(message["role"], avatar=_chat_avatar(message["role"])):
                st.markdown(message["content"])

                if message["role"] == "assistant":
                    _render_answer_metadata(message)

                    # Voice audio playback
                    if voice_mode and msg_idx in audio_store:
                        should_autoplay = msg_idx == autoplay_idx
                        if should_autoplay:
                            _set("autoplay_idx", -1)
                        st.audio(audio_store[msg_idx], format="audio/mp3", autoplay=should_autoplay)
                        if st.button(t("replay"), key=_k(f"replay_{msg_idx}")):
                            _set("autoplay_idx", msg_idx)
                            st.rerun()


def render_hitl_and_input(agent, selected_model: str, prompt_hint: str = "Type your message..."):
    pending = _get("pending_interrupt")
    if pending:
        st.warning(f"**{t('action_required')}:** {pending.get('summary', '')}")
        col_ok, col_cancel, _ = st.columns([2, 2, 6])
        with col_ok:
            if st.button(t("confirm"), type="primary", key=_k("hitl_confirm")):
                _stream_resume(agent, "confirmed", selected_model)
        with col_cancel:
            if st.button(t("cancel"), key=_k("hitl_cancel")):
                _stream_resume(agent, "rejected", selected_model)
        st.caption(t("confirm_cancel_hint"))

    if _get("voice_mode"):
        # Audio input always available — works for normal messages AND HITL confirmations.
        audio_key = _k(f"audio_input_{_get('total_requests', 0)}")
        audio_data = st.audio_input(t("recording_hint"), key=audio_key)
        if audio_data:
            from backend.voice import transcribe_audio

            with st.spinner(t("transcribing")):
                try:
                    text = transcribe_audio(audio_data.read())
                    if text:
                        st.caption(f'"{text}"')
                        return text
                    st.warning(t("no_speech_detected"))
                except Exception as exc:
                    st.error(t("transcription_failed").format(error=exc))
        return None

    # Chat input always enabled — HITL intent is resolved in handle_user_turn.
    return st.chat_input(prompt_hint)


def _stream_resume(agent, decision: str, selected_model: str):
    bind_request_context()
    result = TurnResult()
    with st.chat_message("assistant", avatar=_chat_avatar("assistant")):
        raw_text = st.write_stream(agent.stream_resume(decision, _get("thread_id"), result))
        _save_assistant_turn(result, raw_text)
    st.rerun()


def handle_user_turn(user_prompt: str, agent, selected_model: str, is_synthetic: bool = False):
    """
    is_synthetic: True for messages generated by the app itself (the booking
    form's consolidated submission, an installer's completion-notes form) —
    never organic user chat text. These skip the semantic cache lookup (no
    reason to embed a one-off transactional message full of PII into the
    cache) and the booking-intent heuristic below (a synthetic booking
    submission would otherwise re-trigger "show the booking form").
    """
    bind_request_context()

    if _get("total_requests", 0) >= 50:
        st.error(t("session_limit_reached"))
        st.stop()

    if time.time() - _get("last_request_time", 0.0) < 2:
        st.warning(t("please_wait_between_messages"))
        st.stop()

    _set("last_request_time", time.time())
    _set("total_requests", _get("total_requests", 0) + 1)

    # ── HITL text/voice intent check ─────────────────────────────────────────
    pending = _get("pending_interrupt")
    if pending:
        intent = _detect_hitl_intent(user_prompt)
        if intent:
            # Show user's confirmation/cancellation in chat history, then resume.
            msgs = _get("messages", [])
            msgs.append({"role": "user", "content": user_prompt})
            _set("messages", msgs)
            with st.chat_message("user", avatar=_chat_avatar("user")):
                st.markdown(user_prompt)
            _stream_resume(agent, intent, selected_model)
            return
        # Ambiguous message while HITL is pending — prompt without touching the graph.
        with st.chat_message("assistant", avatar=_chat_avatar("assistant")):
            st.markdown(t("hitl_pending_prompt"))
        return

    # ── Booking-intent heuristic (client role, organic messages only) ────────
    if (
        not is_synthetic
        and st.session_state.get("role") == "client"
        and _detect_booking_intent(user_prompt)
    ):
        msgs = _get("messages", [])
        msgs.append({"role": "user", "content": user_prompt})
        msgs.append(
            {
                "role": "assistant",
                "content": t("booking_intent_redirect"),
                "source_info": None,
                "tools": [],
                "run_id": None,
                "is_general_llm": False,
                "is_cache_hit": False,
            }
        )
        _set("messages", msgs)
        st.session_state[_k("show_booking_form")] = True
        st.rerun()
        return

    # ── Normal turn ───────────────────────────────────────────────────────────
    messages = _get("messages", [])
    messages.append({"role": "user", "content": user_prompt})
    _set("messages", messages)

    with st.chat_message("user", avatar=_chat_avatar("user")):
        st.markdown(user_prompt)

    # Cache check (primary path) — skipped for synthetic/transactional messages.
    cached_answer = None if is_synthetic else _cache_lookup(user_prompt)
    if cached_answer:
        with st.chat_message("assistant", avatar=_chat_avatar("assistant")):
            st.markdown(cached_answer)
            st.success(t("cache_hit"))
        msgs = _get("messages", [])
        msgs.append(
            {
                "role": "assistant",
                "content": cached_answer,
                "source_info": None,
                "tools": [],
                "run_id": None,
                "is_general_llm": False,
                "is_cache_hit": True,
            }
        )
        _set("messages", msgs)
        st.rerun()
        return

    result = TurnResult()
    with st.chat_message("assistant", avatar=_chat_avatar("assistant")):
        raw_text = st.write_stream(agent.stream_ask(user_prompt, _get("thread_id"), result))
        _save_assistant_turn(result, raw_text, user_question=user_prompt)

    st.rerun()


def _catalogue_page_link(source_info: str):
    if not source_info:
        return None
    for chunk in source_info.split(" ; "):
        if "catalogue" not in chunk.lower():
            continue
        fname = re.search(r"(\S+\.pdf)", chunk)
        page = re.search(r"pages?\s+(\d+)", chunk)
        if fname and page:
            return fname.group(1), int(page.group(1))
    return None


def _render_answer_metadata(message: dict):
    if message.get("is_cache_hit"):
        st.success(t("cache_hit"))
    if message.get("source_info") and not message.get("is_cache_hit"):
        # The raw chunk/source dump (filenames, page ranges, collection
        # names) used to be shown verbatim here — removed for a cleaner
        # look; the answer itself now ends with the relevant webpage's
        # full URL when applicable (see generate_answer's prompt), and the
        # catalogue-PDF deep link below still works for PDF-sourced answers.
        cat = _catalogue_page_link(message["source_info"])
        if cat:
            filename, page_num = cat
            pdf_url = f"/app/static/catalogue/{_url_quote(filename)}#page={page_num}"
            st.markdown(
                f'<a href="{pdf_url}" target="_blank" style="font-size:0.8rem;color:#14532D;">'
                f"{t('open_catalogue')} {page_num}</a>",
                unsafe_allow_html=True,
            )
    if message.get("is_general_llm"):
        st.info(t("general_llm"))


def _save_assistant_turn(result: TurnResult, raw_text: str, user_question: str = ""):
    captured_tools = result.tool_calls
    run_id = result.run_id
    _set("pending_interrupt", result.interrupt)

    # The agent already logged the turn to usage_log; this is the session's running total.
    _set("total_cost", _get("total_cost", 0.0) + result.cost_usd)
    _set("last_query_cost", result.cost_usd)
    _set("total_tokens", _get("total_tokens", 0) + result.usage["total_tokens"])

    is_general_llm = any(tc.get("tool") == "general_door_knowledge_tool" for tc in captured_tools)
    is_cache_hit = any("CACHE_HIT" in str(tc.get("output", "")) for tc in captured_tools)

    raw_text = raw_text.replace("GENERAL_LLM_SPLIT", "").replace("CACHE_HIT", "").strip()

    if "SOURCE_DATA_SPLIT" in raw_text:
        main_answer, source_info = raw_text.split("SOURCE_DATA_SPLIT", 1)
        source_info = source_info.strip()
    else:
        main_answer, source_info = raw_text, None

    if not source_info:
        for tc in captured_tools:
            if tc.get("tool") == "search_technical_manuals":
                tool_out = str(tc.get("output", ""))
                if "SOURCE_DATA_SPLIT" in tool_out:
                    source_info = (
                        tool_out.split("SOURCE_DATA_SPLIT", 1)[1].replace("CACHE_HIT", "").strip()
                    )
                break

    main_answer = main_answer.strip()
    pending_interrupt = result.interrupt
    if not main_answer and pending_interrupt and pending_interrupt.get("summary"):
        # The graph paused for confirmation without saying anything first — use the
        # interrupt's summary as the bubble content so it's part of the permanent
        # chat history, not just the transient warning banner shown while pending.
        main_answer = pending_interrupt["summary"]

    msg = {
        "role": "assistant",
        "content": main_answer,
        "source_info": source_info,
        "tools": captured_tools,
        "run_id": run_id,
        "is_general_llm": is_general_llm,
        "is_cache_hit": is_cache_hit,
    }
    _render_answer_metadata(msg)

    # Generate TTS and store for post-rerun playback.
    # Do NOT call st.audio() here — st.rerun() would kill it immediately.
    if _get("voice_mode") and main_answer.strip():
        from backend.voice import text_to_speech

        try:
            audio_bytes = text_to_speech(main_answer)
            if audio_bytes:
                current_msgs = _get("messages", [])
                new_msg_idx = len(current_msgs)  # index this msg will occupy
                audio_store = _get("audio_store", {})
                audio_store[new_msg_idx] = audio_bytes
                _set("audio_store", audio_store)
                _set("autoplay_idx", new_msg_idx)
        except Exception:
            st.caption(t("audio_error"))

    if user_question and not is_cache_hit and _is_cache_eligible(captured_tools):
        with contextlib.suppress(Exception):
            qa_submit(
                question=user_question,
                answer=main_answer.strip(),
                source_role=st.session_state.get("role", "unknown"),
            )

    if captured_tools:
        history = _get("tool_calls_history", [])
        history.extend(captured_tools)
        _set("tool_calls_history", history)

    messages = _get("messages", [])
    messages.append(msg)
    _set("messages", messages)
