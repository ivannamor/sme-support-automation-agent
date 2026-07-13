"""Client chat page."""

import sys
from pathlib import Path
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_DIR = Path(__file__).resolve().parents[1]
for p in (str(PROJECT_ROOT), str(APP_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from ui_shared import (
    require_auth, get_agent, init_chat_session, t,
    render_sidebar, render_chat_history,
    render_hitl_and_input, handle_user_turn,
    render_page_header, render_booking_form,
    render_client_pending_estimates, render_client_installation_status,
    render_chat_autoscroll,
)

st.set_page_config(page_title="Client Portal", page_icon=None, layout="wide")

require_auth(allowed_roles=["client"])
init_chat_session("client")

render_page_header(t("client_title"))

selected_model = render_sidebar("client")
agent          = get_agent(selected_model, "client")

with st.container(key="chat_zone", height="stretch"):
    render_client_pending_estimates()
    render_client_installation_status()

    render_chat_history(agent)

    render_booking_form(agent, selected_model)

    user_prompt = render_hitl_and_input(
        agent, selected_model,
        prompt_hint=t("client_sub"),
    )

    if user_prompt:
        handle_user_turn(user_prompt, agent, selected_model)

    render_chat_autoscroll()
