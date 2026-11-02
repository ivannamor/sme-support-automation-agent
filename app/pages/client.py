"""Client chat page."""

import streamlit as st

from ui import (
    get_agent,
    handle_user_turn,
    init_chat_session,
    render_booking_form,
    render_chat_autoscroll,
    render_chat_history,
    render_client_installation_status,
    render_client_pending_estimates,
    render_hitl_and_input,
    render_page_header,
    render_sidebar,
    require_auth,
    t,
)

st.set_page_config(page_title="Client Portal", page_icon=None, layout="wide")

require_auth(allowed_roles=["client"])
init_chat_session("client")

render_page_header(t("client_title"))

selected_model = render_sidebar("client")
agent = get_agent(selected_model, "client")

with st.container(key="chat_zone", height="stretch"):
    render_client_pending_estimates()
    render_client_installation_status()

    render_chat_history(agent)

    render_booking_form(agent, selected_model)

    user_prompt = render_hitl_and_input(
        agent,
        selected_model,
        prompt_hint=t("client_sub"),
    )

    if user_prompt:
        handle_user_turn(user_prompt, agent, selected_model)

    render_chat_autoscroll()
