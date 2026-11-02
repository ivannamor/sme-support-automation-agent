"""Session state, the auth gate and the agent cache.

All per-chat session state is namespaced by role so that switching between
pages does not bleed conversation history or thread IDs across roles.
Global state (role, username, ux_lang) stays un-prefixed."""

import uuid

import streamlit as st

from backend.assistant import initialize_agent
from backend.log_context import role_var, username_var, verified_client_email_var
from ui.i18n import t
from ui.layout import inject_global_css

ROLE_LABEL = {"client": "Client", "installer": "Installer", "manager": "Manager"}


_CHAT_DEFAULTS = {
    "messages": [],
    "tool_calls_history": [],
    "total_cost": 0.0,
    "total_tokens": 0,
    "last_query_cost": 0.0,
    "total_requests": 0,
    "last_request_time": 0.0,
    "thread_id": None,
    "pending_interrupt": None,
    "audio_store": {},  # {msg_idx: mp3_bytes} for voice playback
    "autoplay_idx": -1,  # index of the message that should autoplay on next render
}


def _k(base_key: str) -> str:
    role = st.session_state.get("current_page_role", "unknown")
    return f"{role}__{base_key}"


def _get(key: str, default=None):
    return st.session_state.get(_k(key), default)


def _set(key: str, value):
    st.session_state[_k(key)] = value


def require_auth(allowed_roles: list[str] | None = None):
    inject_global_css()
    role = st.session_state.get("role")
    if not role:
        st.warning(t("please_login"))
        if st.button(t("go_login")):
            st.switch_page("pages/login.py")
        st.stop()
    if allowed_roles and role not in allowed_roles:
        st.error(t("access_denied").format(roles=", ".join(allowed_roles)))
        st.stop()
    bind_request_context()


def bind_request_context() -> None:
    """
    Copies this session's identity into the ContextVars tools read
    (username_var, verified_client_email_var) and usage accounting reads
    (role_var). Called on every page run via require_auth(), so direct tool
    and LLM calls from dashboard buttons get the right identity too — not
    only agent turns.
    """
    username_var.set(st.session_state.get("username", "unknown"))
    role_var.set(st.session_state.get("role"))
    verified_client_email_var.set(st.session_state.get("verified_client_email", ""))


# Shared by every session of a role/model — safe only because the agent keeps
# no per-request state (each turn's metadata goes into its own TurnResult).
@st.cache_resource(show_spinner=False)
def get_agent(model_name: str, role: str):
    return initialize_agent(model_name=model_name, role=role)


def init_chat_session(role: str):
    st.session_state.current_page_role = role
    for key, default in _CHAT_DEFAULTS.items():
        if _k(key) not in st.session_state:
            v = (
                dict(default)
                if isinstance(default, dict)
                else (list(default) if isinstance(default, list) else default)
            )
            st.session_state[_k(key)] = v
    if not _get("thread_id"):
        _set("thread_id", str(uuid.uuid4()))


def reset_chat_session():
    for key, default in _CHAT_DEFAULTS.items():
        v = (
            dict(default)
            if isinstance(default, dict)
            else (list(default) if isinstance(default, list) else default)
        )
        st.session_state[_k(key)] = v
    _set("thread_id", str(uuid.uuid4()))
