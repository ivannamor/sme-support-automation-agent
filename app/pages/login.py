"""Login and role selection page."""

import sys
from pathlib import Path

import streamlit as st

# TODO PR 11: drop with the app/ restructure (imports ui_shared as top-level).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
APP_DIR = Path(__file__).resolve().parents[1]
for p in (str(PROJECT_ROOT), str(APP_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from ui_shared import inject_global_css

from backend.auth import authenticate

st.set_page_config(
    page_title="Assistant",
    page_icon=None,
    layout="centered",
)
inject_global_css()

st.title("Assistant")
st.caption("Select your role to continue.")

# ── already authenticated → redirect ─────────────────────────────────────────

if st.session_state.get("role"):
    role = st.session_state.role
    target = {
        "client": "pages/client.py",
        "installer": "pages/installer.py",
        "manager": "pages/manager.py",
    }.get(role)
    if target:
        st.switch_page(target)

# ── role selection ────────────────────────────────────────────────────────────

st.divider()
col_client, col_staff = st.columns(2)

with col_client:
    st.subheader("Client")
    st.write("Ask about products, request installation bookings, and get estimates.")
    if st.button("Continue as Client", use_container_width=True, type="primary"):
        st.session_state.role = "client"
        st.session_state.username = "guest"
        st.switch_page("pages/client.py")

with col_staff:
    st.subheader("Staff Login")
    st.write("Installers and managers authenticate with a username and password.")

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submitted = st.form_submit_button("Login", use_container_width=True)

    if submitted:
        if not username or not password:
            st.error("Please enter both username and password.")
        else:
            role = authenticate(username, password)
            if role:
                st.session_state.role = role
                st.session_state.username = username
                target = {
                    "installer": "pages/installer.py",
                    "manager": "pages/manager.py",
                }.get(role, "pages/manager.py")
                st.switch_page(target)
            else:
                st.error("Invalid credentials.")
