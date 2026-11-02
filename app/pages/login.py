"""Login and role selection page."""

import streamlit as st

from backend.auth import authenticate
from ui import inject_global_css

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
            result = authenticate(username, password)
            if result.status == "ok":
                st.session_state.role = result.role
                st.session_state.username = username
                target = {
                    "installer": "pages/installer.py",
                    "manager": "pages/manager.py",
                }.get(result.role, "pages/manager.py")
                st.switch_page(target)
            elif result.status == "locked":
                st.error("Too many failed attempts. Try again in 15 minutes.")
            elif result.status == "default_password":
                st.error(
                    "This account still has the default password. Ask an administrator "
                    "to set a new one with scripts/set_password."
                )
            else:
                st.error("Invalid credentials.")
