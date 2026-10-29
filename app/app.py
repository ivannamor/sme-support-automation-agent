"""
Entry point — configures app navigation and initializes the database.

Streamlit executes this router file on every rerun. Navigation is hidden
(position="hidden") so no default multipage nav ever renders — this avoids
the flash of unstyled page links that a auto-discovered `pages/` nav would
otherwise show before our CSS loads. Each page still gates itself with
require_auth() based on st.session_state.role.

Pages:
  - pages/login.py      → role selection / staff login
  - pages/client.py     → client chat portal
  - pages/installer.py  → installer jobs + chat
  - pages/manager.py    → manager dashboard + chat
"""

import sys
from pathlib import Path

import streamlit as st

# Streamlit only puts app/ on sys.path. TODO PR 11: drop with the app/ restructure.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.database import init_db


@st.cache_resource(show_spinner=False)
def _init_db_once() -> bool:
    init_db()
    return True


_init_db_once()

pg = st.navigation(
    [
        st.Page("pages/login.py", title="Login", default=True),
        st.Page("pages/client.py", title="Client Portal"),
        st.Page("pages/installer.py", title="Installer Portal"),
        st.Page("pages/manager.py", title="Manager Portal"),
    ],
    position="hidden",
)
pg.run()
