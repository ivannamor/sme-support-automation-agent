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

import datetime
import sys
from pathlib import Path

import streamlit as st

# Streamlit only puts app/ on sys.path. TODO PR 11: drop with the app/ restructure.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.checkpointing import prune_stale_threads
from backend.database import init_db, prune_conversation_log


@st.cache_resource(show_spinner=False)
def _init_db_once() -> bool:
    init_db()
    return True


# Retention: idle conversation checkpoints (7 days) and archived conversations
# (365 days) — see "Personal Data & Retention" in the README. The TTL re-runs
# it on the first page load of each day, so a long-running process keeps
# deleting expired data, not only at startup.
@st.cache_resource(show_spinner=False, ttl=datetime.timedelta(days=1))
def _prune_expired_data_daily() -> bool:
    prune_stale_threads()
    prune_conversation_log()
    return True


_init_db_once()
_prune_expired_data_daily()

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
