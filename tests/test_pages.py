"""
Smoke test of the real Streamlit pages: each one must open without an exception,
both empty and with a ticket on the board. Buttons and chat turns aren't covered.

The outbound fixture proves that opening a page never tries to reach a paid API.
"""

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from backend.database import create_pending_booking
from evaluation.sandbox import sandbox

PAGES = [
    ("app/pages/login.py", None),
    ("app/pages/client.py", "client"),
    ("app/pages/installer.py", "installer"),
    ("app/pages/manager.py", "manager"),
]


@pytest.mark.parametrize("with_ticket", [False, True], ids=["empty", "one-ticket"])
@pytest.mark.parametrize(("path", "role"), PAGES, ids=[role or "login" for _, role in PAGES])
def test_page_opens(outbound, path, role, with_ticket):
    st.cache_resource.clear()  # an agent cached by another test holds a closed sandbox
    with sandbox():
        if with_ticket:
            create_pending_booking("IRIS", "2030-01-15", "10:00", client_name="Mario Rossi")

        page = AppTest.from_file(path, default_timeout=120)
        if role:
            page.session_state.role = role
            page.session_state.username = role
        page.run()

    assert not page.exception, [e.value for e in page.exception]
    assert not page.error, [e.value for e in page.error]
    assert outbound == []
