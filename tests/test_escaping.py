"""
Tests for utils/escaping.py, and for the ticket card: what a client typed must
reach Streamlit escaped.
"""

import re

import pytest
import streamlit as st

from app import ui_shared
from utils.escaping import html_md_escape, md_escape, md_no_images


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("IRIS 80", "IRIS 80"),
        ("![x](http://evil/a.png)", r"\!\[x\]\(http\://evil/a.png\)"),
        ("[click](http://evil)", r"\[click\]\(http\://evil\)"),
        ("<b>hi</b>", r"\<b\>hi\</b\>"),
        (":red[big] $x$", r"\:red\[big\] \$x\$"),
        ("a_b*c", r"a\_b\*c"),
        (r"back\slash", r"back\\slash"),
    ],
)
def test_md_escape(text, expected):
    assert md_escape(text) == expected


def test_html_md_escape_neutralizes_html_and_markdown():
    assert (
        html_md_escape('<img src=x onerror="a()">')
        == "&lt;img src=x onerror=&quot;a\\(\\)&quot;&gt;"
    )
    assert html_md_escape("**x**") == r"\*\*x\*\*"
    assert html_md_escape("R&D") == "R&amp;D"


def test_escaping_accepts_non_strings():
    assert md_escape(42) == "42"
    assert html_md_escape(None) == "None"


def test_md_no_images_keeps_the_formatting():
    assert md_no_images("**ok** ![x](http://evil/a.png)") == r"**ok** \![x](http://evil/a.png)"
    assert md_no_images("**ok** [link](http://a)") == "**ok** [link](http://a)"


ATTACK = '<img src=x onerror="a()">![t](http://evil/a.png)'


def test_ticket_card_escapes_what_the_client_typed(monkeypatch):
    shown = []
    monkeypatch.setattr(st, "markdown", lambda text, **kwargs: shown.append(text))
    monkeypatch.setattr(st, "caption", lambda text, **kwargs: shown.append(text))
    monkeypatch.setattr(ui_shared, "get_ticket_log", lambda _: [])
    monkeypatch.setattr(ui_shared, "get_product", lambda _: None)

    ui_shared.render_ticket_card(
        {
            "id": 1,
            "ticket_status": "new",
            "ticket_phase": "survey",
            "ticket_type": "window",
            "quantity": 1,
            "model": ATTACK,
            "date_str": ATTACK,
            "time_str": ATTACK,
            "client_name": ATTACK,
            "client_email": ATTACK,
            "client_phone": ATTACK,
            "client_address": ATTACK,
            "notes": ATTACK,
            "height_cm": 100.0,
            "width_cm": 100.0,
            "glass_type": ATTACK,
        }
    )

    rendered = "\n".join(shown)
    assert rendered.count(r"\!\[t\]") == 9  # every field above but the height and width
    assert not re.search(r"(?<!\\)<img", rendered)
    assert not re.search(r"(?<!\\)!\[", rendered)
