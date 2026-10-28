"""
Tests for backend/client_auth.py — one-time email codes for client sessions.
SMTP is always mocked; the clock is controlled via client_auth._now.
"""

import pytest

from backend import client_auth
from backend.database import get_connection


@pytest.fixture
def demo_mode(test_db, monkeypatch):
    """No SMTP, DEMO_MODE on → request_code() hands back the code directly."""
    monkeypatch.setattr(client_auth, "is_email_configured", lambda: False)
    monkeypatch.setenv("DEMO_MODE", "true")


@pytest.fixture
def clock(monkeypatch):
    now = {"t": 1_000_000.0}
    monkeypatch.setattr(client_auth, "_now", lambda: now["t"])
    return now


def test_demo_code_verifies_once(demo_mode):
    result = client_auth.request_code("mario@example.com")
    assert result.status == "demo"
    assert len(result.demo_code) == 6 and result.demo_code.isdigit()

    assert client_auth.verify_code("mario@example.com", result.demo_code)
    assert not client_auth.verify_code("mario@example.com", result.demo_code)  # single use


def test_email_is_normalized(demo_mode):
    code = client_auth.request_code("  Mario@Example.COM ").demo_code
    assert client_auth.verify_code("mario@example.com", code)


def test_code_is_bound_to_its_email(demo_mode):
    code = client_auth.request_code("mario@example.com").demo_code
    assert not client_auth.verify_code("someone.else@example.com", code)


def test_only_a_hash_is_stored(demo_mode):
    code = client_auth.request_code("mario@example.com").demo_code
    conn = get_connection()
    row = conn.execute("SELECT code_hash FROM client_verification_codes").fetchone()
    conn.close()
    assert code not in row["code_hash"]


def test_code_expires(demo_mode, clock):
    code = client_auth.request_code("mario@example.com").demo_code
    clock["t"] += client_auth.CODE_TTL_SECONDS + 1
    assert not client_auth.verify_code("mario@example.com", code)


def test_too_many_wrong_attempts_kill_the_code(demo_mode):
    code = client_auth.request_code("mario@example.com").demo_code
    wrong = "000000" if code != "000000" else "111111"
    for _ in range(client_auth.MAX_ATTEMPTS):
        assert not client_auth.verify_code("mario@example.com", wrong)
    assert not client_auth.verify_code("mario@example.com", code)


def test_resend_cooldown(demo_mode, clock):
    assert client_auth.request_code("mario@example.com").status == "demo"
    assert client_auth.request_code("mario@example.com").status == "cooldown"
    clock["t"] += client_auth.RESEND_COOLDOWN_SECONDS + 1
    assert client_auth.request_code("mario@example.com").status == "demo"


def test_new_code_replaces_old_one(demo_mode, clock):
    old = client_auth.request_code("mario@example.com").demo_code
    clock["t"] += client_auth.RESEND_COOLDOWN_SECONDS + 1
    new = client_auth.request_code("mario@example.com").demo_code
    if old != new:
        assert not client_auth.verify_code("mario@example.com", old)
    assert client_auth.verify_code("mario@example.com", new)


@pytest.mark.parametrize("email", ["", "not-an-email", "a@b", "two@@example.com"])
def test_invalid_email_rejected(demo_mode, email):
    assert client_auth.request_code(email).status == "invalid_email"


def test_no_smtp_and_no_demo_mode_is_unavailable(test_db, monkeypatch):
    monkeypatch.setattr(client_auth, "is_email_configured", lambda: False)
    monkeypatch.delenv("DEMO_MODE", raising=False)
    result = client_auth.request_code("mario@example.com")
    assert result.status == "email_unavailable"
    assert result.demo_code is None


def test_code_is_emailed_when_smtp_configured(test_db, monkeypatch):
    sent = {}

    def _fake_send(to, subject, body):
        sent.update(to=to, body=body)
        return True

    monkeypatch.setattr(client_auth, "is_email_configured", lambda: True)
    monkeypatch.setattr(client_auth, "send_email", _fake_send)
    monkeypatch.setenv("DEMO_MODE", "true")  # ignored when real email works

    result = client_auth.request_code("mario@example.com")
    assert result.status == "sent"
    assert result.demo_code is None  # never shown on screen when emailed
    assert sent["to"] == "mario@example.com"

    code = next(word for word in sent["body"].split() if word.isdigit() and len(word) == 6)
    assert client_auth.verify_code("mario@example.com", code)


def test_failed_send_leaves_no_usable_code(test_db, monkeypatch):
    monkeypatch.setattr(client_auth, "is_email_configured", lambda: True)
    monkeypatch.setattr(client_auth, "send_email", lambda to, subject, body: False)
    assert client_auth.request_code("mario@example.com").status == "email_unavailable"
    conn = get_connection()
    count = conn.execute("SELECT COUNT(*) FROM client_verification_codes").fetchone()[0]
    conn.close()
    assert count == 0
