"""
Authorization tests for tools that act on existing tickets: client tools
must only touch tickets owned by the session's *verified* email, installer
tools only tickets assigned to the calling installer. Identity comes from
ContextVars set by the UI, never from LLM-supplied arguments.
"""

import pytest

import tools
from backend import database as db
from backend.log_context import username_var, verified_client_email_var


class _FakeInvokeLLM:
    def invoke(self, *args, **kwargs):
        return type("Resp", (), {"content": "Structured text."})()


@pytest.fixture
def as_client():
    """Sets verified_client_email_var for one test, then restores it."""
    tokens = []

    def _set(email: str):
        tokens.append(verified_client_email_var.set(email))

    yield _set
    for token in reversed(tokens):
        verified_client_email_var.reset(token)


def _awaiting_client_ticket(client_email: str) -> int:
    bid = db.create_pending_booking(
        model="IRIS",
        date_str="2026-08-01",
        time_str="10:00",
        ticket_phase="survey",
        client_email=client_email,
        client_address="Via Roma 1",
    )
    db.resolve_booking(bid, "approved", "manager1")
    db.claim_booking(bid, "inst_owner")
    db.save_survey_measurements(bid, 150, 100, 8, "standard", changed_by="inst_owner")
    db.resolve_survey_estimate(bid, 850.0, "manager1")
    return bid


# ── client: find_my_tickets_by_email_tool ─────────────────────────────────────


def test_find_my_tickets_requires_verified_email(test_db, as_client):
    _awaiting_client_ticket("mario@example.com")
    as_client("")
    result = tools.find_my_tickets_by_email_tool.invoke({})
    assert "not verified" in result
    assert "TKT-" not in result


def test_find_my_tickets_shows_only_own_tickets(test_db, as_client):
    mine = _awaiting_client_ticket("mario@example.com")
    theirs = _awaiting_client_ticket("luigi@example.com")
    as_client("mario@example.com")
    result = tools.find_my_tickets_by_email_tool.invoke({})
    assert db.ticket_code(mine) in result
    assert db.ticket_code(theirs) not in result


def test_find_my_tickets_ignores_llm_supplied_email(test_db, as_client):
    # The tool takes no email argument any more, so a prompt-injected
    # "look up luigi@example.com" has nothing to pass it.
    assert "client_email" not in tools.find_my_tickets_by_email_tool.args


# ── client: respond_to_estimate_tool ──────────────────────────────────────────


def test_respond_to_estimate_requires_verified_email(test_db, as_client):
    bid = _awaiting_client_ticket("mario@example.com")
    as_client("")
    result = tools.respond_to_estimate_tool.invoke({"booking_id": bid, "response": "reject"})
    assert "not verified" in result
    assert db.get_booking(bid)["ticket_status"] == "awaiting_client"


def test_respond_to_estimate_refuses_someone_elses_ticket(test_db, as_client):
    bid = _awaiting_client_ticket("victim@example.com")
    as_client("attacker@example.com")
    result = tools.respond_to_estimate_tool.invoke({"booking_id": bid, "response": "reject"})
    assert "not currently awaiting a response from you" in result
    assert db.get_booking(bid)["ticket_status"] == "awaiting_client"


def test_respond_to_estimate_same_answer_for_missing_and_foreign_ticket(test_db, as_client):
    bid = _awaiting_client_ticket("victim@example.com")
    as_client("attacker@example.com")
    foreign = tools.respond_to_estimate_tool.invoke({"booking_id": bid, "response": "reject"})
    missing = tools.respond_to_estimate_tool.invoke({"booking_id": 9999, "response": "reject"})
    assert foreign.replace(db.ticket_code(bid), "X") == missing.replace(db.ticket_code(9999), "X")


def test_respond_to_estimate_owner_can_reject(test_db, as_client):
    bid = _awaiting_client_ticket("Mario@Example.com")
    as_client("mario@example.com")
    result = tools.respond_to_estimate_tool.invoke({"booking_id": bid, "response": "reject"})
    assert "declined" in result
    assert db.get_booking(bid)["ticket_status"] == "cancelled"


def test_respond_to_estimate_owner_can_accept(test_db, as_client, monkeypatch):
    bid = _awaiting_client_ticket("mario@example.com")
    as_client("mario@example.com")
    monkeypatch.setattr(tools, "_compute_travel", lambda addr: {"duration_hours": 0.5})
    monkeypatch.setattr(
        tools,
        "book_installation_tool",
        type("FakeBooking", (), {"invoke": staticmethod(lambda args: "CONFIRMED: booked.")})(),
    )
    result = tools.respond_to_estimate_tool.invoke(
        {
            "booking_id": bid,
            "response": "accept",
            "new_date_str": "2030-01-10",
            "new_time_str": "09:00",
        }
    )
    assert "accepted" in result
    booking = db.get_booking(bid)
    assert booking["ticket_status"] == "open"
    assert booking["date_str"] == "2030-01-10"


# ── installer: ownership of booking_id ────────────────────────────────────────


def _open_ticket() -> int:
    bid = db.create_pending_booking(
        model="IRIS",
        date_str="2026-08-01",
        time_str="10:00",
        ticket_phase="installation",
        client_email="mario@example.com",
    )
    db.resolve_booking(bid, "approved", "manager1")
    return bid


def test_complete_job_refuses_unassigned_ticket(test_db, monkeypatch):
    monkeypatch.setattr(tools, "get_llm", lambda *a, **k: _FakeInvokeLLM())
    bid = _open_ticket()
    db.claim_booking(bid, "inst_a")
    username_var.set("inst_b")
    result = tools.complete_job_tool.invoke({"notes": "done", "booking_id": bid})
    assert "not assigned to you" in result
    assert "mario@example.com" not in result  # no client email leaked
    assert db.get_job_completions(status="pending") == []


def test_complete_job_allowed_for_assigned_installer(test_db, monkeypatch):
    monkeypatch.setattr(tools, "get_llm", lambda *a, **k: _FakeInvokeLLM())
    bid = _open_ticket()
    db.claim_booking(bid, "inst_a")
    username_var.set("inst_a")
    result = tools.complete_job_tool.invoke({"notes": "done", "booking_id": bid})
    assert "submitted to the manager" in result


def test_send_alert_refuses_unassigned_ticket(test_db, monkeypatch):
    monkeypatch.setattr(tools, "get_llm", lambda *a, **k: _FakeInvokeLLM())
    monkeypatch.setattr(tools, "send_manager_alert", lambda subject, body: True)
    bid = _open_ticket()
    username_var.set("inst_a")  # never claimed it
    result = tools.send_alert_to_manager_tool.invoke({"message": "broken", "booking_id": bid})
    assert "not assigned to you" in result
    assert db.get_booking(bid)["ticket_status"] != "blocked"


def test_send_alert_without_ticket_still_allowed(test_db, monkeypatch):
    monkeypatch.setattr(tools, "get_llm", lambda *a, **k: _FakeInvokeLLM())
    monkeypatch.setattr(tools, "send_manager_alert", lambda subject, body: True)
    username_var.set("inst_a")
    result = tools.send_alert_to_manager_tool.invoke({"message": "van broke down"})
    assert "sent to the manager" in result
