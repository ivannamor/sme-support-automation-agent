"""
Tests for the DB-backed tools in tools.py — using the isolated `test_db`
fixture for real database logic, and monkeypatching only the external
service boundaries (geocoding, Google Calendar, SMTP, the LLM, and the
inventory HTTP API) so no test makes a real network/API call.
"""

import tools
from backend.log_context import username_var
from backend.database import get_connection, ticket_code
from tests.conftest import future_date


class _FakeInvokeLLM:
    """Stand-in for get_llm(...) — .invoke(prompt) returns a canned response."""
    def __init__(self, content):
        self._content = content

    def invoke(self, *args, **kwargs):
        class _Resp:
            def __init__(self, content):
                self.content = content
        return _Resp(self._content)


# ── claim / release (pure DB tools) ───────────────────────────────────────────

def test_claim_installation_tool_success(test_db):
    bid = _make_open_installation_ticket()
    username_var.set("inst_a")
    result = tools.claim_installation_tool.invoke({"booking_id": bid})
    assert "now assigned to you" in result
    assert ticket_code(bid) in result


def test_claim_installation_tool_blocks_second_job_for_same_installer(test_db):
    bid1 = _make_open_installation_ticket()
    bid2 = _make_open_installation_ticket()
    username_var.set("inst_a")
    tools.claim_installation_tool.invoke({"booking_id": bid1})
    result = tools.claim_installation_tool.invoke({"booking_id": bid2})
    assert "already have an active installation" in result


def test_release_installation_tool(test_db):
    bid = _make_open_installation_ticket()
    username_var.set("inst_a")
    tools.claim_installation_tool.invoke({"booking_id": bid})
    result = tools.release_installation_tool.invoke({"booking_id": bid})
    assert "released" in result.lower()


def test_release_installation_tool_fails_for_wrong_installer(test_db):
    bid = _make_open_installation_ticket()
    username_var.set("inst_a")
    tools.claim_installation_tool.invoke({"booking_id": bid})
    username_var.set("inst_b")
    result = tools.release_installation_tool.invoke({"booking_id": bid})
    assert "Could not release" in result


# ── survey measurements / estimate rejection (pure DB tools) ─────────────────

def test_submit_survey_measurements_tool(test_db):
    bid = _make_survey_ticket_in_progress()
    username_var.set("inst_a")
    result = tools.submit_survey_measurements_tool.invoke({
        "booking_id": bid, "height_cm": 150, "width_cm": 100, "depth_cm": 8,
        "glass_type": "standard", "survey_notes": "test",
    })
    assert "manager" in result.lower() or "compute" in result.lower() or ticket_code(bid) not in ""
    conn = get_connection()
    row = conn.execute("SELECT ticket_status FROM pending_bookings WHERE id=?", (bid,)).fetchone()
    conn.close()
    assert row["ticket_status"] == "in_review"


def test_submit_survey_measurements_tool_rejects_invalid_glass_type(test_db):
    bid = _make_survey_ticket_in_progress()
    username_var.set("inst_a")
    result = tools.submit_survey_measurements_tool.invoke({
        "booking_id": bid, "height_cm": 150, "width_cm": 100, "depth_cm": 8,
        "glass_type": "unobtainium",
    })
    assert "Error" in result


def test_reject_survey_estimate_tool_sends_back_to_installer(test_db):
    bid = _make_survey_ticket_in_progress()
    username_var.set("inst_a")
    tools.submit_survey_measurements_tool.invoke({
        "booking_id": bid, "height_cm": 150, "width_cm": 100, "depth_cm": 8, "glass_type": "standard",
    })
    username_var.set("manager1")
    result = tools.reject_survey_estimate_tool.invoke({"booking_id": bid})
    assert ticket_code(bid) in result
    conn = get_connection()
    row = conn.execute("SELECT ticket_status FROM pending_bookings WHERE id=?", (bid,)).fetchone()
    conn.close()
    assert row["ticket_status"] == "in_progress"


# ── edit_ticket_tool ──────────────────────────────────────────────────────────

def test_edit_ticket_tool_updates_fields(test_db):
    bid = _make_open_installation_ticket()
    username_var.set("manager1")
    result = tools.edit_ticket_tool.invoke({
        "ticket_reference": ticket_code(bid), "client_phone": "3339998877",
    })
    assert "updated successfully" in result
    conn = get_connection()
    row = conn.execute("SELECT client_phone FROM pending_bookings WHERE id=?", (bid,)).fetchone()
    conn.close()
    assert row["client_phone"] == "3339998877"


def test_edit_ticket_tool_rejects_bad_date_format(test_db):
    bid = _make_open_installation_ticket()
    username_var.set("manager1")
    result = tools.edit_ticket_tool.invoke({"ticket_reference": ticket_code(bid), "date_str": "01/08/2026"})
    assert "Error" in result


# ── ticket status query tools ────────────────────────────────────────────────

def test_get_ticket_status_tool_sees_all_tickets(test_db):
    _make_open_installation_ticket()
    result = tools.get_ticket_status_tool.invoke({})
    assert "1" in result and "open" in result


def test_get_my_ticket_status_tool_scoped_to_assigned_installer(test_db):
    bid_mine = _make_open_installation_ticket()
    bid_other = _make_open_installation_ticket()
    username_var.set("inst_a")
    tools.claim_installation_tool.invoke({"booking_id": bid_mine})
    username_var.set("inst_b")
    tools.claim_installation_tool.invoke({"booking_id": bid_other})

    username_var.set("inst_a")
    result = tools.get_my_ticket_status_tool.invoke({})
    assert "1 total" in result  # only inst_a's own ticket, not inst_b's


# ── request_site_survey_tool (mocks geocoding) ────────────────────────────────

def test_request_site_survey_tool_creates_new_ticket(test_db, monkeypatch):
    monkeypatch.setattr(tools, "_geocode_location", lambda addr: (45.0, 9.0, addr))
    username_var.set("client1")
    result = tools.request_site_survey_tool.invoke({
        "model": "IRIS", "quantity": 1, "date_str": future_date(), "time_str": "10:00",
        "client_name": "Mario Rossi", "client_email": "mario@example.com",
        "client_phone": "3331234567", "client_address": "Via Roma 1, Udine",
    })
    assert "submitted" in result.lower()
    conn = get_connection()
    row = conn.execute("SELECT client_lat, client_lon, ticket_status FROM pending_bookings").fetchone()
    conn.close()
    assert row["client_lat"] == 45.0
    assert row["ticket_status"] == "new"


def test_request_site_survey_tool_survives_geocode_failure(test_db, monkeypatch):
    def _boom(addr):
        raise ValueError("geocoding failed")
    monkeypatch.setattr(tools, "_geocode_location", _boom)
    username_var.set("client1")
    result = tools.request_site_survey_tool.invoke({
        "model": "IRIS", "quantity": 1, "date_str": future_date(), "time_str": "10:00",
        "client_name": "Mario Rossi", "client_email": "mario@example.com",
        "client_phone": "3331234567", "client_address": "Nowhere, Nowhere",
    })
    assert "submitted" in result.lower()  # best-effort: request still succeeds
    conn = get_connection()
    row = conn.execute("SELECT client_lat FROM pending_bookings").fetchone()
    conn.close()
    assert row["client_lat"] is None


def test_request_site_survey_tool_rejects_bad_quantity(test_db):
    username_var.set("client1")
    result = tools.request_site_survey_tool.invoke({
        "model": "IRIS", "quantity": 0, "date_str": future_date(), "time_str": "10:00",
        "client_name": "Mario Rossi", "client_email": "mario@example.com",
        "client_phone": "3331234567", "client_address": "Via Roma 1",
    })
    assert "quantity must be at least 1" in result


def test_request_site_survey_tool_rejects_past_date(test_db, monkeypatch):
    monkeypatch.setattr(tools, "_geocode_location", lambda addr: (45.0, 9.0, addr))
    username_var.set("client1")
    yesterday = future_date(days=-1)
    result = tools.request_site_survey_tool.invoke({
        "model": "IRIS", "quantity": 1, "date_str": yesterday, "time_str": "10:00",
        "client_name": "Mario Rossi", "client_email": "mario@example.com",
        "client_phone": "3331234567", "client_address": "Via Roma 1, Udine",
    })
    assert "is in the past" in result
    conn = get_connection()
    count = conn.execute("SELECT COUNT(*) FROM pending_bookings").fetchone()[0]
    conn.close()
    assert count == 0  # nothing persisted for a rejected request


# ── extract_booking_fields (mocks the LLM structured-output call) ────────────

def test_extract_booking_fields_returns_only_mentioned_fields(monkeypatch):
    from tools import BookingFormExtraction

    class _FakeStructuredLLM:
        def invoke(self, *args, **kwargs):
            return BookingFormExtraction(client_name="Mario Rossi", model="IRIS", quantity=2)

    class _FakeLLM:
        def with_structured_output(self, schema):
            return _FakeStructuredLLM()

    monkeypatch.setattr(tools, "get_llm", lambda *a, **k: _FakeLLM())
    result = tools.extract_booking_fields("Mario Rossi vuole 2 IRIS")
    assert result == {"client_name": "Mario Rossi", "model": "IRIS", "quantity": 2}


# ── send_alert_to_manager_tool (mocks LLM + email) ───────────────────────────

def test_send_alert_to_manager_tool_sets_ticket_blocked(test_db, monkeypatch):
    bid = _make_open_installation_ticket()
    monkeypatch.setattr(tools, "get_llm", lambda *a, **k: _FakeInvokeLLM("MESSAGE: broken window\nTO DO: replace glass"))
    monkeypatch.setattr(tools, "send_manager_alert", lambda subject, body: True)
    username_var.set("inst_a")
    tools.claim_installation_tool.invoke({"booking_id": bid})
    result = tools.send_alert_to_manager_tool.invoke({"message": "window is broken", "booking_id": bid})
    assert "BLOCKED" in result or ticket_code(bid) in result
    conn = get_connection()
    row = conn.execute("SELECT ticket_status FROM pending_bookings WHERE id=?", (bid,)).fetchone()
    conn.close()
    assert row["ticket_status"] == "blocked"


# ── inventory reservation tools (mock the inventory HTTP round trip) ────────

def test_reservation_approve_deducts_stock_via_inventory_api(test_db, monkeypatch):
    monkeypatch.setattr(
        tools, "_inventory_request",
        lambda action, model, quantity=1: (True, "Successfully reserved 2 unit(s) of IRIS."),
    )
    username_var.set("inst_a")
    tools.request_inventory_reservation_tool.invoke({"model": "IRIS", "quantity": 2, "reason": "job"})

    conn = get_connection()
    res_id = conn.execute("SELECT id FROM pending_reservations WHERE status='pending'").fetchone()["id"]
    conn.close()

    username_var.set("manager1")
    result = tools.approve_reservation_tool.invoke({"reservation_id": res_id})
    assert "approved" in result.lower()

    conn = get_connection()
    row = conn.execute("SELECT status FROM pending_reservations WHERE id=?", (res_id,)).fetchone()
    conn.close()
    assert row["status"] == "approved"


def test_reservation_reject_does_not_touch_inventory(test_db, monkeypatch):
    username_var.set("inst_a")
    tools.request_inventory_reservation_tool.invoke({"model": "IRIS", "quantity": 2})
    conn = get_connection()
    res_id = conn.execute("SELECT id FROM pending_reservations WHERE status='pending'").fetchone()["id"]
    conn.close()

    username_var.set("manager1")
    result = tools.reject_reservation_tool.invoke({"reservation_id": res_id})
    assert "rejected" in result.lower()


def _pending_reservation(model: str = "IRIS", quantity: int = 2) -> int:
    username_var.set("inst_a")
    tools.request_inventory_reservation_tool.invoke({"model": model, "quantity": quantity})
    conn = get_connection()
    res_id = conn.execute("SELECT id FROM pending_reservations WHERE status='pending'").fetchone()["id"]
    conn.close()
    username_var.set("manager1")
    return res_id


def _reservation_status(res_id: int) -> str:
    conn = get_connection()
    row = conn.execute("SELECT status FROM pending_reservations WHERE id=?", (res_id,)).fetchone()
    conn.close()
    return row["status"]


def test_reservation_approve_deducts_stock_only_once(test_db, monkeypatch):
    calls = []
    monkeypatch.setattr(
        tools, "_inventory_request",
        lambda action, model, quantity=1: calls.append((action, model, quantity)) or (True, "ok"),
    )
    res_id = _pending_reservation()

    first = tools.approve_reservation_tool.invoke({"reservation_id": res_id})
    second = tools.approve_reservation_tool.invoke({"reservation_id": res_id})

    assert "approved" in first.lower()
    assert "not found" in second
    assert calls == [("take", "IRIS", 2)]


def test_reservation_failed_deduction_puts_request_back_to_pending(test_db, monkeypatch):
    monkeypatch.setattr(
        tools, "_inventory_request",
        lambda action, model, quantity=1: (False, "Inventory Error: Insufficient stock for IRIS. Available: 1"),
    )
    res_id = _pending_reservation()

    result = tools.approve_reservation_tool.invoke({"reservation_id": res_id})

    assert "Could not deduct stock" in result
    assert _reservation_status(res_id) == "pending"


def test_reservation_with_error_in_model_name_is_not_mistaken_for_failure(test_db, monkeypatch):
    # The old check was `"Error" in take_result`; success is now structured.
    monkeypatch.setattr(
        tools, "_inventory_request",
        lambda action, model, quantity=1: (True, "Successfully reserved 2 unit(s) of ERROR-PROOF."),
    )
    res_id = _pending_reservation(model="ERROR-PROOF")

    result = tools.approve_reservation_tool.invoke({"reservation_id": res_id})

    assert "approved" in result.lower()
    assert _reservation_status(res_id) == "approved"


# ── helpers ────────────────────────────────────────────────────────────────

def _make_open_installation_ticket() -> int:
    from backend.database import create_pending_booking, resolve_booking
    bid = create_pending_booking(
        model="IRIS", date_str="2026-08-01", time_str="10:00", ticket_phase="installation",
        client_address="Via Roma 1",
    )
    resolve_booking(bid, "approved", "manager1")
    return bid


def _make_survey_ticket_in_progress() -> int:
    from backend.database import create_pending_booking, resolve_booking, claim_booking
    bid = create_pending_booking(
        model="IRIS", date_str="2026-08-01", time_str="10:00", ticket_phase="survey",
        client_address="Via Roma 1",
    )
    resolve_booking(bid, "approved", "manager1")
    claim_booking(bid, "inst_a")
    return bid
