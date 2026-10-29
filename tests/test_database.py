"""
Tests for backend/database.py — schema, the full ticket lifecycle (+ its
audit log), clients, config, usage stats, business analytics, and the
inventory reservation flow. Every test uses the isolated `test_db` fixture
(see conftest.py) — the real dev database is never touched.
"""

from backend import database as db

# ── schema / migrations ───────────────────────────────────────────────────────


def test_init_db_is_idempotent(test_db):
    # Running init_db twice (without reset) must not raise or duplicate seed data.
    db.init_db(reset=False)
    db.init_db(reset=False)
    assert len(db.get_inventory()) == 7  # same seeded product count both times


def test_init_db_seeds_inventory_with_installation_hours(test_db):
    product = db.get_product("IRIS")
    assert product is not None
    assert product["installation_hours"] == 1.5


# ── config ─────────────────────────────────────────────────────────────────


def test_get_set_config_roundtrip(test_db):
    assert db.get_config("some_key", "default") == "default"
    db.set_config("some_key", "value1")
    assert db.get_config("some_key") == "value1"
    db.set_config("some_key", "value2")  # upsert, not insert-fails
    assert db.get_config("some_key") == "value2"


# ── clients ────────────────────────────────────────────────────────────────


def test_upsert_client_inserts_then_updates_by_name_and_email(test_db):
    cid1 = db.upsert_client(full_name="Mario Rossi", email="mario@example.com", phone="123")
    cid2 = db.upsert_client(full_name="Mario Rossi", email="mario@example.com", phone="456")
    assert cid1 == cid2  # same (name, email) = same client, updated not duplicated
    clients = db.get_clients()
    assert len(clients) == 1
    assert clients[0]["full_name"] == "Mario Rossi"
    assert clients[0]["phone"] == "456"


def test_upsert_client_same_email_different_name_stays_separate(test_db):
    # A shared inbox (or, in practice, the same person submitting requests
    # under different names) must not silently overwrite an earlier,
    # differently named client's row — this is the exact bug this test
    # guards against.
    cid1 = db.upsert_client(full_name="Ivan Namor", email="shared@example.com", phone="111")
    cid2 = db.upsert_client(full_name="Giada Lesizza", email="shared@example.com", phone="222")
    assert cid1 != cid2
    clients = {c["full_name"]: c for c in db.get_clients()}
    assert len(clients) == 2
    assert clients["Ivan Namor"]["phone"] == "111"
    assert clients["Giada Lesizza"]["phone"] == "222"


# ── ticket lifecycle + audit log ──────────────────────────────────────────────


def test_create_pending_booking_logs_new_status(test_db):
    booking_id = db.create_pending_booking(
        model="IRIS",
        date_str="2026-08-01",
        time_str="10:00",
        requested_by="client1",
        ticket_phase="survey",
        client_address="Via Roma 1",
    )
    ticket = db.get_all_tickets()[0]
    assert ticket["id"] == booking_id
    assert ticket["ticket_status"] == "new"

    log = db.get_ticket_log(booking_id)
    assert len(log) == 1
    assert log[0]["ticket_status"] == "new"
    assert log[0]["changed_by"] == "client1"


def test_resolve_booking_approve_sets_open_and_logs(test_db):
    bid = db.create_pending_booking(
        model="IRIS", date_str="2026-08-01", time_str="10:00", ticket_phase="survey"
    )
    updated = db.resolve_booking(bid, "approved", "manager1")
    assert updated
    ticket = db.get_all_tickets()[0]
    assert ticket["ticket_status"] == "open"
    assert ticket["status"] == "approved"
    log = db.get_ticket_log(bid)
    assert log[-1]["ticket_status"] == "open"
    assert log[-1]["changed_by"] == "manager1"


def test_resolve_booking_reject_excludes_from_get_all_tickets(test_db):
    bid = db.create_pending_booking(
        model="IRIS", date_str="2026-08-01", time_str="10:00", ticket_phase="survey"
    )
    db.resolve_booking(bid, "rejected", "manager1", rejection_reason="no slots")
    assert db.get_all_tickets() == []  # get_all_tickets excludes status='rejected'
    log = db.get_ticket_log(bid)
    assert log[-1]["ticket_status"] == "rejected"
    assert log[-1]["note"] == "no slots"


def test_claim_and_release_booking(test_db):
    bid = db.create_pending_booking(
        model="IRIS", date_str="2026-08-01", time_str="10:00", ticket_phase="survey"
    )
    db.resolve_booking(bid, "approved", "manager1")

    assert db.claim_booking(bid, "inst_a")
    ticket = db.get_all_tickets()[0]
    assert ticket["ticket_status"] == "in_progress"
    assert ticket["assigned_to"] == "inst_a"

    # A second installer can't claim while inst_a is already active elsewhere —
    # but claiming the SAME already-assigned ticket again should simply fail
    # (assigned_to IS NULL guard), not silently reassign it.
    assert not db.claim_booking(bid, "inst_b")

    assert db.release_booking(bid, "inst_a")
    ticket = db.get_all_tickets()[0]
    assert ticket["ticket_status"] == "open"
    assert ticket["assigned_to"] is None

    log = db.get_ticket_log(bid)
    statuses = [e["ticket_status"] for e in log]
    assert statuses == ["new", "open", "in_progress", "open"]


def test_full_survey_to_installation_lifecycle(test_db):
    bid = db.create_pending_booking(
        model="IRIS",
        date_str="2026-08-01",
        time_str="10:00",
        requested_by="client1",
        ticket_phase="survey",
        client_address="Via Roma 1",
        quantity=2,
        client_email="client1@example.com",
    )
    db.resolve_booking(bid, "approved", "manager1")
    db.claim_booking(bid, "inst_a")
    assert db.save_survey_measurements(bid, 150, 100, 8, "standard", changed_by="inst_a")

    ticket = db.get_all_tickets()[0]
    assert ticket["ticket_status"] == "in_review"
    assert ticket["height_cm"] == 150

    assert db.resolve_survey_estimate(bid, 850.0, "manager1")
    ticket = db.get_all_tickets()[0]
    assert ticket["ticket_status"] == "awaiting_client"
    assert ticket["estimated_price"] == 850.0

    assert db.resolve_client_estimate_response(
        bid,
        "accept",
        "2026-08-10",
        "11:00",
        changed_by="client1@example.com",
        client_email="client1@example.com",
    )
    ticket = db.get_all_tickets()[0]
    assert ticket["ticket_phase"] == "installation"
    assert ticket["ticket_status"] == "open"
    assert ticket["survey_date_str"] == "2026-08-01"  # original survey date preserved
    assert ticket["date_str"] == "2026-08-10"  # new installation date

    # NOTE (discovered by this test, not assumed): resolve_client_estimate_response's
    # accept path resets ticket_status to 'open' but does NOT clear assigned_to —
    # it's still 'inst_a' from the survey claim. claim_booking requires
    # assigned_to IS NULL, so a second claim (by inst_a or anyone else) fails here,
    # even though the ticket_status suggests it's available. The original surveyor
    # effectively keeps the job without needing to re-claim it.
    assert not db.claim_booking(bid, "inst_b")
    ticket = db.get_all_tickets()[0]
    assert ticket["assigned_to"] == "inst_a"

    job_id = db.create_job_completion("inst_a", "All done.", booking_id=bid)
    ticket = db.get_all_tickets()[0]
    assert ticket["ticket_status"] == "in_review"

    assert db.resolve_job_completion(job_id, "approved", "manager1")
    ticket = db.get_all_tickets()[0]
    assert ticket["ticket_status"] == "closed"

    log = db.get_ticket_log(bid)
    statuses = [e["ticket_status"] for e in log]
    assert statuses == [
        "new",
        "open",
        "in_progress",
        "in_review",
        "awaiting_client",
        "open",
        "in_review",
        "closed",
    ]


def test_reject_survey_report_sends_back_to_in_progress(test_db):
    bid = db.create_pending_booking(
        model="IRIS", date_str="2026-08-01", time_str="10:00", ticket_phase="survey"
    )
    db.resolve_booking(bid, "approved", "manager1")
    db.claim_booking(bid, "inst_a")
    db.save_survey_measurements(bid, 150, 100, 8, "standard", changed_by="inst_a")
    assert db.reject_survey_report(bid, "manager1")
    ticket = db.get_all_tickets()[0]
    assert ticket["ticket_status"] == "in_progress"


def _make_awaiting_client_ticket(client_email: str = "client1@example.com") -> int:
    bid = db.create_pending_booking(
        model="IRIS",
        date_str="2026-08-01",
        time_str="10:00",
        ticket_phase="survey",
        client_email=client_email,
    )
    db.resolve_booking(bid, "approved", "manager1")
    db.claim_booking(bid, "inst_a")
    db.save_survey_measurements(bid, 150, 100, 8, "standard", changed_by="inst_a")
    db.resolve_survey_estimate(bid, 850.0, "manager1")
    return bid


def test_resolve_client_estimate_response_reject_cancels_ticket(test_db):
    bid = _make_awaiting_client_ticket()
    assert db.resolve_client_estimate_response(
        bid,
        "reject",
        changed_by="client1@example.com",
        client_email="client1@example.com",
    )
    ticket = db.get_all_tickets()[0]
    assert ticket["ticket_status"] == "cancelled"
    assert ticket["client_estimate_response"] == "rejected"


def test_resolve_client_estimate_response_email_match_is_case_insensitive(test_db):
    bid = _make_awaiting_client_ticket("Mario.Rossi@Example.com")
    assert db.resolve_client_estimate_response(
        bid, "reject", client_email="mario.rossi@example.com"
    )


def test_resolve_client_estimate_response_refuses_other_email(test_db):
    bid = _make_awaiting_client_ticket("victim@example.com")
    assert not db.resolve_client_estimate_response(
        bid, "reject", client_email="attacker@example.com"
    )
    assert db.get_all_tickets()[0]["ticket_status"] == "awaiting_client"


def test_resolve_client_estimate_response_refuses_ticket_without_email(test_db):
    # A ticket with no email on file can't be "owned" by anyone — an empty
    # verified email must never match it.
    bid = _make_awaiting_client_ticket("")
    assert not db.resolve_client_estimate_response(bid, "reject", client_email="")


def test_find_tickets_by_email_is_case_insensitive_and_ignores_blank(test_db):
    _make_awaiting_client_ticket("Mario@Example.com")
    assert len(db.find_tickets_by_email("mario@example.com")) == 1
    assert db.find_tickets_by_email("   ") == []


def test_create_alert_with_booking_id_sets_blocked(test_db):
    bid = db.create_pending_booking(
        model="IRIS", date_str="2026-08-01", time_str="10:00", ticket_phase="installation"
    )
    db.create_alert(
        sent_by="inst_a",
        raw_message="broken window",
        structured_message="Broken window on site",
        booking_id=bid,
    )
    ticket = db.get_all_tickets()[0]
    assert ticket["ticket_status"] == "blocked"
    log = db.get_ticket_log(bid)
    assert log[-1]["ticket_status"] == "blocked"
    assert log[-1]["changed_by"] == "inst_a"


def test_update_ticket_coordinates(test_db):
    bid = db.create_pending_booking(
        model="IRIS", date_str="2026-08-01", time_str="10:00", ticket_phase="survey"
    )
    db.update_ticket_coordinates(bid, 45.123, 9.456)
    ticket = db.get_all_tickets()[0]
    assert ticket["client_lat"] == 45.123
    assert ticket["client_lon"] == 9.456


def test_update_ticket_details_only_changes_given_fields(test_db):
    bid = db.create_pending_booking(
        model="IRIS",
        date_str="2026-08-01",
        time_str="10:00",
        ticket_phase="survey",
        client_name="Old Name",
        client_phone="000",
    )
    updated = db.update_ticket_details(bid, "manager1", client_phone="999")
    assert updated
    ticket = db.get_all_tickets()[0]
    assert ticket["client_phone"] == "999"
    assert ticket["client_name"] == "Old Name"  # untouched

    log = db.get_ticket_log(bid)
    assert "client_phone" in log[-1]["note"]


def test_update_ticket_details_ignores_non_editable_fields(test_db):
    bid = db.create_pending_booking(
        model="IRIS", date_str="2026-08-01", time_str="10:00", ticket_phase="survey"
    )
    # ticket_status is deliberately not in _EDITABLE_TICKET_FIELDS
    updated = db.update_ticket_details(bid, "manager1", ticket_status="closed")
    assert not updated  # no editable field was actually provided
    ticket = db.get_all_tickets()[0]
    assert ticket["ticket_status"] == "new"  # unchanged


def test_update_ticket_details_returns_false_for_missing_ticket(test_db):
    assert not db.update_ticket_details(999999, "manager1", notes="x")


# ── get_all_tickets ordering ───────────────────────────────────────────────


def test_get_all_tickets_orders_blocked_before_closed(test_db):
    b1 = db.create_pending_booking(
        model="IRIS", date_str="2026-08-01", time_str="10:00", ticket_phase="installation"
    )
    db.resolve_booking(b1, "approved", "manager1")
    db.claim_booking(b1, "inst_a")
    job_id = db.create_job_completion("inst_a", "done", booking_id=b1)
    db.resolve_job_completion(job_id, "approved", "manager1")  # closed

    b2 = db.create_pending_booking(
        model="IRIS", date_str="2026-08-01", time_str="10:00", ticket_phase="installation"
    )
    db.create_alert("inst_b", "problem", "Problem on site", booking_id=b2)  # blocked

    tickets = db.get_all_tickets()
    statuses = [t["ticket_status"] for t in tickets]
    assert statuses.index("blocked") < statuses.index("closed")


# ── usage stats / business analytics ──────────────────────────────────────────


def test_get_usage_stats_aggregates_by_role(test_db):
    db.log_usage(
        role="client", model="gpt-4o-mini", input_tokens=100, output_tokens=50, cost_usd=0.01
    )
    db.log_usage(
        role="client", model="gpt-4o-mini", input_tokens=200, output_tokens=100, cost_usd=0.02
    )
    db.log_usage(
        role="manager", model="gpt-4o-mini", input_tokens=10, output_tokens=5, cost_usd=0.001
    )

    stats = db.get_usage_stats("month")
    assert stats["by_role"]["client"]["requests"] == 2
    assert stats["totals"]["requests"] == 3
    assert round(stats["totals"]["cost_usd"], 3) == 0.031


def test_get_business_analytics_only_counts_closed_with_ticket_log(test_db):
    # A ticket closed WITHOUT going through the logged lifecycle (simulating a
    # pre-ticket_log legacy row) must NOT be picked up — ticket_log is the
    # source of truth for "when did this close".
    bid_legacy = db.create_pending_booking(
        model="IRIS", date_str="2026-08-01", time_str="10:00", ticket_phase="installation"
    )
    conn = db.get_connection()
    conn.execute("UPDATE pending_bookings SET ticket_status='closed' WHERE id=?", (bid_legacy,))
    conn.execute("DELETE FROM ticket_log WHERE booking_id=?", (bid_legacy,))
    conn.commit()
    conn.close()

    biz = db.get_business_analytics("month")
    assert biz["closed_count"] == 0

    # A properly-closed ticket (via the real flow) IS counted, with revenue/hours.
    bid = db.create_pending_booking(
        model="IRIS",
        date_str="2026-08-01",
        time_str="10:00",
        ticket_phase="survey",
        quantity=2,
        client_email="c1@example.com",
    )
    db.resolve_booking(bid, "approved", "manager1")
    db.claim_booking(bid, "inst_a")
    db.save_survey_measurements(bid, 150, 100, 8, "standard", changed_by="inst_a")
    db.resolve_survey_estimate(bid, 700.0, "manager1")
    db.resolve_client_estimate_response(
        bid,
        "accept",
        "2026-08-10",
        "11:00",
        changed_by="c1",
        client_email="c1@example.com",
    )
    db.claim_booking(bid, "inst_b")
    job_id = db.create_job_completion("inst_b", "done", booking_id=bid)
    db.resolve_job_completion(job_id, "approved", "manager1")

    biz = db.get_business_analytics("month")
    assert biz["closed_count"] == 1
    assert biz["total_revenue"] == 700.0
    assert biz["total_hours"] == 2 * 1.5  # quantity * installation_hours


# ── inventory reservation flow ────────────────────────────────────────────────


def test_reservation_flow_pending_approve_reject(test_db):
    r1 = db.create_pending_reservation(
        model="IRIS", quantity=2, requested_by="inst_a", reason="job A"
    )
    r2 = db.create_pending_reservation(model="STILOS", quantity=1, requested_by="inst_b")

    pending = db.get_pending_reservations()
    assert {r["id"] for r in pending} == {r1, r2}

    assert db.resolve_reservation(r1, "approved", "manager1")
    assert db.resolve_reservation(r2, "rejected", "manager1")
    assert db.get_pending_reservations() == []

    # Resolving an already-resolved reservation again must fail (WHERE status='pending' guard).
    assert not db.resolve_reservation(r1, "approved", "manager1")
