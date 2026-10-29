"""
Tests for tools.py's pure/deterministic helper functions — no DB, no network,
no LLM calls needed for any of these.
"""

import pytest

from backend.database import ticket_code
from tools import (
    _format_duration,
    _format_ticket_details,
    _parse_ticket_reference,
    _summarize_tickets,
    compute_appointment_duration,
    format_sources,
)

# ── _parse_ticket_reference ───────────────────────────────────────────────────

@pytest.mark.parametrize("ref,expected", [
    ("TKT-00007", 7),
    ("TKT-00123", 123),
    ("7", 7),
    ("tkt-00042", 42),
    ("ticket #9", 9),
])
def test_parse_ticket_reference_extracts_id(ref, expected):
    assert _parse_ticket_reference(ref) == expected


def test_parse_ticket_reference_returns_none_when_no_digits():
    assert _parse_ticket_reference("no digits here") is None


# ── _format_duration ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("seconds,expected", [
    (30, "1 min"),          # rounds up to at least 1 minute
    (90, "2 min"),
    (3600, "1 h 0 min"),
    (3660, "1 h 1 min"),
    (7290, "2 h 2 min"),
])
def test_format_duration(seconds, expected):
    assert _format_duration(seconds) == expected


# ── compute_appointment_duration ──────────────────────────────────────────────

def test_compute_appointment_duration_scales_with_quantity_and_round_trip_travel():
    # 3 units * 1.5h/unit + round-trip (2x) of a 1h one-way travel = 4.5 + 2 = 6.5h
    assert compute_appointment_duration(quantity=3, hours_per_unit=1.5, travel_duration_hours=1.0) == 6.5


def test_compute_appointment_duration_zero_travel():
    assert compute_appointment_duration(quantity=2, hours_per_unit=2.0, travel_duration_hours=0.0) == 4.0


# ── _format_ticket_details ────────────────────────────────────────────────────

def test_format_ticket_details_includes_core_fields():
    ticket = {
        "id": 5, "model": "IRIS", "quantity": 2, "ticket_phase": "installation",
        "ticket_status": "in_progress", "client_name": "Mario Rossi",
        "client_email": "mario@example.com", "client_address": "Via Roma 1",
        "date_str": "2026-08-01", "time_str": "10:00", "assigned_to": "installer1",
        "estimated_price": 500.0, "notes": "Fragile glass",
    }
    text = _format_ticket_details(ticket)
    assert ticket_code(5) in text
    assert "IRIS" in text
    assert "Mario Rossi" in text
    assert "mario@example.com" in text
    assert "installer1" in text
    assert "500.00" in text
    assert "Fragile glass" in text


def test_format_ticket_details_omits_absent_optional_fields():
    ticket = {
        "id": 6, "model": "IRIS", "quantity": 1, "date_str": "2026-08-01", "time_str": "10:00",
    }
    text = _format_ticket_details(ticket)
    assert "Email:" not in text
    assert "Assigned installer:" not in text
    assert "Notes:" not in text


# ── _summarize_tickets ─────────────────────────────────────────────────────────

_SAMPLE_TICKETS = [
    {"id": 1, "model": "IRIS", "ticket_status": "new", "client_name": "A", "date_str": "2026-08-01", "time_str": "10:00"},
    {"id": 2, "model": "IRIS", "ticket_status": "closed", "client_name": "B", "date_str": "2026-08-02", "time_str": "11:00"},
    {"id": 3, "model": "STILOS", "ticket_status": "in_progress", "client_name": "C", "date_str": "2026-08-03", "time_str": "12:00"},
]


def test_summarize_tickets_by_reference_finds_single_ticket():
    result = _summarize_tickets(_SAMPLE_TICKETS, ticket_reference="TKT-00002", status_filter=None)
    assert "TKT-00002" in result
    assert "closed" in result.lower()


def test_summarize_tickets_by_reference_not_found():
    result = _summarize_tickets(_SAMPLE_TICKETS, ticket_reference="TKT-00099", status_filter=None)
    assert "No ticket found" in result


def test_summarize_tickets_by_status_filter():
    result = _summarize_tickets(_SAMPLE_TICKETS, ticket_reference=None, status_filter="new")
    assert "1 ticket(s)" in result
    assert "TKT-00001" in result
    assert "TKT-00002" not in result


def test_summarize_tickets_counts_by_status_when_no_filters():
    result = _summarize_tickets(_SAMPLE_TICKETS, ticket_reference=None, status_filter=None)
    assert "3 total" in result
    assert "new: 1" in result
    assert "closed: 1" in result
    assert "in_progress: 1" in result


def test_summarize_tickets_model_filter_scopes_everything():
    result = _summarize_tickets(_SAMPLE_TICKETS, ticket_reference=None, status_filter=None, model_filter="IRIS")
    assert "2 total" in result
    assert "STILOS" not in result


def test_summarize_tickets_model_filter_with_no_matches():
    result = _summarize_tickets(_SAMPLE_TICKETS, ticket_reference=None, status_filter=None, model_filter="MAGIS 40")
    assert "No tickets" in result


# ── format_sources ─────────────────────────────────────────────────────────────

class _FakeDoc:
    def __init__(self, metadata):
        self.metadata = metadata


def test_format_sources_with_page_number():
    docs = [_FakeDoc({"source": "manual.pdf", "page": 12})]
    result = format_sources(docs)
    assert "manual.pdf" in result
    assert "page 12" in result


def test_format_sources_with_page_range():
    docs = [_FakeDoc({"source": "manual.pdf", "start_page": 3, "end_page": 5})]
    result = format_sources(docs)
    assert "pages 3-5" in result


def test_format_sources_with_record_number():
    docs = [_FakeDoc({"source": "faq.jsonl", "record_number": 2})]
    result = format_sources(docs)
    assert "record 2" in result
