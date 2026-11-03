"""
Startup order and shutdown of run.py, with the two services replaced by fakes.
"""

import signal

import pytest

import run


class FakeProcess:
    def __init__(self, name, events):
        self.name = name
        self.events = events

    def wait(self):
        # Streamlit runs until someone interrupts it; stopping a process is instant.
        if self.name == "streamlit" and f"{self.name} stopped" not in self.events:
            raise KeyboardInterrupt
        return 0

    def terminate(self):
        self.events.append(f"{self.name} stopped")


@pytest.fixture
def events(monkeypatch):
    recorded = []

    def fake_popen(command, cwd):
        name = "streamlit" if "streamlit" in command else "api"
        recorded.append(f"{name} started")
        return FakeProcess(name, recorded)

    monkeypatch.setattr(run, "ensure_inventory_api_key", lambda: None)
    monkeypatch.setattr(run, "wait_for_api", lambda process: True)
    monkeypatch.setattr(run.subprocess, "Popen", fake_popen)
    monkeypatch.setattr(run.signal, "signal", lambda signum, handler: None)
    monkeypatch.setattr("backend.database.init_db", lambda: recorded.append("db ready"))
    return recorded


def test_database_is_ready_before_the_api_starts(events):
    run.start_services()
    assert events[:3] == ["db ready", "api started", "streamlit started"]


def test_an_interrupt_stops_both_services(events):
    run.start_services()
    assert sorted(events[3:]) == ["api stopped", "streamlit stopped"]


def test_sigterm_is_treated_like_ctrl_c():
    with pytest.raises(KeyboardInterrupt):
        run.stop_on_sigterm(signal.SIGTERM, None)
