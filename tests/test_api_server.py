"""
Tests for app/api_server.py — the inventory API. Runs against the isolated
test DB (the API reads through backend.database.get_connection) via FastAPI's
in-process TestClient; no server is started.
"""

import threading

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app import api_server as api
from backend.database import get_connection

API_KEY = "test-key"
AUTH = {"X-API-Key": API_KEY}


@pytest.fixture
def client(test_db, monkeypatch):
    monkeypatch.setenv("INVENTORY_API_KEY", API_KEY)
    return TestClient(api.app)


def _set_stock(model: str, stock: int, backorder: int = 0) -> None:
    conn = get_connection()
    conn.execute(
        "UPDATE inventory SET stock = ?, backorder_count = ? WHERE model = ?",
        (stock, backorder, model),
    )
    conn.commit()
    conn.close()


def _stock(model: str) -> int:
    conn = get_connection()
    row = conn.execute("SELECT stock FROM inventory WHERE model = ?", (model,)).fetchone()
    conn.close()
    return row["stock"]


# ── auth ──────────────────────────────────────────────────────────────────────

def test_missing_key_is_rejected(client):
    assert client.get("/inventory/IRIS").status_code == 403


def test_wrong_key_is_rejected(client):
    assert client.get("/inventory/IRIS", headers={"X-API-Key": "nope"}).status_code == 403


def test_unconfigured_server_fails_closed(test_db, monkeypatch):
    monkeypatch.delenv("INVENTORY_API_KEY", raising=False)
    client = TestClient(api.app)
    # Even with no key configured AND no key sent, nothing is let through.
    assert client.get("/inventory/IRIS").status_code == 503
    assert client.post("/inventory/take", json={"model": "IRIS", "quantity": 1}).status_code == 503


# ── health ────────────────────────────────────────────────────────────────────

def test_health_is_public_and_leaks_nothing(client, test_db):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert str(test_db.parent) not in response.text


def test_health_reports_unavailable_db(client, test_db):
    conn = get_connection()
    conn.execute("DROP TABLE inventory")
    conn.commit()
    conn.close()
    assert client.get("/health").status_code == 503


# ── get ───────────────────────────────────────────────────────────────────────

def test_get_item(client):
    _set_stock("IRIS", 7, backorder=2)
    response = client.get("/inventory/iris", headers=AUTH)
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "IRIS"
    assert body["stock"] == 7
    assert body["backorder"] == 2


def test_get_unknown_item_is_404(client):
    assert client.get("/inventory/NOPE", headers=AUTH).status_code == 404


# ── take ──────────────────────────────────────────────────────────────────────

def test_take_decrements_stock(client):
    _set_stock("IRIS", 5)
    response = client.post("/inventory/take", json={"model": "iris", "quantity": 2}, headers=AUTH)
    assert response.status_code == 200
    assert response.json()["new_stock"] == 3
    assert _stock("IRIS") == 3


def test_take_more_than_available_is_409_and_changes_nothing(client):
    _set_stock("IRIS", 1)
    response = client.post("/inventory/take", json={"model": "IRIS", "quantity": 2}, headers=AUTH)
    assert response.status_code == 409
    assert "Available: 1" in response.json()["detail"]
    assert _stock("IRIS") == 1


def test_take_unknown_item_is_404(client):
    response = client.post("/inventory/take", json={"model": "NOPE", "quantity": 1}, headers=AUTH)
    assert response.status_code == 404


@pytest.mark.parametrize("body", [
    {"model": "IRIS", "quantity": 0},
    {"model": "IRIS", "quantity": -3},
    {"model": "IRIS", "quantity": api.MAX_QUANTITY + 1},
    {"model": "", "quantity": 1},
    {"model": "IRIS'; DROP TABLE inventory;--", "quantity": 1},
    {"quantity": 1},
])
def test_take_rejects_invalid_bodies(client, body):
    _set_stock("IRIS", 5)
    assert client.post("/inventory/take", json=body, headers=AUTH).status_code == 422
    assert _stock("IRIS") == 5


def test_query_string_parameters_are_no_longer_accepted(client):
    _set_stock("IRIS", 5)
    response = client.post("/inventory/take", params={"model": "IRIS", "quantity": 1}, headers=AUTH)
    assert response.status_code == 422
    assert _stock("IRIS") == 5


def test_concurrent_takes_never_oversell(client):
    # 10 threads race for 5 units. With a SELECT-then-UPDATE several of them
    # could pass the stock check together; the atomic UPDATE lets exactly 5 win.
    _set_stock("IRIS", 5)
    workers = 10
    barrier = threading.Barrier(workers)
    outcomes = []
    lock = threading.Lock()

    def worker():
        barrier.wait()
        try:
            api.take_item(api.StockChange(model="IRIS", quantity=1))
            outcome = "ok"
        except HTTPException as exc:
            outcome = exc.status_code
        with lock:
            outcomes.append(outcome)

    threads = [threading.Thread(target=worker) for _ in range(workers)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert outcomes.count("ok") == 5
    assert outcomes.count(409) == 5
    assert _stock("IRIS") == 0


# ── backorder ─────────────────────────────────────────────────────────────────

def test_backorder_accumulates(client):
    _set_stock("IRIS", 0, backorder=1)
    response = client.post("/inventory/backorder", json={"model": "IRIS", "quantity": 3}, headers=AUTH)
    assert response.status_code == 200
    assert response.json()["backorder_total"] == 4


def test_backorder_unknown_item_is_404(client):
    response = client.post("/inventory/backorder", json={"model": "NOPE", "quantity": 1}, headers=AUTH)
    assert response.status_code == 404
