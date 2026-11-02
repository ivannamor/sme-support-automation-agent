"""
Tests for the migration runner (backend/database/migrate.py) and the first
migration. Each test uses its own throwaway database, created by hand so that
it can start from any state.
"""

import sqlite3

import pytest

from backend import database as db
from backend.database import connection
from backend.database.migrate import (
    DatabaseNewerThanCodeError,
    Migration,
    load_migrations,
    run_migrations,
)


@pytest.fixture
def empty_db(tmp_path, monkeypatch):
    path = tmp_path / "company.db"
    monkeypatch.setattr(connection, "DB_PATH", path)
    return path


def _sql(path, query):
    conn = sqlite3.connect(path)
    try:
        return conn.execute(query).fetchall()
    finally:
        conn.close()


def _version(path) -> int:
    return _sql(path, "PRAGMA user_version")[0][0]


def _schema(path) -> dict[str, set[str]]:
    """Each table's column names. Their order differs on upgraded databases, which is fine."""
    tables = [
        row[0]
        for row in _sql(path, "SELECT name FROM sqlite_master WHERE type='table'")
        if not row[0].startswith("sqlite_")
    ]
    return {t: {row[1] for row in _sql(path, f"PRAGMA table_info({t})")} for t in tables}


def _create_table(conn):
    conn.execute("CREATE TABLE t (x)")


def _add_row(conn):
    conn.execute("INSERT INTO t VALUES (1)")


CREATE = Migration(1, "m0001_create", _create_table)
ADD_ROW = Migration(2, "m0002_add_row", _add_row)


def _fails_halfway(conn):
    conn.execute("INSERT INTO t VALUES (99)")
    conn.execute("ALTER TABLE t ADD COLUMN y")
    raise RuntimeError("boom")


# ── the runner ───────────────────────────────────────────────────────────────


def test_applies_pending_migrations_in_order(empty_db):
    assert run_migrations([CREATE, ADD_ROW]) == 2
    assert _version(empty_db) == 2
    assert _sql(empty_db, "SELECT x FROM t") == [(1,)]


def test_each_migration_runs_only_once(empty_db):
    run_migrations([CREATE, ADD_ROW])
    run_migrations([CREATE, ADD_ROW])
    assert _sql(empty_db, "SELECT x FROM t") == [(1,)]


def test_only_the_new_migration_runs_on_a_later_start(empty_db):
    run_migrations([CREATE])
    assert _version(empty_db) == 1
    run_migrations([CREATE, ADD_ROW])
    assert _version(empty_db) == 2
    assert _sql(empty_db, "SELECT x FROM t") == [(1,)]


def test_a_failed_migration_leaves_nothing_behind(empty_db):
    run_migrations([CREATE])
    failing = Migration(2, "m0002_fails", _fails_halfway)

    with pytest.raises(RuntimeError, match="boom"):
        run_migrations([CREATE, failing])

    assert _version(empty_db) == 1
    assert _sql(empty_db, "SELECT x FROM t") == []
    assert _schema(empty_db)["t"] == {"x"}


def test_it_can_retry_after_a_failure(empty_db):
    run_migrations([CREATE])
    with pytest.raises(RuntimeError):
        run_migrations([CREATE, Migration(2, "m0002_fails", _fails_halfway)])
    assert run_migrations([CREATE, ADD_ROW]) == 2


def test_refuses_a_database_newer_than_the_code(empty_db):
    run_migrations([CREATE, ADD_ROW])
    with pytest.raises(DatabaseNewerThanCodeError, match="version 2"):
        run_migrations([CREATE])
    assert _sql(empty_db, "SELECT x FROM t") == [(1,)]


# ── backups ──────────────────────────────────────────────────────────────────


def test_backs_up_an_existing_database_before_changing_it(empty_db):
    run_migrations([CREATE])
    run_migrations([CREATE, ADD_ROW])

    backup = empty_db.with_name("company.db.bak-v1")
    assert _version(backup) == 1
    assert _sql(backup, "SELECT x FROM t") == []  # the state before the migration


def test_a_new_database_needs_no_backup(empty_db):
    run_migrations([CREATE])
    assert list(empty_db.parent.glob("*.bak-*")) == []


def test_nothing_is_backed_up_when_there_is_nothing_to_do(empty_db):
    run_migrations([CREATE])
    run_migrations([CREATE])
    assert list(empty_db.parent.glob("*.bak-*")) == []


def test_the_backup_includes_writes_still_in_the_wal(empty_db):
    run_migrations([CREATE])
    writer = connection.get_connection()  # stays open, so the WAL isn't checkpointed
    writer.execute("INSERT INTO t VALUES (7)")
    writer.commit()
    try:
        run_migrations([CREATE, ADD_ROW])
    finally:
        writer.close()
    assert _sql(empty_db.with_name("company.db.bak-v1"), "SELECT x FROM t") == [(7,)]


# ── finding the migrations ───────────────────────────────────────────────────


def test_real_migrations_are_numbered_without_gaps():
    versions = [migration.version for migration in load_migrations()]
    assert versions == list(range(1, len(versions) + 1))
    assert versions


def test_a_gap_in_the_numbering_is_rejected(monkeypatch, tmp_path):
    from backend.database import migrations

    (tmp_path / "m0001_a.py").write_text("def upgrade(conn): pass\n")
    (tmp_path / "m0003_c.py").write_text("def upgrade(conn): pass\n")
    monkeypatch.setattr(migrations, "__path__", [str(tmp_path)])
    with pytest.raises(ValueError, match="no gaps"):
        load_migrations()


# ── the first migration ──────────────────────────────────────────────────────


OLD_DATABASE = """
CREATE TABLE inventory (
    id INTEGER PRIMARY KEY AUTOINCREMENT, model TEXT UNIQUE NOT NULL, category TEXT NOT NULL,
    stock INTEGER NOT NULL DEFAULT 0, price REAL NOT NULL DEFAULT 0.0,
    backorder_count INTEGER NOT NULL DEFAULT 0
);
INSERT INTO inventory (model, category, stock, price) VALUES ('IRIS', 'wood', 42, 999.0);
CREATE TABLE clients (
    id INTEGER PRIMARY KEY AUTOINCREMENT, full_name TEXT NOT NULL DEFAULT '',
    email TEXT UNIQUE NOT NULL DEFAULT '', phone TEXT NOT NULL DEFAULT '',
    address TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
);
INSERT INTO clients (full_name, email) VALUES ('Mario Rossi', 'mario@example.com');
CREATE TABLE pending_bookings (
    id INTEGER PRIMARY KEY AUTOINCREMENT, requested_by TEXT NOT NULL DEFAULT 'client',
    model TEXT NOT NULL, date_str TEXT NOT NULL, time_str TEXT NOT NULL,
    notes TEXT NOT NULL DEFAULT '', status TEXT NOT NULL DEFAULT 'pending',
    created_at TEXT NOT NULL DEFAULT (datetime('now')), resolved_at TEXT, resolved_by TEXT
);
INSERT INTO pending_bookings (model, date_str, time_str, notes)
VALUES ('IRIS', '2026-01-15', '10:00', 'a ticket from before migrations');
"""


@pytest.fixture
def old_db(empty_db):
    conn = sqlite3.connect(empty_db)
    conn.executescript(OLD_DATABASE)
    conn.close()
    return empty_db


def test_a_new_database_reaches_the_latest_version(empty_db):
    db.init_db()
    assert _version(empty_db) == load_migrations()[-1].version
    assert len(db.get_inventory()) == 7


def test_an_old_database_reaches_the_same_schema_as_a_new_one(old_db, tmp_path, monkeypatch):
    db.init_db()
    upgraded = _schema(old_db)

    fresh = tmp_path / "fresh.db"
    monkeypatch.setattr(connection, "DB_PATH", fresh)
    db.init_db()
    assert upgraded == _schema(fresh)


def test_an_old_database_keeps_its_data(old_db):
    db.init_db()

    ticket = db.get_booking(1)
    assert ticket["notes"] == "a ticket from before migrations"
    assert ticket["ticket_status"] == "new"  # a new column, with its default
    assert ticket["ticket_type"] == "window"

    product = db.get_product("IRIS")
    assert product["stock"] == 42  # runtime stock isn't overwritten by the seed
    assert product["installation_hours"] == 1.5

    assert _sql(old_db, "SELECT full_name FROM clients") == [("Mario Rossi",)]


def test_the_old_email_only_constraint_on_clients_is_replaced(old_db):
    db.init_db()
    # The same email under another name used to overwrite the first client.
    db.upsert_client("Anna Bianchi", "mario@example.com")
    assert len(_sql(old_db, "SELECT id FROM clients")) == 2


def test_an_old_database_gets_a_backup_first(old_db):
    db.init_db()
    backup = old_db.with_name("company.db.bak-v0")
    assert _version(backup) == 0
    assert "ticket_status" not in _schema(backup)["pending_bookings"]


# ── what still runs at every start ───────────────────────────────────────────


def test_seeding_still_runs_when_the_schema_is_up_to_date(test_db):
    conn = db.get_connection()
    conn.execute("DELETE FROM installation_costs WHERE key = 'vat_rate'")
    conn.execute("DELETE FROM inventory WHERE model = 'IRIS'")
    conn.commit()
    conn.close()

    db.init_db()

    assert db.get_costs()["vat_rate"] == 0.22
    assert db.get_product("IRIS") is not None


def test_clients_missing_from_the_directory_are_restored_at_start(test_db):
    db.create_pending_booking(
        "IRIS", "2030-01-15", "10:00", client_name="Mario Rossi", client_email="m@example.com"
    )
    db.init_db()
    assert [c["full_name"] for c in db.get_clients()] == ["Mario Rossi"]


def test_reset_restores_the_seeded_inventory(test_db):
    conn = db.get_connection()
    conn.execute("UPDATE inventory SET stock = 12345 WHERE model = 'IRIS'")
    conn.commit()
    conn.close()
    first_id = min(item["id"] for item in db.get_inventory())

    db.init_db(reset=True)

    assert db.get_product("IRIS")["stock"] != 12345
    assert len(db.get_inventory()) == 7
    assert min(item["id"] for item in db.get_inventory()) == first_id
