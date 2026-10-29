"""
Tests for backend/checkpointing.py — the process-wide SqliteSaver and the
pruning of idle threads. conftest.py's _isolate_storage points the
checkpoint file at a temporary path for every test.
"""

import datetime
import operator
import sqlite3
from typing import Annotated

from langgraph.graph import START, StateGraph
from typing_extensions import TypedDict

from backend import checkpointing


class _State(TypedDict):
    log: Annotated[list[str], operator.add]


def _graph():
    builder = StateGraph(_State)
    builder.add_node("step", lambda state: {"log": ["step"]})
    builder.add_edge(START, "step")
    return builder.compile(checkpointer=checkpointing.get_checkpointer())


def _config(thread_id: str) -> dict:
    return {"configurable": {"thread_id": thread_id}}


def _set_last_active(thread_id: str, days_ago: int) -> None:
    when = datetime.datetime.now(datetime.UTC) - datetime.timedelta(days=days_ago)
    conn = sqlite3.connect(checkpointing.CHECKPOINT_DB_PATH)
    conn.execute(
        "UPDATE thread_activity SET last_active_at = ? WHERE thread_id = ?",
        (when.isoformat(), thread_id),
    )
    conn.commit()
    conn.close()


def test_checkpointer_is_shared():
    assert checkpointing.get_checkpointer() is checkpointing.get_checkpointer()


def test_state_survives_a_new_checkpointer_on_the_same_file():
    """What a restart looks like: a fresh SqliteSaver still finds the thread."""
    _graph().invoke({"log": ["hello"]}, _config("t1"))

    checkpointing.get_checkpointer().conn.close()
    checkpointing._saver = None

    assert _graph().get_state(_config("t1")).values["log"] == ["hello", "step"]


def test_prune_deletes_only_idle_threads():
    graph = _graph()
    for thread_id in ("old", "recent"):
        graph.invoke({"log": [thread_id]}, _config(thread_id))
        checkpointing.touch_thread(thread_id)
    _set_last_active("old", days_ago=10)

    assert checkpointing.prune_stale_threads(retention_days=7) == 1

    assert graph.get_state(_config("old")).values == {}
    assert graph.get_state(_config("recent")).values["log"] == ["recent", "step"]
    # Already pruned: a second run has nothing left to delete.
    assert checkpointing.prune_stale_threads(retention_days=7) == 0


def test_prune_reads_retention_days_from_env(monkeypatch):
    _graph().invoke({"log": ["x"]}, _config("t1"))
    checkpointing.touch_thread("t1")
    _set_last_active("t1", days_ago=10)

    monkeypatch.setenv("CHECKPOINT_RETENTION_DAYS", "30")
    assert checkpointing.prune_stale_threads() == 0

    monkeypatch.setenv("CHECKPOINT_RETENTION_DAYS", "5")
    assert checkpointing.prune_stale_threads() == 1


def test_touch_thread_refreshes_activity():
    checkpointing.touch_thread("t1")
    _set_last_active("t1", days_ago=10)
    checkpointing.touch_thread("t1")  # the thread is used again

    assert checkpointing.prune_stale_threads(retention_days=7) == 0
