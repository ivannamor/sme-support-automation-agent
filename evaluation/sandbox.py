"""
A throwaway environment for eval runs: a freshly seeded company database and
an empty checkpointer in a temp directory, so nothing an eval does (archived
conversations, usage rows, a booking an agent might attempt) reaches the real
db/company.db. A fresh database rather than a copy also keeps runs comparable:
the inventory seed is deterministic, the dev data isn't.

The Chroma knowledge base is NOT copied: retrieval reads it, and nothing on
the agent's path writes to it (the Q&A cache is fed from the UI, not the agent).
"""

import contextlib
import io
import os
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from backend import checkpointing, database


def disable_tracing() -> None:
    """Eval runs don't upload LangSmith traces: they'd swamp the production project."""
    for key in ("LANGCHAIN_TRACING_V2", "LANGSMITH_TRACING"):
        os.environ[key] = "false"


@contextmanager
def sandbox() -> Iterator[Path]:
    saved = (database.DB_PATH, checkpointing.CHECKPOINT_DB_PATH, checkpointing._saver)
    with tempfile.TemporaryDirectory(prefix="evals_", ignore_cleanup_errors=True) as tmp:
        tmp_dir = Path(tmp)
        database.DB_PATH = tmp_dir / "company.db"
        checkpointing.CHECKPOINT_DB_PATH = tmp_dir / "checkpoints.db"
        checkpointing._saver = None
        try:
            with contextlib.redirect_stdout(io.StringIO()):  # init_db prints a banner
                database.init_db(reset=True)
            yield tmp_dir
        finally:
            if checkpointing._saver is not None:
                checkpointing._saver.conn.close()
            database.DB_PATH, checkpointing.CHECKPOINT_DB_PATH, checkpointing._saver = saved
