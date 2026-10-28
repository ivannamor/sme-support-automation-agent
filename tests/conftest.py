"""
Shared pytest fixtures: an isolated SQLite test database, a scripted fake
chat model for testing backend/graph.py without a real LLM API, and a
deterministic fake embeddings model for testing backend/qa_cache.py without
real OpenAI calls.
"""

import datetime
import hashlib
import math
import os
import sys
from pathlib import Path
from typing import List

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend import database


@pytest.fixture(autouse=True, scope="session")
def _disable_tracing():
    """
    Keeps the suite fully offline: a developer's .env may enable LangSmith
    tracing, which would otherwise upload a run for every graph/LLM call the
    tests make.
    """
    keys = ("LANGCHAIN_TRACING_V2", "LANGSMITH_TRACING")
    saved = {k: os.environ.get(k) for k in keys}
    for k in keys:
        os.environ[k] = "false"
    yield
    for k, v in saved.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def future_date(days: int = 30) -> str:
    """
    A YYYY-MM-DD date safely in the future. Use it wherever the code under
    test rejects past dates, so the test can't silently expire.
    """
    return (datetime.date.today() + datetime.timedelta(days=days)).isoformat()


@pytest.fixture
def test_db(tmp_path, monkeypatch):
    """
    Points backend.database at a throwaway SQLite file and initializes a
    fresh schema — get_connection() reads DB_PATH as a module global at call
    time, so monkeypatching it here fully isolates every test from the real
    dev database (db/company.db is never opened by any test).
    """
    db_path = tmp_path / "test.db"
    monkeypatch.setattr(database, "DB_PATH", db_path)
    database.init_db(reset=True)
    yield db_path


# ── scripted fake chat model (for backend/graph.py tests) ────────────────────

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult


class ScriptedFakeChatModel(BaseChatModel):
    """
    Minimal fake chat model that replays a pre-programmed sequence of
    AIMessage responses (with or without tool_calls), one per invocation —
    lets us drive backend/graph.py's compiled StateGraph through every branch
    (plain answer, read-only tool loop, HITL confirm/execute, HITL cancel)
    without calling a real LLM API. bind_tools() is a no-op returning self,
    matching how build_graph() calls llm.bind_tools(tools, parallel_tool_calls=False).
    """
    responses: List[AIMessage] = []

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        response = self.responses.pop(0)
        return ChatResult(generations=[ChatGeneration(message=response)])

    @property
    def _llm_type(self) -> str:
        return "scripted-fake-chat-model"


@pytest.fixture
def scripted_llm():
    def _make(responses: List[AIMessage]) -> ScriptedFakeChatModel:
        return ScriptedFakeChatModel(responses=list(responses))
    return _make


# ── deterministic fake embeddings (for backend/qa_cache.py tests) ────────────

from langchain_core.embeddings import Embeddings


class FakeEmbeddings(Embeddings):
    """
    Deterministic, offline stand-in for OpenAIEmbeddings — hashes each text
    into a fixed-size float vector, so identical/near-identical strings
    produce identical/similar vectors without any real API call. Good enough
    for testing Chroma-backed similarity search logic (qa_cache.py), not for
    testing actual semantic quality (that's what evaluation/evaluate_rag.py
    with the real embeddings model is for).
    """
    DIMENSIONS = 32

    def _vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.strip().lower().encode("utf-8")).digest()
        raw = [b / 255.0 for b in digest[: self.DIMENSIONS]]
        # Unit-length, like real OpenAI embeddings — keeps Chroma's relevance
        # scores within [0, 1].
        norm = math.sqrt(sum(x * x for x in raw)) or 1.0
        return [x / norm for x in raw]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)
