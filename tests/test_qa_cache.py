"""
Tests for backend/qa_cache.py — the semantic Q&A cache. Uses the isolated
test_db fixture for the qa_pending SQL table, and a deterministic FakeEmbeddings
model (see conftest.py) against a throwaway Chroma directory, so no test makes
a real OpenAI embeddings call.
"""

import pytest

from backend import qa_cache
from tests.conftest import FakeEmbeddings


@pytest.fixture
def isolated_qa_cache(test_db, tmp_path, monkeypatch):
    """Combines the SQL test_db fixture with an isolated Chroma dir + fake embeddings."""
    monkeypatch.setattr(qa_cache, "DB_DIRECTORY", str(tmp_path / "qa_chroma"))
    monkeypatch.setattr(qa_cache, "get_embeddings", lambda: FakeEmbeddings())
    yield


def test_submit_for_review_adds_to_pending_queue(isolated_qa_cache):
    added = qa_cache.submit_for_review(
        "What is IRIS?", "IRIS is a window model.", source_role="client"
    )
    assert added is True
    pending = qa_cache.get_pending()
    assert len(pending) == 1
    assert pending[0]["question"] == "What is IRIS?"


def test_submit_for_review_skips_exact_duplicate_pending(isolated_qa_cache):
    qa_cache.submit_for_review("What is IRIS?", "Answer 1", source_role="client")
    added_again = qa_cache.submit_for_review("What is IRIS?", "Answer 1", source_role="client")
    assert added_again is False
    assert len(qa_cache.get_pending()) == 1


def test_approve_writes_to_cache_and_lookup_finds_it(isolated_qa_cache):
    qa_cache.submit_for_review("What is IRIS?", "IRIS is a window model.", source_role="client")
    pending_id = qa_cache.get_pending()[0]["id"]

    approved = qa_cache.approve(pending_id, approved_by="manager1")
    assert approved is True
    assert qa_cache.get_pending() == []  # no longer pending

    result = qa_cache.lookup("What is IRIS?")
    assert result == "IRIS is a window model."


def test_lookup_returns_none_for_unrelated_question(isolated_qa_cache):
    qa_cache.submit_for_review("What is IRIS?", "IRIS is a window model.", source_role="client")
    qa_cache.approve(qa_cache.get_pending()[0]["id"], approved_by="manager1")

    # A totally unrelated question must not hash-collide into a false hit —
    # FakeEmbeddings is deterministic per-string, not semantically similar.
    result = qa_cache.lookup("How do I reset my password on an unrelated system?")
    assert result is None


def test_reject_removes_from_pending_without_caching(isolated_qa_cache):
    qa_cache.submit_for_review("What is IRIS?", "IRIS is a window model.", source_role="client")
    pending_id = qa_cache.get_pending()[0]["id"]

    rejected = qa_cache.reject(pending_id)
    assert rejected is True
    assert qa_cache.get_pending() == []
    assert qa_cache.lookup("What is IRIS?") is None


def test_submit_for_review_skips_when_already_cached(isolated_qa_cache):
    qa_cache.submit_for_review("What is IRIS?", "IRIS is a window model.", source_role="client")
    qa_cache.approve(qa_cache.get_pending()[0]["id"], approved_by="manager1")

    # Submitting the same question again should be skipped — an approved
    # cache entry already covers it (checked via _approved_cache_hit).
    added = qa_cache.submit_for_review(
        "What is IRIS?", "A slightly different phrasing of the answer.", source_role="client"
    )
    assert added is False
    assert qa_cache.get_pending() == []


def test_similarity_is_the_true_cosine(isolated_qa_cache):
    """The collection stores squared-L2 distances; _closest must turn them into cosine."""
    qa_cache.submit_for_review("What is IRIS?", "IRIS is a window model.", source_role="client")
    qa_cache.approve(qa_cache.get_pending()[0]["id"])
    embeddings = FakeEmbeddings()
    a = embeddings.embed_query("What is IRIS?")
    b = embeddings.embed_query("How do I clean IRIS?")
    cosine = sum(x * y for x, y in zip(a, b, strict=True))

    match = qa_cache._closest("How do I clean IRIS?")

    assert match is not None
    assert match[1] == pytest.approx(cosine, abs=1e-5)
