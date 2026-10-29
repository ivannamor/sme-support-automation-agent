"""Tests for backend/llm_provider.py's client settings."""

from backend.llm_provider import MAX_RETRIES, REQUEST_TIMEOUT_SECONDS, get_embeddings, get_llm


def test_llm_has_a_timeout_and_retries():
    llm = get_llm()
    assert llm.request_timeout == REQUEST_TIMEOUT_SECONDS
    assert llm.max_retries == MAX_RETRIES


def test_callers_can_override_the_timeout():
    assert get_llm(timeout=120).request_timeout == 120


def test_embeddings_client_is_shared_and_has_a_timeout():
    embeddings = get_embeddings()
    assert embeddings is get_embeddings()
    assert embeddings.request_timeout == REQUEST_TIMEOUT_SECONDS
    assert embeddings.max_retries == MAX_RETRIES
