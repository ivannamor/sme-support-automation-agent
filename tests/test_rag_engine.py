"""
Tests for rag/rag_engine.py: its pure functions (deduplication, product-name
detection) and the pipeline's plumbing — one embedding request per search,
parallel LLM calls still counted — on a throwaway Chroma directory with fake
embeddings and a fake LLM. Answer *quality* is left to the agent evals
(evaluation/run_evals.py).
"""

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from backend.usage_tracking import track_usage
from rag import rag_engine
from rag.rag_engine import COLLECTIONS, deduplicate, detect_product
from tests.conftest import FakeEmbeddings, UsageFakeChatModel, fake_llm_with_usage


def test_deduplicate_removes_exact_duplicates():
    doc1 = Document(
        page_content="Same content here" * 20,
        metadata={"source": "a.pdf", "collection": "catalogue_specs"},
    )
    doc2 = Document(
        page_content="Same content here" * 20,
        metadata={"source": "a.pdf", "collection": "catalogue_specs"},
    )
    result = deduplicate([doc1, doc2])
    assert len(result) == 1


def test_deduplicate_keeps_distinct_documents():
    doc1 = Document(page_content="Content A", metadata={"source": "a.pdf"})
    doc2 = Document(page_content="Content B", metadata={"source": "b.pdf"})
    result = deduplicate([doc1, doc2])
    assert len(result) == 2


def test_detect_product_matches_known_product_case_insensitively():
    # IRIS is a real seeded product name in this project's catalogue.
    assert detect_product("Quali sono le specifiche del modello iris?") == "IRIS"


def test_detect_product_matches_multi_word_product_with_loose_spacing():
    # "MAGIS 40" is a real known product; the query may omit the space.
    assert detect_product("dimmi tutto su magis40") == "MAGIS 40"


def test_detect_product_returns_none_when_no_product_mentioned():
    assert detect_product("Come funziona la garanzia in generale?") is None


# ── pipeline plumbing ─────────────────────────────────────────────────────────


class CountingEmbeddings(FakeEmbeddings):
    def __init__(self) -> None:
        self.document_calls: list[list[str]] = []
        self.query_calls: list[str] = []

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_calls.append(list(texts))
        return super().embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        self.query_calls.append(text)
        return super().embed_query(text)


class RagFakeChatModel(UsageFakeChatModel):
    """Answers each of the pipeline's prompts in kind."""

    def _next_reply(self, messages) -> AIMessage:
        prompt = messages[-1].content
        if "alternative search queries" in prompt:
            return AIMessage(content="clean glass\nglass cleaning\nwindow care")
        if prompt.startswith("Translate"):
            return AIMessage(content="come pulire il vetro")
        return AIMessage(content="final answer")


QUESTION = "how do I clean the glass?"


@pytest.fixture
def rag_store(tmp_path, monkeypatch):
    """Maintenance + glossary collections on disk; the FAQ collection stays empty."""
    embeddings = CountingEmbeddings()
    monkeypatch.setattr(rag_engine, "DB_DIRECTORY", str(tmp_path / "chroma"))
    monkeypatch.setattr(rag_engine, "get_embeddings", lambda: embeddings)
    monkeypatch.setattr(rag_engine, "BM25_CACHE", {})
    monkeypatch.setattr(rag_engine, "rerank", lambda query, docs, top_n=8: docs[:top_n])
    rag_engine.load_collection.cache_clear()
    docs = {
        COLLECTIONS["maintenance"]: ["Clean the glass with water.", "Lubricate the hinges."],
        COLLECTIONS["glossary"]: ["Vetro: glass pane.", "Guarnizione: seal."],
    }
    for name, texts in docs.items():
        rag_engine.load_collection(name).add_texts(
            texts, metadatas=[{"source": f"{name}-{i}"} for i in range(len(texts))]
        )
    embeddings.document_calls.clear()
    yield embeddings
    rag_engine.load_collection.cache_clear()


@pytest.fixture
def rag_llm(monkeypatch):
    monkeypatch.setattr(
        rag_engine,
        "get_llm",
        lambda **kwargs: fake_llm_with_usage("openai/gpt-4o-mini", 10, 5, cls=RagFakeChatModel),
    )


def test_search_embeds_all_query_variants_in_one_request(rag_store, rag_llm):
    answer, docs = rag_engine.get_assistant_response(QUESTION)

    assert answer == "final answer"
    assert docs
    # Before: one embedding request per query variant per collection.
    [batch] = rag_store.document_calls
    assert batch[0] == QUESTION
    assert {"clean glass", "window care", "come pulire il vetro"} <= set(batch)
    assert rag_store.query_calls == []


def test_search_by_vector_matches_search_by_text(rag_store):
    name = COLLECTIONS["maintenance"]
    vector = rag_store.embed_documents(["clean the glass"])[0]

    by_vector = rag_engine.retrieve_collection(vector, name, k=2)
    by_text = rag_engine.load_collection(name).similarity_search("clean the glass", k=2)

    assert [d.page_content for d in by_vector] == [d.page_content for d in by_text]


def test_parallel_llm_calls_are_counted_in_the_turn(rag_store, rag_llm, test_db):
    """Multi-query and translation run in worker threads — they must still reach the tracker."""
    with track_usage("client") as usage:
        rag_engine.get_assistant_response(QUESTION)

    # Multi-query + translation + answer, 10 input tokens each.
    assert usage.totals()["input_tokens"] == 30
