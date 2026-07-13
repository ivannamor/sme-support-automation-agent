"""
Light-coverage tests for rag/rag_engine.py's pure functions — deduplication
and product-name detection. The full hybrid retrieval pipeline (vector + BM25
+ multi-query + reranking) needs real, populated Chroma collections and is
intentionally left to manual/RAGAS-based evaluation (evaluation/evaluate_rag.py)
rather than unit tests here.
"""

from langchain_core.documents import Document

from rag.rag_engine import deduplicate, detect_product


def test_deduplicate_removes_exact_duplicates():
    doc1 = Document(page_content="Same content here" * 20, metadata={"source": "a.pdf", "collection": "catalogue_specs"})
    doc2 = Document(page_content="Same content here" * 20, metadata={"source": "a.pdf", "collection": "catalogue_specs"})
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
