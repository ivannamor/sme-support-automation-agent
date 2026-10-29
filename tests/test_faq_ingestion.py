"""
Tests for ingestion/build_faq_and_glossary_rag.py — re-running it must
replace a collection's records, not add a second copy of each, so the
optional FAQs can be edited and rebuilt at will.
"""

from pathlib import Path

from ingestion import build_faq_and_glossary_rag as ingest
from tests.conftest import FakeEmbeddings

EXAMPLE_FAQ = Path(__file__).resolve().parents[1] / "data" / "faqs" / "faq.example.jsonl"


def test_rebuilding_replaces_the_records(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "get_embeddings", lambda: FakeEmbeddings())
    faq = tmp_path / "faq.jsonl"
    faq.write_text(
        '{"question": "a?", "answer": "b"}\n{"question": "c?", "answer": "d"}\n', encoding="utf-8"
    )
    db_directory = str(tmp_path / "chroma")

    ingest.build_jsonl_collection(str(faq), "faq", "faqs", db_directory=db_directory)
    db = ingest.build_jsonl_collection(str(faq), "faq", "faqs", db_directory=db_directory)

    assert db._collection.count() == 2


def test_delete_collection_empties_it(tmp_path, monkeypatch):
    monkeypatch.setattr(ingest, "get_embeddings", lambda: FakeEmbeddings())
    faq = tmp_path / "faq.jsonl"
    faq.write_text('{"question": "a?", "answer": "b"}\n', encoding="utf-8")
    db_directory = str(tmp_path / "chroma")
    db = ingest.build_jsonl_collection(str(faq), "faq", "faqs", db_directory=db_directory)

    ingest.delete_collection("faqs", db_directory)

    assert "faqs" not in [c.name for c in db._client.list_collections()]


def test_example_faq_file_is_valid():
    records = ingest.load_jsonl(str(EXAMPLE_FAQ))
    assert records
    assert all({"question", "answer"} <= record.keys() for record in records)
