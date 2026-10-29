"""
Semantic QA cache backed by a dedicated Chroma collection.

Questions are embedded and stored; answers live in Chroma document metadata.
Similarity threshold is intentionally tight so only near-identical questions hit
the cache and avoid retriggering the full RAG pipeline.
"""

import logging
from datetime import UTC, datetime
from pathlib import Path

from langchain_chroma import Chroma

from backend.database import get_connection
from backend.llm_provider import get_embeddings

logger = logging.getLogger("QACache")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_DIRECTORY = str(PROJECT_ROOT / "db" / "company_rag_db")
COLLECTION_NAME = "qa_cache"
SIMILARITY_THRESHOLD = 0.88  # cosine similarity; 1.0 = identical
# 0.88 catches clear rephrasing of the same question (e.g. "What is X?" / "Tell me about X?")
# while keeping "What is X?" and "How do I install X?" apart (~0.78-0.83).


# ── internal helpers ──────────────────────────────────────────────────────────


def _store() -> Chroma:
    return Chroma(
        persist_directory=DB_DIRECTORY,
        embedding_function=get_embeddings(),
        collection_name=COLLECTION_NAME,
    )


def _approved_cache_hit(question: str) -> bool:
    """True if an approved entry already covers this question."""
    try:
        results = _store().similarity_search_with_relevance_scores(question, k=1)
        if results:
            _, score = results[0]
            return score >= SIMILARITY_THRESHOLD
    except Exception as exc:
        logger.warning(f"QA_CACHE_DUP_CHECK_FAILED: {exc}")
    return False


# ── public API ────────────────────────────────────────────────────────────────


def lookup(query: str) -> str | None:
    """
    Returns the cached answer if a sufficiently similar question exists,
    or None if the cache has no relevant entry.
    """
    try:
        results = _store().similarity_search_with_relevance_scores(query, k=1)
        if results:
            doc, score = results[0]
            if score >= SIMILARITY_THRESHOLD:
                logger.info(f"QA_CACHE_HIT: score={score:.4f} q='{query[:70]}'")
                return doc.metadata.get("answer", "")
    except Exception as exc:
        logger.warning(f"QA_CACHE_LOOKUP_FAILED: {exc}")
    return None


def submit_for_review(question: str, answer: str, source_role: str = "") -> bool:
    """
    Adds a Q&A pair to the qa_pending queue for manager approval.

    Skips silently when:
    - An approved cache entry already covers the question (similarity ≥ threshold)
    - An exact-match entry is already waiting in the pending queue
    """
    # Guard 1: already in approved cache
    if _approved_cache_hit(question):
        logger.debug(f"QA_PENDING_SKIP_CACHED: q='{question[:70]}'")
        return False

    # Guard 2: exact duplicate already pending
    conn = get_connection()
    existing = conn.execute(
        "SELECT id FROM qa_pending WHERE status = 'pending' AND question = ?",
        (question,),
    ).fetchone()
    conn.close()
    if existing:
        logger.debug(f"QA_PENDING_SKIP_EXACT_DUP: q='{question[:70]}'")
        return False

    conn = get_connection()
    conn.execute(
        "INSERT INTO qa_pending (question, answer, source_role) VALUES (?, ?, ?)",
        (question, answer, source_role),
    )
    conn.commit()
    conn.close()
    logger.info(f"QA_PENDING_ADDED: role={source_role} q='{question[:70]}'")
    return True


def approve(pending_id: int, approved_by: str = "manager") -> bool:
    """
    Writes the Q&A pair into the Chroma cache and marks it approved.
    Rejects silently (marks rejected) if a near-duplicate is already cached.
    """
    conn = get_connection()
    row = conn.execute(
        "SELECT question, answer FROM qa_pending WHERE id = ? AND status = 'pending'",
        (pending_id,),
    ).fetchone()

    if not row:
        conn.close()
        return False

    question, answer = row["question"], row["answer"]

    # Final guard: may have been cached by a parallel approval
    if _approved_cache_hit(question):
        conn.execute("UPDATE qa_pending SET status = 'rejected' WHERE id = ?", (pending_id,))
        conn.commit()
        conn.close()
        logger.info(f"QA_CACHE_APPROVE_SKIPPED_DUP: id={pending_id}")
        return False

    try:
        # Embed the QUESTION; store the ANSWER in metadata.
        _store().add_texts(
            texts=[question],
            metadatas=[
                {
                    "answer": answer,
                    "approved_by": approved_by,
                    "approved_at": datetime.now(UTC).isoformat(),
                }
            ],
        )
    except Exception as exc:
        conn.close()
        logger.error(f"QA_CACHE_CHROMA_WRITE_FAILED: {exc}", exc_info=True)
        return False

    conn.execute("UPDATE qa_pending SET status = 'approved' WHERE id = ?", (pending_id,))
    conn.commit()
    conn.close()
    logger.info(f"QA_CACHE_APPROVED: id={pending_id} by={approved_by}")
    return True


def reject(pending_id: int) -> bool:
    conn = get_connection()
    cursor = conn.execute(
        "UPDATE qa_pending SET status = 'rejected' WHERE id = ? AND status = 'pending'",
        (pending_id,),
    )
    updated = cursor.rowcount > 0
    conn.commit()
    conn.close()
    return updated


def get_pending() -> list:
    conn = get_connection()
    rows = conn.execute(
        """
        SELECT id, question, answer, source_role, created_at
        FROM qa_pending
        WHERE status = 'pending'
        ORDER BY created_at DESC
        """,
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]
