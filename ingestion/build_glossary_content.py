"""
Generates data/glossary/glossary.jsonl — the 50 most relevant technical terms
found across the already-ingested catalogue/installation/maintenance content,
each with a short LLM-generated definition grounded in that same content.

Run once (and again whenever the source PDFs/website content changes
meaningfully):

    python -m ingestion.build_glossary_content

Then rebuild the glossary Chroma collection with the existing loader:

    python -m ingestion.build_faq_and_glossary_rag

This does NOT re-scrape or re-parse anything — it reads the real chunks
already sitting in the catalogue_specs/installation_guides/maintenance_guides
Chroma collections (built earlier in this project by
build_catalogue_rag.py/build_installation_rag.py), so the terms/definitions
are grounded in the actual documentation, not invented from the model's own
knowledge of doors and windows in general.
"""

import json
import re
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma

from backend.company_config import COMPANY_CONTEXT, RAG_DB_DIRECTORY
from backend.llm_provider import get_embeddings, get_llm

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_DIRECTORY = str(RAG_DB_DIRECTORY)
GLOSSARY_PATH = PROJECT_ROOT / "data" / "glossary" / "glossary.jsonl"

SOURCE_COLLECTIONS = [
    COMPANY_CONTEXT["catalogue_collection_name"],
    COMPANY_CONTEXT["installation_collection_name"],
    COMPANY_CONTEXT["maintenance_collection_name"],
]

TARGET_TERM_COUNT = 50
BATCH_SIZE = 6  # chunks per term-extraction LLM call


def load_all_chunks() -> list[str]:
    """Pulls every document's page_content already stored in each source
    collection — no new embedding/query calls, just a raw fetch."""
    embeddings = get_embeddings()
    chunks = []
    for collection_name in SOURCE_COLLECTIONS:
        try:
            store = Chroma(
                persist_directory=DB_DIRECTORY,
                embedding_function=embeddings,
                collection_name=collection_name,
            )
            data = store.get()
            chunks.extend(data.get("documents", []))
            print(f"  loaded {len(data.get('documents', []))} chunks from '{collection_name}'")
        except Exception as exc:
            print(f"  skipping '{collection_name}': {exc}")
    return chunks


_TERM_EXTRACTION_PROMPT = """
Sei un esperto del settore serramenti/edilizia. Dal testo seguente, estrai fino
a 10 termini tecnici specifici del settore (materiali, componenti, misure
tecniche, certificazioni, procedure) che meriterebbero una voce di glossario.

Regole:
- NON includere nomi di prodotto o marchi commerciali.
- Un termine per riga, nient'altro (niente definizioni, niente numerazione).
- Se il testo non contiene termini tecnici rilevanti, rispondi con una riga vuota.

Testo:
{chunk}
"""

_DEFINITION_PROMPT = """
Sei un esperto del settore serramenti/edilizia. Scrivi una definizione breve
(1-2 frasi) del termine "{term}", basandoti SOLO sul contesto fornito qui sotto.
Se il contesto non è sufficiente per una definizione precisa, scrivi la
definizione più accurata possibile restando aderente a quanto scritto.

<context>
{context}
</context>

Definizione di "{term}":
"""


def extract_candidate_terms(chunks: list[str], llm) -> Counter:
    counts: Counter = Counter()
    for i in range(0, len(chunks), BATCH_SIZE):
        batch_text = "\n\n---\n\n".join(chunks[i : i + BATCH_SIZE])[:6000]
        try:
            raw = llm.invoke(_TERM_EXTRACTION_PROMPT.format(chunk=batch_text)).content
        except Exception as exc:
            print(f"  batch {i // BATCH_SIZE} extraction failed: {exc}")
            continue
        for line in raw.splitlines():
            term = line.strip(" -*\t").strip()
            if term and len(term) < 60:
                counts[term.lower()] += 1
        print(
            f"  batch {i // BATCH_SIZE + 1}/{(len(chunks) - 1) // BATCH_SIZE + 1} processed, "
            f"{len(counts)} distinct candidate terms so far"
        )
    return counts


def define_term(term: str, chunks: list[str], llm) -> str:
    relevant = [c for c in chunks if re.search(re.escape(term), c, re.IGNORECASE)]
    context = "\n\n".join(relevant[:3]) if relevant else "(nessun contesto specifico trovato)"
    try:
        return llm.invoke(_DEFINITION_PROMPT.format(term=term, context=context)).content.strip()
    except Exception as exc:
        return f"(definizione non disponibile: {exc})"


def main():
    print("Loading chunks from existing collections:")
    chunks = load_all_chunks()
    if not chunks:
        print(
            "No chunks found in any source collection — run the catalogue/installation/"
            "maintenance ingestion scripts first."
        )
        return

    llm = get_llm(temperature=0, max_tokens=300)

    print(f"\nExtracting candidate terms from {len(chunks)} chunks (batches of {BATCH_SIZE}):")
    counts = extract_candidate_terms(chunks, llm)

    top_terms = [term for term, _ in counts.most_common(TARGET_TERM_COUNT)]
    print(f"\nSelected top {len(top_terms)} terms by frequency across batches.")

    print("Generating grounded definitions:")
    records = []
    for i, term in enumerate(top_terms, start=1):
        definition = define_term(term, chunks, llm)
        records.append({"term": term.title(), "definition": definition})
        print(f"  [{i}/{len(top_terms)}] {term}")

    GLOSSARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(GLOSSARY_PATH, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(records)} glossary entries to {GLOSSARY_PATH}")
    print(
        "Next: python ingestion/build_faq_and_glossary_rag.py  (to (re)build the glossary Chroma collection)"
    )


if __name__ == "__main__":
    main()
