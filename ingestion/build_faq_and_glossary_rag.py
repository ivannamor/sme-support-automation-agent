import json
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document

from backend.company_config import COMPANY_CONTEXT
from backend.llm_provider import get_embeddings

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DB_DIRECTORY = PROJECT_ROOT / "db" / "company_rag_db"


def load_jsonl(file_path: str) -> list[dict[str, Any]]:
    records = []

    with open(file_path, encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} in {file_path}") from exc

    return records


def record_to_text(record: dict[str, Any]) -> str:
    """
    Schema-agnostic conversion.
    No hardcoded keys.
    """

    return json.dumps(record, ensure_ascii=False, indent=2)


def jsonl_to_documents(
    file_path: str,
    source_type: str,
) -> list[Document]:
    records = load_jsonl(file_path)
    source_name = os.path.basename(file_path)

    documents = []

    for i, record in enumerate(records):
        page_content = f"Source type: {source_type}\nRecord number: {i}\n\n{record_to_text(record)}"

        metadata = {
            "source": source_name,
            "source_type": source_type,
            "record_number": i,
        }

        documents.append(
            Document(
                page_content=page_content,
                metadata=metadata,
            )
        )

    return documents


def build_jsonl_collection(
    file_path: str,
    source_type: str,
    collection_name: str,
    db_directory: str = "./company_rag_db",
) -> Chroma:
    documents = jsonl_to_documents(
        file_path=file_path,
        source_type=source_type,
    )

    print(f"Loaded {len(documents)} JSONL records from {file_path}")

    embeddings = get_embeddings()

    vector_db = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=db_directory,
        collection_name=collection_name,
    )

    print(f"Stored collection: {collection_name}")
    print(f"Database directory: {db_directory}")

    return vector_db


if __name__ == "__main__":
    json_faq_path = DATA_DIR / "faqs" / "faq.jsonl"
    json_glossary_path = DATA_DIR / "glossary" / "glossary.jsonl"

    # The FAQ half is independent of the glossary half — isolate it so a
    # missing/not-yet-created data/faqs/faq.jsonl doesn't prevent the
    # glossary collection from being built.
    try:
        build_jsonl_collection(
            file_path=str(json_faq_path),
            source_type="faq",
            collection_name=COMPANY_CONTEXT["faq_collection_name"],
            db_directory=str(DB_DIRECTORY),
        )
    except FileNotFoundError:
        print(f"Skipping FAQ collection — {json_faq_path} not found yet.")

    build_jsonl_collection(
        file_path=str(json_glossary_path),
        source_type="glossary",
        collection_name=COMPANY_CONTEXT["glossary_collection_name"],
        db_directory=str(DB_DIRECTORY),
    )
