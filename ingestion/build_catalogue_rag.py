# Web-based catalogue ingestion
# This script reads product URLs from data/website.txt and stores the scraped content in Chroma.

import contextlib
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document

from backend.company_config import RAG_DB_DIRECTORY
from backend.llm_provider import get_embeddings
from ingestion.webscraping_utils import build_web_documents, load_urls_from_file

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DB_DIRECTORY = RAG_DB_DIRECTORY
CATALOGUE_COLLECTION = "catalogue_specs"
WEBSITE_URLS_PATH = DATA_DIR / "website.txt"


def build_chroma_db(
    chunks: list[Document],
    db_directory: str = str(DB_DIRECTORY),
    collection_name: str = CATALOGUE_COLLECTION,
) -> Chroma:
    embeddings = get_embeddings()

    existing_db = Chroma(
        persist_directory=db_directory,
        embedding_function=embeddings,
        collection_name=collection_name,
    )
    with contextlib.suppress(Exception):
        existing_db.delete_collection()

    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=db_directory,
        collection_name=collection_name,
    )


def build_knowledge_base(
    urls_path: str | Path = WEBSITE_URLS_PATH,
    db_directory: str = str(DB_DIRECTORY),
    collection_name: str = CATALOGUE_COLLECTION,
    chunk_size: int = 1600,
) -> None:
    urls = load_urls_from_file(urls_path)
    if not urls:
        raise FileNotFoundError(f"No URLs found in {urls_path}")

    print(f"Scraping {len(urls)} webpages for catalogue content...")
    documents = build_web_documents(urls=urls, chunk_size=chunk_size)

    if not documents:
        raise RuntimeError("No catalogue documents were generated from the provided URLs.")

    print(f"Created {len(documents)} catalogue chunks.")
    build_chroma_db(
        chunks=documents,
        db_directory=db_directory,
        collection_name=collection_name,
    )
    print(f"Knowledge base created at: {db_directory}")


if __name__ == "__main__":
    build_knowledge_base()
