import sys
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_chroma import Chroma

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.llm_provider import get_embeddings
from ingestion.webscraping_utils import build_web_documents, load_urls_from_file

DATA_DIR = PROJECT_ROOT / "data"
DB_DIRECTORY = PROJECT_ROOT / "db" / "company_rag_db"
CATALOGUE_COLLECTION = "catalogue_specs"
WEBSITE_URLS_PATH = DATA_DIR / "website.txt"

# Web-based catalogue ingestion
# This script reads product URLs from data/website.txt and stores the scraped content in Chroma.


load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.llm_provider import get_embeddings
from ingestion.webscraping_utils import build_web_documents, load_urls_from_file

DATA_DIR = PROJECT_ROOT / "data"
DB_DIRECTORY = PROJECT_ROOT / "db" / "company_rag_db"
CATALOGUE_COLLECTION = "catalogue_specs"
WEBSITE_URLS_PATH = DATA_DIR / "website.txt"


def build_chroma_db(
    chunks: List[Document],
    db_directory: str = str(DB_DIRECTORY),
    collection_name: str = CATALOGUE_COLLECTION,
) -> Chroma:
    embeddings = get_embeddings()

    existing_db = Chroma(
        persist_directory=db_directory,
        embedding_function=embeddings,
        collection_name=collection_name,
    )
    try:
        existing_db.delete_collection()
    except Exception:
        pass

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
