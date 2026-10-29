import contextlib
import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from llama_parse import LlamaParse
from pydantic import BaseModel

from backend.company_config import COMPANY_CONTEXT
from backend.llm_provider import get_embeddings

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
DB_DIRECTORY = PROJECT_ROOT / "db" / "company_rag_db"

# ============================================================
# 1. Data model
# ============================================================

class ParsedPage(BaseModel):
    page: int
    text: str
    source: str

# ============================================================
# 2. Parse PDF page by page
# ============================================================

def parse_pdf_pages(
    pdf_path: str,
    parsing_instruction: str = (
        "This is a technical installation manual. "
        "Preserve headings, steps, warnings, numbered lists, measurements, "
        "tools, procedures, troubleshooting notes, and page structure."
    ),
) -> list[ParsedPage]:
    """
    Parses a PDF page by page using LlamaParse.
    """

    parser = LlamaParse(
        api_key=os.getenv("LLAMA_CLOUD_API_KEY"),
        result_type="markdown",
        parsing_instruction=parsing_instruction,
        verbose=True,
    )

    llama_docs = parser.load_data(pdf_path)

    pages: list[ParsedPage] = []
    source_name = os.path.basename(pdf_path)

    for i, doc in enumerate(llama_docs, start=1):
        page_number = doc.metadata.get("page", i) if hasattr(doc, "metadata") else i

        pages.append(
            ParsedPage(
                page=int(page_number),
                text=doc.text or "",
                source=source_name,
            )
        )

    return pages

# ============================================================
# 3. Chunking: group consecutive pages up to a target size
# ============================================================

def split_installation_manual_into_chunks(
    pages: list[ParsedPage],
    chunk_size: int = 1600,
    doc_type: str = "installation",
) -> list[Document]:
    """
    Splits a manual into page-aligned chunks.

    Pages are accumulated in order until the running text reaches
    chunk_size, then flushed as one chunk. This keeps step-by-step
    instructions on the same page together (unlike splitting mid-page)
    while still avoiding one giant chunk per PDF, which drowns out
    other documents during retrieval/reranking.

    doc_type tags each chunk as "installation" (professional installation
    procedures, installer/manager-only) or "maintenance" (product care/
    maintenance guidance, accessible to everyone) — the two ingested from
    separate data/ subfolders into separate Chroma collections.
    """

    if not pages:
        raise ValueError("No pages were parsed from the manual.")

    label = "Installation Manual" if doc_type == "installation" else "Maintenance Manual"
    documents: list[Document] = []
    current_pages: list[ParsedPage] = []
    current_length = 0

    def flush() -> None:
        if not current_pages:
            return

        text = "\n\n".join(
            f"--- PAGE {page.page} ---\n{page.text}"
            for page in current_pages
        )
        start_page, end_page = current_pages[0].page, current_pages[-1].page

        documents.append(
            Document(
                page_content=(
                    f"Document type: {doc_type}_manual\n"
                    f"Section: {label} (pages {start_page}-{end_page})\n"
                    f"Source: {current_pages[0].source}\n"
                    f"Pages: {start_page}-{end_page}\n\n"
                    f"{text}"
                ),
                metadata={
                    "source_type": f"{doc_type}_manual",
                    "doc_type": doc_type,
                    "section_name": f"Pages {start_page}-{end_page}",
                    "start_page": start_page,
                    "end_page": end_page,
                    "source": current_pages[0].source,
                    "chunk_id": len(documents),
                },
            )
        )

    for page in pages:
        current_pages.append(page)
        current_length += len(page.text)

        if current_length >= chunk_size:
            flush()
            current_pages = []
            current_length = 0

    flush()

    return documents

# ============================================================
# 4. Build Chroma collection
# ============================================================

def build_installation_manual_collection(
    pdf_paths: list[str],
    db_directory: str = str(DB_DIRECTORY),
    collection_name: str = "installation_guides",
    doc_type: str = "installation",
    parsing_instruction: str = (
        "This is a technical installation manual. "
        "Preserve headings, steps, warnings, numbered lists, measurements, "
        "tools, procedures, troubleshooting notes, and page structure."
    ),
) -> Chroma:
    """
    Parses the given PDFs and stores them in a Chroma collection. Called once
    per data/ subfolder (installation manuals vs. maintenance manuals) with a
    different collection_name/doc_type/parsing_instruction, so the two stay
    in separate collections — installation procedures are installer/manager-only,
    maintenance guidance is accessible to everyone (see tools.py's
    search_technical_manuals vs. search_installation_manuals_tool).
    """

    if not pdf_paths:
        raise ValueError("No PDF files were provided for ingestion.")

    all_chunks: list[Document] = []

    for pdf_path in pdf_paths:
        pages = parse_pdf_pages(pdf_path, parsing_instruction=parsing_instruction)
        manual_docs = split_installation_manual_into_chunks(pages, doc_type=doc_type)
        all_chunks.extend(manual_docs)

        print(f"\nCreated {len(manual_docs)} chunks from {Path(pdf_path).name}.")

        for doc in manual_docs:
            print(
                f"- {doc.metadata['section_name']} "
                f"(pages {doc.metadata['start_page']}-{doc.metadata['end_page']})"
            )

    embeddings = get_embeddings()

    existing_db = Chroma(
        persist_directory=db_directory,
        embedding_function=embeddings,
        collection_name=collection_name,
    )
    with contextlib.suppress(Exception):
        existing_db.delete_collection()

    vector_db = Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        persist_directory=db_directory,
        collection_name=collection_name,
    )

    print(f"\nManuals stored in collection: {collection_name}")
    print(f"Database directory: {db_directory}")

    return vector_db

# ============================================================
# 5. Main execution
# ============================================================

if __name__ == "__main__":
    installation_dir = DATA_DIR / "installation manuals"
    maintenance_dir  = DATA_DIR / "maintainance manuals"

    installation_pdfs = sorted(str(p) for p in installation_dir.glob("*.pdf"))
    maintenance_pdfs  = sorted(str(p) for p in maintenance_dir.glob("*.pdf"))

    if not installation_pdfs and not maintenance_pdfs:
        raise FileNotFoundError(
            f"No PDF files found in {installation_dir} or {maintenance_dir}."
        )

    if installation_pdfs:
        build_installation_manual_collection(
            pdf_paths=installation_pdfs,
            db_directory=str(DB_DIRECTORY),
            collection_name=COMPANY_CONTEXT["installation_collection_name"],
            doc_type="installation",
            parsing_instruction=(
                "This is a technical installation manual for certified installers. "
                "Preserve headings, steps, warnings, numbered lists, measurements, "
                "tools, procedures, troubleshooting notes, and page structure."
            ),
        )
    else:
        print(f"No PDFs found in {installation_dir} — skipping installation_guides.")

    if maintenance_pdfs:
        build_installation_manual_collection(
            pdf_paths=maintenance_pdfs,
            db_directory=str(DB_DIRECTORY),
            collection_name=COMPANY_CONTEXT["maintenance_collection_name"],
            doc_type="maintenance",
            parsing_instruction=(
                "This is a product care/maintenance manual for end customers. "
                "Preserve headings, cleaning/care steps, product recommendations, "
                "warnings, and page structure."
            ),
        )
    else:
        print(f"No PDFs found in {maintenance_dir} — skipping maintenance_guides.")