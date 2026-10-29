from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DB_DIRECTORY = PROJECT_ROOT / "db" / "company_rag_db"

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

collections = {
    "catalogue_specs": "show me product details",
    "installation_guides": "how do I install this",
    "faqs": "what should I know",
    "glossary": "what does this term mean",
}

for name, query in collections.items():
    print(f"\n--- {name} ---")

    db = Chroma(
        persist_directory=str(DB_DIRECTORY),
        embedding_function=embeddings,
        collection_name=name,
    )

    results = db.similarity_search(query, k=2)

    print(f"Query: {query}")
    print(f"Results: {len(results)}")

    for i, r in enumerate(results, 1):
        print(f"\nResult {i}")
        print("Metadata:", r.metadata)
        print("Content:", r.page_content[:200])
