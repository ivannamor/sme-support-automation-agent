import os
from typing import Dict, List


def _env(name: str, default: str) -> str:
    return os.getenv(name, default)


COMPANY_CONTEXT: Dict[str, object] = {
    "company_name": _env("COMPANY_NAME", "Your Company"),
    "company_short_name": _env("COMPANY_SHORT_NAME", "the company"),
    "domain_name": _env("DOMAIN_NAME", "your company"),
    "product_family": _env("PRODUCT_FAMILY", "your products"),
    "product_family_plural": _env("PRODUCT_FAMILY_PLURAL", "products"),
    "knowledge_domain": _env("KNOWLEDGE_DOMAIN", "products, services, and documentation"),
    "catalogue_label": _env("CATALOGUE_LABEL", "catalogue"),
    "assistant_name": _env("ASSISTANT_NAME", "Assistant"),
    "rag_db_directory_name": _env("RAG_DB_DIRECTORY_NAME", "company_rag_db"),
    "database_file_name": _env("DATABASE_FILE_NAME", "company.db"),
    "catalogue_collection_name": _env("CATALOGUE_COLLECTION_NAME", "catalogue_specs"),
    "installation_collection_name": _env("INSTALLATION_COLLECTION_NAME", "installation_guides"),
    "maintenance_collection_name": _env("MAINTENANCE_COLLECTION_NAME", "maintenance_guides"),
    "faq_collection_name": _env("FAQ_COLLECTION_NAME", "faqs"),
    "glossary_collection_name": _env("GLOSSARY_COLLECTION_NAME", "glossary"),
    # Language the ingested documentation/catalogue is actually written in —
    # used to add a same-language retrieval query when a client asks in a
    # different UI language, since cross-lingual embedding/BM25 matching is
    # measurably weaker than same-language matching (see rag_engine.py's
    # translate_for_retrieval).
    "content_language": _env("CONTENT_LANGUAGE", "English"),
    "product_code_pattern": _env(
        "PRODUCT_CODE_PATTERN",
        r"\b[A-Z]{1,6}\d{2,4}(?:[A-Z0-9-]+)*\b",
    ),
}

COMPANY_CONTEXT["product_code_prefixes"] = [
    prefix.strip().upper()
    for prefix in _env("PRODUCT_CODE_PREFIXES", "ABC,XYZ").split(",")
    if prefix.strip()
]


def get_company_context() -> Dict[str, object]:
    return COMPANY_CONTEXT
