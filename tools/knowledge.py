"""Documentation search and the general-knowledge fallback."""

from langchain.tools import tool
from pydantic import BaseModel, Field

from backend.company_config import COMPANY_CONTEXT
from backend.llm_provider import get_llm
from rag.rag_engine import get_assistant_response
from tools.common import logger


def format_sources(docs) -> str:
    """
    Converts retrieved RAG documents into a compact source string.
    """

    sources = []

    for doc in docs:
        metadata = doc.metadata

        source = metadata.get("source", "Unknown source")
        section = metadata.get("section_name")
        product = metadata.get("product_name")
        page = metadata.get("page")
        start_page = metadata.get("start_page")
        end_page = metadata.get("end_page")
        record = metadata.get("record_number")
        collection = metadata.get("collection")

        if page is not None:
            page_info = f"page {page}"
        elif start_page is not None and end_page is not None:
            page_info = f"pages {start_page}-{end_page}"
        elif record is not None:
            page_info = f"record {record}"
        else:
            page_info = "location unknown"

        parts = [source, page_info]

        if collection:
            parts.append(f"collection: {collection}")
        if section:
            parts.append(f"section: {section}")
        if product:
            parts.append(f"product: {product}")

        sources.append(" | ".join(parts))

    # Remove duplicates while preserving order
    unique_sources = list(dict.fromkeys(sources))

    return " ; ".join(unique_sources)


# Two hard-separated document sets, matching the two data/ subfolders ingested
# by ingestion/build_installation_rag.py into two Chroma collections:
# - "public": product catalogue, maintenance/care guidance, FAQs, glossary —
#   safe for clients. Never includes professional installation procedures.
# - "installation": step-by-step installation manuals — installer/manager only.
# allowed_collections is a hard whitelist enforced inside route_query_hybrid,
# so a client's search_technical_manuals call can never surface installation
# content regardless of query wording — this isn't just a prompt instruction.
_PUBLIC_COLLECTIONS = [
    COMPANY_CONTEXT["catalogue_collection_name"],
    COMPANY_CONTEXT["faq_collection_name"],
    COMPANY_CONTEXT["glossary_collection_name"],
    COMPANY_CONTEXT["maintenance_collection_name"],
]
_INSTALLATION_COLLECTIONS = [
    COMPANY_CONTEXT["installation_collection_name"],
    COMPANY_CONTEXT["faq_collection_name"],
    COMPANY_CONTEXT["glossary_collection_name"],
]


_QUERY_EXPANSION_PROMPT = """Rewrite the user's question into a clear, fully self-contained, specific
question suitable for a document search — expand any short or colloquial
phrasing ("tell me about X", "info on X", "what about X") into a complete
question naming what should be covered (characteristics, specifications,
materials, options), without adding facts not implied by the original
question. If the question is about a product series/collection/category
rather than a single item, explicitly ask about "the products belonging to
the X series/collection" — a series query answered with only one product's
chunk is incomplete. If the question is already a clear, fully-specified
question, return it unchanged. Preserve the question's original language.

Question: {query}

Rewritten question (output ONLY the rewritten question, nothing else):"""


def _expand_query(query: str) -> str:
    """
    Deterministically enforces the "expand terse/colloquial queries before
    searching" rule inside the tool itself, rather than leaving it to the
    calling agent's discretion — a short query like "tell me about X" was
    observed to retrieve noticeably worse (and sometimes unusably worse)
    results than an equivalent fuller phrasing of the exact same question,
    since retrieval ranking is sensitive to query wording. Best-effort: on
    failure, falls back to the original query rather than blocking search.
    """
    try:
        llm = get_llm(temperature=0, max_tokens=100)
        expanded = llm.invoke(_QUERY_EXPANSION_PROMPT.format(query=query)).content.strip()
        return expanded or query
    except Exception as exc:
        logger.warning(f"QUERY_EXPANSION_FAILED: {exc} — using the original query")
        return query


@tool
def search_technical_manuals(query: str) -> str:
    """
    Searches the product catalogue, maintenance/care guidance, FAQs, and glossary.

    Use this for:
    - product specifications
    - maintenance and care instructions
    - technical definitions
    - documentation-based FAQs

    Does NOT cover professional installation procedures — those are only
    available to installers/managers via search_installation_manuals_tool.
    """
    try:
        answer, docs = get_assistant_response(
            query=_expand_query(query),
            k=8,
            top_n=16,
            use_bm25=True,
            use_multi_query=True,
            allowed_collections=_PUBLIC_COLLECTIONS,
        )

        sources = format_sources(docs)

        return f"{answer}\n\nSOURCE_DATA_SPLIT {sources}"

    except Exception as exc:
        logger.error(f"RAG_TOOL_FAILURE: {str(exc)}", exc_info=True)
        return "Error: The technical documentation system is currently unavailable."


@tool
def search_installation_manuals_tool(query: str) -> str:
    """
    Searches professional installation manuals (step-by-step installation
    procedures). Installer/manager-only — not available to clients.

    Use this for installation steps, mounting procedures, tools required,
    and technical installation troubleshooting.
    """
    try:
        answer, docs = get_assistant_response(
            query=_expand_query(query),
            k=8,
            top_n=16,
            use_bm25=True,
            use_multi_query=True,
            allowed_collections=_INSTALLATION_COLLECTIONS,
        )

        sources = format_sources(docs)

        return f"{answer}\n\nSOURCE_DATA_SPLIT {sources}"

    except Exception as exc:
        logger.error(f"INSTALLATION_RAG_TOOL_FAILURE: {str(exc)}", exc_info=True)
        return "Error: The installation documentation system is currently unavailable."


_SCOPE_CHECK_PROMPT = f"""
Decide whether the question below is related to {COMPANY_CONTEXT["company_name"]}, its {COMPANY_CONTEXT["product_family"]},
its services, documentation, operations, or the broader domain of {COMPANY_CONTEXT["knowledge_domain"]}.

Reply with only YES or NO.

Question: {{question}}
"""

_GENERAL_ANSWER_PROMPT = f"""
You are a specialist assistant for {COMPANY_CONTEXT["company_name"]} with broad general knowledge about its {COMPANY_CONTEXT["product_family"]} and related services.
Answer the following question clearly and factually.

Rules:
- Stay focused on the company's offerings, documentation, and domain knowledge.
- Do NOT invent specific product codes, brand specs, or certification numbers unless they are present in the question.
- Keep the answer practical and concise.

Question: {{question}}

Answer:
"""


class GeneralKnowledgeInput(BaseModel):
    question: str = Field(
        description="A general question about doors, hardware, building systems, or construction materials."
    )


@tool(args_schema=GeneralKnowledgeInput)
def general_door_knowledge_tool(question: str) -> str:
    """
    Answers general questions about doors, hardware, building systems, and construction
    using the LLM's broad knowledge — NOT the product documentation.

    Use this ONLY as a last resort when:
    - search_technical_manuals returned insufficient information, OR
    - the question is clearly about general door/building knowledge with no specific
      product or installation procedure in scope.

    Do NOT use this for product specifications, installation steps, inventory,
    bookings, or anything covered by other tools.
    """
    # temperature=0.2 for tone parity with the main conversational assistant —
    # this is explicitly NOT grounded in company documentation (a general-knowledge
    # fallback), so a slightly less rigid tone is appropriate here too.
    llm = get_llm(temperature=0.2, max_tokens=400)

    # Safety net: scope check
    try:
        scope_reply = (
            llm.invoke(_SCOPE_CHECK_PROMPT.format(question=question)).content.strip().upper()
        )
        if not scope_reply.startswith("YES"):
            return (
                "I can only answer questions related to doors, hardware, building systems, "
                "and construction. Please ask something in those areas."
            )
    except Exception as exc:
        logger.warning(f"GENERAL_TOOL_SCOPE_CHECK_FAILED: {exc}")
        # Fail closed — don't answer if scope check errors
        return "Unable to verify whether this question is in scope. Please try rephrasing."

    # Generate answer
    try:
        answer = llm.invoke(_GENERAL_ANSWER_PROMPT.format(question=question)).content.strip()
    except Exception as exc:
        logger.error(f"GENERAL_TOOL_ANSWER_FAILED: {exc}", exc_info=True)
        return "An error occurred while generating the answer. Please try again."

    logger.info(f"GENERAL_TOOL_USED: question='{question[:80]}'")
    return f"{answer}\n\nGENERAL_LLM_SPLIT"
