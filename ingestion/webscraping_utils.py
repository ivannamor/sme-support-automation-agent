import re
import types
import urllib.request
from html import unescape
from pathlib import Path
from urllib.parse import urlparse

from langchain_core.documents import Document

# A product page lives at /prodotto/<slug>/. Everything else (series/category
# listing pages such as /serie-pvc/ or /complementi/) is not a single sellable
# product and gets no product_name metadata.
PRODUCT_PATH_RE = re.compile(r"^/prodotto/([^/]+)/?$")


def slug_to_product_name(slug: str) -> str:
    return slug.replace("-", " ").upper()


def product_name_from_url(url: str) -> str | None:
    match = PRODUCT_PATH_RE.match(urlparse(url).path)
    return slug_to_product_name(match.group(1)) if match else None


try:
    import requests  # type: ignore
except ImportError:  # pragma: no cover - exercised in minimal environments

    class _RequestsCompat:
        @staticmethod
        def get(url: str, **kwargs):
            timeout = kwargs.get("timeout", 20)
            headers = kwargs.get("headers", {}) or {"User-Agent": "Mozilla/5.0"}
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                body = response.read().decode("utf-8", errors="ignore")
                return types.SimpleNamespace(text=body, raise_for_status=lambda: None)

    requests = _RequestsCompat()  # type: ignore


def load_urls_from_file(urls_path: str | Path) -> list[str]:
    path = Path(urls_path)
    urls = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        urls.append(line)
    return urls


def fetch_url_content(url: str) -> str:
    try:
        response = requests.get(url, timeout=20, headers={"User-Agent": "Mozilla/5.0"})
        if hasattr(response, "text"):
            return response.text
    except Exception:
        pass

    request = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        return response.read().decode("utf-8", errors="ignore")


def extract_text_from_html(html: str, url: str) -> str:
    cleaned = re.sub(r"<script.*?</script>", " ", html, flags=re.I | re.S)
    cleaned = re.sub(r"<style.*?</style>", " ", cleaned, flags=re.I | re.S)
    cleaned = re.sub(r"<[^>]+>", "\n", cleaned)
    cleaned = unescape(cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    lines = [line.strip() for line in cleaned.splitlines() if line.strip()]
    content = "\n".join(lines)
    return f"Source URL: {url}\n\n{content}".strip()


def build_web_documents(urls: list[str], chunk_size: int = 1600) -> list[Document]:
    documents: list[Document] = []
    for url in urls:
        try:
            html = fetch_url_content(url)
            text = extract_text_from_html(html, url)
        except Exception as exc:
            text = f"Source URL: {url}\n\nError while fetching page: {exc}"

        if not text.strip():
            continue

        product_name = product_name_from_url(url)

        chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
        for index, chunk in enumerate(chunks):
            metadata = {
                "source": url,
                "source_type": "webpage",
                "chunk_id": index,
                "chunk_type": "webpage_chunk",
            }
            if product_name:
                metadata["product_name"] = product_name

            documents.append(Document(page_content=chunk, metadata=metadata))
    return documents
