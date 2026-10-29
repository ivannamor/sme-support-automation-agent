import json
from pathlib import Path

from ingestion.webscraping_utils import load_urls_from_file, product_name_from_url

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
WEBSITE_URLS_PATH = DATA_DIR / "website.txt"
CATALOGUE_SECTIONS_PATH = PROJECT_ROOT / "sections" / "catalogue" / "catalogue_sections.json"


def url_to_section(url: str) -> dict:
    product_name = product_name_from_url(url)

    if product_name:
        return {
            "url": url,
            "doc_type": "product",
            "product_name": product_name,
        }

    return {
        "url": url,
        "doc_type": "category",
        "product_name": None,
    }


def build_catalogue_sections(urls_path: Path = WEBSITE_URLS_PATH) -> list:
    urls = load_urls_from_file(urls_path)
    return [url_to_section(url) for url in urls]


def main() -> None:
    sections = build_catalogue_sections()

    CATALOGUE_SECTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CATALOGUE_SECTIONS_PATH, "w", encoding="utf-8") as file:
        json.dump(sections, file, ensure_ascii=False, indent=2)

    product_count = sum(1 for section in sections if section["doc_type"] == "product")
    print(
        f"Wrote {len(sections)} sections ({product_count} products) to: {CATALOGUE_SECTIONS_PATH}"
    )


if __name__ == "__main__":
    main()
