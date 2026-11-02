"""Inventory seed data, built from the reviewed catalogue section map."""

import json
from pathlib import Path

from backend.company_config import COMPANY_CONTEXT
from backend.database.connection import PROJECT_ROOT

# This is the reviewed section map used by the catalogue RAG ingestion.
CATALOGUE_SECTIONS_PATH = PROJECT_ROOT / "sections" / "catalogue" / "catalogue_sections.json"


def load_catalogue_products(
    sections_path: Path = CATALOGUE_SECTIONS_PATH,
) -> list[str]:
    """
    Loads all product_name values from the reviewed catalogue sections JSON.

    Only sections with doc_type == "product" are used.
    This keeps the SQL inventory aligned with the catalogue RAG metadata.
    """

    if not sections_path.exists():
        raise FileNotFoundError(
            f"Catalogue sections file not found: {sections_path}\n"
            "Run/fix the catalogue ingestion first, or check the sections path."
        )

    with open(sections_path, encoding="utf-8") as file:
        sections = json.load(file)

    products: list[str] = []

    for section in sections:
        if section.get("doc_type") != "product":
            continue

        product_name = section.get("product_name")

        if not product_name:
            continue

        product_name = str(product_name).strip()

        if product_name.lower() == "null":
            continue

        products.append(product_name)

    # Remove duplicates while preserving order.
    return list(dict.fromkeys(products))


def infer_category(model: str) -> str:
    """
    Assigns a simple inventory category from the product name.
    This is mock business data, but stable enough for testing tools.
    """

    model_upper = model.upper()

    if model_upper.startswith(tuple(COMPANY_CONTEXT["product_code_prefixes"])):
        return "product"

    if model_upper.startswith("ABC"):
        return "product"

    if model_upper.startswith("XYZ"):
        return "accessory"

    return "unknown"


def generate_mock_price(model: str) -> float:
    """
    Generates deterministic mock prices.
    Same model -> same price every run.
    """

    model_upper = model.upper()

    if model_upper.startswith("ABC100"):
        base_price = 280
    elif model_upper.startswith("ABC200"):
        base_price = 240
    elif model_upper.startswith("ABC300"):
        base_price = 190
    elif model_upper.startswith(tuple(COMPANY_CONTEXT["product_code_prefixes"])):
        base_price = 140
    elif model_upper.startswith("XYZ"):
        base_price = 45
    else:
        base_price = 100

    variation = sum(ord(char) for char in model) % 90
    return round(base_price + variation, 2)


def generate_mock_stock(model: str) -> int:
    """
    Generates deterministic mock stock.
    Same model -> same stock every run.
    """

    return sum(ord(char) for char in model) % 18


def build_inventory_seed_items() -> list[tuple]:
    """
    Builds seed rows for the inventory table from catalogue products.
    """

    products = load_catalogue_products()

    items = []

    for product in products:
        items.append(
            (
                product,
                infer_category(product),
                generate_mock_stock(product),
                generate_mock_price(product),
                0,
            )
        )

    return items
