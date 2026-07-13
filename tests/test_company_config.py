import importlib
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def test_company_config_defaults_are_generalized():
    module = importlib.import_module("backend.assistant")
    assert hasattr(module, "COMPANY_CONTEXT")
    assert module.COMPANY_CONTEXT["domain_name"] == "your company"
    assert module.COMPANY_CONTEXT["product_family"] == "your products"
