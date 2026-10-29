import importlib


def test_company_config_defaults_are_generalized():
    module = importlib.import_module("backend.assistant")
    assert hasattr(module, "COMPANY_CONTEXT")
    assert module.COMPANY_CONTEXT["domain_name"] == "your company"
    assert module.COMPANY_CONTEXT["product_family"] == "your products"
