"""
LLM Provider Module - OpenRouter Integration

This module provides a unified interface for using OpenRouter with LangChain,
supporting multiple models with easy switching.
"""

from typing import Any, TypedDict

from backend.secrets_provider import get_secret, require_secret

# Fails loudly at import time if missing — same behavior as before.
OPENROUTER_API_KEY = require_secret("OPENROUTER_API_KEY")


# ============================================================
# MODEL CONFIGURATION
# ============================================================


class ModelInfo(TypedDict):
    name: str
    provider: str
    input_price: float  # $ per 1K tokens
    output_price: float  # $ per 1K tokens
    context_window: int


# Available models on OpenRouter
# Format: "owner/model-name"
OPENROUTER_MODELS: dict[str, ModelInfo] = {
    "openai/gpt-4o-mini": {
        "name": "GPT-4o Mini",
        "provider": "OpenAI",
        "input_price": 0.00015,  # $ per 1K tokens
        "output_price": 0.0006,  # $ per 1K tokens
        "context_window": 128000,
    },
    "google/gemini-2.5-flash": {
        "name": "Gemini 2.5 Flash",
        "provider": "Google",
        "input_price": 0.0003,  # $ per 1K tokens
        "output_price": 0.0025,  # $ per 1K tokens
        "context_window": 1048576,
    },
    "meta-llama/llama-3.3-70b-instruct": {
        "name": "Llama 3.3 70B Instruct",
        "provider": "Meta",
        "input_price": 0.0001,  # $ per 1K tokens
        "output_price": 0.00032,  # $ per 1K tokens
        "context_window": 131072,
    },
    "openai/gpt-4o": {
        "name": "GPT-4o",
        "provider": "OpenAI",
        "input_price": 0.0025,  # $ per 1K tokens
        "output_price": 0.01,  # $ per 1K tokens
        "context_window": 128000,
    },
}

# Default model
DEFAULT_MODEL = get_secret("OPENROUTER_MODEL", "openai/gpt-4o-mini")


# ============================================================
# LLM FACTORY
# ============================================================


def get_llm(
    model_name: str | None = None,
    temperature: float = 0,
    max_tokens: int = 700,
    streaming: bool = False,
    **kwargs,
) -> Any:
    """
    Get a LangChain LLM instance configured for OpenRouter.

    Args:
        model_name: The model identifier (e.g., 'anthropic/claude-3.5-sonnet')
                   If None, uses the default model.
        temperature: Sampling temperature (0-2)
        max_tokens: Maximum tokens to generate
        **kwargs: Additional parameters to pass to the ChatOpenAI constructor

    Returns:
        A LangChain ChatOpenAI instance configured for OpenRouter
    """
    from langchain_openai import ChatOpenAI
    from pydantic import SecretStr

    # Use default if not specified
    if model_name is None:
        model_name = DEFAULT_MODEL

    # Validate model
    if model_name not in OPENROUTER_MODELS:
        raise ValueError(
            f"Unknown model: {model_name}. Available models: {list(OPENROUTER_MODELS.keys())}"
        )

    # Create the LLM with OpenRouter configuration
    # Note: OpenRouter is compatible with OpenAI API, so we use ChatOpenAI
    # with the base_url pointing to OpenRouter
    llm = ChatOpenAI(
        model=model_name,
        temperature=temperature,
        # Field aliases (max_tokens / openai_api_key), the names the type checker sees.
        max_completion_tokens=max_tokens,
        api_key=SecretStr(OPENROUTER_API_KEY),
        base_url="https://openrouter.ai/api/v1",
        streaming=streaming,
        **kwargs,
    )

    return llm


def get_model_info(model_name: str | None = None) -> ModelInfo:
    """
    Get information about a specific model or the default model.

    Args:
        model_name: The model identifier. If None, returns info for default model.

    Returns:
        Dictionary containing model information
    """
    if model_name is None:
        model_name = DEFAULT_MODEL

    if model_name not in OPENROUTER_MODELS:
        raise ValueError(f"Unknown model: {model_name}")

    return OPENROUTER_MODELS[model_name]


def calculate_cost(model_name: str, input_tokens: int, output_tokens: int) -> float:
    """
    Calculate the cost for a given number of tokens.

    Args:
        model_name: The model identifier
        input_tokens: Number of input tokens
        output_tokens: Number of output tokens

    Returns:
        Cost in USD
    """
    if model_name not in OPENROUTER_MODELS:
        raise ValueError(f"Unknown model: {model_name}")

    model_info = OPENROUTER_MODELS[model_name]

    input_cost = (input_tokens / 1000) * model_info["input_price"]
    output_cost = (output_tokens / 1000) * model_info["output_price"]

    return input_cost + output_cost


# ============================================================
# EMBEDDINGS (still using OpenAI for embeddings)
# ============================================================


def get_embeddings() -> Any:
    """
    Get an embeddings instance. Still uses OpenAI as OpenRouter
    doesn't provide free embeddings.

    Returns:
        OpenAIEmbeddings instance
    """
    from langchain_openai import OpenAIEmbeddings

    return OpenAIEmbeddings(model="text-embedding-3-small")
