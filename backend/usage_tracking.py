"""
LLM token and cost accounting from the counts the provider reports.

get_llm() attaches USAGE_RECORDER to every model it creates, so every call is
seen — the agent's own, the safety guard's, those made inside tools and those
made directly by dashboard buttons. Each call is priced at the model that
actually served it (tools use the default model, whatever the chat uses).

Where the usage goes:
  - inside track_usage() (one chat turn): added to that turn's UsageTracker,
    written to usage_log as one row per model when the turn ends;
  - outside it, with a role bound in role_var (a dashboard button): written
    to usage_log as one row per call;
  - neither (offline scripts, tests): not recorded.
"""

import logging
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.messages.utils import count_tokens_approximately

from backend.database import log_usage
from backend.llm_provider import OPENROUTER_MODELS, calculate_cost
from backend.log_context import role_var

logger = logging.getLogger("UsageTracking")

UNKNOWN_MODEL = "unknown"


@dataclass
class ModelUsage:
    input_tokens: int = 0
    output_tokens: int = 0


def _cost(model: str, usage: ModelUsage) -> float:
    if model not in OPENROUTER_MODELS:
        logger.warning(f"USAGE_UNPRICED: no price for model {model!r}, counted as $0")
        return 0.0
    return calculate_cost(model, usage.input_tokens, usage.output_tokens)


def _log_rows(role: str, by_model: dict[str, ModelUsage]) -> None:
    for model, usage in by_model.items():
        log_usage(
            role=role,
            model=model,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cost_usd=_cost(model, usage),
        )


class UsageTracker:
    """Token usage of one chat turn, per model."""

    def __init__(self) -> None:
        self.by_model: dict[str, ModelUsage] = {}
        # Tools may run in worker threads.
        self._lock = threading.Lock()

    def add(self, model: str, input_tokens: int, output_tokens: int) -> None:
        with self._lock:
            usage = self.by_model.setdefault(model, ModelUsage())
            usage.input_tokens += input_tokens
            usage.output_tokens += output_tokens

    def totals(self) -> dict[str, int]:
        input_tokens = sum(u.input_tokens for u in self.by_model.values())
        output_tokens = sum(u.output_tokens for u in self.by_model.values())
        return {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }

    def cost_usd(self) -> float:
        return sum(_cost(model, usage) for model, usage in self.by_model.items())


_current_tracker: ContextVar[UsageTracker | None] = ContextVar("usage_tracker", default=None)


@contextmanager
def track_usage(role: str) -> Iterator[UsageTracker]:
    """
    Collects every LLM call made inside the block (worker threads included —
    they run in a copy of this context) and logs it to usage_log on exit.
    A logging failure never breaks the caller.
    """
    tracker = UsageTracker()
    token = _current_tracker.set(tracker)
    try:
        yield tracker
    finally:
        try:
            _current_tracker.reset(token)
        except ValueError:
            # A chat generator closed from another context (e.g. garbage-collected).
            _current_tracker.set(None)
        try:
            _log_rows(role, tracker.by_model)
        except Exception as exc:
            logger.error(f"USAGE_LOG_FAILED: {exc}")


class UsageRecorder(BaseCallbackHandler):
    """
    Reads each call's token counts from the response's usage_metadata. A
    provider that sends none gets an approximate count instead (logged as
    USAGE_ESTIMATED) — over-counting beats hiding a cost.
    """

    def __init__(self) -> None:
        # run_id → (model, approximate input tokens), from start to end of a call.
        self._calls: dict[UUID, tuple[str, int]] = {}
        self._lock = threading.Lock()

    def on_chat_model_start(
        self, serialized, messages, *, run_id, parent_run_id=None, **kwargs
    ) -> None:
        params = kwargs.get("invocation_params") or {}
        model = params.get("model") or params.get("model_name") or UNKNOWN_MODEL
        estimate = sum(count_tokens_approximately(batch) for batch in messages)
        with self._lock:
            self._calls[run_id] = (model, estimate)

    def on_llm_error(self, error, *, run_id, parent_run_id=None, **kwargs) -> None:
        with self._lock:
            self._calls.pop(run_id, None)

    def on_llm_end(self, response, *, run_id, parent_run_id=None, **kwargs) -> None:
        with self._lock:
            model, input_estimate = self._calls.pop(run_id, (UNKNOWN_MODEL, 0))

        messages = [
            g.message for gens in response.generations for g in gens if hasattr(g, "message")
        ]
        usage: Any = next((m.usage_metadata for m in messages if m.usage_metadata), None)
        if usage:
            input_tokens, output_tokens = usage["input_tokens"], usage["output_tokens"]
        else:
            logger.warning(f"USAGE_ESTIMATED: {model} reported no token usage")
            input_tokens, output_tokens = input_estimate, count_tokens_approximately(messages)

        tracker = _current_tracker.get()
        if tracker is not None:
            tracker.add(model, input_tokens, output_tokens)
            return
        role = role_var.get()
        if role is None:
            return
        try:
            _log_rows(role, {model: ModelUsage(input_tokens, output_tokens)})
        except Exception as exc:
            logger.error(f"USAGE_LOG_FAILED: {exc}")


USAGE_RECORDER = UsageRecorder()
