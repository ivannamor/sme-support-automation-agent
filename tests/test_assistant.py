"""
Tests for backend/assistant.py's DoorAssistantAgent — focused on the fact
that one instance is shared by every Streamlit session of a role
(st.cache_resource), so per-turn state must never leak between sessions.
A routing fake chat model replaces the real LLM; no API calls are made.
"""

import datetime
import threading
import types
from typing import List, Optional

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.outputs import ChatGeneration, ChatResult

import backend.assistant as assistant
from backend.assistant import DoorAssistantAgent, TurnResult


_SURVEY_CALL = {
    "name": "request_site_survey_tool",
    "id": "call_survey",
    "args": {
        "model": "IRIS", "quantity": 1, "date_str": "2030-01-01", "time_str": "10:00",
        "client_name": "Alice", "client_email": "alice@example.com",
        "client_phone": "1", "client_address": "Via A 1",
    },
}


class RoutingFakeChatModel(BaseChatModel):
    """
    Answers by looking at the latest message: a human message mentioning
    "survey" → a mutating tool call (pauses the graph for HITL), anything else
    (incl. a tool result after confirm/cancel) → a plain reply. Optionally
    waits on a barrier so concurrent turns are guaranteed to overlap, and
    records every system prompt it receives.
    """
    barrier: Optional[threading.Barrier] = None
    system_prompts: List[str] = []

    model_config = {"arbitrary_types_allowed": True}

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs) -> ChatResult:
        self.system_prompts.append(next(m.content for m in messages if isinstance(m, SystemMessage)))
        if self.barrier:
            self.barrier.wait()
        last = messages[-1]
        if isinstance(last, HumanMessage) and "survey" in last.content:
            response = AIMessage(content="", tool_calls=[_SURVEY_CALL])
        else:
            response = AIMessage(content="Hello!")
        return ChatResult(generations=[ChatGeneration(message=response)])

    @property
    def _llm_type(self) -> str:
        return "routing-fake-chat-model"


@pytest.fixture
def make_agent(monkeypatch):
    def _make(llm: RoutingFakeChatModel) -> DoorAssistantAgent:
        monkeypatch.setattr(assistant, "get_llm", lambda **kwargs: llm)
        monkeypatch.setattr(DoorAssistantAgent, "is_input_safe", lambda self, q: True)
        return DoorAssistantAgent(role="client")
    return _make


def test_concurrent_turns_on_shared_agent_do_not_leak(make_agent):
    """
    Two sessions talk to the same agent at the same time: Alice triggers a
    HITL confirmation, Bob doesn't. Each must see only its own result —
    previously both read shared agent attributes, so Bob got Alice's
    confirmation prompt (with her personal data).
    """
    llm = RoutingFakeChatModel(barrier=threading.Barrier(2, timeout=10), system_prompts=[])
    agent = make_agent(llm)
    after_stream = threading.Barrier(2, timeout=10)
    results = {}
    errors = []

    def turn(thread_id: str, query: str):
        try:
            result = TurnResult()
            list(agent.stream_ask(query, thread_id, result))
            after_stream.wait()  # both turns finished before either result is read
            results[thread_id] = result
        except Exception as exc:  # surfaced below — threads swallow exceptions
            errors.append(exc)

    threads = [
        threading.Thread(target=turn, args=("alice", "I want to book a survey")),
        threading.Thread(target=turn, args=("bob", "hi")),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors
    assert results["alice"].interrupt is not None
    assert results["alice"].interrupt["tool"] == "request_site_survey_tool"
    assert [tc["tool"] for tc in results["alice"].tool_calls] == ["request_site_survey_tool"]

    assert results["bob"].interrupt is None
    assert results["bob"].tool_calls == []


def test_resume_fills_its_own_result(make_agent):
    agent = make_agent(RoutingFakeChatModel(system_prompts=[]))
    ask_result = TurnResult()
    list(agent.stream_ask("I want to book a survey", "t1", ask_result))
    assert ask_result.interrupt is not None

    resume_result = TurnResult()
    list(agent.stream_resume("rejected", "t1", resume_result))
    assert resume_result.interrupt is None
    assert ask_result.interrupt is not None  # earlier result left untouched


def test_system_prompt_date_is_current_on_every_turn(make_agent, monkeypatch):
    """The agent lives for the whole process — the date must not freeze at construction."""
    llm = RoutingFakeChatModel(system_prompts=[])
    agent = make_agent(llm)

    def _fake_today(day: datetime.date):
        fake_date = type("FakeDate", (datetime.date,), {"today": classmethod(lambda cls: day)})
        monkeypatch.setattr(assistant, "datetime", types.SimpleNamespace(date=fake_date))

    _fake_today(datetime.date(2030, 1, 1))
    list(agent.stream_ask("hi", "t1", TurnResult()))
    _fake_today(datetime.date(2030, 1, 2))
    list(agent.stream_ask("hi again", "t1", TurnResult()))

    assert "CURRENT DATE: January 01, 2030" in llm.system_prompts[0]
    assert "CURRENT DATE: January 02, 2030" in llm.system_prompts[1]
