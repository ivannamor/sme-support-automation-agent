"""
Tests for backend/graph.py — the pure routing/summary helpers directly, and
the compiled StateGraph's four execution paths using a scripted fake LLM
(no real API calls): plain answer, read-only tool loop, HITL confirm+execute,
and HITL confirm+cancel.
"""

from langchain.tools import tool
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from backend.graph import build_graph, _make_is_mutating, _confirmation_summary


# ── _make_is_mutating ─────────────────────────────────────────────────────────

def test_is_mutating_matches_named_tools():
    is_mutating = _make_is_mutating({"book_installation_tool", "edit_ticket_tool"})
    assert is_mutating({"name": "book_installation_tool", "args": {}})
    assert is_mutating({"name": "edit_ticket_tool", "args": {}})
    assert not is_mutating({"name": "search_technical_manuals", "args": {}})


def test_is_mutating_inventory_take_and_backorder_only():
    is_mutating = _make_is_mutating(set())
    assert is_mutating({"name": "inventory_api_tool", "args": {"action": "take"}})
    assert is_mutating({"name": "inventory_api_tool", "args": {"action": "backorder"}})
    assert not is_mutating({"name": "inventory_api_tool", "args": {"action": "get"}})


# ── _confirmation_summary ─────────────────────────────────────────────────────

def test_confirmation_summary_known_tools_produce_readable_text():
    cases = [
        ({"name": "book_installation_tool", "args": {"model": "IRIS", "date_str": "2026-08-01", "time_str": "10:00", "purpose": "Installation"}}, "IRIS"),
        ({"name": "inventory_api_tool", "args": {"action": "take", "model": "IRIS", "quantity": 2}}, "Reserve"),
        ({"name": "inventory_api_tool", "args": {"action": "backorder", "model": "IRIS", "quantity": 2}}, "backorder"),
        ({"name": "approve_booking_tool", "args": {"booking_id": 7}}, "TKT-00007"),
        ({"name": "reject_job_tool", "args": {"job_id": 3}}, "#3"),
    ]
    for tc, expected_fragment in cases:
        summary = _confirmation_summary(tc)
        assert expected_fragment in summary


def test_confirmation_summary_edit_ticket_lists_changed_fields_only():
    tc = {
        "name": "edit_ticket_tool",
        "args": {
            "ticket_reference": "TKT-00005", "client_phone": "3331234567",
            "client_name": None, "date_str": None, "notes": None,
        },
    }
    summary = _confirmation_summary(tc)
    assert "TKT-00005" in summary
    assert "client_phone" in summary
    assert "3331234567" in summary
    assert "client_name" not in summary  # None fields must not appear


def test_confirmation_summary_unknown_tool_falls_back_to_generic_text():
    summary = _confirmation_summary({"name": "some_future_tool", "args": {"x": 1}})
    assert "some_future_tool" in summary


# ── compiled graph integration (scripted fake LLM, no real API) ─────────────

@tool
def _fake_readonly_tool(query: str) -> str:
    """A fake read-only tool for graph tests."""
    return f"result for {query}"


@tool
def _fake_mutating_tool(value: str) -> str:
    """A fake mutating tool for graph tests."""
    return f"executed with {value}"


_TEST_TOOLS = [_fake_readonly_tool, _fake_mutating_tool]
_TEST_MUTATING = {"_fake_mutating_tool"}


def _build_test_graph(scripted_llm_fixture, responses):
    llm = scripted_llm_fixture(responses)
    return build_graph(
        llm=llm, tools=_TEST_TOOLS, system_prompt="You are a test assistant.",
        checkpointer=MemorySaver(), mutating_tool_names=_TEST_MUTATING,
    )


def test_graph_plain_answer_reaches_end_with_no_tool_call(scripted_llm):
    graph = _build_test_graph(scripted_llm, [AIMessage(content="Hello there!")])
    config = {"configurable": {"thread_id": "t1"}}
    result = graph.invoke({"messages": [HumanMessage(content="hi")], "pending_tool_call": None}, config=config)
    assert result["messages"][-1].content == "Hello there!"
    assert graph.get_state(config).interrupts == ()


def test_graph_readonly_tool_loops_back_to_agent(scripted_llm):
    tool_call_msg = AIMessage(content="", tool_calls=[
        {"name": "_fake_readonly_tool", "args": {"query": "IRIS"}, "id": "call_1"}
    ])
    final_msg = AIMessage(content="Here is the answer based on the tool.")
    graph = _build_test_graph(scripted_llm, [tool_call_msg, final_msg])
    config = {"configurable": {"thread_id": "t2"}}
    result = graph.invoke({"messages": [HumanMessage(content="hi")], "pending_tool_call": None}, config=config)
    assert result["messages"][-1].content == "Here is the answer based on the tool."
    # The ToolMessage produced by the read-only tool node must be in history.
    tool_messages = [m for m in result["messages"] if type(m).__name__ == "ToolMessage"]
    assert any("result for IRIS" in m.content for m in tool_messages)


def test_graph_mutating_tool_confirmed_executes_and_resumes(scripted_llm):
    tool_call_msg = AIMessage(content="", tool_calls=[
        {"name": "_fake_mutating_tool", "args": {"value": "X"}, "id": "call_2"}
    ])
    final_msg = AIMessage(content="Done, it's confirmed.")
    graph = _build_test_graph(scripted_llm, [tool_call_msg, final_msg])
    config = {"configurable": {"thread_id": "t3"}}

    result = graph.invoke({"messages": [HumanMessage(content="do it")], "pending_tool_call": None}, config=config)
    # Graph paused for confirmation — no final answer yet.
    state = graph.get_state(config)
    assert len(state.interrupts) == 1
    interrupt_payload = state.interrupts[0].value
    assert interrupt_payload["tool"] == "_fake_mutating_tool"

    result = graph.invoke(Command(resume="confirmed"), config=config)
    assert result["messages"][-1].content == "Done, it's confirmed."
    tool_messages = [m for m in result["messages"] if type(m).__name__ == "ToolMessage"]
    assert any("executed with X" in m.content for m in tool_messages)


def test_graph_mutating_tool_cancelled_skips_execution(scripted_llm):
    tool_call_msg = AIMessage(content="", tool_calls=[
        {"name": "_fake_mutating_tool", "args": {"value": "Y"}, "id": "call_3"}
    ])
    final_msg = AIMessage(content="Okay, cancelled as requested.")
    graph = _build_test_graph(scripted_llm, [tool_call_msg, final_msg])
    config = {"configurable": {"thread_id": "t4"}}

    graph.invoke({"messages": [HumanMessage(content="do it")], "pending_tool_call": None}, config=config)
    result = graph.invoke(Command(resume="rejected"), config=config)

    assert result["messages"][-1].content == "Okay, cancelled as requested."
    tool_messages = [m for m in result["messages"] if type(m).__name__ == "ToolMessage"]
    assert any("cancelled by the user" in m.content.lower() for m in tool_messages)
    # The mutating tool itself must never have actually run.
    assert not any("executed with Y" in m.content for m in tool_messages)
