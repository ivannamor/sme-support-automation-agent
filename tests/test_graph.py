"""
Tests for backend/graph.py — the pure routing/summary helpers directly, and
the compiled StateGraph's four execution paths using a scripted fake LLM
(no real API calls): plain answer, read-only tool loop, HITL confirm+execute,
and HITL confirm+cancel.
"""

from langchain.tools import tool
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.utils import count_tokens_approximately
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from backend.graph import _confirmation_summary, _make_is_mutating, _trim_history, build_graph

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
        (
            {
                "name": "book_installation_tool",
                "args": {
                    "model": "IRIS",
                    "date_str": "2026-08-01",
                    "time_str": "10:00",
                    "purpose": "Installation",
                },
            },
            "IRIS",
        ),
        (
            {
                "name": "inventory_api_tool",
                "args": {"action": "take", "model": "IRIS", "quantity": 2},
            },
            "Reserve",
        ),
        (
            {
                "name": "inventory_api_tool",
                "args": {"action": "backorder", "model": "IRIS", "quantity": 2},
            },
            "backorder",
        ),
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
            "ticket_reference": "TKT-00005",
            "client_phone": "3331234567",
            "client_name": None,
            "date_str": None,
            "notes": None,
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
        llm=llm,
        tools=_TEST_TOOLS,
        system_prompt="You are a test assistant.",
        checkpointer=MemorySaver(),
        mutating_tool_names=_TEST_MUTATING,
    )


def test_graph_plain_answer_reaches_end_with_no_tool_call(scripted_llm):
    graph = _build_test_graph(scripted_llm, [AIMessage(content="Hello there!")])
    config = {"configurable": {"thread_id": "t1"}}
    result = graph.invoke(
        {"messages": [HumanMessage(content="hi")], "pending_tool_call": None}, config=config
    )
    assert result["messages"][-1].content == "Hello there!"
    assert graph.get_state(config).interrupts == ()


def test_graph_readonly_tool_loops_back_to_agent(scripted_llm):
    tool_call_msg = AIMessage(
        content="",
        tool_calls=[{"name": "_fake_readonly_tool", "args": {"query": "IRIS"}, "id": "call_1"}],
    )
    final_msg = AIMessage(content="Here is the answer based on the tool.")
    graph = _build_test_graph(scripted_llm, [tool_call_msg, final_msg])
    config = {"configurable": {"thread_id": "t2"}}
    result = graph.invoke(
        {"messages": [HumanMessage(content="hi")], "pending_tool_call": None}, config=config
    )
    assert result["messages"][-1].content == "Here is the answer based on the tool."
    # The ToolMessage produced by the read-only tool node must be in history.
    tool_messages = [m for m in result["messages"] if type(m).__name__ == "ToolMessage"]
    assert any("result for IRIS" in m.content for m in tool_messages)


def test_graph_mutating_tool_confirmed_executes_and_resumes(scripted_llm):
    tool_call_msg = AIMessage(
        content="",
        tool_calls=[{"name": "_fake_mutating_tool", "args": {"value": "X"}, "id": "call_2"}],
    )
    final_msg = AIMessage(content="Done, it's confirmed.")
    graph = _build_test_graph(scripted_llm, [tool_call_msg, final_msg])
    config = {"configurable": {"thread_id": "t3"}}

    result = graph.invoke(
        {"messages": [HumanMessage(content="do it")], "pending_tool_call": None}, config=config
    )
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
    tool_call_msg = AIMessage(
        content="",
        tool_calls=[{"name": "_fake_mutating_tool", "args": {"value": "Y"}, "id": "call_3"}],
    )
    final_msg = AIMessage(content="Okay, cancelled as requested.")
    graph = _build_test_graph(scripted_llm, [tool_call_msg, final_msg])
    config = {"configurable": {"thread_id": "t4"}}

    graph.invoke(
        {"messages": [HumanMessage(content="do it")], "pending_tool_call": None}, config=config
    )
    result = graph.invoke(Command(resume="rejected"), config=config)

    assert result["messages"][-1].content == "Okay, cancelled as requested."
    tool_messages = [m for m in result["messages"] if type(m).__name__ == "ToolMessage"]
    assert any("cancelled by the user" in m.content.lower() for m in tool_messages)
    # The mutating tool itself must never have actually run.
    assert not any("executed with Y" in m.content for m in tool_messages)


# ── history trimming ──────────────────────────────────────────────────────────


def _turn(n: int, with_tool: bool = False, size: int = 400) -> list:
    """One conversation turn of roughly `size` characters per message."""
    text = f"turn {n} " + "x" * size
    messages = [HumanMessage(content=text)]
    if with_tool:
        call_id = f"call_{n}"
        messages += [
            AIMessage(
                content="",
                tool_calls=[{"name": "_fake_readonly_tool", "args": {"query": "q"}, "id": call_id}],
            ),
            ToolMessage(content=text, tool_call_id=call_id),
        ]
    messages.append(AIMessage(content=text))
    return messages


def _assert_tool_results_have_their_calls(messages: list) -> None:
    call_ids = set()
    for m in messages:
        if isinstance(m, AIMessage):
            call_ids |= {tc["id"] for tc in m.tool_calls}
        if isinstance(m, ToolMessage):
            assert m.tool_call_id in call_ids, "ToolMessage sent without its tool call"


def test_trim_history_keeps_everything_under_budget():
    messages = _turn(1) + _turn(2, with_tool=True) + [HumanMessage(content="now")]
    assert _trim_history(messages, max_tokens=100_000) == messages


def test_trim_history_drops_oldest_whole_turns():
    messages = []
    for n in range(10):
        messages += _turn(n, with_tool=n % 2 == 0)
    messages.append(HumanMessage(content="latest question"))

    trimmed = _trim_history(messages, max_tokens=800)

    assert len(trimmed) < len(messages)
    assert trimmed == messages[-len(trimmed) :]  # a suffix: newest kept, order preserved
    assert isinstance(trimmed[0], HumanMessage)  # cut only at a turn boundary
    assert trimmed[-1].content == "latest question"
    assert count_tokens_approximately(trimmed) <= 800
    _assert_tool_results_have_their_calls(trimmed)


def test_trim_history_never_cuts_the_current_turn():
    """A tool-heavy current turn over budget is still sent whole, even with no history."""
    messages = _turn(1) + _turn(2, with_tool=True, size=5000)[:-1]  # turn 2 in progress
    current = messages[2:]

    assert _trim_history(messages, max_tokens=100) == current


def test_trim_history_without_human_messages_is_unchanged():
    messages = [AIMessage(content="hello")]
    assert _trim_history(messages, max_tokens=1) == messages


def test_graph_sends_trimmed_history_but_keeps_full_state(scripted_llm):
    seen: list[list] = []

    class RecordingLLM(type(scripted_llm([]))):
        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            seen.append(messages)
            return super()._generate(messages, stop, run_manager, **kwargs)

    responses = [AIMessage(content="reply " + "y" * 400) for _ in range(6)]
    graph = build_graph(
        llm=RecordingLLM(responses=responses),
        tools=_TEST_TOOLS,
        system_prompt="You are a test assistant.",
        checkpointer=MemorySaver(),
        mutating_tool_names=_TEST_MUTATING,
        max_history_tokens=500,
    )
    config = {"configurable": {"thread_id": "trim"}}
    for n in range(6):
        graph.invoke({"messages": [HumanMessage(content=f"q{n} " + "x" * 400)]}, config=config)

    last_call = seen[-1]
    assert isinstance(last_call[0], SystemMessage)
    assert last_call[-1].content.startswith("q5")
    assert len(last_call) - 1 < len(graph.get_state(config).values["messages"]) - 1
    assert len(graph.get_state(config).values["messages"]) == 12  # nothing lost from state
