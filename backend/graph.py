"""
Explicit LangGraph StateGraph for the assistant.

Flow:
  START → agent → route_after_agent
    ├─ "tools"   → tools_node → agent  (read-only loop)
    ├─ "confirm" → confirm_node [interrupt] → execute_confirmed → agent
    └─ "__end__" → END

Mutating actions (inventory take/backorder, booking) pass through confirm_node
and are paused via interrupt() until the user explicitly approves or cancels.
All read-only tool calls (RAG, availability check, inventory get, travel time)
execute immediately without interruption.
"""

import logging
from collections.abc import Callable, Mapping
from typing import Annotated, Any, Literal

from langchain_core.messages import AIMessage, BaseMessage, SystemMessage, ToolMessage
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import Command, interrupt
from typing_extensions import TypedDict

from backend.database import ticket_code

logger = logging.getLogger("DoorGraph")

_MUTATING_INVENTORY_ACTIONS = {"take", "backorder"}

# Default mutating tool names when no role-specific override is provided.
_DEFAULT_MUTATING_TOOL_NAMES = {"book_installation_tool", "request_site_survey_tool"}


class AgentState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    pending_tool_call: dict | None   # tool call waiting for HITL confirmation


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_is_mutating(mutating_tool_names: set[str]):
    def _is_mutating(tool_call: Mapping[str, Any]) -> bool:
        name = tool_call.get("name", "")
        if name == "inventory_api_tool":
            return tool_call.get("args", {}).get("action") in _MUTATING_INVENTORY_ACTIONS
        return name in mutating_tool_names
    return _is_mutating


def _confirmation_summary(tc: Mapping[str, Any]) -> str:
    name = tc["name"]
    args = tc.get("args", {})
    if name == "book_installation_tool":
        purpose = args.get("purpose", "Installation")
        return (
            f"Book {purpose.lower()} of **{args.get('model')}** "
            f"on **{args.get('date_str')}** at **{args.get('time_str')}**?"
        )
    if name == "inventory_api_tool":
        action = args.get("action", "")
        verb = "Reserve" if action == "take" else "Place backorder for"
        return f"{verb} **{args.get('quantity', 1)}× {args.get('model')}**?"
    if name == "request_site_survey_tool":
        return (
            f"Submit site survey request for **{args.get('quantity', 1)}× {args.get('model')}** "
            f"on **{args.get('date_str')}** at **{args.get('time_str')}**?  \n"
            f"Client: {args.get('client_name')} · {args.get('client_email')} · {args.get('client_phone')}  \n"
            f"Address: {args.get('client_address')}"
        )
    if name == "submit_survey_measurements_tool":
        return (
            f"Submit survey measurements for **{ticket_code(args.get('booking_id'))}** — "
            f"{args.get('height_cm')}×{args.get('width_cm')}×{args.get('depth_cm')}cm, "
            f"glass: **{args.get('glass_type')}**?"
        )
    if name == "respond_to_estimate_tool":
        response = args.get("response", "")
        if response == "accept":
            return (
                f"Accept the estimate for **{ticket_code(args.get('booking_id'))}** and schedule "
                f"installation on **{args.get('new_date_str')}** at **{args.get('new_time_str')}**?"
            )
        return f"Reject the estimate for **{ticket_code(args.get('booking_id'))}**? This cancels the ticket."
    if name == "request_inventory_reservation_tool":
        return f"Submit reservation request for **{args.get('quantity', 1)}× {args.get('model')}**?"
    if name == "update_installation_costs_tool":
        return f"Update cost variable **{args.get('key')}** to **{args.get('value')}**?"
    if name == "complete_job_tool":
        ref = f" (booking #{args.get('booking_id')})" if args.get("booking_id") else ""
        email_note = f" — client email: {args.get('client_email')}" if args.get("client_email") else ""
        return f"Submit job completion report{ref}{email_note}?\n\n> {args.get('notes', '')}"
    if name == "send_estimate_to_client_tool":
        return f"Send price estimate to **{args.get('client_email')}**?"
    if name == "send_alert_to_manager_tool":
        ref = f" (job: {args.get('job_reference')})" if args.get("job_reference") else ""
        return f"Send this alert to the manager{ref}?\n\n> {args.get('message', '')}"
    if name == "claim_installation_tool":
        return f"Start working on installation **{ticket_code(args.get('booking_id'))}** and mark it as in progress?"
    if name == "approve_booking_tool":
        return f"Approve booking **{ticket_code(args.get('booking_id'))}** and add it to the calendar?"
    if name == "reject_booking_tool":
        return f"Reject booking **{ticket_code(args.get('booking_id'))}**? This cannot be undone."
    if name == "approve_job_tool":
        email_note = f" — thank-you email to **{args['client_email']}**" if args.get("client_email") else ""
        return f"Approve job completion report **#{args.get('job_id')}**{email_note}?"
    if name == "reject_job_tool":
        return f"Reject job completion report **#{args.get('job_id')}**? This cannot be undone."
    if name == "edit_ticket_tool":
        changed = {
            k: v for k, v in args.items()
            if k != "ticket_reference" and v is not None
        }
        field_list = ", ".join(f"**{k}** → {v}" for k, v in changed.items())
        return f"Update {args.get('ticket_reference')}: {field_list}?"
    return f"Execute **{name}** with args `{args}`?"


# ── graph factory ─────────────────────────────────────────────────────────────

def build_graph(
    llm,
    tools: list,
    system_prompt: str | Callable[[], str],
    checkpointer=None,
    mutating_tool_names: set | None = None,
):
    """
    Compiles and returns the assistant StateGraph.

    Args:
        llm:           A streaming-enabled ChatOpenAI (or compatible) instance.
        tools:         Full list of LangChain tools available to the agent.
        system_prompt: System prompt prepended on every agent invocation — a
                       string, or a zero-arg callable re-evaluated on each call
                       (for content that changes over time, e.g. today's date).
        checkpointer:  LangGraph checkpointer (MemorySaver, SqliteSaver, …).
                       Required for interrupt()/resume to work.
    """
    tools_by_name = {t.name: t for t in tools}
    _is_mutating = _make_is_mutating(
        mutating_tool_names if mutating_tool_names is not None else _DEFAULT_MUTATING_TOOL_NAMES
    )

    # Bind tools to the LLM; disable parallel tool calls so the HITL gate only
    # ever needs to handle one action at a time.
    llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)

    # ── nodes ─────────────────────────────────────────────────────────────────

    def agent_node(state: AgentState) -> dict:
        prompt = system_prompt() if callable(system_prompt) else system_prompt
        messages = [SystemMessage(content=prompt)] + state["messages"]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    def route_after_agent(
        state: AgentState,
    ) -> Literal["tools", "confirm", "__end__"]:
        last = state["messages"][-1]
        tool_calls = getattr(last, "tool_calls", None) or []
        if not tool_calls:
            return "__end__"
        for tc in tool_calls:
            if _is_mutating(tc):
                return "confirm"
        return "tools"

    # Read-only tool calls execute immediately via LangGraph's ToolNode.
    tool_node = ToolNode(tools)

    def confirm_node(state: AgentState) -> Command:
        """
        Pauses the graph and surfaces the pending action to the UI.

        The UI receives the interrupt payload via graph.get_state().interrupts
        and renders confirm / cancel buttons. When the user decides:
          - confirmed → Command(resume="confirmed") continues to execute_confirmed
          - rejected  → Command(resume="rejected")  injects a cancellation
                        ToolMessage and loops back to agent for a natural reply
        """
        last = state["messages"][-1]
        # Routed here only after the agent emitted a mutating tool call.
        if not isinstance(last, AIMessage):
            raise TypeError(f"confirm_node expects an AIMessage, got {type(last).__name__}")
        mutating_tc = next(tc for tc in last.tool_calls if _is_mutating(tc))

        decision = interrupt({
            "tool": mutating_tc["name"],
            "args": mutating_tc["args"],
            "summary": _confirmation_summary(mutating_tc),
            "tool_call_id": mutating_tc["id"],
        })

        if decision == "confirmed":
            return Command(
                goto="execute_confirmed",
                update={"pending_tool_call": mutating_tc},
            )

        cancelled_msg = ToolMessage(
            content="Action cancelled by the user.",
            tool_call_id=mutating_tc["id"],
        )
        return Command(
            goto="agent",
            update={
                "messages": [cancelled_msg],
                "pending_tool_call": None,
            },
        )

    def execute_confirmed_node(state: AgentState) -> dict:
        tc = state["pending_tool_call"]
        if tc is None:
            raise RuntimeError("execute_confirmed reached without a pending tool call")
        tool = tools_by_name[tc["name"]]
        try:
            result = tool.invoke(tc["args"])
        except Exception as exc:
            logger.error(f"EXECUTE_CONFIRMED_ERROR: {tc['name']} — {exc}", exc_info=True)
            result = f"Error executing {tc['name']}: {exc}"
        return {
            "messages": [ToolMessage(content=str(result), tool_call_id=tc["id"])],
            "pending_tool_call": None,
        }

    # ── graph assembly ────────────────────────────────────────────────────────

    builder = StateGraph(AgentState)

    builder.add_node("agent", agent_node)
    builder.add_node("tools", tool_node)
    builder.add_node("confirm", confirm_node)
    builder.add_node("execute_confirmed", execute_confirmed_node)

    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", route_after_agent)
    builder.add_edge("tools", "agent")
    builder.add_edge("execute_confirmed", "agent")

    return builder.compile(checkpointer=checkpointer)
