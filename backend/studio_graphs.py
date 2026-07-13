"""
Entry points for LangGraph Studio (via `langgraph dev`, see langgraph.json).

Each function builds a full DoorAssistantAgent for one role and returns its
compiled graph — no logic is duplicated here, this just exposes the same
graphs the app already runs for interactive visualization/debugging in
Studio. The checkpointer built into DoorAssistantAgent is ignored by the
`langgraph dev` server, which injects its own persistence layer instead.
"""

from backend.assistant import DoorAssistantAgent


def build_client_graph():
    return DoorAssistantAgent(role="client").executor


def build_installer_graph():
    return DoorAssistantAgent(role="installer").executor


def build_manager_graph():
    return DoorAssistantAgent(role="manager").executor
