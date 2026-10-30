"""
Tests for the agent eval harness (evaluation/). The golden set is validated
here, so a malformed case fails CI; everything else runs on fake LLMs — no
API calls. The evals themselves (real calls) are run by hand.
"""

import textwrap

import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

import backend.assistant as assistant
import tools
from backend import database
from backend.assistant import DoorAssistantAgent
from evaluation.calibrate import load_labelled_pairs, score
from evaluation.compare import changed_cases
from evaluation.dataset import NO_TOOL, KnowledgeCase, RoutingCase, load_cases
from evaluation.judge import (
    args_match,
    get_judge_llm,
    judge_fact,
    judge_run,
    knowledge_checks,
    routing_checks,
    summarize,
)
from evaluation.run_evals import run_knowledge, run_routing
from evaluation.sandbox import sandbox
from tests.conftest import fake_llm_with_usage

MINI = "openai/gpt-4o-mini"
IRIS_URL = "https://www.marinig.com/prodotto/iris/"


def _judge(*replies: str):
    return fake_llm_with_usage(MINI, 10, 2, replies=[AIMessage(content=r) for r in replies])


def _routing_case(**overrides) -> RoutingCase:
    fields = dict(
        id="r",
        role="manager",
        message="Quante MAGIS 40 abbiamo?",
        expect_tool=("inventory_api_tool",),
        expect_args={"action": "get", "model": "MAGIS 40"},
    )
    return RoutingCase(**{**fields, **overrides})


def _knowledge_case(**overrides) -> KnowledgeCase:
    fields = dict(
        id="k",
        role="client",
        language="it",
        question="Di che materiale è IRIS?",
        facts=("IRIS is made of laminated wood.",),
        reference="IRIS è in legno lamellare.",
        expect_sources=(IRIS_URL,),
    )
    return KnowledgeCase(**{**fields, **overrides})


# ── the golden set ────────────────────────────────────────────────────────────


def test_golden_set_is_valid():
    cases = load_cases()
    assert len(cases) >= 40
    assert {c.kind for c in cases} == {"routing", "knowledge"}


def _write_golden(tmp_path, body: str):
    path = tmp_path / "golden.yaml"
    path.write_text("cases:\n" + textwrap.indent(textwrap.dedent(body), "  "), encoding="utf-8")
    return path


@pytest.mark.parametrize(
    ("body", "error"),
    [
        (
            """
            - id: a
              kind: routing
              role: client
              message: hi
              expect_tool: inventory_api_tool
            """,
            "not a client tool",
        ),
        (
            """
            - id: a
              kind: knowledge
              role: client
              language: it
              question: q
              reference: r
            """,
            "missing facts",
        ),
        (
            """
            - id: a
              kind: routing
              role: client
              message: hi
              expect_tool: no_tool
              expected_tool: no_tool
            """,
            "unknown field",
        ),
        (
            """
            - id: a
              kind: routing
              role: client
              message: hi
              expect_tool: no_tool
            - id: a
              kind: routing
              role: client
              message: hello
              expect_tool: no_tool
            """,
            "duplicated",
        ),
    ],
)
def test_malformed_cases_are_rejected(tmp_path, body, error):
    with pytest.raises(ValueError, match=error):
        load_cases(_write_golden(tmp_path, body))


# ── checks ────────────────────────────────────────────────────────────────────


def test_args_match_ignores_case_and_decimal_comma():
    assert args_match({"model": "MAGIS 40", "value": 0.35}, {"model": "magis 40 ", "value": "0,35"})
    assert not args_match({"quantity": 2}, {"quantity": 3})
    assert not args_match({"quantity": 2}, {})


def test_routing_checks():
    case = _routing_case(forbid_tool=("book_installation_tool",))
    right = {
        "tool_calls": [
            {"name": "inventory_api_tool", "args": {"action": "get", "model": "magis 40"}}
        ]
    }
    forbidden = {"tool_calls": [{"name": "book_installation_tool", "args": {}}]}

    assert routing_checks(case, right) == {
        "decision": "inventory_api_tool",
        "tool_ok": True,
        "forbidden": False,
        "args_ok": True,
    }
    assert routing_checks(case, forbidden)["forbidden"]
    assert routing_checks(case, {"tool_calls": []})["decision"] == NO_TOOL


def test_retrieval_hit_needs_one_expected_source():
    case = _knowledge_case()
    found = {"tool_calls": [], "searches": [{"query": "q", "docs": [{"source": IRIS_URL}]}]}
    missed = {"tool_calls": [], "searches": [{"query": "q", "docs": [{"source": "other.pdf"}]}]}
    assert knowledge_checks(case, found)["retrieval_hit"]
    assert not knowledge_checks(case, missed)["retrieval_hit"]
    assert "retrieval_hit" not in knowledge_checks(_knowledge_case(expect_sources=()), found)


@pytest.mark.parametrize(
    ("reply", "verdict", "reason"),
    [
        ("YES\nIt says so.", True, "It says so."),
        ("**NO**\nNot mentioned.", False, "Not mentioned."),
        ("NO: the reply gives 600 mm.", False, "the reply gives 600 mm."),
        ("Not sure", False, "unparsed: Not sure"),
        ("", False, "unparsed: "),  # cut off while thinking
    ],
)
def test_judge_fact_verdicts(reply, verdict, reason):
    result = judge_fact(_judge(reply), "q", "a", "fact")
    assert result["verdict"] is verdict
    assert result["reason"] == reason


def test_judge_has_room_to_think_before_answering():
    assert get_judge_llm("anthropic/claude-haiku-5.5").max_tokens >= 1000


def test_knowledge_run_passes_only_with_every_fact_and_the_right_language():
    case = _knowledge_case(facts=("f1", "f2"))
    run = {"reply": "…", "tool_calls": [], "searches": []}

    assert judge_run(_judge("YES\n.", "YES\n.", "it"), case, run)["passed"]
    assert not judge_run(_judge("YES\n.", "NO\n.", "it"), case, run)["passed"]
    wrong_language = judge_run(_judge("YES\n.", "YES\n.", "en"), case, run)
    assert not wrong_language["passed"]
    assert wrong_language["language"] == "en"


def test_routing_facts_are_judged_only_for_a_text_reply():
    case = _routing_case(
        expect_tool=(NO_TOOL, "inventory_api_tool"), expect_args={}, facts=("Asks.",)
    )
    text_reply = {"reply": "Which model?", "tool_calls": []}
    tool_call = {"reply": "", "tool_calls": [{"name": "inventory_api_tool", "args": {}}]}

    assert not judge_run(_judge("NO\nIt doesn't ask."), case, text_reply)["passed"]
    assert "facts" not in judge_run(_judge(), case, tool_call)


def test_a_run_that_errored_fails():
    assert judge_run(_judge(), _knowledge_case(), {"error": "Timeout"}) == {
        "passed": False,
        "error": "Timeout",
    }


def test_summary():
    cases = [_routing_case(), _knowledge_case()]
    results = [
        {"id": "r", "runs": [{"checks": {"passed": True, "tool_ok": True, "args_ok": True}}]},
        {
            "id": "k",
            "runs": [
                {"checks": {"passed": True, "language_ok": True, "retrieval_hit": True,
                            "facts": [{"verdict": True}], "ragas": {"faithfulness": 1.0}}},
                {"checks": {"passed": False, "language_ok": True, "retrieval_hit": False,
                            "facts": [{"verdict": False}], "ragas": {"faithfulness": 0.5}}},
            ],
        },
    ]  # fmt: skip
    summary = summarize(cases, results)

    assert summary["pass_rate"] == pytest.approx(2 / 3, abs=1e-3)
    assert summary["cases"] == {"r": 1.0, "k": 0.5}
    assert summary["knowledge"]["fact_recall"] == 0.5
    assert summary["knowledge"]["retrieval_hit_rate"] == 0.5
    assert summary["knowledge"]["faithfulness"] == 0.75


def test_compare_lists_changed_cases_worst_first():
    a = {"cases": {"x": 1.0, "y": 0.5, "z": 1.0}}
    b = {"cases": {"x": 0.0, "y": 1.0, "z": 1.0}}
    assert changed_cases(a, b) == [("x", 1.0, 0.0), ("y", 0.5, 1.0)]


def test_calibration_pairs_are_labelled():
    pairs = load_labelled_pairs()
    assert len(pairs) >= 20
    assert {p["label"] for p in pairs} == {True, False}
    assert all(p["question"] and p["answer"] and p["fact"] for p in pairs)


def test_calibration_score_counts_false_yes_and_false_no():
    pairs = [{"label": True}, {"label": False}, {"label": False}, {"label": True}]
    verdicts = [
        {"verdict": True, "reason": ""},
        {"verdict": True, "reason": ""},  # false yes
        {"verdict": False, "reason": ""},
        {"verdict": False, "reason": "unparsed: ?"},  # false no
    ]
    assert score(pairs, verdicts) == {
        "agreement": 0.5,
        "false_yes": 1,
        "false_no": 1,
        "unparsed": 1,
    }


# ── running the agent ─────────────────────────────────────────────────────────


def test_sandbox_uses_a_fresh_database_and_restores_the_real_one():
    real = database.DB_PATH
    with sandbox() as tmp:
        assert database.DB_PATH.parent == tmp
        assert database.get_inventory()  # seeded
    assert real == database.DB_PATH


def _agent_with(monkeypatch, role: str, *replies: AIMessage) -> DoorAssistantAgent:
    chat = fake_llm_with_usage(MINI, 100, 10, replies=list(replies))
    guard = fake_llm_with_usage(MINI, 5, 1, replies=[AIMessage(content="SAFE")])
    monkeypatch.setattr(assistant, "get_llm", lambda **kw: chat if kw.get("streaming") else guard)
    return DoorAssistantAgent(role=role)


def test_run_routing_records_the_first_decision(monkeypatch):
    call = {
        "name": "inventory_api_tool",
        "args": {"action": "get", "model": "MAGIS 40"},
        "id": "c1",
    }
    with sandbox():
        agent = _agent_with(monkeypatch, "manager", AIMessage(content="", tool_calls=[call]))
        run = run_routing(agent, _routing_case())

    assert run["tool_calls"] == [{"name": "inventory_api_tool", "args": call["args"]}]
    assert run["tokens"] == 110
    assert routing_checks(_routing_case(), run)["args_ok"]


def test_run_knowledge_records_the_retrieved_documents(monkeypatch):
    monkeypatch.setattr(tools, "_expand_query", lambda query: query)
    doc = Document(
        page_content="legno lamellare",
        metadata={"source": IRIS_URL, "collection": "catalogue_specs"},
    )
    monkeypatch.setattr(tools, "get_assistant_response", lambda **kwargs: ("answer", [doc]))
    search = {"name": "search_technical_manuals", "args": {"query": "IRIS"}, "id": "c1"}

    with sandbox():
        agent = _agent_with(
            monkeypatch,
            "client",
            AIMessage(content="", tool_calls=[search]),
            AIMessage(content="IRIS è in legno lamellare."),
        )
        run = run_knowledge(agent, _knowledge_case())

    assert run["reply"] == "IRIS è in legno lamellare."
    assert [tc["name"] for tc in run["tool_calls"]] == ["search_technical_manuals"]
    assert run["searches"] == [
        {
            "query": "IRIS",
            "docs": [
                {
                    "source": IRIS_URL,
                    "page": None,
                    "collection": "catalogue_specs",
                    "content": "legno lamellare",
                }
            ],
        }
    ]
    assert tools.get_assistant_response.__name__ == "<lambda>"  # the recorder was removed
    assert knowledge_checks(_knowledge_case(), run)["retrieval_hit"]
