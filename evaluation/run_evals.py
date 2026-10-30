"""
Agent evals, phase 1 of 2: runs every golden case (evaluation/datasets/
golden.yaml) against the real agent and saves what happened — replies, tool
calls, retrieved documents, cost — to evaluation/results/<date>_<label>.json.
Nothing is scored here: `python -m evaluation.judge <file>` does that, so the
judge can be changed or re-run without paying for the agent again.

    python -m evaluation.run_evals --label quick                 # one run per case
    python -m evaluation.run_evals --label baseline --repeats 3  # for comparisons
    python -m evaluation.run_evals --label debug --only kb-iris-profile,edge-price

Makes real API calls. Runs in a sandbox (evaluation/sandbox.py): the real
company database is never touched.

  routing    — only the agent's first decision: one LLM call with the real
               system prompt and tools. No tool runs.
  knowledge  — a whole chat turn through DoorAssistantAgent.stream_ask. Tools
               that change data stop at the HITL confirmation and are never
               confirmed, so they never run either.
"""

import argparse
import datetime
import json
import subprocess
import sys
import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

import tools
from backend.assistant import DoorAssistantAgent, TurnResult
from backend.llm_provider import DEFAULT_MODEL
from backend.log_context import username_var
from backend.usage_tracking import track_usage
from evaluation.dataset import Case, KnowledgeCase, RoutingCase, load_cases
from evaluation.sandbox import disable_tracing, sandbox

RESULTS_DIR = Path(__file__).resolve().parent / "results"
EVAL_USERNAME = "eval-user"


def _text(content: Any) -> str:
    if isinstance(content, list):
        return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in content)
    return str(content)


def run_routing(agent: DoorAssistantAgent, case: RoutingCase) -> dict:
    """The agent's first decision, as the graph's agent node makes it (backend/graph.py)."""
    llm = agent.llm.bind_tools(agent.tools, parallel_tool_calls=False)
    messages = [SystemMessage(content=agent._build_system_prompt()), HumanMessage(case.message)]
    with track_usage(agent.role) as usage:
        reply = llm.invoke(messages)
    return {
        "reply": _text(reply.content),
        "tool_calls": [{"name": tc["name"], "args": tc["args"]} for tc in reply.tool_calls],
        "cost_usd": usage.cost_usd(),
        "tokens": usage.totals()["total_tokens"],
    }


def _doc_record(doc) -> dict:
    meta = doc.metadata
    return {
        "source": meta.get("source"),
        "page": meta.get("page", meta.get("start_page")),
        "collection": meta.get("collection"),
        "content": doc.page_content,
    }


@contextmanager
def recording_retrieval() -> Iterator[list[dict]]:
    """Records every RAG search the agent's tools make, with the documents it returned."""
    searches: list[dict] = []
    original = tools.get_assistant_response

    def recording(*args, **kwargs):
        answer, docs = original(*args, **kwargs)
        query = kwargs.get("query", args[0] if args else "")
        searches.append({"query": query, "docs": [_doc_record(d) for d in docs]})
        return answer, docs

    tools.get_assistant_response = recording
    try:
        yield searches
    finally:
        tools.get_assistant_response = original


def run_knowledge(agent: DoorAssistantAgent, case: KnowledgeCase) -> dict:
    result = TurnResult()
    with recording_retrieval() as searches:
        reply = "".join(agent.stream_ask(case.question, f"eval-{uuid.uuid4().hex}", result))
    return {
        "reply": reply,
        "tool_calls": [{"name": tc["tool"], "args": tc["input"]} for tc in result.tool_calls],
        "searches": searches,
        "interrupt": result.interrupt,
        "cost_usd": result.cost_usd,
        "tokens": result.usage["total_tokens"],
    }


def run_case(agent: DoorAssistantAgent, case: Case) -> dict:
    start = time.monotonic()
    try:
        if isinstance(case, RoutingCase):
            run = run_routing(agent, case)
        else:
            run = run_knowledge(agent, case)
    except Exception as exc:  # one broken case must not lose the whole run
        run = {"error": f"{type(exc).__name__}: {exc}"}
    run["seconds"] = round(time.monotonic() - start, 2)
    return run


def _git_commit() -> str:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return f"{commit}-dirty" if dirty else commit


def _summary_line(case: Case, run: dict) -> str:
    if "error" in run:
        return f"ERROR {run['error'][:80]}"
    tools_called = [tc["name"] for tc in run["tool_calls"]] or ["no_tool"]
    return f"{', '.join(tools_called)} · ${run['cost_usd']:.4f} · {run['seconds']}s"


def run_all(cases: list[Case], model: str, repeats: int) -> list[dict]:
    """Every case `repeats` times, in a sandbox. One agent per role, as in the app."""
    results = []
    with sandbox():
        username_var.set(EVAL_USERNAME)
        agents: dict[str, DoorAssistantAgent] = {}
        for i, case in enumerate(cases, 1):
            agent = agents.get(case.role) or DoorAssistantAgent(model_name=model, role=case.role)
            agents[case.role] = agent
            runs = []
            for r in range(repeats):
                run = run_case(agent, case)
                runs.append(run)
                print(f"[{i}/{len(cases)}] {case.id} #{r + 1}: {_summary_line(case, run)}")
            results.append({"id": case.id, "kind": case.kind, "runs": runs})
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--label", required=True, help="names the results file")
    parser.add_argument("--repeats", type=int, default=1, help="runs per case (3 to compare)")
    parser.add_argument("--only", help="comma-separated case ids")
    parser.add_argument("--kind", choices=["routing", "knowledge"])
    parser.add_argument("--model", default=DEFAULT_MODEL, help="the agent's model")
    parser.add_argument("--force", action="store_true", help="overwrite an existing file")
    args = parser.parse_args(argv)

    cases = load_cases()
    if args.only:
        wanted = set(args.only.split(","))
        unknown = wanted - {c.id for c in cases}
        if unknown:
            sys.exit(f"Unknown case id(s): {sorted(unknown)}")
        cases = [c for c in cases if c.id in wanted]
    if args.kind:
        cases = [c for c in cases if c.kind == args.kind]

    created = datetime.datetime.now()
    out_path = RESULTS_DIR / f"{created:%Y-%m-%d}_{args.label}.json"
    if out_path.exists() and not args.force:
        sys.exit(f"{out_path} already exists: pick another --label, or pass --force.")

    disable_tracing()
    print(f"Running {len(cases)} case(s) × {args.repeats} with {args.model}…")
    results = run_all(cases, args.model, args.repeats)

    output = {
        "label": args.label,
        "created_at": created.isoformat(timespec="seconds"),
        "commit": _git_commit(),
        "model": args.model,
        "repeats": args.repeats,
        "agent_cost_usd": sum(run.get("cost_usd", 0) for c in results for run in c["runs"]),
        "cases": results,
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nAgent cost: ${output['agent_cost_usd']:.4f}. Saved to {out_path}")
    print(f"Next: python -m evaluation.judge {out_path}")


if __name__ == "__main__":
    main()
