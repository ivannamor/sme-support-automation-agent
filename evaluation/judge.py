"""
Agent evals, phase 2 of 2: scores a results file written by run_evals, and
writes the scores back into it.

    python -m evaluation.judge evaluation/results/2026-10-30_baseline.json
    python -m evaluation.judge <file> --no-ragas      # skip RAGAS (cheaper, faster)

Checks per run:
  routing    — the first decision is an expected one; arguments match
               expect_args; a forbid_tool call is counted separately as the
               serious mistake it is; for a text reply, the facts.
  knowledge  — every fact holds (LLM judge, one call per fact); the reply is
               in the question's language; retrieval found an expected
               source. RAGAS adds faithfulness, answer relevancy and context
               precision/recall against the hand-written reference.

A run passes when its decision (routing) or its facts and language
(knowledge) are right. Retrieval and RAGAS are diagnostics: they say WHERE a
failure comes from, but a right answer from the "wrong" document still passes.

The judge model stays fixed across runs (JUDGE_MODEL), at temperature 0:
changing it changes every score, so results judged by different models can't
be compared. It was chosen by calibration against human labels — see
evaluation/calibrate.py.
"""

import argparse
import datetime
import json
import re
import statistics
from pathlib import Path
from typing import Any

from backend.llm_provider import get_embeddings, openrouter_chat_model
from backend.usage_tracking import track_usage
from evaluation.dataset import NO_TOOL, Case, KnowledgeCase, RoutingCase, load_cases
from evaluation.sandbox import disable_tracing, sandbox

JUDGE_MODEL = "anthropic/claude-sonnet-5.5"

# Far more than a verdict and a one-sentence reason need: the candidate models
# think first, and their reasoning counts against this limit. At 300, some
# replies were cut off before the verdict was even written.
_JUDGE_MAX_TOKENS = 2000

UNPARSED = "unparsed"
_VERDICT = re.compile(r"\W*(YES|NO)\b\W*", re.IGNORECASE)

_FACT_PROMPT = """You are grading the reply of an AI assistant for a door and window company.

User message:
{question}

Assistant reply:
{answer}

Statement to check:
{fact}

Does the reply satisfy the statement? Judge the meaning, not the wording: the reply may be in any language and phrase things differently. Extra information doesn't matter unless it contradicts the statement. When the statement says the reply must NOT do something, answer YES only if the reply avoids it. Answer NO if the reply is vaguer than the statement, or gives it only as an uncertain guess.

Answer YES or NO on the first line, then one short sentence explaining why."""

_LANGUAGE_PROMPT = """Which language is the following text written in? Ignore product names and quoted technical terms. Reply with only the ISO 639-1 code (for example: it, en, de).

Text:
{text}"""


def get_judge_llm(model: str = JUDGE_MODEL, max_tokens: int = _JUDGE_MAX_TOKENS):
    return openrouter_chat_model(model, temperature=0, max_tokens=max_tokens)


def _content(llm, prompt: str) -> str:
    raw = llm.invoke(prompt).content
    if isinstance(raw, list):
        raw = "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in raw)
    return str(raw).strip()


def judge_fact(llm, question: str, answer: str, fact: str) -> dict:
    """{"fact", "verdict": True/False, "reason"} — an unreadable verdict counts as NO."""
    text = _content(llm, _FACT_PROMPT.format(question=question, answer=answer, fact=fact))
    match = _VERDICT.match(text)
    if not match:
        return {"fact": fact, "verdict": False, "reason": f"{UNPARSED}: {text[:200]}"}
    return {
        "fact": fact,
        "verdict": match[1].upper() == "YES",
        "reason": text[match.end() :].strip(),
    }


def detect_language(llm, text: str) -> str:
    return _content(llm, _LANGUAGE_PROMPT.format(text=text[:1500])).lower()[:2]


# ── deterministic checks ──────────────────────────────────────────────────────


def _same_value(expected: Any, actual: Any) -> bool:
    if isinstance(expected, int | float) and not isinstance(expected, bool):
        try:
            return float(str(actual).replace(",", ".")) == float(expected)
        except (TypeError, ValueError):
            return False
    return str(actual).strip().casefold() == str(expected).strip().casefold()


def args_match(expected: dict, actual: dict) -> bool:
    return all(_same_value(value, actual.get(key)) for key, value in expected.items())


def decision(run: dict) -> str:
    calls = run.get("tool_calls") or []
    return calls[0]["name"] if calls else NO_TOOL


def retrieved_sources(run: dict) -> set[str]:
    return {doc["source"] for search in run.get("searches", []) for doc in search["docs"]}


def routing_checks(case: RoutingCase, run: dict) -> dict:
    chosen = decision(run)
    checks: dict[str, Any] = {
        "decision": chosen,
        "tool_ok": chosen in case.expect_tool,
        "forbidden": chosen in case.forbid_tool,
    }
    if case.expect_args and chosen != NO_TOOL:
        checks["args_ok"] = args_match(case.expect_args, run["tool_calls"][0]["args"])
    return checks


def knowledge_checks(case: KnowledgeCase, run: dict) -> dict:
    checks: dict[str, Any] = {"tools": [tc["name"] for tc in run["tool_calls"]]}
    if case.expect_sources:
        checks["retrieval_hit"] = bool(retrieved_sources(run) & set(case.expect_sources))
    return checks


def judge_run(llm, case: Case, run: dict) -> dict:
    if "error" in run:
        return {"passed": False, "error": run["error"]}
    if isinstance(case, RoutingCase):
        checks = routing_checks(case, run)
        if case.facts and checks["decision"] == NO_TOOL:
            checks["facts"] = [judge_fact(llm, case.message, run["reply"], f) for f in case.facts]
        checks["passed"] = (
            checks["tool_ok"]
            and checks.get("args_ok", True)
            and all(f["verdict"] for f in checks.get("facts", []))
        )
        return checks

    checks = knowledge_checks(case, run)
    checks["facts"] = [judge_fact(llm, case.question, run["reply"], f) for f in case.facts]
    checks["language"] = detect_language(llm, run["reply"])
    checks["language_ok"] = checks["language"] == case.language
    checks["passed"] = checks["language_ok"] and all(f["verdict"] for f in checks["facts"])
    return checks


# ── RAGAS ─────────────────────────────────────────────────────────────────────

RAGAS_METRICS = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")


def ragas_scores(model: str, items: list[tuple[KnowledgeCase, dict]]) -> list[dict]:
    """RAGAS metrics per (case, run), for runs that retrieved something."""
    from ragas import evaluate
    from ragas.dataset_schema import EvaluationDataset, EvaluationResult, SingleTurnSample
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

    samples = [
        SingleTurnSample(
            user_input=case.question,
            response=run["reply"],
            retrieved_contexts=[d["content"] for s in run["searches"] for d in s["docs"]],
            reference=case.reference,
        )
        for case, run in items
    ]
    # RAGAS' multi-step prompts need far more room than a verdict: a truncated
    # reply makes a sub-score silently come back as 0.
    result = evaluate(
        dataset=EvaluationDataset(samples=samples),  # type: ignore[arg-type]
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=LangchainLLMWrapper(get_judge_llm(model, max_tokens=4000)),
        embeddings=LangchainEmbeddingsWrapper(get_embeddings()),
    )
    assert isinstance(result, EvaluationResult)  # not an Executor: return_executor is off
    rows = result.to_pandas().to_dict(orient="records")
    return [
        {
            m: (None if row.get(m) is None or row[m] != row[m] else round(row[m], 3))
            for m in RAGAS_METRICS
        }
        for row in rows
    ]


# ── summary ───────────────────────────────────────────────────────────────────


def _rate(values: list[bool]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


def summarize(cases: list[Case], results: list[dict]) -> dict:
    by_id = {c.id: c for c in cases}
    routing: list[dict] = []
    knowledge: list[dict] = []
    per_case = {}
    for entry in results:
        checks = [run["checks"] for run in entry["runs"]]
        per_case[entry["id"]] = _rate([c["passed"] for c in checks])
        (routing if isinstance(by_id[entry["id"]], RoutingCase) else knowledge).extend(checks)

    facts = [f["verdict"] for c in knowledge for f in c.get("facts", [])]
    judged = [f for c in routing + knowledge for f in c.get("facts", [])]
    summary: dict[str, Any] = {
        "pass_rate": _rate([c["passed"] for c in routing + knowledge]),
        # Counted as NO, so any of these pulls the scores down: find out why before trusting them.
        "unparsed_verdicts": sum(f.get("reason", "").startswith(UNPARSED) for f in judged),
        "routing": {
            "runs": len(routing),
            "pass_rate": _rate([c["passed"] for c in routing]),
            "tool_accuracy": _rate([c.get("tool_ok", False) for c in routing]),
            "args_accuracy": _rate([c["args_ok"] for c in routing if "args_ok" in c]),
            "forbidden_calls": sum(c.get("forbidden", False) for c in routing),
        },
        "knowledge": {
            "runs": len(knowledge),
            "pass_rate": _rate([c["passed"] for c in knowledge]),
            "fact_recall": _rate(facts),
            "language_match": _rate([c.get("language_ok", False) for c in knowledge]),
            "retrieval_hit_rate": _rate(
                [c["retrieval_hit"] for c in knowledge if "retrieval_hit" in c]
            ),
        },
        "cases": per_case,
    }
    for metric in RAGAS_METRICS:
        values = [
            c["ragas"][metric] for c in knowledge if c.get("ragas", {}).get(metric) is not None
        ]
        if values:
            summary["knowledge"][metric] = round(statistics.mean(values), 3)
    return summary


def print_summary(summary: dict) -> None:
    print(f"\nOverall pass rate: {summary['pass_rate']}")
    if summary["unparsed_verdicts"]:
        print(f"⚠️  {summary['unparsed_verdicts']} unreadable verdict(s), counted as NO.")
    for section in ("routing", "knowledge"):
        print(f"\n{section}")
        for key, value in summary[section].items():
            print(f"  {key:20s} {value}")
    failing = {k: v for k, v in summary["cases"].items() if v is not None and v < 1}
    if failing:
        print("\nCases not always passing:")
        for case_id, rate in sorted(failing.items(), key=lambda kv: kv[1]):
            print(f"  {rate:.2f}  {case_id}")


def judge_file(path: Path, model: str = JUDGE_MODEL, use_ragas: bool = True) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    cases = {c.id: c for c in load_cases()}
    llm = get_judge_llm(model)

    with sandbox(), track_usage("eval") as usage:
        ragas_items: list[tuple[KnowledgeCase, dict]] = []
        for entry in data["cases"]:
            case = cases[entry["id"]]
            for run in entry["runs"]:
                run["checks"] = judge_run(llm, case, run)
                if (
                    use_ragas
                    and isinstance(case, KnowledgeCase)
                    and "error" not in run
                    and any(s["docs"] for s in run.get("searches", []))
                ):
                    ragas_items.append((case, run))
            print(f"  judged {entry['id']}")
        if ragas_items:
            print(f"Running RAGAS on {len(ragas_items)} run(s)…")
            for (_, run), scores in zip(ragas_items, ragas_scores(model, ragas_items), strict=True):
                run["checks"]["ragas"] = scores

    data["judge"] = {
        "model": model,
        "judged_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "ragas": use_ragas,
        "cost_usd": round(usage.cost_usd(), 4),
    }
    data["summary"] = summarize(list(cases.values()), data["cases"])
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return data


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Score an eval results file.")
    parser.add_argument("results", type=Path)
    parser.add_argument("--judge-model", default=JUDGE_MODEL)
    parser.add_argument("--no-ragas", action="store_true")
    args = parser.parse_args(argv)

    disable_tracing()
    data = judge_file(args.results, args.judge_model, use_ragas=not args.no_ragas)
    print_summary(data["summary"])
    print(f"\nJudge cost: ${data['judge']['cost_usd']:.4f}. Scores written to {args.results}")


if __name__ == "__main__":
    main()
