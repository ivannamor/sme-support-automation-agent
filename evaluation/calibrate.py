"""
Chooses the eval judge: candidate models judge the (answer, fact) pairs in
evaluation/calibration/pairs.yaml, which a human has labelled, and are scored
on how often they agree with the human.

    python -m evaluation.calibrate --models anthropic/claude-haiku-5.5,google/gemini-3.8-flash

Pick the cheapest model with ≥95% agreement, looking at false YES above all:
a judge that passes wrong answers hides regressions, while a false NO at
least gets looked at. The chosen model goes into judge.JUDGE_MODEL.

Results are saved to evaluation/calibration/results.json.
"""

import argparse
import json
from pathlib import Path

import yaml

from backend.usage_tracking import track_usage
from evaluation.judge import UNPARSED, get_judge_llm, judge_fact
from evaluation.sandbox import disable_tracing, sandbox

CALIBRATION_DIR = Path(__file__).resolve().parent / "calibration"
PAIRS_PATH = CALIBRATION_DIR / "pairs.yaml"


def load_labelled_pairs(path: Path = PAIRS_PATH) -> list[dict]:
    """Pairs with a human label (YAML yes/no → True/False); unlabelled ones are skipped."""
    pairs = yaml.safe_load(path.read_text(encoding="utf-8"))["pairs"]
    for pair in pairs:
        if pair.get("label") not in (True, False, None):
            raise ValueError(f"calibration pair {pair['id']}: label must be yes, no or empty")
    return [p for p in pairs if p.get("label") is not None]


def score(pairs: list[dict], verdicts: list[dict]) -> dict:
    labels = [p["label"] for p in pairs]
    judged = [v["verdict"] for v in verdicts]
    return {
        "agreement": round(
            sum(h == j for h, j in zip(labels, judged, strict=True)) / len(pairs), 3
        ),
        "false_yes": sum(j and not h for h, j in zip(labels, judged, strict=True)),
        "false_no": sum(h and not j for h, j in zip(labels, judged, strict=True)),
        "unparsed": sum(v["reason"].startswith(UNPARSED) for v in verdicts),
    }


def calibrate(model: str, pairs: list[dict]) -> dict:
    llm = get_judge_llm(model)
    with track_usage("eval") as usage:
        verdicts = [judge_fact(llm, p["question"], p["answer"], p["fact"]) for p in pairs]
    return {
        **score(pairs, verdicts),
        "cost_usd": round(usage.cost_usd(), 5),
        "disagreements": [
            {"id": p["id"], "human": p["label"], "judge": v["verdict"], "reason": v["reason"]}
            for p, v in zip(pairs, verdicts, strict=True)
            if p["label"] != v["verdict"]
        ],
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare judge models against human labels.")
    parser.add_argument("--models", required=True, help="comma-separated OpenRouter model ids")
    args = parser.parse_args(argv)

    pairs = load_labelled_pairs()
    print(f"{len(pairs)} labelled pairs.")
    disable_tracing()
    results = {}
    with sandbox():
        for model in args.models.split(","):
            print(f"Judging with {model}…")
            results[model] = calibrate(model, pairs)

    print(
        f"\n{'model':40s} {'agree':>6s} {'falseYES':>9s} {'falseNO':>8s} {'unread':>7s} {'cost $':>9s}"
    )
    for model, r in results.items():
        print(
            f"{model:40s} {r['agreement']:>6.1%} {r['false_yes']:>9d} "
            f"{r['false_no']:>8d} {r['unparsed']:>7d} {r['cost_usd']:>9.5f}"
        )
        for d in r["disagreements"]:
            print(f"    {d['id']}: human={d['human']} judge={d['judge']} — {d['reason'][:100]}")

    out = CALIBRATION_DIR / "results.json"
    out.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
