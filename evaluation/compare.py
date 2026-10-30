"""
Compares two judged results files, e.g. a baseline and an experiment:

    python -m evaluation.compare evaluation/results/A.json evaluation/results/B.json

Prints every summary metric side by side, then the cases whose pass rate
changed. Read the deltas against the noise floor: the same configuration run
twice (an A/A comparison) also moves a little, and a change smaller than that
isn't evidence of anything.
"""

import argparse
import json
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "summary" not in data:
        raise SystemExit(f"{path} isn't judged yet: python -m evaluation.judge {path}")
    return data


def _fmt(value) -> str:
    return "—" if value is None else f"{value:.4g}" if isinstance(value, float) else str(value)


def _cost_per_run(data: dict) -> float:
    """Per run, so files with different --repeats compare."""
    runs = sum(len(c["runs"]) for c in data["cases"])
    return round(data["agent_cost_usd"] / runs, 5) if runs else 0.0


def metric_rows(a: dict, b: dict) -> list[tuple[str, Any, Any]]:
    rows: list[tuple[str, Any, Any]] = [("pass_rate", a.get("pass_rate"), b.get("pass_rate"))]
    for section in ("routing", "knowledge"):
        keys = list(dict.fromkeys([*a.get(section, {}), *b.get(section, {})]))
        rows += [
            (f"{section}.{k}", a.get(section, {}).get(k), b.get(section, {}).get(k)) for k in keys
        ]
    return rows


def changed_cases(a: dict, b: dict) -> list[tuple[str, float | None, float | None]]:
    """Cases whose pass rate differs (or that exist in one file only), worst drop first."""
    ids = list(dict.fromkeys([*a["cases"], *b["cases"]]))
    changed = [(i, a["cases"].get(i), b["cases"].get(i)) for i in ids]
    changed = [row for row in changed if row[1] != row[2]]
    return sorted(changed, key=lambda r: (r[2] or 0) - (r[1] or 0))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compare two judged eval results.")
    parser.add_argument("a", type=Path)
    parser.add_argument("b", type=Path)
    args = parser.parse_args(argv)
    a, b = _load(args.a), _load(args.b)

    if a["judge"]["model"] != b["judge"]["model"]:
        print(
            f"⚠️  Different judges ({a['judge']['model']} vs {b['judge']['model']}): "
            "scores aren't comparable."
        )
    print(f"A: {args.a.name}  ({a['model']}, ×{a['repeats']}, {a['commit']})")
    print(f"B: {args.b.name}  ({b['model']}, ×{b['repeats']}, {b['commit']})\n")

    print(f"{'metric':32s} {'A':>8s} {'B':>8s} {'Δ':>8s}")
    rows = metric_rows(a["summary"], b["summary"])
    rows.append(("agent_cost_per_run_usd", _cost_per_run(a), _cost_per_run(b)))
    for name, va, vb in rows:
        both_numbers = isinstance(va, int | float) and isinstance(vb, int | float)
        delta = f"{vb - va:+.4g}" if both_numbers and vb != va else ""
        print(f"{name:32s} {_fmt(va):>8s} {_fmt(vb):>8s} {delta:>8s}")

    changed = changed_cases(a["summary"], b["summary"])
    if changed:
        print("\nCases whose pass rate changed:")
        for case_id, ra, rb in changed:
            print(f"  {_fmt(ra):>6s} → {_fmt(rb):<6s} {case_id}")
    else:
        print("\nNo case changed pass rate.")


if __name__ == "__main__":
    main()
