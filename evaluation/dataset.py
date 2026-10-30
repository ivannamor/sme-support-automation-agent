"""
Loads and validates the golden set (evaluation/datasets/golden.yaml).

Validation runs in CI (tests/test_evals.py), so a malformed case — a typo in
a tool name, a missing field, a duplicated id — fails the build instead of
silently scoring 0 in the next eval run.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

GOLDEN_PATH = Path(__file__).resolve().parent / "datasets" / "golden.yaml"

NO_TOOL = "no_tool"
ROLES = ("client", "installer", "manager")
LANGUAGES = ("it", "en", "de")


@dataclass(frozen=True)
class RoutingCase:
    id: str
    role: str
    message: str
    expect_tool: tuple[str, ...]  # any one is accepted; NO_TOOL = a plain text reply
    forbid_tool: tuple[str, ...] = ()
    expect_args: dict[str, Any] = field(default_factory=dict)
    facts: tuple[str, ...] = ()
    kind: str = "routing"

    @property
    def prompt(self) -> str:
        return self.message


@dataclass(frozen=True)
class KnowledgeCase:
    id: str
    role: str
    language: str
    question: str
    facts: tuple[str, ...]
    reference: str
    expect_sources: tuple[str, ...] = ()  # any one is enough; empty: not in the documents
    evidence: str = ""
    kind: str = "knowledge"

    @property
    def prompt(self) -> str:
        return self.question


Case = RoutingCase | KnowledgeCase

_ROUTING_FIELDS = {
    "id",
    "kind",
    "role",
    "message",
    "expect_tool",
    "forbid_tool",
    "expect_args",
    "facts",
}
_KNOWLEDGE_FIELDS = {
    "id",
    "kind",
    "role",
    "language",
    "question",
    "expect_sources",
    "facts",
    "evidence",
    "reference",
}


def _as_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    return tuple(value) if isinstance(value, list) else (value,)


def _role_tools() -> dict[str, set[str]]:
    # Imported here: the agent module needs the API keys set (CI uses dummies).
    from backend.assistant import DoorAssistantAgent

    return {
        role: {t.name for t in factory()}
        for role, factory in DoorAssistantAgent._ROLE_TOOLS.items()
    }


def _parse(raw: dict, role_tools: dict[str, set[str]]) -> Case:
    case_id = raw.get("id") or "<missing id>"

    def fail(reason: str) -> ValueError:
        return ValueError(f"golden case {case_id}: {reason}")

    kind: str = raw.get("kind", "")
    allowed = {"routing": _ROUTING_FIELDS, "knowledge": _KNOWLEDGE_FIELDS}.get(kind)
    if allowed is None:
        raise fail(f"unknown kind {kind!r}")
    unknown = set(raw) - allowed
    if unknown:
        raise fail(f"unknown field(s) {sorted(unknown)}")
    role: str = raw.get("role", "")
    if role not in ROLES:
        raise fail(f"unknown role {role!r}")

    if kind == "routing":
        if not raw.get("message"):
            raise fail("missing message")
        expect = _as_tuple(raw.get("expect_tool"))
        forbid = _as_tuple(raw.get("forbid_tool"))
        if not expect:
            raise fail("missing expect_tool")
        for name in expect + forbid:
            if name != NO_TOOL and name not in role_tools[role]:
                raise fail(f"{name!r} is not a {role} tool")
        if set(expect) & set(forbid):
            raise fail("a tool is both expected and forbidden")
        return RoutingCase(
            id=case_id,
            role=role,
            message=raw["message"],
            expect_tool=expect,
            forbid_tool=forbid,
            expect_args=raw.get("expect_args") or {},
            facts=_as_tuple(raw.get("facts")),
        )

    for required in ("language", "question", "facts", "reference"):
        if not raw.get(required):
            raise fail(f"missing {required}")
    if raw["language"] not in LANGUAGES:
        raise fail(f"unknown language {raw['language']!r}")
    return KnowledgeCase(
        id=case_id,
        role=role,
        language=raw["language"],
        question=raw["question"],
        facts=_as_tuple(raw["facts"]),
        reference=raw["reference"],
        expect_sources=_as_tuple(raw.get("expect_sources")),
        evidence=raw.get("evidence", ""),
    )


def load_cases(path: Path = GOLDEN_PATH) -> list[Case]:
    """Every case of the golden set, validated. Raises ValueError on the first bad one."""
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    role_tools = _role_tools()
    cases = [_parse(raw, role_tools) for raw in data["cases"]]
    ids = [c.id for c in cases]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    if duplicates:
        raise ValueError(f"duplicated golden case id(s): {duplicates}")
    return cases
