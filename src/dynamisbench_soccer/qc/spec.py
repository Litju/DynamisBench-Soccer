"""QC rule specification (``qc/rules.toml``) and the runner."""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dynamisbench_soccer.registry.models import V01_TESTS, Registry
from dynamisbench_soccer.signals.schema import SignalBundle


class Severity(StrEnum):
    REJECT = "reject"
    FLAG = "flag"
    NOTE = "note"


class Status(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    NOT_APPLICABLE = "not_applicable"


class Rule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^QC-[A-Z]\d{2}$")
    name: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    level: Literal["signal", "trial"]
    applies_to: list[str] = Field(min_length=1)
    severity: Severity
    basis: Literal["evidence", "metadata"]
    claims: list[str]
    rationale: str = Field(min_length=20)
    params: dict[str, float] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _basis(self) -> Rule:
        unknown = set(self.applies_to) - set(V01_TESTS)
        if unknown:
            raise ValueError(f"{self.id}: unknown tests {sorted(unknown)}")
        if self.basis == "evidence" and not self.claims:
            raise ValueError(f"{self.id}: evidence-based rules must cite registry claims")
        return self


class RuleSet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str
    rule: list[Rule]

    def by_id(self, rule_id: str) -> Rule:
        return next(r for r in self.rule if r.id == rule_id)


@dataclass(frozen=True)
class Outcome:
    status: Status
    detail: str = ""
    values: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Result:
    rule_id: str
    name: str
    severity: Severity
    status: Status
    detail: str
    values: Mapping[str, float]


def load(path: Path) -> RuleSet:
    return RuleSet.model_validate(tomllib.loads(path.read_text(encoding="utf-8")))


def validate(rules: RuleSet, registry: Registry | None = None) -> list[str]:
    from dynamisbench_soccer.qc.checks import EVALUATORS

    problems = []
    ids = [r.id for r in rules.rule]
    if len(ids) != len(set(ids)):
        problems.append("duplicate rule ids")
    missing = set(ids) - set(EVALUATORS)
    extra = set(EVALUATORS) - set(ids)
    if missing:
        problems.append(f"rules without evaluator: {sorted(missing)}")
    if extra:
        problems.append(f"evaluators without rule: {sorted(extra)}")
    if registry is not None:
        known = {c.id for c in registry.claims}
        for r in rules.rule:
            for cid in r.claims:
                if cid not in known:
                    problems.append(f"{r.id}: unknown registry claim {cid}")
    return problems


def run(
    bundle: SignalBundle,
    rules: RuleSet,
    reference: SignalBundle | None = None,
    overrides: Mapping[str, Mapping[str, float]] | None = None,
) -> list[Result]:
    """Evaluate every rule that applies to the bundle's test."""
    from dynamisbench_soccer.qc.checks import EVALUATORS

    results = []
    for rule in rules.rule:
        if bundle.meta.test_id not in rule.applies_to:
            continue
        params = {**rule.params, **(overrides or {}).get(rule.id, {})}
        outcome = EVALUATORS[rule.id](bundle, params, reference)
        results.append(
            Result(
                rule.id,
                rule.name,
                rule.severity,
                outcome.status,
                outcome.detail,
                dict(outcome.values),
            )
        )
    return results


def verdict(results: list[Result]) -> Literal["accept", "accept_with_flags", "reject"]:
    failed = [r for r in results if r.status is Status.FAIL]
    if any(r.severity is Severity.REJECT for r in failed):
        return "reject"
    if failed:
        return "accept_with_flags"
    return "accept"
