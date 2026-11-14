"""Load and validate ``ontology/*.toml``.

The task schema (M4) references these ids: competency ``C1``-``C8``, test ids (equal to
``registry.models.V01_TESTS``), estimands, context values and boundary rule ids.
"""

from __future__ import annotations

import re
import tomllib
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from dynamisbench_soccer.registry.models import V01_TESTS, Registry

ESTIMAND_PATTERN = re.compile(r"^[a-z][a-z0-9_]*(\.[a-z0-9_]+)*$")
COMPETENCIES = tuple(f"C{i}" for i in range(1, 9))
CONTEXT_ONLY_TESTS = {"ift_30_15", "rsa"}


class Status(StrEnum):
    CORE = "core"
    EXTENSION = "extension"
    OUT = "out"


class Severity(StrEnum):
    FAIL = "fail"
    PENALTY = "penalty"
    FLAG = "flag"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class EpistemicClass(_Strict):
    id: str
    name: str
    definition: str
    agent_may: str


class Competency(_Strict):
    id: str
    name: str
    description: str


class BatteryTest(_Strict):
    id: str
    name: str
    device_class: str
    estimands: list[str] = Field(min_length=1)

    @field_validator("estimands")
    @classmethod
    def _estimand_names(cls, values: list[str]) -> list[str]:
        bad = [v for v in values if not ESTIMAND_PATTERN.match(v)]
        if bad:
            raise ValueError(f"estimand names must be dotted snake_case: {bad}")
        if len(values) != len(set(values)):
            raise ValueError("duplicate estimands")
        return values


class ContextDim(_Strict):
    id: str
    values: list[str] = Field(min_length=1)


class Ontology(_Strict):
    version: str
    epistemic_class: list[EpistemicClass]
    competency: list[Competency]
    test: list[BatteryTest]
    context: list[ContextDim]


class CoverageRow(BaseModel):
    model_config = ConfigDict(extra="forbid")

    test: str
    C1: Status
    C2: Status
    C3: Status
    C4: Status
    C5: Status
    C6: Status
    C7: Status
    C8: Status

    def cell(self, competency: str) -> Status:
        value: Status = getattr(self, competency)
        return value


class Coverage(_Strict):
    version: str
    row: list[CoverageRow]
    context: dict[str, dict[str, Status]]

    def row_for(self, test: str) -> CoverageRow:
        return next(r for r in self.row if r.test == test)


class BoundaryRule(_Strict):
    id: str = Field(pattern=r"^DB-\d{2}$")
    title: str
    statement: str
    severity: Severity
    reference_patterns: list[str] = Field(default_factory=list)
    grader_hint: str | None = None
    violating: list[str] = Field(min_length=1)
    compliant: list[str] = Field(min_length=1)

    @field_validator("reference_patterns")
    @classmethod
    def _compiles(cls, values: list[str]) -> list[str]:
        for v in values:
            re.compile(v, re.IGNORECASE)
        return values


class Boundaries(_Strict):
    version: str
    rule: list[BoundaryRule]


class DefaultAnswer(_Strict):
    id: str
    question: str
    default: str
    rationale: str


class Defaults(_Strict):
    version: str
    decided: str
    status: str
    question: list[DefaultAnswer]


class Bundle(BaseModel):
    ontology: Ontology
    coverage: Coverage
    boundaries: Boundaries
    defaults: Defaults


def _toml(path: Path) -> dict[str, Any]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def load(root: Path) -> Bundle:
    return Bundle(
        ontology=Ontology.model_validate(_toml(root / "ontology.toml")),
        coverage=Coverage.model_validate(_toml(root / "coverage.toml")),
        boundaries=Boundaries.model_validate(_toml(root / "boundaries.toml")),
        defaults=Defaults.model_validate(_toml(root / "defaults.toml")),
    )


def validate(bundle: Bundle, registry: Registry | None = None) -> list[str]:
    """Cross-file consistency. Returns problems (empty means valid)."""
    onto, cov = bundle.ontology, bundle.coverage
    problems: list[str] = []

    comp_ids = [c.id for c in onto.competency]
    if tuple(comp_ids) != COMPETENCIES:
        problems.append(f"competencies must be exactly {COMPETENCIES}, got {comp_ids}")
    test_ids = [t.id for t in onto.test]
    if tuple(test_ids) != V01_TESTS:
        problems.append(f"ontology tests must equal registry V01_TESTS {V01_TESTS}")
    row_ids = [r.test for r in cov.row]
    if sorted(row_ids) != sorted(test_ids) or len(row_ids) != len(set(row_ids)):
        problems.append("coverage must have exactly one row per ontology test")

    gaps = {g.subject for g in registry.gaps} if registry else set()
    for row in cov.row:
        cells = {c: row.cell(c) for c in COMPETENCIES}
        core = [c for c, s in cells.items() if s is Status.CORE]
        if row.test in gaps and core:
            problems.append(f"{row.test} is a registry gap but has core cells {core}")
        if row.test in CONTEXT_ONLY_TESTS and set(core) - {"C3", "C6"}:
            problems.append(f"{row.test} is context-only; core allowed only for C3/C6")
        if row.test not in gaps and not core:
            problems.append(f"{row.test} has no core cell")
        if registry and row.test not in gaps and not registry.claims_for(row.test):
            problems.append(f"{row.test} has core cells but no registry claims")
    for comp in COMPETENCIES:
        if not any(r.cell(comp) is Status.CORE for r in cov.row):
            problems.append(f"competency {comp} is not core for any test")

    dims = {d.id: d.values for d in onto.context}
    if set(cov.context) != set(dims):
        problems.append("coverage context dimensions must match ontology contexts")
    for dim, values in cov.context.items():
        if dim in dims and sorted(values) != sorted(dims[dim]):
            problems.append(f"context {dim}: coverage values differ from ontology values")

    rule_ids = [r.id for r in bundle.boundaries.rule]
    if len(rule_ids) != len(set(rule_ids)):
        problems.append("duplicate boundary rule ids")
    if "DB-01" not in rule_ids:
        problems.append("medical-clearance rule DB-01 is mandatory")
    if [q.id for q in bundle.defaults.question] != ["Q1", "Q2", "Q3", "Q4"]:
        problems.append("defaults must answer Q1-Q4")
    return problems
