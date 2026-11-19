"""Evidence-based admission contract for new device families (RES-702).

A proposal is a request to *start* qualifying a family; it is never a qualification. The
contract checks that the proposal carries what the existing ledger demands of every family
(all fifteen dimensions with owners, a paradigm-valid method set, an independent reference
plan, registry-backed metrology evidence, explicit non-equivalences, a data-rights statement)
and that it cannot alter the qualified scope: the eight families and fifteen subfamilies are
fixed constants of the ledger and an admission never edits them. An admitted candidate starts
at NOT_STARTED and can only advance through the same ledger gates.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from dynamisbench_soccer.qualification.ledger import (
    CONTINUOUS_FILTERS,
    DIMENSIONS,
    FAMILIES,
    ISSUE_PATTERN,
    METHODS,
    PARADIGMS,
    SUBFAMILIES,
)

FAMILY_PATTERN = re.compile(r"[a-z][a-z0-9_]{2,40}\Z")
Decision = Literal["ADMISSION_READY", "REJECTED"]


class NonEquivalence(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    against: str = Field(min_length=3, description="existing subfamily id")
    quantity: str = Field(min_length=3)
    study_plan: str = Field(min_length=10)


class ReferencePlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    dimension: str
    kind: Literal["independent_test", "analytical_reference", "empirical_study", "external_review"]
    implementation: str = Field(min_length=3, description="what computes the reference")
    independent_of: str = Field(min_length=3, description="the engine it verifies")


class Proposal(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    family: str
    subfamilies: tuple[str, ...] = Field(min_length=1)
    paradigm: str
    engine_id: str = Field(min_length=2)
    methods: tuple[str, ...] = ()
    owners: dict[str, str]
    registry_claims: tuple[str, ...] = ()
    registry_gaps: tuple[str, ...] = ()
    references: tuple[ReferencePlan, ...] = ()
    non_equivalences: tuple[NonEquivalence, ...] = ()
    data_rights_statement: str = Field(default="", description="who may use which data and how")
    scope_impact: Literal["none"] = "none"
    requested_ratings: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _shape(self) -> Proposal:
        if any(not s.startswith(self.family + ".") for s in self.subfamilies):
            raise ValueError("every subfamily id must start with '<family>.'")
        return self


@dataclass(frozen=True)
class AdmissionResult:
    family: str
    decision: Decision
    problems: tuple[str, ...]
    initial_state: str
    scope_changed: bool

    def report(self) -> dict[str, object]:
        return {
            "family": self.family,
            "decision": self.decision,
            "initial_state": self.initial_state,
            "problems": list(self.problems),
            "scope_changed": self.scope_changed,
            "qualification": "admission starts qualification; it confers none",
        }


def load(path: Path) -> Proposal:
    return Proposal.model_validate(tomllib.loads(path.read_text(encoding="utf-8")))


def evaluate(
    proposal: Proposal,
    *,
    registry_claim_ids: set[str],
    registry_gap_subjects: set[str],
) -> AdmissionResult:
    problems: list[str] = []
    if not FAMILY_PATTERN.fullmatch(proposal.family):
        problems.append("family id must be lowercase snake_case (3-41 characters)")
    if proposal.family in FAMILIES:
        problems.append(f"{proposal.family} is an existing qualified family; its scope is fixed")
    clash = set(proposal.subfamilies) & SUBFAMILIES
    if clash:
        problems.append(f"subfamilies collide with the qualified scope: {sorted(clash)}")
    if proposal.paradigm not in PARADIGMS:
        problems.append(f"unknown acquisition paradigm {proposal.paradigm!r}")
    elif set(proposal.methods) - METHODS[proposal.paradigm]:
        problems.append("processing methods are not valid for the declared paradigm")
    if proposal.paradigm == "discrete_event" and CONTINUOUS_FILTERS & set(proposal.methods):
        problems.append("a discrete-event family cannot declare continuous-signal filters")
    if set(proposal.owners) != set(DIMENSIONS):
        problems.append("owning issues are required for every dimension A-O")
    elif any(not ISSUE_PATTERN.fullmatch(v) for v in proposal.owners.values()):
        problems.append("every dimension owner must be a RES-<n> issue")
    if proposal.requested_ratings:
        problems.append(
            "an admission cannot request ratings; ratings are earned through ledger evidence"
        )
    if not proposal.registry_claims and not proposal.registry_gaps:
        problems.append("admission needs registry-backed metrology evidence (claims or gaps)")
    unknown_claims = set(proposal.registry_claims) - registry_claim_ids
    if unknown_claims:
        problems.append(f"unknown registry claims {sorted(unknown_claims)}")
    unknown_gaps = set(proposal.registry_gaps) - registry_gap_subjects
    if unknown_gaps:
        problems.append(f"unknown registry gaps {sorted(unknown_gaps)}")
    if not proposal.references:
        problems.append("an independent reference plan is required (dimension J)")
    for ref in proposal.references:
        if ref.dimension not in DIMENSIONS:
            problems.append(f"reference targets unknown dimension {ref.dimension!r}")
        if ref.implementation.strip() == ref.independent_of.strip() or (
            ref.independent_of.strip() == proposal.engine_id
            and ref.implementation.strip() == proposal.engine_id
        ):
            problems.append("a reference may not be computed by the engine it verifies")
        if "vendor" in ref.implementation.lower():
            problems.append("vendor output is never an independent reference")
    if not any(r.dimension == "J" for r in proposal.references):
        problems.append("at least one reference plan must target dimension J")
    if not proposal.non_equivalences:
        problems.append("declare non-equivalences against at least one qualified subfamily")
    for item in proposal.non_equivalences:
        if item.against not in SUBFAMILIES:
            problems.append(f"non-equivalence names an unknown subfamily {item.against!r}")
    if len(proposal.data_rights_statement.strip()) < 20:
        problems.append("a data-rights statement (source, license, permitted use) is required")
    return AdmissionResult(
        proposal.family,
        "REJECTED" if problems else "ADMISSION_READY",
        tuple(problems),
        "NOT_STARTED",
        scope_changed=False,
    )
