"""Computed per-subfamily qualification states and the M3 gate (RES-675).

Nothing here is declared. A state is a pure function of the ledger's per-dimension ratings,
evidence kinds and blockers, so changing the evidence changes the state and an unproven
subfamily can never read as verified. Rules follow docs/m3-architecture.md.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from dynamisbench_soccer.qualification.ledger import (
    DIMENSIONS,
    M3_DIMENSIONS,
    Ledger,
    Rating,
    Subfamily,
    validate,
)

State = Literal[
    "NOT_STARTED",
    "IN_PROGRESS",
    "SPECIFIED",
    "ENGINEERING_VERIFIED",
    "SCIENTIFICALLY_QUALIFIED",
]
NON_ANALYTICAL = frozenset({"empirical_study", "external_review"})
INDEPENDENT = frozenset({"independent_test", "analytical_reference", *NON_ANALYTICAL})
REQUIRED_EXPERIMENT: dict[str, str] = {
    "A": "audited native raw-data acquisition from the device",
    "B": "measured calibration, axes and geometry records for the device",
    "C": "measured clock, latency and synchronisation characterisation",
    "D": "empirical signal-integrity study with real fault occurrences",
    "E": "method comparison on measured data with accepted sensitivity bounds",
    "F": "device-specific reconstruction checked against a criterion measure",
    "G": "protocol event detection checked against adjudicated events",
    "H": "metric-definition review against the protocol literature",
    "I": "measurement-uncertainty budget from device metrology",
    "J": "independent reference implementation or empirical criterion",
    "K": "fixtures covering measured fault classes",
    "L": "paired measured cross-device agreement study (at least 20 independent subjects)",
    "M": "seeded bundle chain with provenance for every processing stage",
    "N": "AI task set with qualified grader controls (M4/M5)",
    "O": "sourced limitation statement reviewed by a domain expert",
}


@dataclass(frozen=True)
class SubfamilyState:
    subfamily: str
    family: str
    state: State
    verified_m3: tuple[str, ...]
    unverified_m3: tuple[str, ...]
    ratings: dict[str, str]
    blockers: tuple[str, ...]
    owners: dict[str, str]
    required_experiments: dict[str, str]
    independent_reference: bool
    non_analytical_reference: bool


def subfamily_state(model: Ledger, sub: Subfamily) -> SubfamilyState:
    ratings = sub.ratings
    evidence = [e for e in model.evidence if e.subfamily == sub.id]
    verified = tuple(
        d for d in M3_DIMENSIONS if ratings[d] in {Rating.VERIFIED, Rating.NOT_APPLICABLE}
    )
    unverified = tuple(d for d in M3_DIMENSIONS if d not in verified)
    independent = any(
        e.dimension == "J" and e.kind in INDEPENDENT and e.independence_from == sub.engine_id
        for e in evidence
    )
    non_analytical = any(e.dimension == "J" and e.kind in NON_ANALYTICAL for e in evidence)
    all_verified = all(ratings[d] in {Rating.VERIFIED, Rating.NOT_APPLICABLE} for d in DIMENSIONS)
    if all(ratings[d] == Rating.ABSENT for d in M3_DIMENSIONS):
        state: State = "NOT_STARTED"
    elif not unverified and not sub.blockers and independent:
        state = (
            "SCIENTIFICALLY_QUALIFIED"
            if all_verified and non_analytical and _reviewed(model, sub)
            else "ENGINEERING_VERIFIED"
        )
    elif all(ratings[d] != Rating.ABSENT for d in M3_DIMENSIONS):
        state = "SPECIFIED"  # every M3 dimension has at least a specification or partial evidence
    else:
        state = "IN_PROGRESS"
    return SubfamilyState(
        sub.id,
        sub.family,
        state,
        verified,
        unverified,
        {d: ratings[d].value for d in DIMENSIONS},
        sub.blockers,
        dict(sub.owners),
        {d: REQUIRED_EXPERIMENT[d] for d in unverified},
        independent,
        non_analytical,
    )


def _reviewed(model: Ledger, sub: Subfamily) -> bool:
    accepted = {
        r.identity
        for r in model.reviewers
        if r.family == sub.family and r.independent and r.decision == "accepted" and r.evidence_ref
    }
    return len(accepted) >= 2


def computed_states(model: Ledger) -> list[SubfamilyState]:
    return [subfamily_state(model, s) for s in sorted(model.subfamilies, key=lambda s: s.id)]


@dataclass(frozen=True)
class M3Gate:
    passed: bool
    total_subfamilies: int
    engineering_verified: int
    by_state: dict[str, int]
    structural_problems: tuple[str, ...]
    blocking: dict[str, tuple[str, ...]]


def m3_gate(model: Ledger, root: object | None = None) -> M3Gate:
    from pathlib import Path

    problems = tuple(validate(model, Path(str(root)) if root is not None else None))
    states = computed_states(model)
    counts: dict[str, int] = {}
    for s in states:
        counts[s.state] = counts.get(s.state, 0) + 1
    verified = counts.get("ENGINEERING_VERIFIED", 0) + counts.get("SCIENTIFICALLY_QUALIFIED", 0)
    return M3Gate(
        passed=verified == len(states) and not problems,
        total_subfamilies=len(states),
        engineering_verified=verified,
        by_state=counts,
        structural_problems=problems,
        blocking={
            s.subfamily: tuple(f"{d}: {s.required_experiments[d]}" for d in s.unverified_m3)
            + tuple(f"blocker: {b}" for b in s.blockers)
            for s in states
            if s.state not in {"ENGINEERING_VERIFIED", "SCIENTIFICALLY_QUALIFIED"}
        },
    )
