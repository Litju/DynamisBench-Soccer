"""The single canonical M3 gate evaluator (RES-692 / RES-675, Policy A).

M3 = computational-reference readiness: for each of the 15 subfamilies, every A-M and O
dimension is verified computationally, with a substantive independent receipt
(``receipts.py``), no open computational blocker and an independent J reference.

It is deliberately silent about physical-device validity. That lives in ``empirical.py`` and
is reported next to, never merged into, the computational state. ``ledger.gate(Stage.M3)``,
``state.m3_gate`` and the ``dbs qualification gate|states`` commands all call this module, so
two answers to the same ledger cannot exist.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from dynamisbench_soccer.qualification import empirical as empirical_mod
from dynamisbench_soccer.qualification import receipts as receipts_mod
from dynamisbench_soccer.qualification.ledger import (
    DIMENSIONS,
    M3_DIMENSIONS,
    Ledger,
    Rating,
    Subfamily,
)

DONE_RATINGS = frozenset({Rating.VERIFIED, Rating.NOT_APPLICABLE})
NON_ANALYTICAL = frozenset({"empirical_study", "external_review"})
INDEPENDENT = frozenset({"independent_test", "analytical_reference", *NON_ANALYTICAL})

# Policy A: what each dimension requires *computationally* at M3.
COMPUTATIONAL: dict[str, str] = {
    "A": "executable admission contract: native/vendor stage, units and integrity refusals tested",
    "B": "declared calibration and geometry records are typed, applied by processing, and "
    "missing or contradictory records are refused",
    "C": "clock, latency and synchronisation models are declared, applied and propagated, and "
    "inconsistent timing is refused",
    "D": "integrity and QC checks reject injected faults and pass clean controls",
    "E": "every processing method variant is compared with an independent reference with "
    "stated sensitivity bounds",
    "F": "reconstructed quantities agree with an independent analytical truth within tolerance",
    "G": "protocol events are recovered against analytically known event times, with eligibility "
    "refusals",
    "H": "metric definitions are bound to sourced registry claims and executable exactly",
    "I": "declared uncertainty propagation is checked against an independent analytical result",
    "J": "an independent reference implementation or closed form that shares no code with the "
    "engine",
    "K": "synthetic fault fixtures cover the declared fault classes with expected verdicts",
    "L": "analytical cross-device studies are executed and classified, never read as "
    "interchangeability",
    "M": "seeded, hash-bound provenance chain for every processing stage, reproduced exactly",
    "O": "limitation statements are sourced and emitted with every result",
}
# What only physical measurement can establish (kept separate from the computational gate).
EMPIRICAL: dict[str, str] = {
    "A": "audited native raw-data acquisition from the device",
    "B": "measured calibration, axes and geometry records for the device",
    "C": "measured clock, latency and synchronisation characterisation",
    "D": "empirical signal-integrity study with real fault occurrences",
    "E": "method comparison on measured data with accepted sensitivity bounds",
    "F": "device-specific reconstruction checked against a criterion measure",
    "G": "protocol event detection checked against adjudicated events",
    "H": "metric-definition review against the protocol literature by a domain expert",
    "I": "measurement-uncertainty budget from device metrology",
    "J": "criterion measure or independent instrument for the physical quantity",
    "K": "fixtures covering measured fault classes",
    "L": "paired measured cross-device agreement study (at least 20 independent subjects)",
    "M": "provenance of measured data back to the acquisition session",
    "O": "limitation statement reviewed by a domain expert",
}


@dataclass(frozen=True)
class Cell:
    dimension: str
    rating: str
    verified: bool
    receipts: tuple[str, ...]
    reason: str | None


@dataclass(frozen=True)
class SubfamilyStatus:
    subfamily: str
    family: str
    state: str
    cells: dict[str, Cell]
    verified_dimensions: tuple[str, ...]
    unverified_dimensions: tuple[str, ...]
    computational_blockers: tuple[str, ...]
    empirical_requirements: dict[str, str]
    empirical_records: tuple[str, ...]
    empirical_dimensions_covered: tuple[str, ...]
    independent_reference: bool
    analytical_only_dimensions: tuple[str, ...]
    empirical_interchangeability: str


@dataclass(frozen=True)
class M3Report:
    passed: bool
    total_subfamilies: int
    engineering_verified: int
    by_state: dict[str, int]
    structural_problems: tuple[str, ...]
    receipt_problems: dict[str, tuple[str, ...]]
    empirical_problems: dict[str, tuple[str, ...]]
    computational_blockers: dict[str, tuple[str, ...]]
    empirical_requirements: dict[str, dict[str, str]]
    subfamilies: tuple[SubfamilyStatus, ...] = field(default=())


def _valid_receipts(
    model: Ledger, receipts: tuple[receipts_mod.Receipt, ...], root: Path
) -> tuple[dict[tuple[str, str], tuple[str, ...]], dict[str, list[str]], list[str]]:
    ids = {s.id for s in model.subfamilies}
    problems, global_errors = receipts_mod.verify_all(receipts, root, ids)
    index: dict[tuple[str, str], list[str]] = {}
    for r in receipts:
        if problems.get(r.id):
            continue
        for sub in r.subfamilies:
            index.setdefault((sub, r.dimension), []).append(r.id)
    return {k: tuple(v) for k, v in index.items()}, problems, global_errors


def receipt_structure_problems(
    model: Ledger,
    receipts: tuple[receipts_mod.Receipt, ...],
    empirical_records: tuple[empirical_mod.EmpiricalRecord, ...],
    root: Path,
) -> list[str]:
    """Problems that make a ledger invalid (used by ``validate`` and by the gate)."""
    index, problems, global_errors = _valid_receipts(model, receipts, root)
    errors = list(global_errors)
    for errs in problems.values():
        errors.extend(errs)
    sourced = {
        (e.subfamily, e.dimension)
        for e in model.evidence
        if e.kind in {"registry_claim", "registry_gap"}
    }
    for sub in model.subfamilies:
        for d in M3_DIMENSIONS:
            if sub.ratings.get(d) == Rating.VERIFIED and not index.get((sub.id, d)):
                errors.append(f"{sub.id}/{d}: rated verified without a valid independent receipt")
            if d == "O" and sub.ratings.get(d) == Rating.VERIFIED and (sub.id, d) not in sourced:
                errors.append(f"{sub.id}/O: limitations verified without a registry-sourced basis")
    ids = {s.id for s in model.subfamilies}
    for errs in empirical_mod.verify_all(empirical_records, ids).values():
        errors.extend(errs)
    return errors


def _independent_reference(model: Ledger, sub: Subfamily) -> bool:
    return any(
        e.subfamily == sub.id
        and e.dimension == "J"
        and e.kind in INDEPENDENT
        and e.independence_from == sub.engine_id
        and e.implementation.strip()
        and e.implementation != sub.engine_id
        for e in model.evidence
    )


def _reviewed(model: Ledger, sub: Subfamily) -> bool:
    accepted = {
        r.identity
        for r in model.reviewers
        if r.family == sub.family and r.independent and r.decision == "accepted" and r.evidence_ref
    }
    return len(accepted) >= 2


def subfamily_status(
    model: Ledger,
    sub: Subfamily,
    index: dict[tuple[str, str], tuple[str, ...]],
    empirical_records: tuple[empirical_mod.EmpiricalRecord, ...] = (),
    empirical_valid: set[str] | None = None,
) -> SubfamilyStatus:
    ratings = sub.ratings
    cells: dict[str, Cell] = {}
    for d in M3_DIMENSIONS:
        rating = ratings[d]
        have = index.get((sub.id, d), ())
        if rating == Rating.VERIFIED and have:
            cells[d] = Cell(d, rating.value, True, have, None)
        elif rating == Rating.NOT_APPLICABLE and any(
            e.subfamily == sub.id and e.dimension == d and e.kind == "applicability_review"
            for e in model.evidence
        ):
            cells[d] = Cell(d, rating.value, True, (), None)
        elif rating == Rating.VERIFIED:
            cells[d] = Cell(d, rating.value, False, (), "verified without a valid receipt")
        else:
            cells[d] = Cell(d, rating.value, False, have, f"rated {rating.value}")
    verified = tuple(d for d in M3_DIMENSIONS if cells[d].verified)
    unverified = tuple(d for d in M3_DIMENSIONS if not cells[d].verified)
    independent = _independent_reference(model, sub)
    blockers = tuple(
        f"{d}: {COMPUTATIONAL[d]} (owner {sub.owners[d]}; {cells[d].reason})" for d in unverified
    ) + tuple(f"blocker: {b}" for b in sub.blockers)
    if not independent:
        blockers = (*blockers, "J: no independent reference implementation recorded")
    mine = [
        r
        for r in empirical_records
        if r.subfamily == sub.id and (empirical_valid is None or r.id in empirical_valid)
    ]
    covered = tuple(d for d in M3_DIMENSIONS if any(r.dimension == d for r in mine))
    empirical_missing = {d: EMPIRICAL[d] for d in M3_DIMENSIONS if d not in covered}
    comps = [c for c in model.comparisons if sub.id in (c.left, c.right)]
    if comps and all(c.verdict == "conditionally_interchangeable" for c in comps):
        interchange = "conditionally_interchangeable (empirical, reviewed)"
    elif any(c.verdict == "not_interchangeable" for c in comps) and not any(
        c.verdict == "unknown" for c in comps
    ):
        interchange = "not_interchangeable"
    else:
        interchange = "unknown"
    analytical_only = tuple(
        d
        for d in verified
        if ratings[d] == Rating.VERIFIED
        and not any(
            e.dimension == d and e.kind in NON_ANALYTICAL
            for e in model.evidence
            if e.subfamily == sub.id
        )
        and d not in covered
    )
    if all(ratings[d] == Rating.ABSENT for d in M3_DIMENSIONS):
        state = "NOT_STARTED"
    elif not unverified and not blockers:
        all_dims = all(ratings[d] in DONE_RATINGS for d in DIMENSIONS)
        state = (
            "SCIENTIFICALLY_QUALIFIED"
            if all_dims and len(covered) == len(M3_DIMENSIONS) and _reviewed(model, sub)
            else "ENGINEERING_VERIFIED"
        )
    elif all(ratings[d] != Rating.ABSENT for d in M3_DIMENSIONS):
        state = "SPECIFIED"
    else:
        state = "IN_PROGRESS"
    return SubfamilyStatus(
        sub.id,
        sub.family,
        state,
        cells,
        verified,
        unverified,
        blockers,
        empirical_missing,
        tuple(r.id for r in mine),
        covered,
        independent,
        analytical_only,
        interchange,
    )


def evaluate_m3(
    model: Ledger,
    receipts: tuple[receipts_mod.Receipt, ...] | None = None,
    empirical_records: tuple[empirical_mod.EmpiricalRecord, ...] | None = None,
    root: Path | None = None,
) -> M3Report:
    from dynamisbench_soccer.qualification.ledger import validate

    root = root if root is not None else Path(".")
    qdir = root / "qualification"
    if receipts is None:
        receipts = receipts_mod.load(qdir / "receipts.toml")
    if empirical_records is None:
        empirical_records = empirical_mod.load(qdir / "empirical.toml")
    index, receipt_errors, _global = _valid_receipts(model, receipts, root)
    ids = {s.id for s in model.subfamilies}
    empirical_errors = empirical_mod.verify_all(empirical_records, ids)
    valid_empirical = {rid for rid, errs in empirical_errors.items() if not errs}
    structural = tuple(validate(model, root, receipts=receipts, empirical=empirical_records))
    statuses = tuple(
        subfamily_status(model, s, index, empirical_records, valid_empirical)
        for s in sorted(model.subfamilies, key=lambda s: s.id)
    )
    counts: dict[str, int] = {}
    for st in statuses:
        counts[st.state] = counts.get(st.state, 0) + 1
    verified = counts.get("ENGINEERING_VERIFIED", 0) + counts.get("SCIENTIFICALLY_QUALIFIED", 0)
    return M3Report(
        passed=verified == len(statuses) and not structural,
        total_subfamilies=len(statuses),
        engineering_verified=verified,
        by_state=counts,
        structural_problems=structural,
        receipt_problems={k: tuple(v) for k, v in receipt_errors.items() if v},
        empirical_problems={k: tuple(v) for k, v in empirical_errors.items() if v},
        computational_blockers={
            st.subfamily: st.computational_blockers
            for st in statuses
            if st.state not in {"ENGINEERING_VERIFIED", "SCIENTIFICALLY_QUALIFIED"}
        },
        empirical_requirements={st.subfamily: st.empirical_requirements for st in statuses},
        subfamilies=statuses,
    )
