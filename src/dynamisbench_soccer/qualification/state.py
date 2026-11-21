"""Computed per-subfamily states and the M3 gate (RES-675), under Policy A.

Nothing here is declared. A state is a pure function of the ledger ratings, the validated
independent receipts and the blockers; the decision itself is made once, in ``gate.py``.
This module only reshapes that report for the ``states`` command and the verification
report. ``ENGINEERING_VERIFIED`` means computational-reference readiness. Physical-device
validity is reported separately (``empirical_*`` fields), never folded into the state.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from dynamisbench_soccer.qualification import empirical as empirical_mod
from dynamisbench_soccer.qualification import receipts as receipts_mod
from dynamisbench_soccer.qualification.gate import (
    COMPUTATIONAL,
    EMPIRICAL,
    INDEPENDENT,
    NON_ANALYTICAL,
    SubfamilyStatus,
    _valid_receipts,
    evaluate_m3,
    subfamily_status,
)
from dynamisbench_soccer.qualification.ledger import Ledger, Subfamily

State = Literal[
    "NOT_STARTED",
    "IN_PROGRESS",
    "SPECIFIED",
    "ENGINEERING_VERIFIED",
    "SCIENTIFICALLY_QUALIFIED",
]
REQUIRED_EXPERIMENT = EMPIRICAL  # backwards-compatible name for the empirical requirements
__all__ = [
    "COMPUTATIONAL",
    "EMPIRICAL",
    "INDEPENDENT",
    "NON_ANALYTICAL",
    "REQUIRED_EXPERIMENT",
    "M3Gate",
    "SubfamilyState",
    "computed_states",
    "m3_gate",
    "subfamily_state",
]


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
    required_computational: dict[str, str]
    empirical_requirements: dict[str, str]
    independent_reference: bool
    non_analytical_reference: bool
    analytical_only_dimensions: tuple[str, ...]
    receipts: dict[str, tuple[str, ...]]
    empirical_records: tuple[str, ...]
    empirical_dimensions_covered: tuple[str, ...]
    empirical_interchangeability: str
    empirical_gaps: tuple[str, ...]


@dataclass(frozen=True)
class M3Gate:
    passed: bool
    total_subfamilies: int
    engineering_verified: int
    by_state: dict[str, int]
    structural_problems: tuple[str, ...]
    blocking: dict[str, tuple[str, ...]]
    empirical_requirements: dict[str, dict[str, str]]
    receipt_problems: dict[str, tuple[str, ...]]
    empirical_problems: dict[str, tuple[str, ...]]


def _shape(model: Ledger, sub: Subfamily, st: SubfamilyStatus) -> SubfamilyState:
    return SubfamilyState(
        sub.id,
        sub.family,
        st.state,  # type: ignore[arg-type]
        st.verified_dimensions,
        st.unverified_dimensions,
        {d: r.value for d, r in sub.ratings.items()},
        st.computational_blockers,
        dict(sub.owners),
        {d: COMPUTATIONAL[d] for d in st.unverified_dimensions},
        st.empirical_requirements,
        st.independent_reference,
        any(
            e.subfamily == sub.id and e.dimension == "J" and e.kind in NON_ANALYTICAL
            for e in model.evidence
        ),
        st.analytical_only_dimensions,
        {d: c.receipts for d, c in st.cells.items() if c.receipts},
        st.empirical_records,
        st.empirical_dimensions_covered,
        st.empirical_interchangeability,
        sub.empirical_gaps,
    )


def computed_states(
    model: Ledger,
    root: Path | None = None,
    *,
    receipts: tuple[receipts_mod.Receipt, ...] | None = None,
    empirical: tuple[empirical_mod.EmpiricalRecord, ...] | None = None,
) -> list[SubfamilyState]:
    root = root if root is not None else Path(".")
    if receipts is None:
        receipts = receipts_mod.load(root / "qualification" / "receipts.toml")
    if empirical is None:
        empirical = empirical_mod.load(root / "qualification" / "empirical.toml")
    index, _problems, _global = _valid_receipts(model, receipts, root)
    ids = {s.id for s in model.subfamilies}
    valid = {rid for rid, errs in empirical_mod.verify_all(empirical, ids).items() if not errs}
    return [
        _shape(model, s, subfamily_status(model, s, index, empirical, valid))
        for s in sorted(model.subfamilies, key=lambda s: s.id)
    ]


def subfamily_state(
    model: Ledger,
    sub: Subfamily,
    root: Path | None = None,
    *,
    receipts: tuple[receipts_mod.Receipt, ...] | None = None,
) -> SubfamilyState:
    """State of ``sub`` as if it replaced the model's entry of the same id (what-if checks)."""
    others = tuple(s for s in model.subfamilies if s.id != sub.id)
    candidate = dataclasses.replace(model, subfamilies=(*others, sub))
    return next(
        s for s in computed_states(candidate, root, receipts=receipts) if s.subfamily == sub.id
    )


def m3_gate(
    model: Ledger,
    root: object | None = None,
    *,
    receipts: tuple[receipts_mod.Receipt, ...] | None = None,
    empirical: tuple[empirical_mod.EmpiricalRecord, ...] | None = None,
) -> M3Gate:
    report = evaluate_m3(model, receipts, empirical, Path(str(root)) if root is not None else None)
    return M3Gate(
        passed=report.passed,
        total_subfamilies=report.total_subfamilies,
        engineering_verified=report.engineering_verified,
        by_state=report.by_state,
        structural_problems=report.structural_problems,
        blocking=report.computational_blockers,
        empirical_requirements=report.empirical_requirements,
        receipt_problems=report.receipt_problems,
        empirical_problems=report.empirical_problems,
    )
