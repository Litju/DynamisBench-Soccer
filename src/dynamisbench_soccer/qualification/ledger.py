"""Stage-aware quality-parity contracts for sports-science AI benchmarks."""

from __future__ import annotations

import tomllib
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

DIMENSIONS = tuple("ABCDEFGHIJKLMNO")
M3_DIMENSIONS = tuple(d for d in DIMENSIONS if d != "N")
FAMILIES = frozenset({
    "force_platform", "lpt_encoder", "load_cell_isometric", "nordic_hip",
    "timing_contact", "radar_laser", "gnss_lps", "imu",
})
PARADIGMS = frozenset({
    "continuous_signal", "discrete_event", "position_track", "inertial"
})
CONTINUOUS_FILTERS = frozenset({
    "butterworth", "fir", "iir", "savgol", "lowpass", "highpass"
})
WEAK_EVIDENCE = frozenset({"adapter_import", "vendor_reproduction"})


class Rating(StrEnum):
    ABSENT = "absent"
    PARTIAL = "partial"
    SPECIFIED = "specified"
    VERIFIED = "verified"
    NOT_APPLICABLE = "not_applicable"


class Stage(StrEnum):
    M3 = "M3"
    M5 = "M5"
    M8 = "M8"


@dataclass(frozen=True)
class Evidence:
    subfamily: str
    dimension: str
    kind: str
    ref: str
    issue: str
    rationale: str
    independence_from: str = ""


@dataclass(frozen=True)
class Subfamily:
    id: str
    family: str
    paradigm: str
    issue: str
    engine_id: str
    methods: tuple[str, ...]
    ratings: dict[str, Rating]
    blockers: tuple[str, ...]


@dataclass(frozen=True)
class Comparison:
    left: str
    right: str
    quantity: str
    verdict: str
    evidence_level: str
    refs: tuple[str, ...]
    reviewer_status: str


@dataclass(frozen=True)
class Reviewer:
    family: str
    identity: str
    independent: bool
    decision: str
    evidence_ref: str


@dataclass(frozen=True)
class Ledger:
    version: str
    subfamilies: tuple[Subfamily, ...]
    evidence: tuple[Evidence, ...]
    comparisons: tuple[Comparison, ...]
    reviewers: tuple[Reviewer, ...]


@dataclass(frozen=True)
class GateResult:
    stage: Stage
    passed: bool
    passed_families: int
    total_families: int
    missing: dict[str, tuple[str, ...]]


def load(path: Path) -> Ledger:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    subfamilies = []
    for row in raw.get("subfamily", []):
        subfamilies.append(Subfamily(
            id=row["id"],
            family=row["family"],
            paradigm=row["paradigm"],
            issue=row["issue"],
            engine_id=row["engine_id"],
            methods=tuple(row.get("methods", [])),
            ratings={str(k): Rating(v) for k, v in row["ratings"].items()},
            blockers=tuple(row.get("blockers", [])),
        ))
    evidence = tuple(Evidence(**e) for e in raw.get("evidence", []))
    comparisons = tuple(
        Comparison(
            left=c["left"],
            right=c["right"],
            quantity=c["quantity"],
            verdict=c["verdict"],
            evidence_level=c["evidence_level"],
            refs=tuple(c.get("refs", [])),
            reviewer_status=c.get("reviewer_status", "pending"),
        )
        for c in raw.get("comparison", [])
    )
    reviewers = tuple(Reviewer(**r) for r in raw.get("reviewer", []))
    return Ledger(
        str(raw.get("version", "")),
        tuple(subfamilies), evidence, comparisons, reviewers,
    )


def validate(model: Ledger, root: Path | None = None) -> list[str]:
    errors: list[str] = []
    ids = [s.id for s in model.subfamilies]
    for key, count in Counter(ids).items():
        if count > 1:
            errors.append(f"duplicate subfamily {key}")
    families = {s.family for s in model.subfamilies}
    if families != FAMILIES:
        errors.append("eight required device families must be represented")
    lookup: dict[tuple[str, str], list[Evidence]] = {}
    for e in model.evidence:
        if e.subfamily not in ids or e.dimension not in DIMENSIONS:
            errors.append(f"unknown evidence target {e.subfamily}/{e.dimension}")
        if not e.ref or not e.rationale or not e.issue:
            errors.append(f"evidence metadata incomplete {e.subfamily}/{e.dimension}")
        if root is not None and e.ref.startswith("repo:"):
            path = (root / e.ref[5:]).resolve()
            if not path.is_relative_to(root.resolve()) or not path.is_file():
                errors.append(f"unresolved repo evidence {e.ref}")
        lookup.setdefault((e.subfamily, e.dimension), []).append(e)
    for s in model.subfamilies:
        if not s.id.startswith(s.family + "."):
            errors.append(f"{s.id} has invalid family prefix")
        if s.paradigm not in PARADIGMS:
            errors.append(f"{s.id} has unrecognized acquisition paradigm")
        if set(s.ratings) != set(DIMENSIONS):
            errors.append(f"{s.id} missing or extraneous A-O dimension ratings")
        if s.paradigm == "discrete_event" and CONTINUOUS_FILTERS.intersection(s.methods):
            errors.append(f"{s.id} cannot filter discrete events as continuous signals")
        for d, value in s.ratings.items():
            refs = lookup.get((s.id, d), [])
            if value != Rating.ABSENT and not refs:
                errors.append(f"{s.id}/{d}: rated without evidence")
            if (
                d != "A"
                and value != Rating.ABSENT
                and refs
                and all(e.kind in WEAK_EVIDENCE for e in refs)
            ):
                errors.append(f"{s.id}/{d}: adapter/vendor-only evidence")
            if value == Rating.NOT_APPLICABLE and not any(
                e.kind == "applicability_review" for e in refs
            ):
                errors.append(f"{s.id}/{d}: N/A without applicability review")
            if value == Rating.VERIFIED:
                accepted = {
                    "independent_test", "analytical_reference",
                    "empirical_study", "external_review",
                }
                if not any(e.kind in accepted for e in refs):
                    errors.append(f"{s.id}/{d}: no independent verification")
                if d == "J" and not any(
                    e.independence_from and e.independence_from != s.engine_id
                    for e in refs
                ):
                    errors.append(f"{s.id}/J: circular reference engine")
    for c in model.comparisons:
        if c.left not in ids or c.right not in ids or c.left == c.right:
            errors.append(f"invalid comparison {c.left}/{c.right}")
        if c.verdict not in {
            "unknown", "not_interchangeable", "conditionally_interchangeable"
        }:
            errors.append(f"invalid comparison verdict {c.left}/{c.right}")
        if c.evidence_level not in {"none", "analytical", "empirical"}:
            errors.append(f"invalid evidence tier {c.left}/{c.right}")
        if c.verdict == "conditionally_interchangeable" and (
            c.evidence_level != "empirical"
            or c.reviewer_status != "accepted"
            or not c.refs
        ):
            errors.append(f"{c.left}/{c.right}: analytical-only interchangeability")
    for r in model.reviewers:
        if r.family not in families:
            errors.append(f"unknown reviewer family {r.family}")
    return errors


def gate(model: Ledger, stage: Stage) -> GateResult:
    required = M3_DIMENSIONS if stage == Stage.M3 else DIMENSIONS
    missing: dict[str, list[str]] = {}
    for s in model.subfamilies:
        for d in required:
            if s.ratings.get(d) not in {Rating.VERIFIED, Rating.NOT_APPLICABLE}:
                missing.setdefault(s.family, []).append(f"{s.id}/{d}")
        if stage == Stage.M8:
            accepted = {
                r.identity
                for r in model.reviewers
                if r.family == s.family
                and r.independent
                and r.decision == "accepted"
                and r.evidence_ref
            }
            if len(accepted) < 2:
                missing.setdefault(s.family, []).append("two independent reviewers")
    families = {s.family for s in model.subfamilies}
    ready = len(families - set(missing))
    structural = validate(model)
    if structural:
        missing.setdefault("__ledger__", []).extend(structural)
    return GateResult(
        stage, ready == len(families) and not structural, ready, len(families),
        {name: tuple(v) for name, v in sorted(missing.items())},
    )


def markdown(model: Ledger) -> str:
    lines = [
        "# Device-family qualification ledger",
        "",
        "<!-- GENERATED by dbs qualification ledger --write -->",
        "",
        "M3 verifies A-M and O; M5 additionally verifies N; M8 adds scientific review.",
        "Specified is NOT verified. Analytical agreement is NOT empirical equivalence.",
        "",
        "| Family | Subfamilies | M3 | M5 | M8 |",
        "| -- | -- | -- | -- | -- |",
    ]
    for family in sorted({s.family for s in model.subfamilies}):
        subs = [s for s in model.subfamilies if s.family == family]
        statuses: dict[Stage, str] = {}
        for stage in Stage:
            required = M3_DIMENSIONS if stage == Stage.M3 else DIMENSIONS
            complete = all(
                s.ratings[d] in {Rating.VERIFIED, Rating.NOT_APPLICABLE}
                for s in subs
                for d in required
            )
            if stage == Stage.M8:
                reviewer_names = {
                    r.identity
                    for r in model.reviewers
                    if r.family == family
                    and r.independent
                    and r.decision == "accepted"
                    and r.evidence_ref
                }
                complete = complete and len(reviewer_names) >= 2
            statuses[stage] = "VERIFIED" if complete else "IN_PROGRESS"
        names = ", ".join(s.id.split(".")[-1] for s in subs)
        lines.append(
            f"| {family} | {names} | {statuses[Stage.M3]} | "
            f"{statuses[Stage.M5]} | {statuses[Stage.M8]} |"
        )
    lines += ["", "## Ratings and blockers", ""]
    for s in model.subfamilies:
        ratings = ", ".join(f"{d}:{s.ratings[d].value}" for d in DIMENSIONS)
        lines.append(f"- {s.id} ({s.issue}): {ratings}")
        lines += [f"  - {b}" for b in s.blockers]
    lines += ["", "## Interchangeability studies", ""]
    for c in model.comparisons:
        lines.append(
            f"- {c.left} vs {c.right} ({c.quantity}): "
            f"{c.verdict}; {c.evidence_level} evidence"
        )
    return "\n".join(lines) + "\n"
