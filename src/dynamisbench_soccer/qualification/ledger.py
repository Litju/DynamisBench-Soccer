"""Stage-aware quality-parity contracts for sports-science AI benchmarks."""

from __future__ import annotations

import re
import tomllib
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

DIMENSIONS = tuple("ABCDEFGHIJKLMNO")
M3_DIMENSIONS = tuple(d for d in DIMENSIONS if d != "N")
FAMILIES = frozenset(
    {
        "force_platform",
        "lpt_encoder",
        "load_cell_isometric",
        "nordic_hip",
        "timing_contact",
        "radar_laser",
        "gnss_lps",
        "imu",
    }
)
SUBFAMILIES = frozenset(
    {
        "force_platform.single",
        "force_platform.dual",
        "lpt_encoder.lpt",
        "lpt_encoder.encoder",
        "load_cell_isometric.load_cell",
        "load_cell_isometric.isometric_rig",
        "nordic_hip.nordic",
        "nordic_hip.hip",
        "timing_contact.gates",
        "timing_contact.contact",
        "radar_laser.radar",
        "radar_laser.laser",
        "gnss_lps.gnss",
        "gnss_lps.lps",
        "imu.inertial",
    }
)
METHODS = {
    "continuous_signal": {"filter", "integrate", "differentiate", "calibrate", "start_detection"},
    "discrete_event": {
        "count_unwrap",
        "event_rate",
        "debounce",
        "pair_events",
        "multi_beam_voting",
        "trigger_latency_correction",
    },
    "position_track": {"resample", "smooth"},
    "inertial": {"orientation_estimation", "gravity_removal"},
}
EVIDENCE_KINDS = frozenset(
    {
        "schema",
        "doc",
        "code",
        "test",
        "registry_claim",
        "registry_gap",
        "issue",
        "adapter_import",
        "vendor_reproduction",
        "applicability_review",
        "independent_test",
        "analytical_reference",
        "empirical_study",
        "external_review",
    }
)
ISSUE_PATTERN = re.compile(r"RES-\d+\Z")
PARADIGMS = frozenset({"continuous_signal", "discrete_event", "position_track", "inertial"})
CONTINUOUS_FILTERS = frozenset({"butterworth", "fir", "iir", "savgol", "lowpass", "highpass"})
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
    implementation: str = ""


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
    owners: dict[str, str]
    # Policy A: what only measured, criterion-referenced evidence can close. Reported next to,
    # never merged into, the computational blockers.
    empirical_gaps: tuple[str, ...] = ()


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
        subfamilies.append(
            Subfamily(
                id=row["id"],
                family=row["family"],
                paradigm=row["paradigm"],
                issue=row["issue"],
                engine_id=row["engine_id"],
                methods=tuple(row.get("methods", [])),
                ratings={str(k): Rating(v) for k, v in row["ratings"].items()},
                blockers=tuple(row.get("blockers", [])),
                owners=row["owners"],
                empirical_gaps=tuple(row.get("empirical_gaps", [])),
            )
        )
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
        tuple(subfamilies),
        evidence,
        comparisons,
        reviewers,
    )


def validate(
    model: Ledger,
    root: Path | None = None,
    *,
    receipts: tuple[Any, ...] | None = None,
    empirical: tuple[Any, ...] | None = None,
) -> list[str]:
    errors: list[str] = []
    ids = [s.id for s in model.subfamilies]
    for key, count in Counter(ids).items():
        if count > 1:
            errors.append(f"duplicate subfamily {key}")
    families = {s.family for s in model.subfamilies}
    if families != FAMILIES:
        errors.append("eight required device families must be represented")
    if set(ids) != SUBFAMILIES:
        errors.append("all 15 required instrument subfamilies must be represented")
    lookup: dict[tuple[str, str], list[Evidence]] = {}
    registry = None
    if root is not None:
        from dynamisbench_soccer.registry.models import load as load_registry

        registry = load_registry(root / "registry")
    for e in model.evidence:
        if e.subfamily not in ids or e.dimension not in DIMENSIONS:
            errors.append(f"unknown evidence target {e.subfamily}/{e.dimension}")
        if e.kind not in EVIDENCE_KINDS:
            errors.append(f"unknown evidence kind {e.kind}")
        if not ISSUE_PATTERN.fullmatch(e.issue):
            errors.append(f"invalid evidence issue {e.issue}")
        if not e.ref.strip() or not e.rationale.strip() or not e.issue:
            errors.append(f"evidence metadata incomplete {e.subfamily}/{e.dimension}")
        if root is not None and e.ref.startswith("repo:"):
            path = (root / e.ref[5:]).resolve()
            if not path.is_relative_to(root.resolve()) or not path.is_file():
                errors.append(f"unresolved repo evidence {e.ref}")
        if (
            e.kind == "registry_claim"
            and registry is not None
            and e.ref not in {c.id for c in registry.claims}
        ):
            errors.append(f"unknown registry claim {e.ref}")
        if (
            e.kind == "registry_gap"
            and registry is not None
            and e.ref not in {g.subject for g in registry.gaps}
        ):
            errors.append(f"unknown registry gap {e.ref}")
        if e.kind == "issue" and not ISSUE_PATTERN.fullmatch(e.ref):
            errors.append(f"invalid issue reference {e.ref}")
        if e.kind in {
            "schema",
            "doc",
            "code",
            "test",
            "adapter_import",
            "independent_test",
            "analytical_reference",
            "applicability_review",
            "external_review",
        } and not e.ref.startswith("repo:"):
            errors.append(f"repository evidence requires repo: reference {e.ref}")
        lookup.setdefault((e.subfamily, e.dimension), []).append(e)
    for s in model.subfamilies:
        if not s.id.startswith(s.family + "."):
            errors.append(f"{s.id} has invalid family prefix")
        if not ISSUE_PATTERN.fullmatch(s.issue) or not s.engine_id.strip():
            errors.append(f"{s.id} missing engine or invalid owning issue")
        if set(s.methods) - METHODS.get(s.paradigm, set()):
            errors.append(f"{s.id} has processing methods invalid for its paradigm")
        if s.paradigm not in PARADIGMS:
            errors.append(f"{s.id} has unrecognized acquisition paradigm")
        if set(s.owners) != set(DIMENSIONS) or any(
            not ISSUE_PATTERN.fullmatch(issue) for issue in s.owners.values()
        ):
            errors.append(f"{s.id} needs valid owning issues for A-O")
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
            if value not in set(Rating):
                errors.append(f"{s.id}/{d}: unknown rating {value}")
            if value == Rating.VERIFIED:
                accepted = {
                    "independent_test",
                    "analytical_reference",
                    "empirical_study",
                    "external_review",
                }
                if not any(e.kind in accepted for e in refs):
                    errors.append(f"{s.id}/{d}: no independent verification")
                if d == "J" and not any(
                    e.kind in accepted
                    and e.independence_from == s.engine_id
                    and e.implementation.strip()
                    and e.implementation != s.engine_id
                    for e in refs
                ):
                    errors.append(f"{s.id}/J: circular reference engine")

    def study_exists(ref: str) -> bool:
        if root is None:
            return bool(ref.strip())
        if ref.startswith("repo:"):
            path = (root / ref[5:]).resolve()
            return path.is_relative_to(root.resolve()) and path.is_file()
        return registry is not None and ref in {c.id for c in registry.claims}

    for e in model.evidence:
        if e.kind == "empirical_study" and not study_exists(e.ref):
            errors.append(f"unresolved empirical study {e.ref}")
    for c in model.comparisons:
        if any(not study_exists(ref) for ref in c.refs):
            errors.append(f"unresolved comparison evidence {c.left}/{c.right}")
        if c.left not in ids or c.right not in ids or c.left == c.right:
            errors.append(f"invalid comparison {c.left}/{c.right}")
        if c.verdict not in {"unknown", "not_interchangeable", "conditionally_interchangeable"}:
            errors.append(f"invalid comparison verdict {c.left}/{c.right}")
        if c.evidence_level not in {"none", "analytical", "empirical"}:
            errors.append(f"invalid evidence tier {c.left}/{c.right}")
        if c.verdict == "conditionally_interchangeable" and (
            c.evidence_level != "empirical" or c.reviewer_status != "accepted" or not c.refs
        ):
            errors.append(f"{c.left}/{c.right}: analytical-only interchangeability")
    for s in model.subfamilies:
        # Policy A: L verified means the analytical cross-device study was executed and
        # classified. The empirical interchangeability verdict may legitimately stay unknown.
        if s.ratings.get("L") == Rating.VERIFIED and any(
            s.id in (c.left, c.right) and c.evidence_level == "none" for c in model.comparisons
        ):
            errors.append(f"{s.id}/L: a declared non-equivalence has no executed study")
    for r in model.reviewers:
        if not r.identity.strip() or not study_exists(r.evidence_ref):
            errors.append(f"unresolved reviewer evidence {r.family}")
        if r.family not in families:
            errors.append(f"unknown reviewer family {r.family}")
    if root is not None or receipts is not None:
        from dynamisbench_soccer.qualification import empirical as empirical_mod
        from dynamisbench_soccer.qualification import receipts as receipts_mod
        from dynamisbench_soccer.qualification.gate import receipt_structure_problems

        base = root if root is not None else Path(".")
        loaded_receipts = (
            receipts
            if receipts is not None
            else receipts_mod.load(base / "qualification" / "receipts.toml")
        )
        loaded_empirical = (
            empirical
            if empirical is not None
            else empirical_mod.load(base / "qualification" / "empirical.toml")
        )
        errors.extend(receipt_structure_problems(model, loaded_receipts, loaded_empirical, base))
    return errors


def gate(
    model: Ledger,
    stage: Stage,
    root: Path = Path("."),
    *,
    receipts: tuple[Any, ...] | None = None,
    empirical: tuple[Any, ...] | None = None,
) -> GateResult:
    if stage == Stage.M3:
        # One canonical evaluator (gate.py) decides M3, for both gate commands.
        from dynamisbench_soccer.qualification.gate import evaluate_m3

        report = evaluate_m3(model, receipts, empirical, root)
        families = {s.family for s in model.subfamilies}
        blocked: dict[str, list[str]] = {}
        for st in report.subfamilies:
            if st.state not in {"ENGINEERING_VERIFIED", "SCIENTIFICALLY_QUALIFIED"}:
                blocked.setdefault(st.family, []).extend(
                    f"{st.subfamily}: {b}" for b in st.computational_blockers
                )
        if report.structural_problems:
            blocked.setdefault("__ledger__", []).extend(report.structural_problems)
        ready = len(families - set(blocked))
        return GateResult(
            Stage.M3,
            report.passed,
            ready,
            len(families),
            {name: tuple(v) for name, v in sorted(blocked.items())},
        )
    # M5 and M8 build on the same canonical evaluator: M5 needs the M3 computational state plus
    # a verified N; M8 needs the full scientific state (measured empirical records, N, review).
    from dynamisbench_soccer.qualification.gate import evaluate_m3

    report = evaluate_m3(model, receipts, empirical, root)
    sub_by_id = {x.id: x for x in model.subfamilies}
    missing: dict[str, list[str]] = {}
    for st in report.subfamilies:
        sub = sub_by_id[st.subfamily]
        if stage == Stage.M5:
            ok = st.state in {
                "ENGINEERING_VERIFIED",
                "SCIENTIFICALLY_QUALIFIED",
            } and sub.ratings.get("N") in {Rating.VERIFIED, Rating.NOT_APPLICABLE}
            reasons = [f"{st.subfamily}: M3 state {st.state} / N {sub.ratings.get('N')}"]
        else:
            ok = st.state == "SCIENTIFICALLY_QUALIFIED"
            reasons = [f"{st.subfamily}: state {st.state}, needs measured records and review"]
        if not ok:
            missing.setdefault(st.family, []).extend(reasons)
    families = {s.family for s in model.subfamilies}
    ready = len(families - set(missing))
    if report.structural_problems:
        missing.setdefault("__ledger__", []).extend(report.structural_problems)
    return GateResult(
        stage,
        ready == len(families) and not report.structural_problems,
        ready,
        len(families),
        {name: tuple(v) for name, v in sorted(missing.items())},
    )


def markdown(model: Ledger) -> str:
    lines = [
        "# Device-family qualification ledger",
        "",
        "<!-- GENERATED by dbs qualification ledger --write -->",
        "",
        "M3 verifies A-M and O computationally (Policy A); M5 additionally verifies N; M8 adds",
        "scientific review. Physical-device validity is a separate, empirical status.",
        "Specified is NOT verified. Analytical agreement is NOT empirical equivalence.",
        "",
        "| Family | Subfamilies | M3 | M5 | M8 |",
        "| -- | -- | -- | -- | -- |",
    ]
    for family in sorted({s.family for s in model.subfamilies}):
        subs = [s for s in model.subfamilies if s.family == family]
        statuses = {
            stage: "VERIFIED"
            if family not in gate(model, stage).missing
            and "__ledger__" not in gate(model, stage).missing
            else "IN_PROGRESS"
            for stage in Stage
        }
        names = ", ".join(s.id.split(".")[-1] for s in subs)
        lines.append(
            f"| {family} | {names} | {statuses[Stage.M3]} | "
            f"{statuses[Stage.M5]} | {statuses[Stage.M8]} |"
        )
    lines += ["", "## Ratings and blockers", ""]
    for s in model.subfamilies:
        ratings = ", ".join(f"{d}:{s.ratings[d].value} ({s.owners[d]})" for d in DIMENSIONS)
        lines.append(f"- {s.id} ({s.issue}): {ratings}")
        lines += [f"  - computational blocker: {b}" for b in s.blockers]
        lines += [f"  - empirical gap: {g}" for g in s.empirical_gaps]
    lines += ["", "## Interchangeability studies", ""]
    for c in model.comparisons:
        lines.append(
            f"- {c.left} vs {c.right} ({c.quantity}): {c.verdict}; {c.evidence_level} evidence"
        )
    return "\n".join(lines) + "\n"
