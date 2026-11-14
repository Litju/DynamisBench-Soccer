"""Device-family qualification ledger (RES-692).

``qualification/ledger.toml`` records, for every in-scope measurement-device family,
the state of each of the fifteen qualification dimensions A-O, the evidence behind it,
the issue that owns the remaining work, and any blockers.

The family qualification state is **computed** from its dimensions by
:func:`compute_state`. It is never declared by hand. If ``declared_state`` is present, it
must equal the computed state, or validation fails.

Invariants enforced by :func:`validate`:

* Every family lists all fifteen dimensions.
* Evidence of kind ``adapter_import`` or ``vendor_reproduction`` never counts. A
  dimension above ``absent`` needs at least one countable evidence item.
* A reference (dimension J) whose implementation is the engine it verifies is rejected,
  and vendor output never counts as a reference.
* A family's processing methods must be valid for its paradigm. For example, discrete
  timing events may not use continuous-signal filters.
* Non-equivalences need a study issue. An ``interchangeable`` or ``convertible`` verdict
  needs evidence and a stated basis.
* Evidence references resolve to files in the repo, registry claims, registry gaps or
  issue ids.
"""

from __future__ import annotations

import re
import tomllib
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from dynamisbench_soccer.registry.models import V01_TESTS, Registry
from dynamisbench_soccer.signals.schema import DeviceClass

ISSUE_PATTERN = re.compile(r"^RES-\d+$")
ID_PATTERN = r"^[a-z][a-z0-9_]*$"

#: The fifteen qualification dimensions. Quality parity means every family meets all of
#: them; it does not mean every family uses the same algorithms.
DIMENSIONS: dict[str, str] = {
    "A": "Acquisition and raw-data contracts",
    "B": "Calibration, units, axes and geometry",
    "C": "Sampling, timestamps and synchronization",
    "D": "Signal or event integrity validation",
    "E": "Filtering or event processing",
    "F": "Device-specific numerical reconstruction",
    "G": "Assessment-specific event detection",
    "H": "Scientific metric definitions and variants",
    "I": "Measurement uncertainty and sensitivity",
    "J": "Independent mathematical or empirical references",
    "K": "Positive, negative and adversarial fixtures",
    "L": "Cross-device agreement and interchangeability",
    "M": "Provenance and reproducibility",
    "N": "AI task families with qualified grading",
    "O": "Explicit scientific limitations",
}


class DimState(StrEnum):
    ABSENT = "absent"
    PARTIAL = "partial"
    SPECIFIED = "specified"
    VERIFIED = "verified"

    @property
    def rank(self) -> int:
        return list(DimState).index(self)


class QualState(StrEnum):
    NOT_STARTED = "NOT_STARTED"
    IN_PROGRESS = "IN_PROGRESS"
    SPECIFIED = "SPECIFIED"
    ENGINEERING_VERIFIED = "ENGINEERING_VERIFIED"
    SCIENTIFICALLY_QUALIFIED = "SCIENTIFICALLY_QUALIFIED"

    @property
    def rank(self) -> int:
        return list(QualState).index(self)


#: The state every in-scope family must compute for M3 to be complete (RES-675).
M3_GATE = QualState.ENGINEERING_VERIFIED


class Paradigm(StrEnum):
    CONTINUOUS_SIGNAL = "continuous_signal"
    DISCRETE_EVENT = "discrete_event"
    POSITION_TRACK = "position_track"
    INERTIAL = "inertial"


class Scope(StrEnum):
    IN_SCOPE = "in_scope"
    PROPOSED = "proposed"  # admitted through the RES-702 contract, not yet in the gate


class EvidenceKind(StrEnum):
    ISSUE = "issue"
    DOC = "doc"
    CODE = "code"
    TEST = "test"
    REGISTRY_CLAIM = "registry_claim"
    REGISTRY_GAP = "registry_gap"
    REVIEW = "review"
    #: Recorded for transparency, but never counts towards any dimension.
    ADAPTER_IMPORT = "adapter_import"
    VENDOR_REPRODUCTION = "vendor_reproduction"


NON_COUNTING_EVIDENCE = frozenset({EvidenceKind.ADAPTER_IMPORT, EvidenceKind.VENDOR_REPRODUCTION})
_PATH_EVIDENCE = frozenset(
    {EvidenceKind.DOC, EvidenceKind.CODE, EvidenceKind.TEST, EvidenceKind.ADAPTER_IMPORT}
)


class ReferenceKind(StrEnum):
    ANALYTICAL_CLOSED_FORM = "analytical_closed_form"
    INDEPENDENT_IMPLEMENTATION = "independent_implementation"
    EMPIRICAL_CRITERION = "empirical_criterion"
    PUBLISHED_WORKED_EXAMPLE = "published_worked_example"
    #: Never a reference (DB-07). Listed only so that a mistaken entry is caught.
    VENDOR_OUTPUT = "vendor_output"


#: Reference kinds that are independent of our own analytical generators. Scientific
#: qualification needs at least one of these.
NON_ANALYTICAL_REFERENCES = frozenset(
    {
        ReferenceKind.INDEPENDENT_IMPLEMENTATION,
        ReferenceKind.EMPIRICAL_CRITERION,
        ReferenceKind.PUBLISHED_WORKED_EXAMPLE,
    }
)


class BlockerKind(StrEnum):
    DEPENDENCY = "dependency"  # engineering work in another issue
    SCIENTIFIC = "scientific"  # missing evidence, data or a source; may not be resolvable


class Verdict(StrEnum):
    INTERCHANGEABLE = "interchangeable"
    CONVERTIBLE = "convertible"
    NOT_INTERCHANGEABLE = "not_interchangeable"


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


IssueId = Annotated[str, Field(pattern=ISSUE_PATTERN.pattern)]


class Evidence(_Strict):
    kind: EvidenceKind
    ref: str = Field(min_length=1)
    note: str = ""


class Blocker(_Strict):
    kind: BlockerKind
    issue: IssueId
    text: str = Field(min_length=10)


class Dimension(_Strict):
    state: DimState
    owner: IssueId
    evidence: list[Evidence] = Field(default_factory=list)
    blockers: list[Blocker] = Field(default_factory=list)
    note: str = ""

    @property
    def counting_evidence(self) -> list[Evidence]:
        return [e for e in self.evidence if e.kind not in NON_COUNTING_EVIDENCE]


class Reference(_Strict):
    id: str = Field(pattern=r"^REF-[A-Z0-9_-]+$")
    kind: ReferenceKind
    implementation: str = Field(min_length=1, description="what produces the reference value")
    verifies: str = Field(min_length=1, description="engine id the reference checks")
    ref: str = Field(min_length=1, description="issue, path, DOI or claim id")

    @property
    def counts(self) -> bool:
        return self.kind is not ReferenceKind.VENDOR_OUTPUT and self.implementation != self.verifies


class Family(_Strict):
    id: str = Field(pattern=ID_PATTERN)
    number: int = Field(ge=1)
    name: str
    scope: Scope
    paradigm: Paradigm
    engine: str = Field(pattern=r"^[a-z][a-z0-9_.]*$", description="device engine id")
    engine_issue: IssueId
    device_classes: list[str] = Field(default_factory=list)
    planned_device_classes: list[str] = Field(default_factory=list)
    processing_methods: list[str] = Field(min_length=1)
    dims: dict[str, Dimension]
    references: list[Reference] = Field(default_factory=list)
    independent_review: list[Evidence] = Field(default_factory=list)
    declared_state: QualState | None = None

    @model_validator(mode="after")
    def _all_dimensions(self) -> Family:
        if list(self.dims) != list(DIMENSIONS):
            missing = [d for d in DIMENSIONS if d not in self.dims]
            extra = [d for d in self.dims if d not in DIMENSIONS]
            raise ValueError(
                f"{self.id}: dims must be exactly A-O in order (missing {missing}, unknown {extra})"
            )
        if not self.device_classes and not self.planned_device_classes:
            raise ValueError(f"{self.id}: needs device_classes or planned_device_classes")
        return self


class ProcessingMethod(_Strict):
    id: str = Field(pattern=ID_PATTERN)
    paradigms: list[Paradigm] = Field(min_length=1)
    description: str


class NonEquivalence(_Strict):
    id: str = Field(pattern=r"^NE-\d{3}$")
    a: str = Field(description="family_id:estimand")
    b: str
    statement: str
    study: IssueId
    verdict: Verdict | None = None
    basis: Literal["analytical", "empirical"] | None = None
    evidence: list[Evidence] = Field(default_factory=list)

    @field_validator("a", "b")
    @classmethod
    def _side(cls, value: str) -> str:
        if not re.match(r"^[a-z][a-z0-9_]*:[a-z][a-z0-9_.]*$", value):
            raise ValueError(f"expected 'family:estimand', got {value!r}")
        return value

    def families(self) -> tuple[str, str]:
        return self.a.split(":", 1)[0], self.b.split(":", 1)[0]


class Assessment(_Strict):
    id: str = Field(pattern=ID_PATTERN)
    name: str
    issue: IssueId
    tests: list[str] = Field(min_length=1)
    device_families: list[str] = Field(min_length=1)


class Principle(_Strict):
    id: str = Field(pattern=r"^QP-\d{2}$")
    text: str
    enforced_by: str


class Admission(_Strict):
    issue: IssueId
    rule: str


class Ledger(_Strict):
    version: str
    admission: Admission
    principle: list[Principle]
    processing_method: list[ProcessingMethod]
    family: list[Family]
    assessment: list[Assessment]
    non_equivalence: list[NonEquivalence]

    def family_by_id(self, family_id: str) -> Family:
        for f in self.family:
            if f.id == family_id:
                return f
        raise KeyError(family_id)

    def in_scope(self) -> list[Family]:
        return [f for f in self.family if f.scope is Scope.IN_SCOPE]


def load(root: Path) -> Ledger:
    with (root / "ledger.toml").open("rb") as fh:
        return Ledger.model_validate(tomllib.load(fh))


def is_blocked(family: Family) -> bool:
    """True when any dimension has a scientific blocker."""
    return any(b.kind is BlockerKind.SCIENTIFIC for d in family.dims.values() for b in d.blockers)


def compute_state(family: Family) -> QualState:
    """Derive the qualification state from dimension states and evidence.

    * ``SCIENTIFICALLY_QUALIFIED``: every dimension verified, at least one counting
      non-analytical reference, and recorded independent review (M8).
    * ``ENGINEERING_VERIFIED`` (the M3 gate): every dimension except N verified, N at
      least specified, and at least one counting reference.
    * ``SPECIFIED``: every dimension at least specified.
    * ``NOT_STARTED``: every dimension absent. Otherwise ``IN_PROGRESS``.
    """
    states = {k: d.state for k, d in family.dims.items()}
    refs = [r for r in family.references if r.counts]
    if all(s is DimState.ABSENT for s in states.values()):
        return QualState.NOT_STARTED
    non_n_verified = all(s is DimState.VERIFIED for k, s in states.items() if k != "N")
    if (
        non_n_verified
        and states["N"] is DimState.VERIFIED
        and family.independent_review
        and any(r.kind in NON_ANALYTICAL_REFERENCES for r in refs)
    ):
        return QualState.SCIENTIFICALLY_QUALIFIED
    if non_n_verified and states["N"].rank >= DimState.SPECIFIED.rank and refs:
        return QualState.ENGINEERING_VERIFIED
    if all(s.rank >= DimState.SPECIFIED.rank for s in states.values()):
        return QualState.SPECIFIED
    return QualState.IN_PROGRESS


def gate_met(ledger: Ledger) -> bool:
    return all(compute_state(f).rank >= M3_GATE.rank for f in ledger.in_scope())


def _check_evidence(
    where: str, ev: Evidence, repo_root: Path, registry: Registry, problems: list[str]
) -> None:
    claim_ids = {c.id for c in registry.claims}
    gap_subjects = {g.subject for g in registry.gaps}
    if ev.kind in _PATH_EVIDENCE:
        path = repo_root / ev.ref
        if Path(ev.ref).is_absolute() or not path.exists():
            problems.append(f"{where}: evidence path does not exist: {ev.ref}")
    elif ev.kind is EvidenceKind.REGISTRY_CLAIM and ev.ref not in claim_ids:
        problems.append(f"{where}: unknown registry claim {ev.ref}")
    elif ev.kind is EvidenceKind.REGISTRY_GAP and ev.ref not in gap_subjects:
        problems.append(f"{where}: unknown registry gap {ev.ref}")
    elif ev.kind is EvidenceKind.ISSUE and not ISSUE_PATTERN.match(ev.ref):
        problems.append(f"{where}: not an issue id: {ev.ref}")


def validate(ledger: Ledger, registry: Registry, repo_root: Path) -> list[str]:
    problems: list[str] = []
    methods = {m.id: m for m in ledger.processing_method}
    if len(methods) != len(ledger.processing_method):
        problems.append("duplicate processing_method ids")
    ids = [f.id for f in ledger.family]
    if len(ids) != len(set(ids)):
        problems.append("duplicate family ids")
    numbers = [f.number for f in ledger.family]
    if len(numbers) != len(set(numbers)):
        problems.append("duplicate family numbers")
    known_classes = {c.value for c in DeviceClass}

    for fam in ledger.family:
        for cls in fam.device_classes:
            if cls not in known_classes:
                problems.append(f"{fam.id}: device class {cls!r} is not in the signal schema")
        for cls in fam.planned_device_classes:
            if cls in known_classes:
                problems.append(
                    f"{fam.id}: planned device class {cls!r} already exists; "
                    "move it to device_classes"
                )
        for mid in fam.processing_methods:
            method = methods.get(mid)
            if method is None:
                problems.append(f"{fam.id}: unknown processing method {mid!r}")
            elif fam.paradigm not in method.paradigms:
                problems.append(
                    f"{fam.id}: processing method {mid!r} is not valid for paradigm "
                    f"{fam.paradigm.value} (allowed: {[p.value for p in method.paradigms]})"
                )

        for key, dim in fam.dims.items():
            where = f"{fam.id}.{key}"
            if dim.state is not DimState.ABSENT and not dim.counting_evidence:
                problems.append(
                    f"{where}: state {dim.state.value} needs countable evidence "
                    "(adapter imports and vendor-number reproduction never count)"
                )
            if dim.state is DimState.VERIFIED and dim.blockers:
                problems.append(f"{where}: verified dimension cannot have open blockers")
            for ev in dim.evidence:
                _check_evidence(where, ev, repo_root, registry, problems)

        if fam.dims["J"].state is DimState.VERIFIED and not any(r.counts for r in fam.references):
            problems.append(f"{fam.id}.J: verified without an independent reference")
        for ref in fam.references:
            if ref.verifies != fam.engine:
                problems.append(f"{fam.id}: reference {ref.id} verifies {ref.verifies!r}, "
                                f"not this family's engine {fam.engine!r}")  # fmt: skip
            if ref.implementation == ref.verifies:
                problems.append(
                    f"{fam.id}: reference {ref.id} validates itself "
                    "(implementation equals the engine it verifies)"
                )
            if ref.kind is ReferenceKind.VENDOR_OUTPUT:
                problems.append(
                    f"{fam.id}: reference {ref.id} is vendor output, which is never a reference"
                )
        for ev in fam.independent_review:
            _check_evidence(f"{fam.id}.review", ev, repo_root, registry, problems)

        computed = compute_state(fam)
        if fam.declared_state is not None and fam.declared_state is not computed:
            problems.append(
                f"{fam.id}: declared_state {fam.declared_state.value} but the dimensions "
                f"compute {computed.value}; states are computed, never declared"
            )
        if fam.scope is Scope.PROPOSED and computed.rank > QualState.SPECIFIED.rank:
            problems.append(f"{fam.id}: a proposed family cannot exceed SPECIFIED")

    fam_ids = set(ids)
    proposed = {f.id for f in ledger.family if f.scope is Scope.PROPOSED}
    used: set[str] = set()
    for a in ledger.assessment:
        for t in a.tests:
            if t not in V01_TESTS:
                problems.append(f"assessment {a.id}: unknown test {t!r}")
        for fid in a.device_families:
            if fid not in fam_ids:
                problems.append(f"assessment {a.id}: unknown device family {fid!r}")
            elif fid in proposed:
                problems.append(f"assessment {a.id}: proposed family {fid!r} has no task cells")
            used.add(fid)
    for fam in ledger.in_scope():
        if fam.id not in used:
            problems.append(f"{fam.id}: in scope but used by no assessment engine")

    ne_ids = [n.id for n in ledger.non_equivalence]
    if len(ne_ids) != len(set(ne_ids)):
        problems.append("duplicate non_equivalence ids")
    for ne in ledger.non_equivalence:
        for fid in ne.families():
            if fid not in fam_ids:
                problems.append(f"{ne.id}: unknown family {fid!r}")
        if ne.a == ne.b:
            problems.append(f"{ne.id}: both sides are the same estimand")
        if ne.verdict in (Verdict.INTERCHANGEABLE, Verdict.CONVERTIBLE) and not (
            ne.evidence and ne.basis
        ):
            problems.append(f"{ne.id}: verdict {ne.verdict.value} needs evidence and a basis")
        for ev in ne.evidence:
            _check_evidence(ne.id, ev, repo_root, registry, problems)
    for fam in ledger.family:
        if fam.dims["L"].state is DimState.VERIFIED:
            open_ne = [
                n.id for n in ledger.non_equivalence if fam.id in n.families() and not n.verdict
            ]
            if open_ne:
                problems.append(f"{fam.id}.L: verified with open non-equivalences {open_ne}")
    return problems
