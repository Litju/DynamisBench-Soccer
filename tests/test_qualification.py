from pathlib import Path

import pydantic
import pytest

from dynamisbench_soccer.qualification.models import (
    DIMENSIONS,
    DimState,
    Evidence,
    EvidenceKind,
    Family,
    Ledger,
    QualState,
    Reference,
    ReferenceKind,
    Scope,
    Verdict,
    compute_state,
    gate_met,
    is_blocked,
    load,
    validate,
)
from dynamisbench_soccer.registry.models import Registry
from dynamisbench_soccer.registry.models import load as load_registry

EXPECTED_FAMILIES = [
    "force_platform",
    "lpt_encoder",
    "load_cell_isometric",
    "nordic_hip",
    "timing_contact",
    "radar_laser",
    "gnss_lps",
    "imu",
]


@pytest.fixture
def ledger(repo_root: Path) -> Ledger:
    return load(repo_root / "qualification")


@pytest.fixture
def registry(repo_root: Path) -> Registry:
    return load_registry(repo_root / "registry")


def _problems(ledger: Ledger, registry: Registry, repo_root: Path) -> list[str]:
    return validate(ledger, registry, repo_root)


def _verify_all(fam: Family, *, n: DimState = DimState.SPECIFIED) -> None:
    """Make every dimension look done, with countable evidence and no blockers."""
    for key, dim in fam.dims.items():
        dim.state = n if key == "N" else DimState.VERIFIED
        dim.blockers = []
        dim.evidence = [Evidence(kind=EvidenceKind.DOC, ref="docs/signal-schema.md")]


def _analytical_ref(fam: Family) -> Reference:
    return Reference(
        id="REF-ANALYTIC",
        kind=ReferenceKind.ANALYTICAL_CLOSED_FORM,
        implementation="dbs.generators.analytical",
        verifies=fam.engine,
        ref="RES-669",
    )


# --- the real ledger -------------------------------------------------------------------


def test_ledger_is_valid(ledger: Ledger, registry: Registry, repo_root: Path) -> None:
    assert _problems(ledger, registry, repo_root) == []


def test_all_declared_families_are_in_scope_and_not_narrowed(ledger: Ledger) -> None:
    assert [f.id for f in ledger.family] == EXPECTED_FAMILIES
    assert [f.number for f in ledger.family] == list(range(1, 9))
    assert all(f.scope is Scope.IN_SCOPE for f in ledger.family)


def test_every_family_has_all_fifteen_dimensions(ledger: Ledger) -> None:
    for fam in ledger.family:
        assert list(fam.dims) == list(DIMENSIONS)


def test_no_family_is_qualified_or_gate_met_today(ledger: Ledger) -> None:
    for fam in ledger.family:
        assert compute_state(fam).rank < QualState.ENGINEERING_VERIFIED.rank
        assert fam.declared_state is None
    assert not gate_met(ledger)


def test_families_without_registry_evidence_are_reported_blocked(ledger: Ledger) -> None:
    for fid in ("lpt_encoder", "gnss_lps", "imu"):
        assert is_blocked(ledger.family_by_id(fid))


def test_discrete_event_family_uses_no_continuous_filter(ledger: Ledger) -> None:
    fam = ledger.family_by_id("timing_contact")
    assert fam.paradigm.value == "discrete_event"
    assert not {"butterworth_zero_lag", "chebyshev_zero_lag", "moving_average"} & set(
        fam.processing_methods
    )


def test_required_non_equivalences_are_declared(ledger: Ledger) -> None:
    pairs = {frozenset((n.a, n.b)) for n in ledger.non_equivalence}
    assert frozenset(("gnss_lps:speed.gnss", "radar_laser:velocity.radar")) in pairs
    assert (
        frozenset(("lpt_encoder:displacement.cable", "force_platform:displacement.com_vertical"))
        in pairs
    )
    assert all(n.verdict is None for n in ledger.non_equivalence)


# --- computed state ------------------------------------------------------------------


def test_engineering_verified_needs_an_independent_reference(ledger: Ledger) -> None:
    fam = ledger.family_by_id("force_platform").model_copy(deep=True)
    _verify_all(fam)
    assert compute_state(fam) is QualState.SPECIFIED  # no reference yet
    fam.references = [_analytical_ref(fam)]
    assert compute_state(fam) is QualState.ENGINEERING_VERIFIED


def test_scientific_qualification_needs_review_grading_and_non_analytical_reference(
    ledger: Ledger,
) -> None:
    fam = ledger.family_by_id("imu").model_copy(deep=True)
    _verify_all(fam, n=DimState.VERIFIED)
    fam.references = [_analytical_ref(fam)]
    assert compute_state(fam) is QualState.ENGINEERING_VERIFIED
    fam.independent_review = [Evidence(kind=EvidenceKind.REVIEW, ref="RES-688 panel")]
    assert compute_state(fam) is QualState.ENGINEERING_VERIFIED  # analytical only
    fam.references.append(
        Reference(
            id="REF-EMPIRICAL",
            kind=ReferenceKind.EMPIRICAL_CRITERION,
            implementation="motion-capture criterion",
            verifies=fam.engine,
            ref="RES-703",
        )
    )
    assert compute_state(fam) is QualState.SCIENTIFICALLY_QUALIFIED


def test_all_absent_is_not_started(ledger: Ledger) -> None:
    fam = ledger.family_by_id("gnss_lps").model_copy(deep=True)
    for dim in fam.dims.values():
        dim.state = DimState.ABSENT
    assert compute_state(fam) is QualState.NOT_STARTED


# --- negative and adversarial controls -------------------------------------------------


def test_hand_set_label_that_disagrees_is_rejected(
    ledger: Ledger, registry: Registry, repo_root: Path
) -> None:
    ledger.family_by_id("force_platform").declared_state = QualState.SCIENTIFICALLY_QUALIFIED
    assert any("states are computed" in p for p in _problems(ledger, registry, repo_root))


def test_adapter_import_alone_never_qualifies(
    ledger: Ledger, registry: Registry, repo_root: Path
) -> None:
    dim = ledger.family_by_id("lpt_encoder").dims["F"]
    dim.state = DimState.VERIFIED
    adapter = "src/dynamisbench_soccer/adapters/generic.py"
    dim.evidence = [Evidence(kind=EvidenceKind.ADAPTER_IMPORT, ref=adapter)]
    problems = _problems(ledger, registry, repo_root)
    assert any("lpt_encoder.F" in p and "countable" in p for p in problems)


def test_vendor_number_reproduction_never_qualifies(
    ledger: Ledger, registry: Registry, repo_root: Path
) -> None:
    dim = ledger.family_by_id("force_platform").dims["J"]
    dim.state = DimState.VERIFIED
    dim.evidence = [Evidence(kind=EvidenceKind.VENDOR_REPRODUCTION, ref="vendor export matched")]
    problems = _problems(ledger, registry, repo_root)
    assert any("force_platform.J" in p and "countable" in p for p in problems)
    assert any("force_platform.J: verified without an independent reference" in p for p in problems)


def test_self_validating_reference_is_rejected(
    ledger: Ledger, registry: Registry, repo_root: Path
) -> None:
    fam = ledger.family_by_id("radar_laser")
    fam.references = [
        Reference(
            id="REF-SELF",
            kind=ReferenceKind.ANALYTICAL_CLOSED_FORM,
            implementation=fam.engine,
            verifies=fam.engine,
            ref="RES-699",
        )
    ]
    assert not fam.references[0].counts
    assert any("validates itself" in p for p in _problems(ledger, registry, repo_root))


def test_vendor_output_is_never_a_reference(
    ledger: Ledger, registry: Registry, repo_root: Path
) -> None:
    fam = ledger.family_by_id("nordic_hip")
    fam.references = [
        Reference(
            id="REF-VENDOR",
            kind=ReferenceKind.VENDOR_OUTPUT,
            implementation="vendor software",
            verifies=fam.engine,
            ref="export.csv",
        )
    ]
    assert not fam.references[0].counts
    assert any("vendor output" in p for p in _problems(ledger, registry, repo_root))


def test_reference_for_another_engine_is_rejected(
    ledger: Ledger, registry: Registry, repo_root: Path
) -> None:
    fam = ledger.family_by_id("imu")
    ref = _analytical_ref(fam)
    ref.verifies = "dbs.engines.device.force_platform"
    fam.references = [ref]
    assert any("not this family's engine" in p for p in _problems(ledger, registry, repo_root))


def test_continuous_filter_on_discrete_events_is_rejected(
    ledger: Ledger, registry: Registry, repo_root: Path
) -> None:
    ledger.family_by_id("timing_contact").processing_methods.append("butterworth_zero_lag")
    assert any(
        "timing_contact" in p and "not valid for paradigm discrete_event" in p
        for p in _problems(ledger, registry, repo_root)
    )


def test_unknown_processing_method_is_rejected(
    ledger: Ledger, registry: Registry, repo_root: Path
) -> None:
    ledger.family_by_id("imu").processing_methods.append("magic_filter")
    assert any("unknown processing method" in p for p in _problems(ledger, registry, repo_root))


def test_missing_dimension_fails_to_load(ledger: Ledger) -> None:
    raw = ledger.family_by_id("imu").model_dump()
    del raw["dims"]["L"]
    with pytest.raises(pydantic.ValidationError, match="dims must be exactly A-O"):
        Family.model_validate(raw)


def test_dangling_evidence_is_rejected(ledger: Ledger, registry: Registry, repo_root: Path) -> None:
    dim = ledger.family_by_id("imu").dims["B"]
    dim.state = DimState.PARTIAL
    dim.evidence = [
        Evidence(kind=EvidenceKind.DOC, ref="docs/does-not-exist.md"),
        Evidence(kind=EvidenceKind.REGISTRY_CLAIM, ref="IM-999"),
        Evidence(kind=EvidenceKind.REGISTRY_GAP, ref="imu"),
    ]
    problems = _problems(ledger, registry, repo_root)
    assert any("path does not exist" in p for p in problems)
    assert any("unknown registry claim IM-999" in p for p in problems)
    assert any("unknown registry gap imu" in p for p in problems)


def test_verified_dimension_with_open_blocker_is_rejected(
    ledger: Ledger, registry: Registry, repo_root: Path
) -> None:
    dim = ledger.family_by_id("gnss_lps").dims["I"]
    dim.state = DimState.VERIFIED
    dim.evidence = [Evidence(kind=EvidenceKind.DOC, ref="docs/signal-schema.md")]
    assert any("open blockers" in p for p in _problems(ledger, registry, repo_root))


def test_interchangeability_without_evidence_is_rejected(
    ledger: Ledger, registry: Registry, repo_root: Path
) -> None:
    ledger.non_equivalence[0].verdict = Verdict.INTERCHANGEABLE
    problems = _problems(ledger, registry, repo_root)
    assert any("NE-001" in p and "needs evidence" in p for p in problems)


def test_verified_agreement_with_open_non_equivalence_is_rejected(
    ledger: Ledger, registry: Registry, repo_root: Path
) -> None:
    dim = ledger.family_by_id("gnss_lps").dims["L"]
    dim.state = DimState.VERIFIED
    dim.blockers = []
    dim.evidence = [Evidence(kind=EvidenceKind.DOC, ref="docs/signal-schema.md")]
    assert any("gnss_lps.L" in p and "open non-equivalences" in p
               for p in _problems(ledger, registry, repo_root))  # fmt: skip


def test_planned_class_that_already_exists_is_rejected(
    ledger: Ledger, registry: Registry, repo_root: Path
) -> None:
    ledger.family_by_id("lpt_encoder").planned_device_classes.append("force_plate")
    assert any("already exists" in p for p in _problems(ledger, registry, repo_root))


def test_unknown_device_class_is_rejected(
    ledger: Ledger, registry: Registry, repo_root: Path
) -> None:
    ledger.family_by_id("imu").device_classes.append("tachyon_sensor")
    assert any("not in the signal schema" in p for p in _problems(ledger, registry, repo_root))


def test_silently_dropping_a_family_from_assessments_is_rejected(
    ledger: Ledger, registry: Registry, repo_root: Path
) -> None:
    for a in ledger.assessment:
        a.device_families = [f for f in a.device_families if f != "gnss_lps"]
    assert any("gnss_lps: in scope but used by no assessment" in p
               for p in _problems(ledger, registry, repo_root))  # fmt: skip


def test_proposed_family_cannot_have_task_cells_or_exceed_specified(
    ledger: Ledger, registry: Registry, repo_root: Path
) -> None:
    fam = ledger.family_by_id("imu")
    fam.scope = Scope.PROPOSED
    _verify_all(fam)
    fam.references = [_analytical_ref(fam)]
    problems = _problems(ledger, registry, repo_root)
    assert any("proposed family 'imu' has no task cells" in p for p in problems)
    assert any("imu: a proposed family cannot exceed SPECIFIED" in p for p in problems)
