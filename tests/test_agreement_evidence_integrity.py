"""RES-703: source evidence (QC, eligibility, execution basis) propagates into every study."""

import pytest

from dynamisbench_soccer.agreement import studies
from dynamisbench_soccer.agreement.studies import EVIDENCE_CLASSES, evidence_class

OK = {"diagnostic_only": False}
BAD = {"diagnostic_only": True}


def test_every_study_has_exactly_one_known_evidence_class_and_none_is_empirical() -> None:
    report = studies.run_all()
    assert report["empirical_device_agreement_studies"] == 0
    for s in report["studies"]:
        assert s["evidence_class"] in EVIDENCE_CLASSES, s["id"]
        assert s["evidence_class"] != "EMPIRICAL_MEASURED", s["id"]
    assert sum(report["evidence_class_counts"].values()) == len(report["studies"])


def test_evidence_class_rules() -> None:
    assert evidence_class(("engine", "engine"), [OK], False) == "ENGINE_TO_ENGINE"
    assert (
        evidence_class(("closed_form", "engine"), [OK], False)
        == "ENGINE_TO_INDEPENDENT_CLOSED_FORM"
    )
    assert evidence_class(("closed_form", "closed_form"), [], False) == "CLOSED_FORM_DEMONSTRATION"
    assert evidence_class(("engine", "engine"), [OK, BAD], False) == "DIAGNOSTIC_ONLY"
    assert evidence_class(("unavailable", "unavailable"), [], False) == "UNSUPPORTED"
    assert evidence_class(("engine", "engine"), [OK], True) == "UNSUPPORTED"


@pytest.mark.parametrize("name", ["force_vs_contact_flight_time", "force_vs_imu_flight_time"])
def test_accepted_force_trials_give_engine_to_engine_evidence_with_full_context(name: str) -> None:
    study = getattr(studies, name)()
    assert study.evidence_class == "ENGINE_TO_ENGINE"
    forces = [e for e in study.source_evidence if e["source"] == "force_platform.single"]
    assert len(forces) == len(studies.TAKEOFF_SPEEDS)
    for e in forces:
        assert e["protocol"] == "sj" and e["protocol_eligible"] is True
        assert e["qc_verdict"] != "reject" and not e["diagnostic_only"]
        assert e["estimand"]
    assert study.statistics is not None and study.classification == "analytical_model_agreement"


@pytest.mark.parametrize("name", ["force_vs_contact_flight_time", "force_vs_imu_flight_time"])
def test_rejected_force_trials_make_the_study_diagnostic_only(name: str) -> None:
    study = getattr(studies, name)(accepted_force=False)
    assert study.evidence_class == "DIAGNOSTIC_ONLY"
    assert study.statistics is None  # no agreement statistics from rejected sources
    assert study.classification == "missing_evidence"
    assert any(e["diagnostic_only"] for e in study.source_evidence)
    assert study.pairs  # numbers retained as diagnostics, not hidden


def test_the_jump_fixture_is_a_squat_jump_not_a_countermovement_jump() -> None:
    """The analytical jump has no unweighting: assessing it as CMJ is protocol-ineligible."""
    from dynamisbench_soccer.assessment import jumps
    from dynamisbench_soccer.engines.force_platform import process
    from dynamisbench_soccer.engines.force_platform_acquisition import PlateSpec
    from dynamisbench_soccer.reference.exporters import export_case

    device = process(
        export_case("force_platform.single", studies._jump(2.6), hz=1000).observation,
        plates=(PlateSpec(plate=1, assume_field_axes=True),),
    )
    signal = jumps.JumpSignal.from_engine(device)
    as_cmj = jumps.assess(signal, jumps.JumpSpec(protocol="cmj"))
    assert not as_cmj.eligible
    assert any(
        c["check"] == "countermovement_present" and not c["passed"] for c in as_cmj.eligibility
    )


def test_execution_basis_and_evidence_class_agree_for_every_study() -> None:
    expected = {
        ("engine", "engine"): {"ENGINE_TO_ENGINE", "DIAGNOSTIC_ONLY"},
        ("closed_form", "engine"): {"ENGINE_TO_INDEPENDENT_CLOSED_FORM", "DIAGNOSTIC_ONLY"},
        ("engine", "closed_form"): {"ENGINE_TO_INDEPENDENT_CLOSED_FORM", "DIAGNOSTIC_ONLY"},
        ("closed_form", "closed_form"): {"CLOSED_FORM_DEMONSTRATION"},
        ("unavailable", "unavailable"): {"UNSUPPORTED"},
    }
    for s in studies.run_all()["studies"]:
        assert s["evidence_class"] in expected[tuple(s["execution_basis"])], s["id"]
