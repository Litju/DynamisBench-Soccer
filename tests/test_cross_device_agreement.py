"""Cross-device agreement statistics, classification and analytical studies (RES-703).

Statistics are compared with independent manual numpy/scipy computations; each study's
quantity is compared with a closed form written in this test.
"""

import ast
import json
import math
from pathlib import Path

import numpy as np
import pytest
from scipy import stats as sps

from dynamisbench_soccer.agreement import stats, studies
from dynamisbench_soccer.agreement.stats import classify, paired_agreement
from dynamisbench_soccer.assessment.isometric import NordicSpec, assess_nordic
from dynamisbench_soccer.cli import main
from dynamisbench_soccer.qualification import ledger as ledger_model
from dynamisbench_soccer.reference.exporters import export_case
from dynamisbench_soccer.reference.motion_models import MonoExponentialSprint

ROOT = Path(__file__).resolve().parents[1]
SPRINT = MonoExponentialSprint()
G = 9.80665


def _round(value, digits=7):
    if isinstance(value, float):
        return round(value, digits)
    if isinstance(value, dict):
        return {k: _round(v, digits) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_round(v, digits) for v in value]
    return value


@pytest.fixture(scope="module")
def report():
    return studies.run_all()


def _study(report, id_):
    return next(s for s in report["studies"] if s["id"] == id_)


def test_paired_agreement_matches_independent_computation() -> None:
    rng = np.random.default_rng(11)
    truth = rng.uniform(20, 60, 40)
    a = truth + rng.normal(0, 0.5, 40)
    b = 1.03 * truth + 0.4 + rng.normal(0, 0.5, 40)  # proportional + constant bias
    result = paired_agreement(a, b, tolerance=5.0, uncertainty_a=0.3, uncertainty_b=0.4)
    d = b - a
    assert result.bias == pytest.approx(d.mean())
    assert result.sd_difference == pytest.approx(d.std(ddof=1))
    assert result.typical_error == pytest.approx(d.std(ddof=1) / math.sqrt(2))
    half = 1.959964 * d.std(ddof=1)
    assert result.limits_of_agreement == pytest.approx((d.mean() - half, d.mean() + half), rel=1e-6)
    fit = sps.linregress((a + b) / 2, d)
    assert result.proportional_bias_slope == pytest.approx(fit.slope)
    assert result.proportional_bias_slope == pytest.approx(0.03, abs=0.02)
    assert result.proportional_bias_p is not None and result.proportional_bias_p < 0.05
    assert result.expanded_uncertainty_of_difference == pytest.approx(2 * math.hypot(0.3, 0.4))
    assert result.ratio_bias_percent == pytest.approx(100 * d.mean() / a.mean())
    assert result.within_tolerance is True
    assert paired_agreement(a, b, tolerance=0.5).within_tolerance is False


def test_statistics_refuse_bad_inputs() -> None:
    with pytest.raises(ValueError, match="equal length"):
        paired_agreement([1, 2, 3], [1, 2])
    with pytest.raises(ValueError, match="at least three"):
        paired_agreement([1, 2], [1, 2])
    with pytest.raises(ValueError, match="finite"):
        paired_agreement([1, 2, np.nan], [1, 2, 3])
    with pytest.raises(ValueError, match="positive"):
        paired_agreement([1, 2, 3], [1, 2, 3], tolerance=0)
    with pytest.raises(ValueError, match="nonnegative"):
        paired_agreement([1, 2, 3], [1, 2, 3], uncertainty_a=-1, uncertainty_b=1)
    constant = paired_agreement([5, 5, 5], [5, 5, 5])
    assert constant.proportional_bias_slope is None and constant.bias == 0
    exact = paired_agreement([1, 2, 3, 4], [1, 2, 3, 4])
    assert exact.proportional_bias_p is None  # undefined, never NaN in the report


def test_classification_separates_the_five_evidence_states() -> None:
    assert classify(origin="none", definitions_match=True).classification == "missing_evidence"
    analytical = classify(origin="analytical", definitions_match=True)
    assert analytical.classification == "analytical_model_agreement"
    mismatch = classify(origin="analytical", definitions_match=False)
    assert mismatch.classification == "definition_mismatch" and mismatch.interchangeable == "no"
    unsupported = classify(origin="analytical", definitions_match=True, unsupported_reason="x")
    assert unsupported.classification == "unsupported_comparison"
    thin = classify(origin="empirical", definitions_match=True, n_pairs=5, data_is_measured=True)
    assert thin.classification == "missing_evidence"
    synthetic_claim = classify(origin="empirical", definitions_match=True, n_pairs=100)
    assert synthetic_claim.classification == "missing_evidence"
    same_subjects = classify(
        origin="empirical", definitions_match=True, n_pairs=100, data_is_measured=True
    )
    assert same_subjects.classification == "missing_evidence"
    enough = classify(
        origin="empirical",
        definitions_match=True,
        n_pairs=30,
        independent_subjects=True,
        data_is_measured=True,
    )
    assert enough.classification == "empirical_device_agreement"


def test_no_classification_ever_declares_interchangeability() -> None:
    for origin in ("none", "analytical", "empirical"):
        for match in (True, False):
            verdict = classify(
                origin=origin,  # type: ignore[arg-type]
                definitions_match=match,
                n_pairs=500,
                independent_subjects=True,
                data_is_measured=True,
            )
            assert verdict.interchangeable in {"unknown", "no"}


def test_synthetic_bundles_are_never_measured_evidence() -> None:
    case = export_case("force_platform.single", hz=500)
    assert stats.is_measured(case.observation.meta) is False
    assert stats.is_measured(case.truth.meta) is False


def test_every_declared_non_equivalence_has_a_study_and_none_claims_empirical_agreement(
    report,
) -> None:
    ledger = ledger_model.load(ROOT / "qualification" / "ledger.toml")
    studied = {(s["left"], s["right"], s["quantity"]) for s in report["studies"]}
    declared = {(c.left, c.right, c.quantity) for c in ledger.comparisons}
    assert len(report["studies"]) == 12 and len(declared) == 11
    assert declared <= studied  # every declared ledger non-equivalence has its own study
    assert report["interchangeable_claims"] == 0
    assert report["empirical_device_agreement_studies"] == 0
    assert not any(s["classification"] == "empirical_device_agreement" for s in report["studies"])
    assert all("missing_evidence" in s["empirical_status"] for s in report["studies"])
    assert report["classification_counts"] == {
        "analytical_model_agreement": 4,
        "definition_mismatch": 7,
        "unsupported_comparison": 1,
    }


def test_flight_time_studies_agree_within_resolution_and_not_beyond(report) -> None:
    for id_, tolerance in (
        ("force_vs_contact_flight_time", 0.004),
        ("force_vs_imu_flight_time", 0.006),
    ):
        s = _study(report, id_)
        assert s["classification"] == "analytical_model_agreement"
        assert abs(s["statistics"]["bias"]) < tolerance
        assert s["statistics"]["within_tolerance"] is True
    truth = [2 * v / G for v in studies.TAKEOFF_SPEEDS]
    forces = [p[0] for p in _study(report, "force_vs_contact_flight_time")["pairs"]]
    assert forces == pytest.approx(truth, abs=2e-3)
    thresholds = _study(report, "force_vs_imu_flight_time")["sensitivities"][
        "error_vs_truth_by_threshold_s"
    ]
    assert set(thresholds) == {"threshold_1", "threshold_3", "threshold_6"}


def test_isometric_force_study_is_exact_only_under_ideal_assumptions(report) -> None:
    s = _study(report, "force_vs_load_cell_isometric_force")
    assert all(a == pytest.approx(b, abs=1e-6) for a, b in s["pairs"])
    assert "not modelled" in s["sensitivities"]["rig_compliance"]
    assert "rigid zero-compliance rig" in s["assumptions"]


def test_radar_laser_is_a_definition_mismatch_with_derived_noise_amplification(report) -> None:
    s = _study(report, "radar_vs_laser_velocity")
    assert s["classification"] == "definition_mismatch"
    noise = np.random.default_rng(5).normal(0, 1.0, 400000)
    derivative = (noise[:-4] - 8 * noise[1:-3] + 8 * noise[3:-1] - noise[4:]) / 12
    assert float(derivative.std()) == pytest.approx(math.sqrt(130) / 12, rel=0.01)
    expected = float(derivative.std()) * 0.01 * 256
    key = "five_point_derivative_noise_sd_m_s_at_256hz_for_0.01m_range_noise"
    assert s["sensitivities"][key] == pytest.approx(expected, rel=0.01)


def test_positional_gate_cable_and_attachment_studies_match_closed_forms(report) -> None:
    latency = 0.1
    positional = _study(report, "radar_vs_positional_speed")
    vmax, tau = SPRINT.vmax_m_s, SPRINT.tau_s
    for t, bias in positional["sensitivities"]["latency_bias_at_t_s"].items():
        expected = vmax * math.exp(-float(t) / tau) * (math.exp(latency / tau) - 1)
        assert bias == pytest.approx(expected, rel=1e-9)
    gates = _study(report, "radar_vs_gates_velocity")
    first = gates["pairs"][0]
    assert first[1] == pytest.approx(5.0 / SPRINT.gate_time_s(5.0), rel=1e-9)
    assert first[0] != pytest.approx(first[1], rel=1e-3)  # not the same quantity
    cable = _study(report, "cable_vs_vertical_displacement")
    for h, c in cable["pairs"]:
        assert c == pytest.approx(math.hypot(0.3, h) - 0.3, rel=1e-12)
        assert c < h  # a tilted cable always under-reads vertical rise
    power = _study(report, "attachment_vs_com_power")
    for fp, lpt in power["pairs"]:
        assert lpt / fp == pytest.approx((75.0 + 20.0) / 75.0, rel=1e-9)
    for s in (positional, gates, cable, power):
        assert s["classification"] == "definition_mismatch" and s["interchangeable"] == "no"


def test_gate_split_times_agree_with_integrated_radar_speed(report) -> None:
    s = _study(report, "gates_vs_radar_laser_timing")
    assert s["classification"] == "analytical_model_agreement"
    assert s["statistics"]["bias"] == pytest.approx(0.0, abs=2e-4)
    assert "start offset adds directly" in s["sensitivities"]["start_offset_effect"]


def test_nordic_raw_versus_vendor_corrected_is_unsupported_and_pooling_is_refused(report) -> None:
    s = _study(report, "nordic_raw_vs_vendor_corrected")
    assert s["classification"] == "unsupported_comparison" and s["pairs"] == []
    assert "refuses to pool" in s["reason"]
    from dynamisbench_soccer.engines.nordic_hip import NordicSpec as EngineSpec
    from dynamisbench_soccer.engines.nordic_hip import SideDeclaration, process

    bundle = export_case("nordic_hip.nordic", hz=1000).observation

    def trial(basis: str):
        spec = EngineSpec(
            protocol="nordic_eccentric",
            force_basis=basis,  # type: ignore[arg-type]
            sides=(
                SideDeclaration(channel="force_left", side="left"),
                SideDeclaration(channel="force_right", side="right"),
            ),
        )
        return process(bundle, spec)

    with pytest.raises(ValueError, match="cannot be combined"):
        assess_nordic([trial("raw_calibrated"), trial("vendor_corrected")], NordicSpec())


def test_committed_report_reproduces_and_ledger_records_analytical_evidence(report) -> None:
    committed = json.loads((ROOT / "docs/verification/cross-device-agreement.json").read_text())
    assert _round(committed) == _round(json.loads(json.dumps(report)))
    ledger = ledger_model.load(ROOT / "qualification" / "ledger.toml")
    assert ledger_model.validate(ledger, ROOT) == []
    for c in ledger.comparisons:
        assert c.verdict == "unknown"  # nothing is upgraded without empirical evidence
        assert c.evidence_level == "analytical"
        assert "repo:docs/verification/cross-device-agreement.json" in c.refs
        assert c.reviewer_status == "pending"


def test_modules_keep_import_boundaries() -> None:
    for folder in ("engines", "reference"):
        for path in (ROOT / "src/dynamisbench_soccer" / folder).glob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    assert not (node.module or "").startswith("dynamisbench_soccer.agreement")


def test_cli_writes_the_report(tmp_path: Path) -> None:
    out = tmp_path / "agreement.json"
    assert main(["agreement", "run", "--out", str(out)]) == 0
    written = json.loads(out.read_text())
    assert written["interchangeable_claims"] == 0 and len(written["studies"]) == 12
