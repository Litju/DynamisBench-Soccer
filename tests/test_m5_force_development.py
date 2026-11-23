"""Inspect development diagnostics without granting scored reference admission."""

import json
from pathlib import Path

import numpy as np
import pytest

from dynamisbench_soccer.engines.load_cell import MethodSpec, process
from dynamisbench_soccer.engines.load_cell_acquisition import CellSpec
from dynamisbench_soccer.m4.grader import ExpectedMethod, MeasurementSubmission, _compare
from dynamisbench_soccer.m4.key_store import digest
from dynamisbench_soccer.m4.task_registry import load_registry
from dynamisbench_soccer.m4.trials import TrialContext, run_episode
from dynamisbench_soccer.signals.schema import BundleMeta, SignalBundle

ROOT = Path(__file__).resolve().parents[1]
CELL = "load-cell-net-torque-ramp"
RIG = "rig-plate-net-peak-pair"


@pytest.fixture(scope="module")
def force_diagnostics(tmp_path_factory):
    catalog = (ROOT / "benchmark/reviewed-slices.json").read_bytes()
    context = TrialContext(ROOT, catalog, digest(catalog), tmp_path_factory.mktemp("force-custody"))
    registry = load_registry(ROOT / "benchmark/tasks")
    source = (ROOT / "scripts/m5_controls/force_development.py").read_bytes()
    results = {}
    for task_id in (CELL, RIG):
        task = registry[task_id, 1]
        report = run_episode(context, task, "diagnostic-formula", source, seed=704)
        assert report.admission.status == "DEVELOPMENT_ONLY_NOT_QUALIFIED"
        assert report.terminal_reason is None
        assert len(report.attempts) == len(task.checkpoints)
        assert all(
            row.grade.outcome == "EXCLUDED_FROM_SCORED_BENCHMARK"
            and row.execution.outcome == "graded"
            and row.execution.workspace_removed
            for row in report.attempts
        )
        artifacts = {a.name: (ROOT / a.path).read_bytes() for a in task.artifacts}
        visible = set()
        for checkpoint in task.checkpoints:
            visible.update(checkpoint.artifacts)
            projection = json.loads(task.candidate_payload(checkpoint.id, artifacts, seed=704))
            assert set(projection["inputs"]) == visible
            assert "reference_profile" not in projection["task"]
        results[task_id] = [
            (
                MeasurementSubmission.model_validate_json(row.execution.stdout),
                json.loads(artifacts[a.name]),
            )
            for row, a in zip(report.attempts, task.artifacts, strict=True)
        ]
    return results


def native_bundle(source):
    meta = BundleMeta.model_validate(source["meta"])
    return SignalBundle(meta, {meta.channels[0].name: np.asarray(source["samples"], dtype=float)})


@pytest.mark.parametrize(
    ("index", "baseline", "gain", "lever"), [(0, 12, 800, 0.35), (1, 10, 600, 0.42)]
)
def test_load_cell_native_ramp_matches_analytical_diagnostic(
    force_diagnostics, index, baseline, gain, lever
):
    result, fixture = force_diagnostics[CELL][index]
    assert (
        result.quantity is not None
        and result.quantity.estimand == "peak_external_knee_torque_above_baseline"
    )
    assert result.quantity.value == pytest.approx(gain * lever, abs=1e-8)
    # A 4000 N/s ramp contributes the area of a 100 ms triangle above baseline.
    expected = {
        "baseline_n": baseline,
        "gross_peak_n": baseline + gain,
        "net_peak_n": gain,
        "onset_s": 0.12,
        "rfd_100ms_n_s": 4000,
        "impulse_100ms_n_s": 4000 * 0.1**2 / 2,
        "lever_arm_m": lever,
    }
    assert {item.field: item.value for item in result.operands} == pytest.approx(expected, abs=1e-8)
    top_right = (gain + 2) * (lever + 0.01)
    assert result.quantity.uncertainty == pytest.approx(top_right - gain * lever, abs=1e-8)

    # Use the native production engine as a separate comparison of the supported point values.
    cell = CellSpec(
        calibration=fixture["setup"].get("analysis_calibration"),
        raw_range=(0, 3.3),
        rail_margin_fraction=0,
        anti_aliasing_hz=100,
    )
    method = MethodSpec(
        onset="absolute_N", onset_absolute_n=3, baseline_window_s=(0, 0.1), windows_ms=(100,)
    )
    production = process(native_bundle(fixture), cell, method)
    assert production.verdict != "reject"
    assert production.metrics["net_peak_torque_n_m"] == pytest.approx(
        result.quantity.value, abs=1e-8
    )
    for key in ("baseline_n", "net_peak_force_n", "rfd_0_100ms_n_s", "net_impulse_0_100ms_n_s"):
        assert production.metrics[key] == pytest.approx(
            expected[
                {
                    "baseline_n": "baseline_n",
                    "net_peak_force_n": "net_peak_n",
                    "rfd_0_100ms_n_s": "rfd_100ms_n_s",
                    "net_impulse_0_100ms_n_s": "impulse_100ms_n_s",
                }[key]
            ],
            abs=1e-8,
        )


def test_force_faults_and_pair_claims(force_diagnostics):
    refusal = [
        ("not_eligible", "missing_calibration"),
        ("not_eligible", "calibration_unit_mismatch"),
        ("not_eligible", "adc_saturation"),
        ("not_eligible", "protocol_geometry_ineligible"),
        ("not_eligible", "wrong_limb"),
        ("insufficient_evidence", "moment_arm_unestablished"),
    ]
    assert [
        (answer.reason, answer.claims[0]) for answer, _ in force_diagnostics[CELL][2:]
    ] == refusal
    for index, (difference, plate_slope, classification) in enumerate(
        [
            (0, 4000, "inside_declared_peak_tolerance"),
            (-15, 815 / 0.25, "outside_declared_peak_tolerance"),
        ]
    ):
        answer, fixture = force_diagnostics[RIG][index]
        assert native_bundle(fixture["rig"]).meta.device.device_class == "isometric_dynamometer"
        assert native_bundle(fixture["plate"]).meta.device.device_class == "force_plate"
        assert answer.quantity is not None
        assert answer.quantity.value == pytest.approx(difference, abs=1e-8)
        assert answer.quantity.uncertainty == 5
        assert set(answer.claims) == {
            "matched_net_vertical_force",
            "analytical_single_pair",
            classification,
            "paired_variance_unavailable",
            "hardware_interchangeability_unknown",
        }
        assert {x.field: x.value for x in answer.operands} == pytest.approx(
            {
                "rig_net_peak_n": 800,
                "plate_net_peak_n": 800 - difference,
                "rig_rfd_100ms_n_s": 4000,
                "plate_rfd_100ms_n_s": plate_slope,
            }
        )
    assert [(answer.reason, answer.claims[0]) for answer, _ in force_diagnostics[RIG][2:]] == [
        ("unsupported_claim", "estimand_mismatch"),
        ("not_eligible", "pair_timing_or_identity_unresolved"),
    ]


def test_force_adversarial_comparison_diagnostics(force_diagnostics):
    known, fixture = force_diagnostics[CELL][0]
    assert known.quantity is not None
    target = ExpectedMethod(
        decision=known.decision,
        reasons=(known.reason,),
        selected_trials=known.selected_trials,
        required_citations=known.citations,
        allowed_citations=known.citations,
        required_claims=known.claims,
        allowed_claims=known.claims,
        quantity=known.quantity,
        operands=known.operands,
    )
    assert _compare(known, target, 1e-8) == ()
    assert _compare(known.model_copy(update={"claims": known.claims[::-1]}), target, 1e-8) == ()
    wrong_lever = known.operands[-1].model_copy(update={"value": 0.42})
    variants = [
        ("derivation", {"operands": (*known.operands[:-1], wrong_lever)}),
        ("trial_selection", {"selected_trials": ("left-cell",)}),
        ("estimand", {"quantity": known.quantity.model_copy(update={"estimand": "muscle_force"})}),
        ("uncertainty", {"quantity": known.quantity.model_copy(update={"uncertainty": None})}),
        (
            "value",
            {
                "quantity": known.quantity.model_copy(
                    update={
                        "value": fixture["meta"]["vendor_reported"]["unrelated_vendor_output"][
                            "value"
                        ],
                    }
                )
            },
        ),
        ("interpretation_boundary", {"claims": (*known.claims, "instrument_equivalent")}),
    ]
    for finding, update in variants:
        assert _compare(known.model_copy(update=update), target, 1e-8) == (finding,)
