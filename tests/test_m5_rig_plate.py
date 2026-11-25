"""Independent native rig and force-plate comparison controls."""

import json
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

from dynamisbench_soccer.engines import force_platform_acquisition, load_cell
from dynamisbench_soccer.m4.measurement import RIG_OPERATION, replay
from dynamisbench_soccer.m4.task_registry import load_registry
from dynamisbench_soccer.signals.schema import BundleMeta, SignalBundle
from dynamisbench_soccer.statistics.reliability import bland_altman, paired_reliability

ROOT = Path(__file__).resolve().parents[1]
TASK = "rig-plate-net-peak-pair"
ABS = 1e-8


def _fixture(index):
    return json.loads(
        (ROOT / f"benchmark/fixtures/m5-force-development/rig-case-{index:02}.json").read_bytes()
    )


def _array_digest(values):
    return sha256((json.dumps(values, separators=(",", ":")) + "\n").encode()).hexdigest()


def _set_samples(record, values):
    record["samples"] = values
    record["meta"]["n_rows"] = len(values)
    record["meta"]["provenance"]["source_sha256"] = _array_digest(values)
    record["meta"]["acquisition"]["sources"][0]["source_sha256"] = _array_digest(values)


def _native(record):
    return SignalBundle(
        BundleMeta.model_validate(record["meta"]),
        {"force": np.array(record["samples"], dtype=np.float64)},
    )


def _closed_form(row, baseline, excursion, rise_samples):
    return baseline + excursion * min(max(row - 120, 0), rise_samples) / rise_samples


@pytest.mark.parametrize(("index", "plate_peak", "rise"), [(1, 800, 200), (2, 815, 250)])
def test_frozen_both_arms_reconstruct_independent_tares_peaks_and_rfd(index, plate_peak, rise):
    data = _fixture(index)
    for name, base, excursion, length in (("rig", 15, 800, 200), ("plate", 690, plate_peak, rise)):
        expected = [_closed_form(row, base, excursion, length) for row in range(651)]
        assert data[name]["samples"] == pytest.approx(expected, abs=ABS, rel=0)
    native_rig = load_cell.process(
        _native(data["rig"]),
        method=load_cell.MethodSpec(
            onset="absolute_N", onset_absolute_n=3, baseline_window_s=(0, 0.101), windows_ms=(100,)
        ),
    )
    native_plate = force_platform_acquisition.acquire(
        _native(data["plate"]),
        (force_platform_acquisition.PlateSpec(plate=1, zero_window_s=(0, 0.101)),),
    )
    assert native_rig.onset.index == 120 and native_rig.metrics["baseline_n"] == pytest.approx(15)
    assert native_rig.metrics["net_peak_force_n"] == pytest.approx(800)
    assert native_rig.metrics["rfd_0_100ms_n_s"] == pytest.approx(4000)
    assert native_plate.plates[0].offset[2] == pytest.approx(690)
    assert float(native_plate.vertical_force_n.max()) == pytest.approx(plate_peak)
    assert np.array_equal(native_rig.time_s, native_plate.time_s)
    np.testing.assert_allclose(native_plate.time_s, np.arange(651) * 0.001, atol=1e-15, rtol=0)
    reading = replay(RIG_OPERATION, data)
    assert reading.reason == "eligible" and reading.value == 800 - plate_peak
    assert reading.radius == 5
    assert dict(reading.operands) == {
        "rig_net_peak_n": 800,
        "plate_net_peak_n": plate_peak,
        "rig_rfd_100ms_n_s": 4000,
        "plate_rfd_100ms_n_s": plate_peak * 1000 / rise,
    }


def _nonlinear_tare_trace(tare, onset, k1, k2, peak_index, peak):
    net = [0.0] * 651
    net[110] = -125
    for i in range(101):
        net[onset + i] = 3 + k1 * i + k2 * i**2
    net[peak_index] = peak
    return [tare + value for value in net]


def _asymmetric_truth(reverse):
    # Two independent 100 ms rises start at distinct exact-threshold samples.
    peaks = [925, 900]
    slopes = [(2 * 100 + 0.005 * 100**2) / 0.1, (1.5 * 100 + 0.003 * 100**2) / 0.1]
    if reverse:
        peaks.reverse()
        slopes.reverse()
    return peaks[0] - peaks[1], {
        "rig_net_peak_n": peaks[0],
        "plate_net_peak_n": peaks[1],
        "rig_rfd_100ms_n_s": slopes[0],
        "plate_rfd_100ms_n_s": slopes[1],
    }


@pytest.mark.parametrize("reverse", [False, True])
def test_asymmetric_independent_onsets_different_peak_times_and_nominal_classification(reverse):
    from dynamisbench_soccer.m4.grader import ExpectedMethod, _replay_measurement

    data = _fixture(1)
    rises = [(126, 2, 0.005, 300, 925), (159, 1.5, 0.003, 500, 900)]
    if reverse:
        rises.reverse()
    rig, plate = (
        _nonlinear_tare_trace(tare, *event) for tare, event in zip((17, 731), rises, strict=True)
    )
    _set_samples(data["rig"], rig)
    _set_samples(data["plate"], plate)
    data["setup"]["peak_difference_limit_n"] = 25
    value, operands = _asymmetric_truth(reverse)
    result = replay(RIG_OPERATION, data)
    assert result.reason == "eligible" and result.value == value
    assert dict(result.operands) == operands
    assert result.radius == 5
    assert "inside_declared_peak_tolerance" in result.claims and abs(value) + 5 > 25
    assert max(a - 17 - b + 731 for a, b in zip(rig, plate, strict=True)) != value

    artifact = "asymmetric-rig-control"
    method = load_registry(ROOT / "benchmark/tasks")[TASK, 1].methods[0]
    claims = [
        "matched_net_vertical_force",
        "analytical_single_pair",
        "inside_declared_peak_tolerance",
        "paired_variance_unavailable",
        "hardware_interchangeability_unknown",
    ]
    expected = ExpectedMethod.model_validate_json(
        json.dumps(
            {
                "decision": "compute",
                "reasons": ["eligible"],
                "selected_trials": [artifact],
                "required_citations": ["DEV-006"],
                "allowed_citations": ["DEV-006"],
                "required_claims": claims,
                "allowed_claims": claims,
                "quantity": {
                    "value": value,
                    "uncertainty": 5,
                    "unit": "N",
                    "dimension": "force",
                    "estimand": "rig_minus_plate_net_peak_force",
                },
                "operands": [
                    {"field": field, "value": number, "artifact": artifact}
                    for field, number in operands.items()
                ],
            }
        )
    )
    payload = json.dumps({"inputs": {artifact: json.dumps(data)}}).encode()
    assert _replay_measurement(expected, method, payload, (artifact,))
    onset = rises[0][0]
    wrong_rfd = expected.operands[2].model_copy(
        update={"value": (rig[onset + 101] - rig[onset + 1]) / 0.1}
    )
    assert not _replay_measurement(
        expected.model_copy(
            update={"operands": (*expected.operands[:2], wrong_rfd, expected.operands[3])}
        ),
        method,
        payload,
        (artifact,),
    )


FAULTS = (
    (3, "unsupported_claim", "estimand_mismatch"),
    (4, "not_eligible", "pair_timing_or_identity_unresolved"),
)


@pytest.mark.parametrize(("index", "reason", "claim"), FAULTS)
def test_frozen_rig_plate_refusals(index, reason, claim):
    result = replay(RIG_OPERATION, _fixture(index))
    assert (result.reason, result.claims) == (reason, (claim,))
    assert result.value is result.radius is None and not result.operands


@pytest.mark.parametrize("arm", ["rig", "plate"])
def test_each_arm_needs_baseline_bracket_and_complete_window(arm):
    data = _fixture(1)
    trace = data[arm]
    tare = trace["samples"][0]
    for samples in (
        [tare] * 100 + [tare + 20] * 551,
        [tare] * 620 + [tare + 20] * 31,
        [tare] * 651,
    ):
        _set_samples(trace, samples)
        assert replay(RIG_OPERATION, data).claims == ("unresolved_pair_rfd_window",)


def test_pair_checks_both_source_hashes_and_same_pair_identity():
    data = _fixture(1)
    for arm in ("rig", "plate"):
        trace = data[arm]["samples"]
        trace[200] += 10
        with pytest.raises(ValueError, match="source SHA-256"):
            replay(RIG_OPERATION, data)
        trace[200] -= 10
    data["plate"]["meta"]["session"]["trial_index"] = 2
    assert replay(RIG_OPERATION, data).claims == ("pair_timing_or_identity_unresolved",)
    data = _fixture(1)
    for arm in ("rig", "plate"):
        data[arm]["meta"]["sampling"]["measured_hz"] = 999
    assert replay(RIG_OPERATION, data).claims == ("pair_timing_or_identity_unresolved",)
    data = _fixture(1)
    data["setup"]["load_transfer"] = "off_axis"
    assert replay(RIG_OPERATION, data).claims == ("estimand_mismatch",)

    task = load_registry(ROOT / "benchmark/tasks")[TASK, 1]
    raw = {item.name: (ROOT / item.path).read_bytes() for item in task.artifacts}
    visible = set()
    for checkpoint in task.checkpoints:
        visible.update(checkpoint.artifacts)
        assert set(json.loads(task.candidate_payload(checkpoint.id, raw, 704))["inputs"]) == visible


def _difference_bound(rig_bound, plate_bound):
    return max(abs(a - b) for a in (-rig_bound, rig_bound) for b in (-plate_bound, plate_bound))


def test_declared_single_pair_bounds_do_not_supply_statistics():
    data = _fixture(2)
    reading = replay(RIG_OPERATION, data)
    assert reading.radius == _difference_bound(2, 3)
    data["setup"]["net_peak_error_n"] = [1, 6]
    assert replay(RIG_OPERATION, data).radius == _difference_bound(1, 6)
    assert "paired_variance_unavailable" in reading.claims
    assert "hardware_interchangeability_unknown" in reading.claims
    with pytest.raises(ValueError, match="at least two"):
        bland_altman([[815, 800]], missing="raise")
    with pytest.raises(ValueError, match="at least two"):
        paired_reliability([[815, 800]], single_measure_icc=0.5, missing="raise")


def test_private_campaign_naive_dispatch_executes_in_linux_runner():
    from dynamisbench_soccer.m4.grader import MeasurementSubmission
    from dynamisbench_soccer.m4.runner import execute_payload

    task = load_registry(ROOT / "benchmark/tasks")[TASK, 1]
    inputs = {item.name: (ROOT / item.path).read_bytes() for item in task.artifacts}
    source = (ROOT / "scripts/m5_controls/naive.py").read_bytes()
    execution = execute_payload(task.candidate_payload("c01", inputs, 704), source)
    assert execution.outcome == "graded" and execution.workspace_removed
    answer = MeasurementSubmission.model_validate_json(execution.stdout)
    assert answer.method == RIG_OPERATION
    assert answer.quantity is not None and answer.quantity.value == -675
