"""Public development controls for the unfinished motion-device reference slices."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from dynamisbench_soccer.engines import imu, lpt
from dynamisbench_soccer.engines.lpt_acquisition import CableSpec, GeometrySpec
from dynamisbench_soccer.m4.grader import MeasurementSubmission
from dynamisbench_soccer.m4.task_registry import load_registry
from dynamisbench_soccer.m4.trials import TrialContext, run_episode
from dynamisbench_soccer.signals.schema import BundleMeta, SignalBundle

ROOT = Path(__file__).resolve().parents[1]
TASKS = (
    ("lpt-vertical-attachment", 1),
    ("encoder-rollover-attachment", 1),
    ("imu-frame-observability", 1),
)


def _fixture(name, checkpoint):
    return json.loads(
        (ROOT / "benchmark/fixtures/m5-motion" / f"{name}-{checkpoint:02d}.json").read_bytes()
    )


def _native(data):
    meta = BundleMeta.model_validate(data["meta"])
    channels = {
        channel.name: np.array(data["channels"][channel.name], dtype=float)
        for channel in meta.channels
    }
    return SignalBundle(meta, channels)


@pytest.fixture(scope="module")
def diagnostic_outputs(tmp_path_factory):
    registry = load_registry(ROOT / "benchmark/tasks")
    catalog = (ROOT / "benchmark/reviewed-slices.json").read_bytes()
    custody = tmp_path_factory.mktemp("empty")
    context = TrialContext(ROOT, catalog, hashlib.sha256(catalog).hexdigest(), custody)
    source = (ROOT / "scripts/m5_controls/motion_diagnostic.py").read_bytes()
    answers = {}
    for key in TASKS:
        task = registry[key]
        inputs = {artifact.name: (ROOT / artifact.path).read_bytes() for artifact in task.artifacts}
        report = run_episode(context, task, "motion-diagnostic", source, 704)
        assert report.admission.status == "DEVELOPMENT_ONLY_NOT_QUALIFIED"
        assert report.terminal_reason is None and len(report.attempts) == len(task.checkpoints)
        seen = set()
        rows = []
        for checkpoint, attempt in zip(task.checkpoints, report.attempts, strict=True):
            seen.update(checkpoint.artifacts)
            assert (
                json.loads(task.candidate_payload(checkpoint.id, inputs, 704))["inputs"].keys()
                == seen
            )
            assert attempt.execution.outcome == "graded" and attempt.execution.workspace_removed
            assert attempt.grade.outcome == "EXCLUDED_FROM_SCORED_BENCHMARK"
            response = MeasurementSubmission.model_validate_json(attempt.execution.stdout)
            assert response.method in {method.id for method in task.methods}
            rows.append(response)
        answers[key[0]] = rows
    assert list(custody.iterdir()) == []
    return answers


def test_lpt_projection_power_geometry_and_qc(diagnostic_outputs):
    task_id = "lpt-vertical-attachment"
    answer, *refusals = diagnostic_outputs[task_id]
    a, v = 2.0, 0.4 + 2 * 0.06  # Prescribed rigid-load polynomial z = 0.4t + t².
    assert answer.quantity.value == pytest.approx(v, abs=1e-12)
    assert answer.quantity.estimand == "bar_vertical_attachment_speed"
    data = _fixture(task_id, 1)
    angle = math.acos(0.8)
    high_speed = 0.8 * v / (math.cos(angle + 0.02) * (1 - 250e-6))
    low_speed = 0.8 * v / (math.cos(angle - 0.02) * (1 + 250e-6))
    assert answer.quantity.uncertainty == pytest.approx(max(high_speed - v, v - low_speed))
    operands = {entry.field: entry.value for entry in answer.operands}
    assert operands["vertical_acceleration_m_s2"] == pytest.approx(a, abs=1e-12)
    assert operands["vertical_displacement_m"] == pytest.approx(0.4 * 0.06 + 0.06**2)
    assert operands["vertical_mass_model_power_w"] == pytest.approx(20 * (9.80665 + a) * v)
    assert [r.reason for r in refusals] == [
        "insufficient_evidence",
        "not_eligible",
        "not_eligible",
        "unsupported_claim",
    ]
    assert [r.claims[0] for r in refusals] == [
        "cable_only",
        "cable_slack",
        "nonincreasing_native_clock",
        "mass_model_distinct_from_plate_power",
    ]

    def process(record):
        return lpt.process(
            _native(record),
            (CableSpec.model_validate(record["cable"]),),
            GeometrySpec.model_validate(record["geometry"]),
            power=lpt.PowerSpec.model_validate(data["power"]),
        )

    engine = process(data)
    assert engine["motion"]["velocity_m_s"][3] == pytest.approx(v)
    assert process(_fixture(task_id, 2))["estimand"] == "cable_displacement"
    assert process(_fixture(task_id, 3))["verdict"] == "reject"
    with pytest.raises(ValueError, match="strictly increasing"):
        process(_fixture(task_id, 4))


def test_encoder_signed_rollover_quantization_and_refusals(diagnostic_outputs):
    task_id = "encoder-rollover-attachment"
    answer, *refusals = diagnostic_outputs[task_id]
    # Four forward electronic counts per interval, reverse physical direction, six intervals.
    assert answer.quantity.value == pytest.approx(-6 * 4 / 400)
    assert answer.quantity.uncertainty == pytest.approx(1 / 400)
    assert {entry.field: entry.value for entry in answer.operands}[
        "mean_vertical_velocity_m_s"
    ] == pytest.approx(-0.5)
    assert [r.claims[0] for r in refusals] == [
        "missing_count_scale",
        "ambiguous_rollover",
        "cable_only",
        "implausible_count_jump",
    ]

    def process(record):
        return lpt.process(
            _native(record),
            (CableSpec.model_validate(record["cable"]),),
            GeometrySpec.model_validate(record["geometry"]),
        )

    engine = process(_fixture(task_id, 1))
    assert engine["motion"]["displacement_m"][-1] == pytest.approx(-0.06)
    with pytest.raises(ValueError, match="missing encoder counts_per_m"):
        process(_fixture(task_id, 2))
    with pytest.raises(ValueError, match="rollover is ambiguous"):
        process(_fixture(task_id, 3))
    assert process(_fixture(task_id, 4))["estimand"] == "cable_displacement"
    assert process(_fixture(task_id, 5))["verdict"] == "reject"


def test_imu_observability_calibration_and_signature(diagnostic_outputs):
    task_id = "imu-frame-observability"
    answer, no_attitude, no_speed, no_position, clipped, com, flight, gap = diagnostic_outputs[
        task_id
    ]
    # The prescribed field-X motion is x = 1.5 + 0.3t + t² at t=0.4 s.
    assert answer.quantity.value == pytest.approx(1.78)
    assert answer.quantity.uncertainty == pytest.approx(0.04 * 0.4**2 / 2)
    operands = {entry.field: entry.value for entry in answer.operands}
    assert operands["free_ax_m_s2"] == pytest.approx(2.0)
    assert operands["free_az_m_s2"] == pytest.approx(1.2)
    assert [r.claims[0] for r in (no_attitude, no_speed, no_position, clipped, com, gap)] == [
        "orientation_unobservable",
        "missing_initial_velocity",
        "missing_initial_position",
        "clipped_instrument",
        "instrument_point_only",
        "native_clock_gap",
    ]
    assert (no_attitude.reason, com.reason) == ("insufficient_evidence", "unsupported_claim")
    assert flight.quantity.estimand == "sensor_threshold_signature_duration"
    assert flight.quantity.value == pytest.approx(0.3 - 0.05)
    assert flight.quantity.uncertainty == pytest.approx(0.1)

    def process(index):
        data = _fixture(task_id, index)
        return imu.process(
            _native(data),
            imu.ImuSpec.model_validate(data["spec"]),
            imu.MethodSpec(integration="declared_initial_velocity"),
        )

    assert process(1).position_m[-1, 0] == pytest.approx(1.78)
    ambiguous = process(2)
    assert np.linalg.norm(ambiguous.specific_force_m_s2[0]) == pytest.approx(9.80665)
    assert ambiguous.free_acceleration_m_s2 is None
    # With the same data, one rotation aligns sensed force with up, while another
    # preserves a nonzero lateral acceleration. Both have zero angular rate.
    same_data = _fixture(task_id, 2)
    spec = imu.ImuSpec.model_validate(same_data["spec"])
    sine = 3 / 9.80665
    cosine = math.sqrt(1 - sine**2)
    tilt = ((1, 0, 0), (0, cosine, -sine), (0, sine, cosine))
    static = imu.process(_native(same_data), spec.model_copy(update={"initial_orientation": tilt}))
    aligned = imu.process(
        _native(same_data),
        spec.model_copy(update={"initial_orientation": ((1, 0, 0), (0, 1, 0), (0, 0, 1))}),
    )
    assert np.linalg.norm(static.free_acceleration_m_s2[0]) < 1e-12
    assert aligned.free_acceleration_m_s2[0, 1] == pytest.approx(3)
    assert process(3).velocity_m_s is None and process(4).position_m is None
    assert process(5).verdict == "reject"
    assert process(7).flights[0].duration_s == pytest.approx(0.25)
    assert process(8).verdict == "reject"
