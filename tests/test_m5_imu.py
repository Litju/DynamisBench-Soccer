"""Native IMU replay against the prespecified instrument-point and signature models."""

import json
import math
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

from dynamisbench_soccer.engines import imu
from dynamisbench_soccer.m4.measurement import IMU_POSITION, IMU_SIGNATURE, replay
from dynamisbench_soccer.m4.task_registry import load_registry
from dynamisbench_soccer.signals.schema import BundleMeta, SignalBundle

ROOT = Path(__file__).resolve().parents[1]
TASK = "imu-frame-observability"


def _fixture(index):
    return json.loads((ROOT / f"benchmark/fixtures/m5-motion/{TASK}-{index:02}.json").read_bytes())


def _clock_time(index):
    return index / 20


def _instrument_x(t):
    return 1.5 + 0.3 * t + t**2


def _signature_span():
    first, last = 1, 6
    return (last - first) / 20, 2 / 20


def _source_digest(channels):
    return sha256((json.dumps(channels, sort_keys=True, indent=2) + "\n").encode()).hexdigest()


def _rehash(data):
    sha = _source_digest(data["channels"])
    data["meta"]["acquisition"]["sources"][0]["source_sha256"] = sha
    data["meta"]["provenance"]["source_sha256"] = sha


def test_frozen_scenarios_source_provenance_and_projection():
    tasks = load_registry(ROOT / "benchmark/tasks")
    original, task = tasks[TASK, 1], tasks[TASK, 2]
    assert task.artifacts == original.artifacts and task.task.convention == "defended"
    raw = {item.name: (ROOT / item.path).read_bytes() for item in task.artifacts}
    released = set()
    for previous, current in zip(original.checkpoints, task.checkpoints, strict=True):
        assert (current.id, current.artifacts, current.corrects) == (
            previous.id,
            previous.artifacts,
            previous.corrects,
        )
        assert current.prompt.startswith(previous.prompt)
        (artifact,) = current.artifacts
        released.add(artifact)
        public = json.loads(task.candidate_payload(current.id, raw, 704))
        assert set(public["inputs"]) == released and set(public) == {
            "task",
            "checkpoint",
            "inputs",
            "seed",
        }
        data = json.loads(raw[artifact])
        source = data["meta"]["acquisition"]["sources"][0]
        assert source["source_sha256"] == data["meta"]["provenance"]["source_sha256"]
        assert source["source_sha256"] == _source_digest(data["channels"])


def test_native_position_calibration_integration_and_bounds():
    data = _fixture(1)
    assert data["channels"]["time"] == [_clock_time(index) for index in range(9)]
    result = replay(IMU_POSITION, data)
    assert result.reason == "eligible"
    assert result.value == pytest.approx(_instrument_x(0.4), abs=1e-9, rel=0)
    assert result.radius == pytest.approx(0.04 * 0.4**2 / 2, abs=1e-9, rel=0)
    assert dict(result.operands) == pytest.approx(
        {
            "free_ax_m_s2": 2,
            "free_ay_m_s2": -1,
            "free_az_m_s2": 1.2,
            "terminal_vx_m_s": 0.3 + 2 * 0.4,
            "elapsed_s": 0.4,
            "peak_corrected_gyro_rad_s": 0,
        },
        abs=1e-9,
        rel=0,
    )
    data["spec"]["accel_misalignment"][0][1] = 0
    uncorrected = replay(IMU_POSITION, data)
    assert uncorrected.value == pytest.approx(result.value)
    assert dict(uncorrected.operands)["free_ay_m_s2"] == pytest.approx(-1.4)


def _equal_observations():
    gravity = 9.80665
    sine = 3 / gravity
    cosine = math.sqrt(1 - sine**2)
    # Both states start with the declared velocity. One is unaccelerated.
    return (
        ((1, 0, 0), (0, cosine, -sine), (0, sine, cosine)),
        ((0, -1, 0), (1, 0, 0), (0, 0, 1)),
        (0, 3, gravity * cosine),
    )


def test_same_observation_allows_distinct_field_acceleration():
    data = _fixture(2)
    meta = BundleMeta.model_validate(data["meta"])
    bundle = SignalBundle(
        meta, {c.name: np.asarray(data["channels"][c.name]) for c in meta.channels}
    )
    tilt, yaw, observation = _equal_observations()
    spec = imu.ImuSpec.model_validate(data["spec"])
    assert np.linalg.norm(observation) == pytest.approx(9.80665, abs=1e-12)
    for rotation, free in ((tilt, (0, 0, 0)), (yaw, (-3, 0, observation[2] - 9.80665))):
        checked = imu.process(bundle, spec.model_copy(update={"initial_orientation": rotation}))
        np.testing.assert_allclose(checked.specific_force_m_s2, [observation] * 9, atol=1e-12)
        np.testing.assert_allclose(checked.free_acceleration_m_s2, [free] * 9, atol=1e-12)
        np.testing.assert_allclose(checked.angular_velocity_rad_s, 0, atol=1e-12)
    assert replay(IMU_POSITION, data).claims == (
        "initial_attitude_unestablished",
        "gravity_norm_not_static",
    )


@pytest.mark.parametrize(
    ("index", "reason", "finding"),
    [
        (2, "insufficient_evidence", "initial_attitude_unestablished"),
        (3, "insufficient_evidence", "initial_velocity_unestablished"),
        (4, "insufficient_evidence", "initial_position_unestablished"),
        (5, "not_eligible", "accelerometer_rail_clipping"),
        (6, "unsupported_claim", "instrument_point_not_com"),
        (8, "not_eligible", "native_clock_gap"),
    ],
)
def test_frozen_imu_refusal(index, reason, finding):
    result = replay(IMU_SIGNATURE if index == 8 else IMU_POSITION, _fixture(index))
    assert result.reason == reason and result.claims[0] == finding
    assert result.value is result.radius is None and not result.operands


def test_signature_uses_complete_outer_edges_and_conditional_radius():
    data = _fixture(7)
    result = replay(IMU_SIGNATURE, data)
    duration, radius = _signature_span()
    assert result.reason == "eligible"
    assert result.value == pytest.approx(duration, abs=1e-9, rel=0)
    assert result.radius == pytest.approx(radius, abs=1e-9, rel=0)
    assert dict(result.operands) == {
        "takeoff_edge_s": 0.05,
        "landing_edge_s": 0.3,
        "threshold_m_s2": 3,
    }
    del data["spec"]["initial_orientation"]
    assert replay(IMU_SIGNATURE, data) == result
    data["flight_min_s"] = 0.3
    assert replay(IMU_SIGNATURE, data).claims == ("incomplete_signature",)


@pytest.mark.parametrize("indices", [(0, 1), (8,), (7,)])
def test_signature_refuses_missing_bracket_or_second_run(indices):
    data = _fixture(7)
    for name in ("specific_force_x", "specific_force_y", "specific_force_z"):
        for index in indices:
            data["channels"][name][index] = data["channels"][name][2]
    _rehash(data)
    assert replay(IMU_SIGNATURE, data).claims == ("incomplete_signature",)


def test_native_contract_rejects_changed_digest_axis_clock_mount_calibration_and_motion():
    data = _fixture(1)
    data["channels"]["angular_velocity_x"][0] += 0.01
    with pytest.raises(ValueError, match="provenance digests"):
        replay(IMU_POSITION, data)
    _rehash(data)
    assert replay(IMU_POSITION, data).claims == ("fixed_attitude_model_unavailable",)
    data = _fixture(1)
    data["meta"]["acquisition"]["alignments"][0]["sync_events"][1] = [0, 0]
    assert replay(IMU_POSITION, data).claims == ("native_clock_gap",)
    data = _fixture(1)
    data["spec"]["placement"] = "shank"
    assert replay(IMU_POSITION, data).claims == ("mounting_or_frame_unestablished",)
    data["spec"]["accel_scale"][0] = 0
    assert replay(IMU_POSITION, data).claims == ("invalid_calibration",)
    data = _fixture(1)
    channels = data["meta"]["channels"]
    channels[1]["axis"], channels[2]["axis"] = channels[2]["axis"], channels[1]["axis"]
    with pytest.raises(ValueError, match="canonical sensor axes"):
        replay(IMU_POSITION, data)


def test_private_episode_rejects_method_substitution_and_future_incompatible_keys():
    from dynamisbench_soccer.m4.custody import AnswerEnvelope
    from dynamisbench_soccer.m4.grader import _private_methods
    from dynamisbench_soccer.m4.measurement import checkpoint_methods
    from dynamisbench_soccer.m4.task_registry import fingerprint

    task = load_registry(ROOT / "benchmark/tasks")[TASK, 2]
    positions, signatures = task.checkpoints[:6], task.checkpoints[6:]
    assert all(checkpoint_methods(task, row.id)[0].id == IMU_POSITION for row in positions)
    assert all(checkpoint_methods(task, row.id)[0].id == IMU_SIGNATURE for row in signatures)
    # A structurally valid refusal serves as a type and domain test without any operator keys.
    refusal = {
        "decision": "refuse",
        "reasons": ["not_eligible"],
        "selected_trials": [],
        "required_citations": ["DEV-002"],
        "allowed_citations": ["DEV-002"],
        "required_claims": [],
        "allowed_claims": [],
        "quantity": None,
        "operands": [],
    }
    keys = {row.id: {checkpoint_methods(task, row.id)[0].id: refusal} for row in task.checkpoints}
    envelope = AnswerEnvelope(task_sha256=fingerprint(task), nonce="a" * 64, checkpoints=keys)
    assert tuple(_private_methods(task, envelope, positions[0])) == (IMU_POSITION,)
    assert tuple(_private_methods(task, envelope, signatures[0])) == (IMU_SIGNATURE,)
    keys["c07"] = {IMU_POSITION: refusal}
    with pytest.raises(ValueError, match="methods disagree"):
        _private_methods(task, envelope.model_copy(update={"checkpoints": keys}), positions[0])
    keys["c07"] = {IMU_SIGNATURE: refusal, IMU_POSITION: refusal}
    with pytest.raises(ValueError, match="methods disagree"):
        _private_methods(task, envelope.model_copy(update={"checkpoints": keys}), positions[0])
    keys.pop("c07")
    with pytest.raises(ValueError, match="cover the declared episode"):
        _private_methods(task, envelope.model_copy(update={"checkpoints": keys}), positions[0])
