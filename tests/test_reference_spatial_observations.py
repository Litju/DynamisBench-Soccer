"""Closed-form checks across the newly admitted raw instrument families."""

import ast
import math
from pathlib import Path

import numpy as np
import pytest

from dynamisbench_soccer.reference.kinematics import G, cmj
from dynamisbench_soccer.reference.motion_models import IsometricRamp
from dynamisbench_soccer.reference.spatial import (
    EccentricCurve,
    GeodeticTrack,
    PlanarPhase,
    PlanarTrajectory,
    RigidBody,
)
from dynamisbench_soccer.reference.spatial_observations import (
    additive_noise,
    bilateral_force_n,
    cable_length_m,
    contact_events,
    dual_plate_force_n,
    encoder_counts,
    gnss_fix,
    imu_body_observation,
    lps_position_m,
)


def test_single_dual_plate_and_nordic_hip_force_partition() -> None:
    motion = cmj()
    left, right = dual_plate_force_n(motion, 0.5, 0.4)
    assert left + right == pytest.approx(75 * G)
    assert left == pytest.approx(0.4 * 75 * G)
    for force in [EccentricCurve().force_n(1), IsometricRamp().force_n(1)]:
        a, b = bilateral_force_n(force, 0.3)
        assert a + b == pytest.approx(force)
        assert a == pytest.approx(0.3 * force)


def test_rigid_cable_and_encoder_quantization_bound() -> None:
    body = RigidBody(PlanarTrajectory((PlanarPhase(1, (0, 0)),)), height_m=1)
    assert cable_length_m(body, 0, (1, 0, 0)) == pytest.approx(math.sqrt(2))
    for length in np.linspace(0, 2, 37):
        counts = encoder_counts(float(length), 1000)
        assert abs(counts / 1000 - length) <= 0.0005 + 1e-12
    assert encoder_counts(0.0005, 1000) == 1


def test_contact_switching_latency_changes_observed_flight() -> None:
    motion = cmj()
    events = contact_events(motion, on_latency_s=0.02, off_latency_s=0.01)
    assert events[0] == (0, 1)
    takeoff = next(t for t, s in events if s == 0)
    landing = next(t for t, s in events if s == 1 and t > takeoff)
    assert landing - takeoff == pytest.approx(motion.flight_time_s() + 0.01)


def test_lps_and_gnss_latency_keep_distinct_coordinate_semantics() -> None:
    body = RigidBody(PlanarTrajectory((PlanarPhase(3, (2, 0)),)))
    np.testing.assert_allclose(lps_position_m(body, 2, latency_s=1), [1, 0, 1])
    track = GeodeticTrack(40, -3, 100, (0.001, 0.002, 1))
    assert gnss_fix(track, 2, latency_s=1) == pytest.approx((40.001, -2.998, 101))


def test_imu_gravity_rotation_bias_and_saturation() -> None:
    body = RigidBody(PlanarTrajectory((PlanarPhase(2, (2, 0)),)), yaw0_rad=math.pi / 2)
    specific, gyro = imu_body_observation(body, 0)
    np.testing.assert_allclose(specific, [0, -2, G], atol=1e-12)
    np.testing.assert_allclose(gyro, [0, 0, 0])
    biased, _ = imu_body_observation(body, 0, bias_m_s2=(0, 1, 0), saturation_m_s2=5)
    np.testing.assert_allclose(biased, [0, -1, 5], atol=1e-12)
    rotating = RigidBody(body.translation, yaw_rate_rad_s=2, sensor_offset_m=(1, 0, 0))
    specific, gyro = imu_body_observation(rotating, 0)
    np.testing.assert_allclose(specific, [-2, 0, G])
    np.testing.assert_allclose(gyro, [0, 0, 2])


def test_seeded_noise_reproduces_bytes_and_does_not_mutate_truth() -> None:
    clean = np.arange(10, dtype=np.float64)
    a = additive_noise(clean, 0.1, seed=3)
    assert a.tobytes() == additive_noise(clean, 0.1, seed=3).tobytes()
    assert a.tobytes() != additive_noise(clean, 0.1, seed=4).tobytes()
    np.testing.assert_array_equal(clean, np.arange(10))


def test_import_boundary_between_references_and_engines() -> None:
    root = Path(__file__).resolve().parents[1] / "src/dynamisbench_soccer"
    for path in root.rglob("*.py"):
        parts = path.relative_to(root).parts
        is_reference = parts[0] == "reference"
        is_engine = "engines" in parts
        if not (is_reference or is_engine):
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            modules = []
            if isinstance(node, ast.ImportFrom):
                parent_parts = ("dynamisbench_soccer", *parts[:-1])
                base = (
                    list(parent_parts[: len(parent_parts) - node.level + 1]) if node.level else []
                )
                base += node.module.split(".") if node.module else []
                modules = [".".join(base), *(".".join([*base, alias.name]) for alias in node.names)]
            elif isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            assert not any(
                (is_reference and ".engines" in m) or (is_engine and ".reference" in m)
                for m in modules
            ), path


@pytest.mark.parametrize(
    "call",
    [
        lambda: encoder_counts(1, 0),
        lambda: bilateral_force_n(1, 2),
        lambda: gnss_fix(GeodeticTrack(0, 0, 0), 0, 1),
        lambda: additive_noise(np.array([np.nan]), 1, 1),
        lambda: contact_events(cmj(), on_latency_s=-1),
    ],
)
def test_invalid_observation_conditions_reject(call) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError):
        call()
