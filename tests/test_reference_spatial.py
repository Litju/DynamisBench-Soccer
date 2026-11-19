"""Independent algebraic checks, not reconstruction through a device engine."""

import math

import numpy as np
import pytest

from dynamisbench_soccer.reference.spatial import (
    EccentricCurve,
    GeodeticTrack,
    PlanarPhase,
    PlanarTrajectory,
    RigidBody,
)


def test_planar_cod_continuity_and_momentum_change() -> None:
    path = PlanarTrajectory((PlanarPhase(1, (2, 0)), PlanarPhase(1, (-2, 2))))
    x, v, a = path.state(2)
    np.testing.assert_allclose(x, [2, 1])
    np.testing.assert_allclose(v, [0, 2])
    np.testing.assert_allclose(a, [-2, 2])
    before = path.state(1 - 1e-9)
    at = path.state(1)
    np.testing.assert_allclose(before[0], at[0], atol=3e-9)
    np.testing.assert_allclose(before[1], at[1], atol=3e-9)


def test_rigid_offset_rotation_velocity_and_centripetal_acceleration() -> None:
    body = RigidBody(
        PlanarTrajectory((PlanarPhase(3, (0, 0)),)), yaw_rate_rad_s=2, sensor_offset_m=(1, 0, 0)
    )
    x, v, a = body.sensor_state(math.pi / 4)
    np.testing.assert_allclose(x, [0, 1, 1], atol=1e-12)
    np.testing.assert_allclose(v, [-2, 0, 0], atol=1e-12)
    np.testing.assert_allclose(a, [0, -4, 0], atol=1e-12)
    assert np.dot(v, a) == pytest.approx(0, abs=1e-12)
    np.testing.assert_allclose(body.rotation(0).T @ body.rotation(0), np.eye(3))


def test_eccentric_breakpoint_and_perpendicular_torque() -> None:
    curve = EccentricCurve()
    assert curve.force_n(0) == 0
    assert curve.force_n(curve.break_time_s) == pytest.approx(400)
    assert curve.torque_nm(curve.break_time_s) == pytest.approx(160)
    assert curve.force_n(curve.end_angle_rad / curve.angular_rate_rad_s) == pytest.approx(0)
    assert curve.force_n(curve.break_time_s / 2) == pytest.approx(200)


def test_geodetic_coordinates_do_not_claim_local_distance() -> None:
    path = GeodeticTrack(40, -3, 100, (0.001, 0.002, 1))
    assert path.position(2) == pytest.approx((40.002, -2.996, 102))
    with pytest.raises(ValueError):
        GeodeticTrack(91, 0, 0)


@pytest.mark.parametrize(
    "factory",
    [
        lambda: PlanarTrajectory(()),
        lambda: PlanarTrajectory((PlanarPhase(0, (0, 0)),)),
        lambda: EccentricCurve(lever_arm_m=0),
        lambda: EccentricCurve(break_angle_rad=2, end_angle_rad=1),
        lambda: RigidBody(PlanarTrajectory((PlanarPhase(1, (0, 0)),)), yaw_rate_rad_s=float("nan")),
    ],
)
def test_invalid_spatial_parameters_rejected(factory) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError):
        factory()


def test_net_sprint_force_power_and_ratio_have_stated_model() -> None:
    from dynamisbench_soccer.reference.motion_models import MonoExponentialSprint

    sprint = MonoExponentialSprint(vmax_m_s=10, tau_s=2)
    assert sprint.f0_n(80) == 400
    assert sprint.pmax_w(80) == 1000
    t = 2 * math.log(2)
    force = 80 * sprint.acceleration_m_s2(t)
    assert force * sprint.velocity_m_s(t) == pytest.approx(sprint.pmax_w(80))
    assert force == pytest.approx(sprint.f0_n(80) * (1 - sprint.velocity_m_s(t) / 10))
    eps = 1e-5
    numerical = (sprint.force_ratio(t + eps) - sprint.force_ratio(t - eps)) / (
        sprint.velocity_m_s(t + eps) - sprint.velocity_m_s(t - eps)
    )
    assert sprint.instantaneous_drf_per_m_s(t) == pytest.approx(numerical, rel=1e-8)
