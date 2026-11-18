"""Independent cases, derivative/convergence checks and model counterexamples."""

import math

import numpy as np
import pytest

from dynamisbench_soccer.reference.clocks import DeviceClock
from dynamisbench_soccer.reference.coordinates import A, F, LocalFrame, ecef_geodetic, geodetic_ecef
from dynamisbench_soccer.reference.kinematics import G, Phase, Trajectory, cmj, hops
from dynamisbench_soccer.reference.motion_models import MonoExponentialSprint
from dynamisbench_soccer.reference.observations import spherical_gate_trigger_s
from dynamisbench_soccer.reference.spatial import (
    EccentricCurve,
    NordicMechanics,
    PlanarPhase,
    PlanarTrajectory,
    RigidBody,
)
from dynamisbench_soccer.reference.spatial_observations import imu_body_observation


def test_initial_fall_and_multiple_rebounds_pair_only_complete_jump_flights() -> None:
    motion = Trajectory(
        75,
        -1,
        (
            Phase("drop", 0.1, -G, False),
            Phase("contact", 0.3, 20, True),
            Phase("rebound", 0.2, -G, False),
            Phase("land", 0.1, 20, True),
        ),
    )
    np.testing.assert_allclose(motion.flight_intervals(), ((0.4, 0.6),), atol=1e-12)
    assert motion.flight_time_s() == pytest.approx(0.2)
    h = hops(5)
    assert len(h.flight_intervals()) == 5
    for takeoff, landing in h.flight_intervals():
        assert landing - takeoff == pytest.approx(4 / G)
        assert h.state(takeoff).velocity_m_s == pytest.approx(2)
        assert h.state(landing).velocity_m_s == pytest.approx(-2)
    assert h.state(h.total_duration_s).velocity_m_s == pytest.approx(0, abs=1e-12)
    assert h.state(h.total_duration_s).displacement_m == pytest.approx(0, abs=1e-12)


def test_grf_and_net_power_are_distinct_and_obey_energy_balance() -> None:
    m = cmj()
    t = 1.6
    state = m.state(t)
    assert m.grf_power_w(t) - m.net_power_w(t) == pytest.approx(75 * G * state.velocity_m_s)
    eps = 1e-5

    def kinetic(t: float) -> float:
        return 75 * m.state(t).velocity_m_s ** 2 / 2

    assert (kinetic(t + eps) - kinetic(t - eps)) / (2 * eps) == pytest.approx(
        m.net_power_w(t), rel=1e-9
    )
    assert m.grf_power_w(m.takeoff_s() + 0.01) == 0
    assert m.net_power_w(m.takeoff_s() + 0.01) < 0


def test_nordic_work_and_torque_closure_with_explicit_actuator() -> None:
    m = NordicMechanics(curve=EccentricCurve(angular_rate_rad_s=1.273372709987525))
    times = np.linspace(0, m.duration_s, 1001)
    resistance = []
    actuator = []
    gravity = []
    for t in times:
        _, r, g, a = m.state(float(t))
        assert g - r + a == pytest.approx(0, abs=1e-12)
        resistance.append(r)
        actuator.append(a)
        gravity.append(g)
    angular_rate = m.curve.angular_rate_rad_s
    assert np.trapezoid(resistance, times) * angular_rate == pytest.approx(
        m.resistance_work_j(), abs=1e-9
    )
    gravity_work = m.segment_mass_kg * G * m.com_radius_m * (1 - math.cos(m.curve.end_angle_rad))
    assert np.trapezoid(gravity, times) * angular_rate == pytest.approx(gravity_work, rel=1e-6)
    assert np.trapezoid(actuator, times) * angular_rate == pytest.approx(
        m.resistance_work_j() - gravity_work, rel=1e-6
    )
    assert m.curve.force_n(m.duration_s) == 0


def test_full_rigid_rotation_derivatives_and_imu_inversion() -> None:
    m = RigidBody(
        PlanarTrajectory((PlanarPhase(4, (1, 2)),)),
        angular_velocity_rad_s=(0.3, -0.5, 0.7),
        sensor_offset_m=(0.2, 0.1, 0.3),
    )
    t, eps = 1.0, 1e-4
    _x, v, a = m.sensor_state(t)
    np.testing.assert_allclose(
        (m.sensor_state(t + eps)[0] - m.sensor_state(t - eps)[0]) / (2 * eps), v, atol=1e-8
    )
    np.testing.assert_allclose(
        (m.sensor_state(t + eps)[1] - m.sensor_state(t - eps)[1]) / (2 * eps), a, atol=1e-8
    )
    r = m.rotation(t)
    np.testing.assert_allclose(r.T @ r, np.eye(3), atol=1e-12)
    specific, gyro = imu_body_observation(m, t)
    np.testing.assert_allclose(r @ specific + [0, 0, -G], a, atol=1e-12)
    np.testing.assert_allclose(gyro, [0.3, -0.5, 0.7])
    x_rot = RigidBody(m.translation, angular_velocity_rad_s=(math.pi / 2, 0, 0)).rotation(1)
    np.testing.assert_allclose(x_rot @ [0, 1, 0], [0, 0, 1], atol=1e-12)


def test_wgs84_known_axes_and_independent_published_topocentric_example() -> None:
    np.testing.assert_allclose(geodetic_ecef(0, 0, 0), [A, 0, 0], atol=1e-9)
    np.testing.assert_allclose(geodetic_ecef(90, 0, 0), [0, 0, A * (1 - F)], atol=1e-9)
    for lat in (-90, -55, 0, 40, 90):
        assert ecef_geodetic(geodetic_ecef(lat, 30, 200)) == pytest.approx((lat, 30, 200), abs=1e-8)
    # PROJ topocentric usage example (GRS80/WGS84 differences < 1 mm here).
    frame = LocalFrame(55, 5, 200)
    np.testing.assert_allclose(
        frame.from_geodetic(53.80939444, 2.12955, 73),
        [-189013.869, -128642.040, -4220.171],
        atol=0.001,
    )
    for point in ([0, 0, 0], [50, -25, 2], [-2000, 1000, 30]):
        np.testing.assert_allclose(
            frame.from_geodetic(*frame.to_geodetic(np.array(point, dtype=np.float64))),
            point,
            atol=3e-9,
        )


def test_clock_synchronization_inverse_and_gate_geometry() -> None:
    clock = DeviceClock(0.125, 80)
    t = np.array([0.0, 1.0, 5.0])
    np.testing.assert_allclose(clock.reference_time(clock.device_time(t)), t, atol=1e-12)
    alignment = clock.alignment(5)
    assert len(alignment.sync_events) == 2
    s = MonoExponentialSprint()
    mid = spherical_gate_trigger_s(s, 10, beam_height_m=1, center_height_m=1, radius_m=0.2)
    high = spherical_gate_trigger_s(s, 10, beam_height_m=1.1, center_height_m=1, radius_m=0.2)
    assert s.position_m(mid) == pytest.approx(9.8, abs=1e-11)
    assert high > mid
    with pytest.raises(ValueError, match="intersect"):
        spherical_gate_trigger_s(s, 10, beam_height_m=2, center_height_m=1, radius_m=0.2)


def test_trapezoid_converges_to_closed_form_sprint_distance() -> None:
    s = MonoExponentialSprint()
    errors = []
    for n in (21, 41, 81):
        t = np.linspace(0, 3, n)
        errors.append(abs(np.trapezoid([s.velocity_m_s(float(v)) for v in t], t) - s.position_m(3)))
    assert 3.9 < errors[0] / errors[1] < 4.1
    assert 3.9 < errors[1] / errors[2] < 4.1


@pytest.mark.parametrize(
    "call",
    [
        lambda: DeviceClock(drift_ppm=-1e6),
        lambda: LocalFrame(91, 0),
        lambda: ecef_geodetic(np.zeros(3)),
        lambda: hops(0),
        lambda: hops(True),
        lambda: RigidBody(
            PlanarTrajectory((PlanarPhase(1, (0, 0)),)),
            initial_rotation=((1, 0, 0), (0, 1, 0), (0, 0, -1)),
        ),
    ],
)
def test_invalid_reference_mechanics_rejected(call) -> None:  # type: ignore[no-untyped-def]
    with pytest.raises(ValueError):
        call()
