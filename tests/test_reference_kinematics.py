"""Independent mechanics invariants for RES-669 truth scenarios."""

import math

import pytest

from dynamisbench_soccer.reference.kinematics import G, Phase, cmj, dj, sj


@pytest.mark.parametrize("factory", [cmj, sj, dj])
def test_takeoff_flight_impulse_and_landing(factory) -> None:
    motion = factory()
    to = motion.takeoff_s()
    land = motion.landing_s()
    assert to < land < motion.total_duration_s
    assert motion.takeoff_velocity_m_s() > 0
    assert motion.flight_time_s() == pytest.approx(
        2 * motion.takeoff_velocity_m_s() / G
    )
    assert motion.jump_height_m() == pytest.approx(
        G * motion.flight_time_s() ** 2 / 8
    )
    assert motion.state(to).vertical_grf_n == 0
    assert motion.state(land).vertical_grf_n > 0
    assert motion.state(land).displacement_m == pytest.approx(
        motion.state(to).displacement_m, abs=1e-12
    )
    delta_v = motion.takeoff_velocity_m_s() - motion.initial_velocity_m_s
    assert motion.impulse_n_s(0.0, to) == pytest.approx(motion.mass_kg * delta_v)


def test_countermovement_is_not_squat_jump() -> None:
    assert cmj().state(1.1).acceleration_m_s2 < 0
    assert sj().state(1.1).acceleration_m_s2 > 0
    assert cmj().jump_height_m() == pytest.approx(2.7**2 / (2 * G))
    assert sj().jump_height_m() == pytest.approx(2.4**2 / (2 * G))


def test_contact_force_nonnegative_and_finite() -> None:
    for motion in (cmj(), sj(), dj()):
        for i in range(100):
            state = motion.state(motion.total_duration_s * (i / 100))
            assert state.vertical_grf_n >= 0
            assert math.isfinite(state.displacement_m)


def test_bad_physics_and_times_fail_closed() -> None:
    with pytest.raises(ValueError, match="normal reaction"):
        Phase("wrong", 1, -G - 1, True).validate()
    with pytest.raises(ValueError, match="ballistic"):
        Phase("fake flight", 1, 0, False).validate()
    with pytest.raises(ValueError, match="outside"):
        cmj().state(-1)
    with pytest.raises(ValueError, match="positive"):
        dj(drop_height_m=-0.1)
