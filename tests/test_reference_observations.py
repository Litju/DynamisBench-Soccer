"""Analytical scenario and observation-model regression controls."""

import math

import pytest

from dynamisbench_soccer.reference.kinematics import G, cmj
from dynamisbench_soccer.reference.motion_models import IsometricRamp, MonoExponentialSprint
from dynamisbench_soccer.reference.observations import (
    force_plate_n,
    gate_trigger_s,
    laser_distance_m,
    load_cell_voltage,
    lpt_cable_length_m,
    radar_radial_velocity_m_s,
    vertical_imu_specific_force_m_s2,
)


def test_isometric_rfd_and_impulse_are_exact() -> None:
    ramp = IsometricRamp()
    assert ramp.force_n(ramp.onset_s) == 0
    assert ramp.force_n(ramp.onset_s + ramp.rise_s) == 2500
    assert ramp.mean_rfd_n_s(0.1) == pytest.approx(6250)
    assert ramp.impulse_n_s() == pytest.approx(2500 * (0.4 / 2 + 0.7 + 0.3 / 2))
    assert load_cell_voltage(
        ramp, ramp.onset_s + ramp.rise_s, volts_per_n=0.001, offset_v=0.2
    ) == pytest.approx(2.7)


def test_sprint_model_and_numerical_gate_inverse() -> None:
    sprint = MonoExponentialSprint()
    assert sprint.position_m(0.0) == 0
    assert sprint.velocity_m_s(0.0) == 0
    assert sprint.acceleration_m_s2(0.0) == 9 / 1.2
    for distance in (5.0, 10.0, 20.0, 30.0):
        t_s = sprint.gate_time_s(distance)
        assert sprint.position_m(t_s) == pytest.approx(distance, abs=1e-11)
        assert laser_distance_m(sprint, t_s, offset_m=1.5) == pytest.approx(distance + 1.5)
        assert gate_trigger_s(sprint, distance, latency_s=0.015) == pytest.approx(t_s + 0.015)


def test_radar_radial_projection_is_not_gnss_speed() -> None:
    sprint = MonoExponentialSprint()
    velocity = sprint.velocity_m_s(2.0)
    result = radar_radial_velocity_m_s(sprint, 2.0, angle_rad=math.pi / 3)
    assert result == pytest.approx(velocity / 2)


def test_observation_models_preserve_distinct_measurands() -> None:
    motion = cmj()
    assert force_plate_n(motion, motion.takeoff_s()) == 0
    assert vertical_imu_specific_force_m_s2(motion, motion.takeoff_s()) == pytest.approx(
        0, abs=1e-10
    )
    sample_t = 1.1
    cable_length = lpt_cable_length_m(motion, sample_t, anchor_x_m=1, anchor_z_m=-1)
    assert cable_length != pytest.approx(motion.state(sample_t).displacement_m)
    assert force_plate_n(motion, 0.5) == pytest.approx(motion.mass_kg * G)


def test_invalid_observation_parameters_rejected() -> None:
    with pytest.raises(ValueError):
        IsometricRamp(peak_n=-1)
    with pytest.raises(ValueError):
        MonoExponentialSprint(tau_s=0)
    with pytest.raises(ValueError):
        load_cell_voltage(IsometricRamp(), 1, volts_per_n=0, offset_v=0)
    with pytest.raises(ValueError):
        gate_trigger_s(MonoExponentialSprint(), -1)
