"""Ideal device observation projections; no empirical device agreement implied."""

from __future__ import annotations

import math

from dynamisbench_soccer.reference.kinematics import Trajectory
from dynamisbench_soccer.reference.motion_models import (
    IsometricRamp,
    MonoExponentialSprint,
)


def force_plate_n(motion: Trajectory, t_s: float) -> float:
    """Ideal vertical GRF from a known COM trajectory."""
    return motion.state(t_s).vertical_grf_n


def vertical_imu_specific_force_m_s2(motion: Trajectory, t_s: float) -> float:
    """Ideal, aligned sensor: specific force equals a plus gravity."""
    return motion.state(t_s).acceleration_m_s2 + motion.gravity_m_s2


def lpt_cable_length_m(
    motion: Trajectory,
    t_s: float,
    *,
    anchor_x_m: float,
    anchor_z_m: float,
) -> float:
    """Distance to COM-attached point, not vertical bar displacement."""
    if not all(math.isfinite(value) for value in (anchor_x_m, anchor_z_m)):
        raise ValueError("cable geometry must be finite")
    height = motion.state(t_s).displacement_m
    return math.hypot(anchor_x_m, height - anchor_z_m)


def load_cell_voltage(
    force: IsometricRamp,
    t_s: float,
    *,
    volts_per_n: float,
    offset_v: float,
) -> float:
    if not all(math.isfinite(v) for v in (volts_per_n, offset_v)):
        raise ValueError("calibration parameters must be finite")
    if volts_per_n == 0:
        raise ValueError("calibration gain must be nonzero")
    return force.force_n(t_s) * volts_per_n + offset_v


def radar_radial_velocity_m_s(
    sprint: MonoExponentialSprint,
    t_s: float,
    *,
    angle_rad: float,
) -> float:
    if not math.isfinite(angle_rad):
        raise ValueError("radar angle must be finite")
    return sprint.velocity_m_s(t_s) * math.cos(angle_rad)


def laser_distance_m(
    sprint: MonoExponentialSprint,
    t_s: float,
    *,
    offset_m: float = 0.0,
) -> float:
    if not math.isfinite(offset_m):
        raise ValueError("laser offset must be finite")
    return sprint.position_m(t_s) + offset_m


def gate_trigger_s(
    sprint: MonoExponentialSprint,
    distance_m: float,
    *,
    latency_s: float = 0.0,
) -> float:
    if not math.isfinite(latency_s):
        raise ValueError("trigger latency must be finite")
    return sprint.gate_time_s(distance_m) + latency_s
