"""Exact idealized spatial and eccentric fixtures; no physiological validity claim."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from dynamisbench_soccer.signals.schema import FloatArray


@dataclass(frozen=True)
class PlanarPhase:
    duration_s: float
    acceleration_m_s2: tuple[float, float]


@dataclass(frozen=True)
class PlanarTrajectory:
    """Continuous position/velocity with piecewise constant planar acceleration."""

    phases: tuple[PlanarPhase, ...]
    initial_position_m: tuple[float, float] = (0, 0)
    initial_velocity_m_s: tuple[float, float] = (0, 0)

    def __post_init__(self) -> None:
        if not self.phases or not all(
            math.isfinite(v) for v in (*self.initial_position_m, *self.initial_velocity_m_s)
        ):
            raise ValueError("finite initial state and nonempty phases required")
        if any(
            not math.isfinite(p.duration_s)
            or p.duration_s <= 0
            or not all(math.isfinite(v) for v in p.acceleration_m_s2)
            for p in self.phases
        ):
            raise ValueError("finite acceleration and positive phase duration required")

    @property
    def duration_s(self) -> float:
        return sum(p.duration_s for p in self.phases)

    def state(self, t_s: float) -> tuple[FloatArray, FloatArray, FloatArray]:
        if not math.isfinite(t_s) or not 0 <= t_s <= self.duration_s:
            raise ValueError("time outside planar trajectory")
        x = np.array(self.initial_position_m, dtype=np.float64)
        v = np.array(self.initial_velocity_m_s, dtype=np.float64)
        elapsed = 0.0
        for i, p in enumerate(self.phases):
            a = np.array(p.acceleration_m_s2, dtype=np.float64)
            dt = min(t_s - elapsed, p.duration_s)
            if t_s < elapsed + p.duration_s or i == len(self.phases) - 1:
                return x + v * dt + a * dt**2 / 2, v + a * dt, a
            x += v * dt + a * dt**2 / 2
            v += a * dt
            elapsed += dt
        raise AssertionError("unreachable")


@dataclass(frozen=True)
class RigidBody:
    """Planar translation and constant yaw rate, with a body-fixed sensor offset."""

    translation: PlanarTrajectory
    yaw0_rad: float = 0
    yaw_rate_rad_s: float = 0
    sensor_offset_m: tuple[float, float, float] = (0, 0, 0)
    height_m: float = 1

    def __post_init__(self) -> None:
        if not all(
            math.isfinite(v)
            for v in (self.yaw0_rad, self.yaw_rate_rad_s, self.height_m, *self.sensor_offset_m)
        ):
            raise ValueError("rigid-body parameters must be finite")

    def rotation(self, t_s: float) -> FloatArray:
        self.translation.state(t_s)
        angle = self.yaw0_rad + self.yaw_rate_rad_s * t_s
        c, s = math.cos(angle), math.sin(angle)
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)

    def sensor_state(self, t_s: float) -> tuple[FloatArray, FloatArray, FloatArray]:
        x, v, a = self.translation.state(t_s)
        r = self.rotation(t_s) @ np.array(self.sensor_offset_m)
        omega = np.array([0.0, 0.0, self.yaw_rate_rad_s])
        return (
            np.array([*x, self.height_m]) + r,
            np.array([*v, 0.0]) + np.cross(omega, r),
            np.array([*a, 0.0]) + np.cross(omega, np.cross(omega, r)),
        )


@dataclass(frozen=True)
class EccentricCurve:
    """Prescribed force-vs-angle ramp and release with a known break point."""

    peak_force_n: float = 400
    break_angle_rad: float = math.pi / 4
    end_angle_rad: float = math.pi / 2
    angular_rate_rad_s: float = 0.2
    lever_arm_m: float = 0.4

    def __post_init__(self) -> None:
        values = (
            self.peak_force_n,
            self.break_angle_rad,
            self.end_angle_rad,
            self.angular_rate_rad_s,
            self.lever_arm_m,
        )
        if (
            not all(math.isfinite(v) and v > 0 for v in values)
            or self.break_angle_rad >= self.end_angle_rad
        ):
            raise ValueError("positive finite parameters and interior break point required")

    @property
    def break_time_s(self) -> float:
        return self.break_angle_rad / self.angular_rate_rad_s

    def force_n(self, t_s: float) -> float:
        if not math.isfinite(t_s) or not 0 <= t_s <= self.end_angle_rad / self.angular_rate_rad_s:
            raise ValueError("time outside eccentric curve")
        angle = self.angular_rate_rad_s * t_s
        fraction = (
            angle / self.break_angle_rad
            if angle <= self.break_angle_rad
            else (self.end_angle_rad - angle) / (self.end_angle_rad - self.break_angle_rad)
        )
        return self.peak_force_n * fraction

    def torque_nm(self, t_s: float) -> float:
        """Torque for a force perpendicular to the declared lever arm."""
        return self.force_n(t_s) * self.lever_arm_m


@dataclass(frozen=True)
class GeodeticTrack:
    """Prescribed WGS84 coordinates; deliberately no local-frame conversion."""

    latitude_deg: float
    longitude_deg: float
    height_m: float
    rates: tuple[float, float, float] = (0, 0, 0)

    def __post_init__(self) -> None:
        if not all(
            math.isfinite(v)
            for v in (self.latitude_deg, self.longitude_deg, self.height_m, *self.rates)
        ):
            raise ValueError("geodetic parameters must be finite")
        self.position(0)

    def position(self, t_s: float) -> tuple[float, float, float]:
        if not math.isfinite(t_s) or t_s < 0:
            raise ValueError("time must be finite and nonnegative")
        lat, lon, height = (
            v + r * t_s
            for v, r in zip(
                (self.latitude_deg, self.longitude_deg, self.height_m), self.rates, strict=True
            )
        )
        if not -90 <= lat <= 90 or not -180 <= lon <= 180 or not math.isfinite(height):
            raise ValueError("coordinates outside geodetic domain")
        return lat, lon, height
