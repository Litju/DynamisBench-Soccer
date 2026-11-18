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
        if len(self.initial_position_m) != 2 or len(self.initial_velocity_m_s) != 2:
            raise ValueError("planar state must have two components")
        if not self.phases or not all(
            math.isfinite(v) for v in (*self.initial_position_m, *self.initial_velocity_m_s)
        ):
            raise ValueError("finite initial state and nonempty phases required")
        if any(
            len(p.acceleration_m_s2) != 2
            or not math.isfinite(p.duration_s)
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
    angular_velocity_rad_s: tuple[float, float, float] | None = None
    initial_rotation: tuple[
        tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]
    ] = ((1, 0, 0), (0, 1, 0), (0, 0, 1))

    def __post_init__(self) -> None:
        r = np.array(self.initial_rotation, dtype=np.float64)
        if (
            r.shape != (3, 3)
            or not np.all(np.isfinite(r))
            or not np.allclose(r.T @ r, np.eye(3), atol=1e-12, rtol=0)
            or not np.isclose(np.linalg.det(r), 1, atol=1e-12, rtol=0)
        ):
            raise ValueError("initial rotation must be proper orthonormal")
        if self.angular_velocity_rad_s is not None and self.yaw_rate_rad_s != 0:
            raise ValueError("declare angular velocity or legacy yaw rate, not both")
        if (
            len(self.sensor_offset_m) != 3
            or len(self.omega_body()) != 3
            or not all(
                math.isfinite(v)
                for v in (
                    self.yaw0_rad,
                    self.yaw_rate_rad_s,
                    self.height_m,
                    *self.sensor_offset_m,
                    *self.omega_body(),
                )
            )
        ):
            raise ValueError("rigid-body parameters must be finite")

    def omega_body(self) -> tuple[float, float, float]:
        return self.angular_velocity_rad_s or (0, 0, self.yaw_rate_rad_s)

    def rotation(self, t_s: float) -> FloatArray:
        self.translation.state(t_s)
        c, s = math.cos(self.yaw0_rad), math.sin(self.yaw0_rad)
        base: FloatArray = np.array(
            [[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64
        ) @ np.array(self.initial_rotation, dtype=np.float64)
        omega = np.array(self.omega_body(), dtype=np.float64)
        rate = float(np.linalg.norm(omega))
        if rate == 0:
            return base
        x, y, z = map(float, omega / rate)
        skew: FloatArray = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=np.float64)
        angle = rate * t_s
        return base @ (np.eye(3) + math.sin(angle) * skew + (1 - math.cos(angle)) * (skew @ skew))

    def sensor_state(self, t_s: float) -> tuple[FloatArray, FloatArray, FloatArray]:
        x, v, a = self.translation.state(t_s)
        r = self.rotation(t_s) @ np.array(self.sensor_offset_m)
        omega = self.rotation(t_s) @ np.array(self.omega_body())
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
        return self.peak_force_n * min(1.0, max(0.0, fraction))

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


@dataclass(frozen=True)
class NordicMechanics:
    """Prescribed rigid-link descent with explicit actuator balancing the load curve.

    Gravity - knee resistance + actuator = I*alpha = 0. This is a bench mechanics
    scenario, not a model of a human hamstring's capacity or post-break motion.
    """

    curve: EccentricCurve = EccentricCurve()
    segment_mass_kg: float = 40
    com_radius_m: float = 0.5
    inertia_kg_m2: float = 12

    def __post_init__(self) -> None:
        if not all(
            math.isfinite(v) and v > 0
            for v in (self.segment_mass_kg, self.com_radius_m, self.inertia_kg_m2)
        ):
            raise ValueError("positive finite segment mechanics required")

    @property
    def duration_s(self) -> float:
        return self.curve.end_angle_rad / self.curve.angular_rate_rad_s

    def state(self, t_s: float) -> tuple[float, float, float, float]:
        from dynamisbench_soccer.reference.kinematics import G

        resistance = self.curve.torque_nm(t_s)
        angle = min(self.curve.end_angle_rad, self.curve.angular_rate_rad_s * t_s)
        gravity = self.segment_mass_kg * G * self.com_radius_m * math.sin(angle)
        return angle, resistance, gravity, resistance - gravity

    def resistance_work_j(self) -> float:
        return self.curve.peak_force_n * self.curve.lever_arm_m * self.curve.end_angle_rad / 2
