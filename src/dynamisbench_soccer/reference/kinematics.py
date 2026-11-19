"""Exact one-dimensional analytical motion fixtures (not athlete physiology)."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

G = 9.80665


@dataclass(frozen=True)
class Phase:
    name: str
    duration_s: float
    acceleration_m_s2: float
    in_contact: bool

    def validate(self, g: float = G) -> None:
        if not math.isfinite(self.duration_s) or self.duration_s <= 0:
            raise ValueError("phase duration must be finite and positive")
        if not math.isfinite(self.acceleration_m_s2):
            raise ValueError("phase acceleration must be finite")
        if self.in_contact and self.acceleration_m_s2 < -g:
            raise ValueError("normal reaction force cannot be negative")
        if not self.in_contact and not math.isclose(self.acceleration_m_s2, -g, abs_tol=1e-12):
            raise ValueError("ballistic flight requires acceleration = -gravity")


@dataclass(frozen=True)
class State:
    time_s: float
    displacement_m: float
    velocity_m_s: float
    acceleration_m_s2: float
    vertical_grf_n: float
    phase_name: str


@dataclass(frozen=True)
class Trajectory:
    mass_kg: float
    initial_velocity_m_s: float
    phases: tuple[Phase, ...]
    gravity_m_s2: float = G
    assessment_id: Literal["cmj", "sj", "dj", "rj_10_5"] | None = None
    prescribed_drop_height_m: float | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.mass_kg) or self.mass_kg <= 0:
            raise ValueError("mass must be finite and positive")
        if not math.isfinite(self.initial_velocity_m_s):
            raise ValueError("initial velocity must be finite")
        if not math.isfinite(self.gravity_m_s2) or self.gravity_m_s2 <= 0:
            raise ValueError("gravity must be finite and positive")
        if not self.phases:
            raise ValueError("at least one phase is required")
        for phase in self.phases:
            phase.validate(self.gravity_m_s2)

    @property
    def total_duration_s(self) -> float:
        return sum(p.duration_s for p in self.phases)

    def state(self, t_s: float) -> State:
        if not math.isfinite(t_s) or not 0 <= t_s <= self.total_duration_s + 1e-10:
            raise ValueError("time outside analytical trajectory")
        t0 = 0.0
        x0 = 0.0
        v0 = self.initial_velocity_m_s
        for idx, phase in enumerate(self.phases):
            end = t0 + phase.duration_s
            if t_s < end - 1e-12 or idx == len(self.phases) - 1:
                dt = min(max(t_s - t0, 0.0), phase.duration_s)
                x = x0 + v0 * dt + 0.5 * phase.acceleration_m_s2 * dt * dt
                v = v0 + phase.acceleration_m_s2 * dt
                if phase.in_contact:
                    force = self.mass_kg * (self.gravity_m_s2 + phase.acceleration_m_s2)
                else:
                    force = 0.0
                return State(t_s, x, v, phase.acceleration_m_s2, force, phase.name)
            x0 += v0 * phase.duration_s + 0.5 * phase.acceleration_m_s2 * phase.duration_s**2
            v0 += phase.acceleration_m_s2 * phase.duration_s
            t0 = end
        raise AssertionError("unreachable")

    def takeoff_s(self) -> float:
        for idx in range(len(self.phases) - 1):
            if self.phases[idx].in_contact and not self.phases[idx + 1].in_contact:
                return sum(p.duration_s for p in self.phases[: idx + 1])
        raise ValueError("trajectory has no take-off event")

    def flight_intervals(self) -> tuple[tuple[float, float], ...]:
        """Paired complete flights; an initial fall is not a rebound jump."""
        elapsed = 0.0
        takeoff = None
        intervals = []
        for previous, current in zip(self.phases, self.phases[1:], strict=False):
            elapsed += previous.duration_s
            if previous.in_contact and not current.in_contact:
                takeoff = elapsed
            elif not previous.in_contact and current.in_contact and takeoff is not None:
                intervals.append((takeoff, elapsed))
                takeoff = None
        return tuple(intervals)

    def landing_s(self) -> float:
        intervals = self.flight_intervals()
        if not intervals:
            raise ValueError("trajectory has no complete jump flight")
        return intervals[0][1]

    def grf_power_w(self, t_s: float) -> float:
        state = self.state(t_s)
        return state.vertical_grf_n * state.velocity_m_s

    def net_power_w(self, t_s: float) -> float:
        state = self.state(t_s)
        return self.mass_kg * state.acceleration_m_s2 * state.velocity_m_s

    def takeoff_velocity_m_s(self) -> float:
        return self.state(self.takeoff_s()).velocity_m_s

    def flight_time_s(self) -> float:
        return self.landing_s() - self.takeoff_s()

    def jump_height_m(self) -> float:
        velocity = self.takeoff_velocity_m_s()
        if velocity <= 0:
            raise ValueError("jump height requires upward take-off velocity")
        return velocity**2 / (2 * self.gravity_m_s2)

    def impulse_n_s(self, start_s: float, end_s: float) -> float:
        if end_s < start_s:
            raise ValueError("impulse interval reversed")
        return self.mass_kg * (self.state(end_s).velocity_m_s - self.state(start_s).velocity_m_s)


def _flight_and_land(takeoff_v_m_s: float, g: float = G) -> tuple[Phase, ...]:
    if takeoff_v_m_s <= 0:
        raise ValueError("take-off velocity must be positive")
    return (
        Phase("flight", 2 * takeoff_v_m_s / g, -g, False),
        Phase("landing_brake", 0.20, takeoff_v_m_s / 0.20, True),
        Phase("quiet_after", 0.30, 0.0, True),
    )


def cmj(mass_kg: float = 75.0) -> Trajectory:
    """Static baseline, unweighting, braking and propulsion."""
    return Trajectory(
        mass_kg,
        0.0,
        (
            Phase("quiet", 1.0, 0.0, True),
            Phase("unweight", 0.25, -2.0, True),
            Phase("brake", 0.2, 2.5, True),
            Phase("propel", 0.3, 9.0, True),
            *_flight_and_land(2.7),
        ),
        assessment_id="cmj",
    )


def sj(mass_kg: float = 75.0) -> Trajectory:
    """Static crouch and propulsion without prior unweighting."""
    return Trajectory(
        mass_kg,
        0.0,
        (
            Phase("quiet", 1.0, 0.0, True),
            Phase("propel", 0.3, 8.0, True),
            *_flight_and_land(2.4),
        ),
        assessment_id="sj",
    )


def dj(mass_kg: float = 75.0, drop_height_m: float = 0.30) -> Trajectory:
    """Contact/rebound from independently known impact velocity."""
    if not math.isfinite(drop_height_m) or drop_height_m <= 0:
        raise ValueError("drop height must be positive and finite")
    impact_speed = math.sqrt(2 * G * drop_height_m)
    rebound_speed = 2.6
    contact_s = 0.18
    acceleration = (rebound_speed + impact_speed) / contact_s
    return Trajectory(
        mass_kg,
        -impact_speed,
        (
            Phase("rebound_contact", contact_s, acceleration, True),
            *_flight_and_land(rebound_speed),
        ),
        assessment_id="dj",
        prescribed_drop_height_m=drop_height_m,
    )


def hops(count: int = 5, mass_kg: float = 75.0) -> Trajectory:
    """Repeated ideal rebounds; each contact reverses incoming velocity to +2 m/s."""
    if isinstance(count, bool) or not isinstance(count, int) or not 1 <= count <= 100:
        raise ValueError("hop count must be an integer in [1,100]")
    v, contact_s = 2.0, 0.2
    phases = [Phase("initial_propulsion", contact_s, v / contact_s, True)]
    for i in range(count):
        phases.append(Phase(f"flight_{i}", 2 * v / G, -G, False))
        if i < count - 1:
            phases.append(Phase(f"rebound_{i}", contact_s, 2 * v / contact_s, True))
    phases += [Phase("final_brake", contact_s, v / contact_s, True)]
    return Trajectory(mass_kg, 0, tuple(phases), assessment_id="rj_10_5")
