"""Exact force histories and sprint motion, not empirical athlete validation."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class IsometricRamp:
    """Idealized external load-cell force ramp, hold, and release."""

    baseline_n: float = 0.0
    peak_n: float = 2500.0
    onset_s: float = 0.5
    rise_s: float = 0.4
    hold_s: float = 0.7
    release_s: float = 0.3

    def __post_init__(self) -> None:
        values = (
            self.baseline_n,
            self.peak_n,
            self.onset_s,
            self.rise_s,
            self.hold_s,
            self.release_s,
        )
        if not all(math.isfinite(v) for v in values):
            raise ValueError("parameters must be finite")
        if self.peak_n < self.baseline_n or self.baseline_n < 0:
            raise ValueError("nonnegative rising external force required")
        if min(self.onset_s, self.rise_s, self.hold_s, self.release_s) <= 0:
            raise ValueError("positive durations required")

    @property
    def duration_s(self) -> float:
        return self.onset_s + self.rise_s + self.hold_s + self.release_s

    def force_n(self, t_s: float) -> float:
        if not math.isfinite(t_s) or not 0 <= t_s <= self.duration_s:
            raise ValueError("time outside observation")
        rise_end = self.onset_s + self.rise_s
        hold_end = rise_end + self.hold_s
        if t_s <= self.onset_s:
            return self.baseline_n
        if t_s < rise_end:
            fraction = (t_s - self.onset_s) / self.rise_s
            return self.baseline_n + (self.peak_n - self.baseline_n) * fraction
        if t_s <= hold_end:
            return self.peak_n
        fraction = (t_s - hold_end) / self.release_s
        return self.peak_n + (self.baseline_n - self.peak_n) * fraction

    def mean_rfd_n_s(self, window_s: float) -> float:
        if not 0 < window_s <= self.rise_s:
            raise ValueError("RFD window outside force-rise interval")
        increase = self.force_n(self.onset_s + window_s) - self.baseline_n
        return increase / window_s

    def impulse_n_s(self) -> float:
        baseline = self.baseline_n * self.onset_s
        ramp = (self.baseline_n + self.peak_n) * self.rise_s / 2
        hold = self.peak_n * self.hold_s
        release = (self.baseline_n + self.peak_n) * self.release_s / 2
        return baseline + ramp + hold + release


@dataclass(frozen=True)
class MonoExponentialSprint:
    """Exact v(t)=Vmax*(1-exp(-t/tau)), x(0)=0, without an F-V model."""

    vmax_m_s: float = 9.0
    tau_s: float = 1.2

    def __post_init__(self) -> None:
        values = (self.vmax_m_s, self.tau_s)
        if not all(math.isfinite(value) and value > 0 for value in values):
            raise ValueError("sprint parameters must be finite and positive")

    @staticmethod
    def _time(t_s: float) -> None:
        if not math.isfinite(t_s) or t_s < 0:
            raise ValueError("time must be finite and nonnegative")

    def velocity_m_s(self, t_s: float) -> float:
        self._time(t_s)
        return -self.vmax_m_s * math.expm1(-t_s / self.tau_s)

    def acceleration_m_s2(self, t_s: float) -> float:
        self._time(t_s)
        return self.vmax_m_s / self.tau_s * math.exp(-t_s / self.tau_s)

    def position_m(self, t_s: float) -> float:
        self._time(t_s)
        return self.vmax_m_s * (t_s + self.tau_s * math.expm1(-t_s / self.tau_s))

    def gate_time_s(self, distance_m: float) -> float:
        """Numerical inverse of exact x(t), not sensor-specific event detection."""
        if not math.isfinite(distance_m) or distance_m < 0:
            raise ValueError("gate distance must be finite and nonnegative")
        if distance_m == 0:
            return 0.0
        lower, upper = 0.0, max(1.0, distance_m / self.vmax_m_s * 3)
        while self.position_m(upper) < distance_m:
            upper *= 2
        for _ in range(96):
            midpoint = (lower + upper) / 2
            if self.position_m(midpoint) < distance_m:
                lower = midpoint
            else:
                upper = midpoint
        return (lower + upper) / 2

    def f0_n(self, mass_kg: float) -> float:
        """Zero-speed NET horizontal force, under an explicit zero-drag model."""
        if not math.isfinite(mass_kg) or mass_kg <= 0:
            raise ValueError("mass must be finite and positive")
        return mass_kg * self.vmax_m_s / self.tau_s

    def pmax_w(self, mass_kg: float) -> float:
        """Maximum net horizontal power for F(v)=F0*(1-v/Vmax), no air drag."""
        return self.f0_n(mass_kg) * self.vmax_m_s / 4

    def force_ratio(self, t_s: float, gravity_m_s2: float = 9.80665) -> float:
        """Ideal ratio with constant vertical support mg; not a gait-cycle model."""
        if not math.isfinite(gravity_m_s2) or gravity_m_s2 <= 0:
            raise ValueError("gravity must be finite and positive")
        a = self.acceleration_m_s2(t_s)
        return a / math.hypot(a, gravity_m_s2)

    def instantaneous_drf_per_m_s(self, t_s: float, gravity_m_s2: float = 9.80665) -> float:
        """dRF/dv under the ideal model; distinct from a fitted regression slope."""
        self.force_ratio(t_s, gravity_m_s2)
        a = self.acceleration_m_s2(t_s)
        return -(gravity_m_s2**2) / (self.tau_s * math.hypot(a, gravity_m_s2) ** 3)
