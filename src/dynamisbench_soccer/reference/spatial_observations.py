"""Conditional instrument observations of separate mathematical truth scenarios."""

from __future__ import annotations

import math
from itertools import pairwise

import numpy as np

from dynamisbench_soccer.reference.kinematics import G, Trajectory
from dynamisbench_soccer.reference.spatial import GeodeticTrack, RigidBody
from dynamisbench_soccer.signals.schema import FloatArray


def dual_plate_force_n(motion: Trajectory, t_s: float, left_share: float) -> tuple[float, float]:
    if not math.isfinite(left_share) or not 0 <= left_share <= 1:
        raise ValueError("plate share must lie in [0,1]")
    total = motion.state(t_s).vertical_grf_n
    return total * left_share, total * (1 - left_share)


def cable_length_m(body: RigidBody, t_s: float, anchor_m: tuple[float, float, float]) -> float:
    if not all(math.isfinite(v) for v in anchor_m):
        raise ValueError("anchor coordinates must be finite")
    return float(np.linalg.norm(body.sensor_state(t_s)[0] - np.array(anchor_m)))


def encoder_counts(length_m: float, counts_per_m: float) -> int:
    """Ideal nearest-count quantization of nonnegative cable length; ties round up."""
    if (
        not math.isfinite(length_m)
        or length_m < 0
        or not math.isfinite(counts_per_m)
        or counts_per_m <= 0
    ):
        raise ValueError("nonnegative finite length and positive finite resolution required")
    return math.floor(length_m * counts_per_m + 0.5)


def bilateral_force_n(total_n: float, left_share: float) -> tuple[float, float]:
    """Declared isometric/Nordic/hip load split; no vendor correction implied."""
    if (
        not math.isfinite(total_n)
        or total_n < 0
        or not math.isfinite(left_share)
        or not 0 <= left_share <= 1
    ):
        raise ValueError("nonnegative force and share in [0,1] required")
    return total_n * left_share, total_n * (1 - left_share)


def contact_events(
    motion: Trajectory, on_latency_s: float = 0, off_latency_s: float = 0
) -> tuple[tuple[float, int], ...]:
    if not all(math.isfinite(v) and v >= 0 for v in (on_latency_s, off_latency_s)):
        raise ValueError("switch latencies must be finite and nonnegative")
    state = motion.phases[0].in_contact
    events = [(0.0, int(state))]
    elapsed = 0.0
    for previous, current in pairwise(motion.phases):
        elapsed += previous.duration_s
        if current.in_contact != previous.in_contact:
            latency = on_latency_s if current.in_contact else off_latency_s
            events.append((elapsed + latency, int(current.in_contact)))
    if any(b[0] <= a[0] for a, b in pairwise(events)):
        raise ValueError("switch latency reverses event chronology")
    return tuple(events)


def lps_position_m(body: RigidBody, t_s: float, latency_s: float = 0) -> FloatArray:
    if not math.isfinite(latency_s) or latency_s < 0:
        raise ValueError("latency must be finite and nonnegative")
    return body.sensor_state(t_s - latency_s)[0]


def gnss_fix(track: GeodeticTrack, t_s: float, latency_s: float = 0) -> tuple[float, float, float]:
    if not math.isfinite(latency_s) or latency_s < 0:
        raise ValueError("latency must be finite and nonnegative")
    return track.position(t_s - latency_s)


def imu_body_observation(
    body: RigidBody,
    t_s: float,
    *,
    bias_m_s2: tuple[float, float, float] = (0, 0, 0),
    saturation_m_s2: float | None = None,
) -> tuple[FloatArray, FloatArray]:
    """Specific force R.T(a-g) plus declared bias; gyro is body-frame yaw rate."""
    if not all(math.isfinite(v) for v in bias_m_s2):
        raise ValueError("IMU bias must be finite")
    if saturation_m_s2 is not None and (not math.isfinite(saturation_m_s2) or saturation_m_s2 <= 0):
        raise ValueError("saturation must be finite and positive")
    acceleration = body.sensor_state(t_s)[2]
    specific = body.rotation(t_s).T @ (acceleration - np.array([0, 0, -G])) + np.array(bias_m_s2)
    if saturation_m_s2 is not None:
        specific = np.clip(specific, -saturation_m_s2, saturation_m_s2)
    return specific, np.array([0, 0, body.yaw_rate_rad_s], dtype=np.float64)


def additive_noise(values: FloatArray, sd: float, seed: int) -> FloatArray:
    """Seeded Gaussian measurement noise; sd has the same units as the input channel."""
    if not np.all(np.isfinite(values)) or not math.isfinite(sd) or sd < 0:
        raise ValueError("finite values and nonnegative finite noise SD required")
    return values + np.random.default_rng(seed).normal(0, sd, size=values.shape)
