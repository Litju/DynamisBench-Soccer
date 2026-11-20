"""Change-of-direction metrics on declared paths (RES-673).

A COD metric is only defined relative to a declared path: its length, turn position and
turn angle are mandatory declarations. Continuous-speed sources give entry, minimum and exit
speeds; gate splits give segment times only (never speeds between gates). The COD deficit
needs an explicit straight-sprint reference time over the same distance.
"""

from __future__ import annotations

import itertools
import math
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from dynamisbench_soccer.assessment.sprint import (
    StartSpec,
    instantaneous_velocity,
    start_time,
)
from dynamisbench_soccer.signals.schema import FloatArray

COD_LIMITATIONS: tuple[str, ...] = (
    "COD metrics are defined only for the declared path (length, turn distance, turn angle); "
    "a different path or turn angle gives different quantities.",
    "Entry, minimum and exit speeds come from a continuous speed source and keep that "
    "source's estimand; gate-based tests give segment times and no speeds.",
    "The COD deficit is total time minus a declared straight-sprint reference time for the "
    "same distance; it is refused without that reference and is not corrected for start type.",
    "Turn events are speed-trace definitions (minimum speed in the turn window), not "
    "foot-contact or kinematic events.",
    "Engineering calculations only: no scientific qualification of any COD metric.",
)


class CodSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    declared_path: str = Field(min_length=3)
    path_length_m: float = Field(gt=0)
    turn_distance_m: float = Field(gt=0)
    turn_angle_deg: float = Field(gt=0, le=180)
    entry_distance_m: float = Field(default=2.0, gt=0)
    turn_window_m: float = Field(default=3.0, gt=0)
    regain_fraction: float = Field(default=0.9, gt=0, lt=1)
    straight_reference_time_s: float | None = Field(default=None, gt=0)
    source: Literal["radar", "laser", "gnss_doppler", "gnss_position", "lps_position"]

    @model_validator(mode="after")
    def valid(self) -> CodSpec:
        if not self.entry_distance_m < self.turn_distance_m < self.path_length_m:
            raise ValueError("entry distance < turn distance < path length required")
        return self


def assess_trace(
    time: FloatArray, speed: FloatArray, spec: CodSpec, start: StartSpec
) -> dict[str, object]:
    instantaneous_velocity(spec.source)
    if (
        time.ndim != 1
        or time.shape != speed.shape
        or len(time) < 3
        or not np.all(np.isfinite(time))
        or not np.all(np.diff(time) > 0)
        or not np.all(np.isfinite(speed))
        or np.any(speed < 0)
    ):
        raise ValueError("COD requires finite increasing time and finite nonnegative speed")
    t0 = start_time(time, speed, start)
    keep = time > t0
    t = np.concatenate(([t0], time[keep]))
    v = np.concatenate(([np.interp(t0, time, speed)], speed[keep]))
    distance = np.concatenate(([0.0], np.cumsum(0.5 * (v[1:] + v[:-1]) * np.diff(t))))
    if distance[-1] < spec.path_length_m:
        raise ValueError("trace ends before the declared path length")
    refusals: dict[str, str] = {}

    def at(d: float) -> float:
        # Invert the integral of linear speed within the crossing interval.
        right = int(np.searchsorted(distance, d))
        left = right - 1
        delta = d - distance[left]
        acceleration = (v[right] - v[left]) / (t[right] - t[left])
        crossing_speed = math.sqrt(max(0.0, v[left] ** 2 + 2 * acceleration * delta))
        return float(t[left] + 2 * delta / (v[left] + crossing_speed))

    finish = at(spec.path_length_m)
    in_course = t < finish
    t = np.concatenate((t[in_course], [finish]))
    v = np.concatenate((v[in_course], [np.interp(finish, time, speed)]))
    distance = np.concatenate((distance[in_course], [spec.path_length_m]))

    turn_idx = np.flatnonzero(
        (distance > 0)
        & (distance < spec.path_length_m)
        & (distance >= spec.turn_distance_m - spec.turn_window_m / 2)
        & (distance <= spec.turn_distance_m + spec.turn_window_m / 2)
    )
    if len(turn_idx) < 3:
        raise ValueError("too few samples in the turn window")
    k = int(turn_idx[np.argmin(v[turn_idx])])
    pre = slice(0, k + 1)
    post = slice(k, len(v))
    pre_peak_i = int(np.argmax(v[pre]))
    post_peak = float(v[post].max())
    regain = np.flatnonzero(v[post] >= spec.regain_fraction * post_peak)
    total = finish - t0
    out: dict[str, object] = {
        "declared_path": spec.declared_path,
        "source": spec.source,
        "turn_angle_deg": spec.turn_angle_deg,
        "total_time_s": total,
        "turn_time_s": float(t[k] - t0),
        "turn_distance_m": float(distance[k]),
        "minimum_speed_m_s": float(v[k]),
        "pre_turn_peak_speed_m_s": float(v[pre_peak_i]),
        "entry_speed_m_s": float(np.interp(at(spec.turn_distance_m - spec.entry_distance_m), t, v)),
        "post_turn_peak_speed_m_s": post_peak,
        "deceleration_time_s": float(t[k] - t[pre_peak_i]),
        "reacceleration_time_s": float(t[k + int(regain[0])] - t[k]) if len(regain) else None,
        "speed_loss_fraction": float(1 - v[k] / v[pre_peak_i]),
        "start": start.kind,
        "milestones_s": {
            f"{spec.turn_distance_m:g}_m": at(spec.turn_distance_m) - t0,
            f"{spec.path_length_m:g}_m": total,
        },
    }
    if spec.straight_reference_time_s is None:
        refusals["cod_deficit"] = "no declared straight-sprint reference time: deficit refused"
        out["cod_deficit_s"] = None
    else:
        out["cod_deficit_s"] = total - spec.straight_reference_time_s
        out["straight_reference_time_s"] = spec.straight_reference_time_s
    out["refusals"] = refusals
    out["limitations"] = list(COD_LIMITATIONS)
    return out


def assess_gates(
    gate_positions_m: list[float], cumulative_times_s: list[float], spec: CodSpec
) -> dict[str, object]:
    """Segment times for a gate-timed COD test; speeds between gates are refused."""
    if len(gate_positions_m) != len(cumulative_times_s) or len(gate_positions_m) < 2:
        raise ValueError("paired gate positions and times (at least two) are required")
    if not all(map(math.isfinite, [*gate_positions_m, *cumulative_times_s])):
        raise ValueError("gate positions and times must be finite")
    if gate_positions_m[0] != 0 or cumulative_times_s[0] != 0:
        raise ValueError("full-course gates must start at position zero and elapsed time zero")
    if any(b <= a for a, b in itertools.pairwise(gate_positions_m)):
        raise ValueError("gate positions must increase")
    if any(b <= a for a, b in itertools.pairwise(cumulative_times_s)):
        raise ValueError("gate times must increase")
    if abs(gate_positions_m[-1] - spec.path_length_m) > 1e-9:
        raise ValueError("the last gate must sit at the declared path length")
    segments = [
        {
            "from_m": a,
            "to_m": b,
            "time_s": t2 - t1,
            "segment_average_speed_m_s": (b - a) / (t2 - t1),
        }
        for (a, t1), (b, t2) in zip(
            zip(gate_positions_m, cumulative_times_s, strict=True),
            list(zip(gate_positions_m, cumulative_times_s, strict=True))[1:],
            strict=False,
        )
    ]
    total = cumulative_times_s[-1] - cumulative_times_s[0]
    return {
        "declared_path": spec.declared_path,
        "source": "gates",
        "total_time_s": total,
        "segments": segments,
        "entry_speed_m_s": None,
        "minimum_speed_m_s": None,
        "refusals": {
            "speeds": "gate segment averages only; entry/minimum/exit speeds are not manufactured"
        },
        "cod_deficit_s": None
        if spec.straight_reference_time_s is None
        else total - spec.straight_reference_time_s,
        "limitations": list(COD_LIMITATIONS),
    }
