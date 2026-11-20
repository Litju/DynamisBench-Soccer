"""Sprint timing, velocity interpretation and conditional F-V modelling (RES-673).

Sources keep their own estimands. Gate splits are segment averages and never become
instantaneous velocity; Doppler radar speed, laser range-rate, GNSS Doppler speed and
GNSS/LPS positional speed are separate definitions and are labelled, not merged. The
force-velocity model is conditional: mono-exponential, flat, zero air drag, net horizontal
force. Fit diagnostics and parameter identifiability decide whether results are reported or
refused. No analytical generator is imported: expected values belong in independent tests.
"""

from __future__ import annotations

import itertools
import json
import math
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy.optimize import least_squares

from dynamisbench_soccer.signals.schema import FloatArray

ENGINE = "sprint-assessment.v1"
Source = Literal["radar", "laser", "gnss_doppler", "gnss_position", "lps_position", "gates"]

SOURCE_ESTIMANDS: dict[str, dict[str, object]] = {
    "radar": {
        "estimand": "Doppler radial speed along the beam, corrected by a declared fixed angle",
        "instantaneous": True,
    },
    "laser": {
        "estimand": "range-rate of the laser distance to the declared aiming point",
        "instantaneous": True,
    },
    "gnss_doppler": {"estimand": "GNSS Doppler speed over ground", "instantaneous": True},
    "gnss_position": {
        "estimand": "horizontal speed differentiated from GNSS positions",
        "instantaneous": True,
    },
    "lps_position": {
        "estimand": "horizontal speed differentiated from LPS positions",
        "instantaneous": True,
    },
    "gates": {
        "estimand": "segment-average velocity between beam crossings",
        "instantaneous": False,
    },
}

LIMITATIONS: tuple[str, ...] = (
    "Gate segment averages are not instantaneous velocities; no instantaneous velocity is "
    "interpolated or inferred from them.",
    "Doppler speed, laser range-rate and positional speed are different estimands with "
    "different latencies and filters. Source labels are kept and no pooled speed is produced.",
    "Distances integrated from speed inherit the speed source's start definition and bias; "
    "the start definition is declared and reported with every milestone.",
    "The F-V model is mono-exponential on flat ground with zero air drag and net horizontal "
    "force. F0 and Pmax change with any of those assumptions, and a good fit does not validate "
    "them. Vendor-computed F-V values are different definitions.",
    "Parameters are reported only when the window covers enough time constants and the fit is "
    "identifiable; otherwise they are refused, never stabilised by priors.",
    "Uncertainty is covariance-based under a local linear model and declared noise "
    "assumptions, plus start-definition sensitivity; not a full measurement-uncertainty budget.",
    "Engineering calculations only: no scientific qualification of any sprint metric.",
)


class StartSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    kind: Literal["trace_origin", "velocity_threshold", "external_trigger"] = "trace_origin"
    threshold_m_s: float = Field(default=0.5, gt=0)
    trigger_s: float | None = None


class FvSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    version: Literal["sprint-assessment.v1"] = "sprint-assessment.v1"
    mass_kg: float = Field(gt=0)
    drag: Literal["none"] = "none"
    fit_t0: bool = False
    window_s: tuple[float, float] | None = None
    min_window_over_tau: float = Field(default=2.0, gt=0)
    max_condition_number: float = Field(default=1e6, gt=1)
    max_correlation: float = Field(default=0.995, gt=0, lt=1)
    max_relative_se: float = Field(default=0.25, gt=0)
    initial_vmax_m_s: float = Field(default=8.0, gt=0)
    initial_tau_s: float = Field(default=1.0, gt=0)


@dataclass(frozen=True)
class FvFit:
    model: str
    vmax_m_s: float | None
    tau_s: float | None
    t0_s: float
    f0_n: float | None
    pmax_w: float | None
    fv_slope_n_per_m_s: float | None
    rmse: float | None
    r_squared: float | None
    n: int
    parameter_se: dict[str, float]
    correlation_vmax_tau: float | None
    condition_number: float | None
    window_over_tau: float | None
    identifiable: bool
    refusals: list[str]
    assumptions: tuple[str, ...]


def source_estimand(source: str) -> dict[str, object]:
    if source not in SOURCE_ESTIMANDS:
        raise ValueError(f"unknown sprint source {source!r}")
    return {"source": source, **SOURCE_ESTIMANDS[source]}


def instantaneous_velocity(source: str) -> None:
    """Explicit refusal API: instantaneous velocity from sources that cannot provide it."""
    if not SOURCE_ESTIMANDS[source]["instantaneous"]:
        raise ValueError(
            f"{source} yields segment averages only; instantaneous velocity is not manufactured"
        )


def start_time(time: FloatArray, speed: FloatArray, start: StartSpec) -> float:
    if start.kind == "trace_origin":
        return float(time[0])
    if start.kind == "external_trigger":
        if start.trigger_s is None:
            raise ValueError("external_trigger start needs a declared trigger time")
        if not time[0] <= start.trigger_s <= time[-1]:
            raise ValueError("trigger time outside the trace")
        return float(start.trigger_s)
    hits = np.flatnonzero(speed >= start.threshold_m_s)
    if len(hits) == 0:
        raise ValueError("speed never reaches the start threshold")
    i = int(hits[0])
    if i == 0:
        return float(time[0])
    # Linear interpolation of the threshold crossing between the bracketing samples.
    frac = (start.threshold_m_s - speed[i - 1]) / (speed[i] - speed[i - 1])
    return float(time[i - 1] + frac * (time[i] - time[i - 1]))


def _validate_trace(time: FloatArray, speed: FloatArray) -> None:
    if (
        time.ndim != 1
        or time.shape != speed.shape
        or len(time) < 3
        or not np.all(np.isfinite(time))
        or not np.all(np.isfinite(speed))
        or not np.all(np.diff(time) > 0)
    ):
        raise ValueError("milestones need finite, strictly increasing time and finite speed")
    if np.any(speed < 0):
        raise ValueError("milestones need nonnegative speed: distance would not be monotonic")


def distance_profile(
    time: FloatArray, speed: FloatArray, start: StartSpec
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Time, speed and distance from the exact (possibly fractional) declared start.

    Speed is piecewise linear between samples, so each interval integrates exactly by the
    trapezoid rule. The profile begins at the declared start itself, not at a sample.
    """
    _validate_trace(time, speed)
    t0 = start_time(time, speed, start)
    after = time > t0
    t = np.concatenate(([t0], time[after]))
    v = np.concatenate(([np.interp(t0, time, speed)], speed[after]))
    if len(t) < 3:
        raise ValueError("too few samples after the start")
    distance = np.concatenate(([0.0], np.cumsum(0.5 * (v[1:] + v[:-1]) * np.diff(t))))
    return t, v, distance


def first_arrival(t: FloatArray, v: FloatArray, distance: FloatArray, d: float) -> float | None:
    """First time the integrated distance reaches ``d``; ``None`` beyond the observed support.

    Inside an interval the linear speed is inverted exactly, so a stationary tail after
    arrival never moves the answer and nothing is extrapolated past the last sample.
    """
    if d < 0 or d > distance[-1] * (1 + 1e-12) + 1e-12:
        return None
    if d == 0:
        return float(t[0])
    d = min(d, float(distance[-1]))
    right = int(np.searchsorted(distance, d, side="left"))
    left = right - 1
    delta = d - float(distance[left])
    dt = float(t[right] - t[left])
    acceleration = float(v[right] - v[left]) / dt
    crossing = math.sqrt(max(0.0, float(v[left]) ** 2 + 2 * acceleration * delta))
    denominator = float(v[left]) + crossing
    return float(min(float(t[right]), float(t[left]) + 2 * delta / denominator))


def milestones(
    time: FloatArray, speed: FloatArray, distances_m: list[float], start: StartSpec
) -> dict[str, float | None]:
    """Elapsed time from the declared start to the first arrival at each distance."""
    t, v, distance = distance_profile(time, speed, start)
    out: dict[str, float | None] = {}
    for d in distances_m:
        arrival = first_arrival(t, v, distance, float(d))
        out[f"{d:g}_m"] = None if arrival is None else arrival - float(t[0])
    return out


def _model_velocity(p: FloatArray, t: FloatArray, fit_t0: bool) -> FloatArray:
    t0 = p[2] if fit_t0 else 0.0
    return np.asarray(p[0] * (1 - np.exp(-np.clip(t - t0, 0, None) / p[1])))


def _model_distance(p: FloatArray, t: FloatArray, fit_t0: bool) -> FloatArray:
    t0 = p[2] if fit_t0 else 0.0
    s = np.clip(t - t0, 0, None)
    return np.asarray(p[0] * (s + p[1] * np.expm1(-s / p[1])))


def _fit(
    kind: str,
    t: FloatArray,
    y: FloatArray,
    spec: FvSpec,
    sigma: float | None,
) -> FvFit:
    model = _model_velocity if kind == "velocity" else _model_distance
    names = ["vmax", "tau"] + (["t0"] if spec.fit_t0 else [])
    p0 = np.array([spec.initial_vmax_m_s, spec.initial_tau_s] + ([0.0] if spec.fit_t0 else []))
    refusals: list[str] = []
    assumptions = (
        "mono-exponential velocity-time model",
        "flat ground, zero air drag (net horizontal force)",
        "constant mass",
        f"start-time parameter {'fitted' if spec.fit_t0 else 'fixed at the declared start'}",
    )
    dof = len(t) - len(p0)
    if dof < 1:
        return FvFit(
            kind,
            None,
            None,
            0.0,
            None,
            None,
            None,
            math.nan,
            None,
            len(t),
            {},
            None,
            None,
            None,
            False,
            [f"{len(t)} points cannot identify {len(p0)} parameters"],
            assumptions,
        )
    result = least_squares(
        lambda p: model(p, t, spec.fit_t0) - y,
        p0,
        bounds=(
            [0.1, 0.05] + ([-2.0] if spec.fit_t0 else []),
            [30, 20] + ([2.0] if spec.fit_t0 else []),
        ),
        x_scale="jac",
    )
    p = result.x
    residual = result.fun
    rmse = float(np.sqrt(np.mean(residual**2)))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = None if ss_tot == 0 else float(1 - np.sum(residual**2) / ss_tot)
    jac = result.jac
    scaled = jac / np.linalg.norm(jac, axis=0)
    cond = float(np.linalg.cond(scaled))
    s2 = float(np.sum(residual**2) / dof) if sigma is None else sigma**2
    try:
        cov = s2 * np.linalg.inv(jac.T @ jac)
    except np.linalg.LinAlgError:
        cov = np.full((len(p), len(p)), np.nan)
        refusals.append("normal matrix singular: parameters not identifiable")
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    corr = (
        float(cov[0, 1] / (se[0] * se[1]))
        if se[0] > 0 and se[1] > 0 and np.isfinite(cov).all()
        else None
    )
    window_over_tau = float((t[-1] - t[0]) / p[1])
    if window_over_tau < spec.min_window_over_tau:
        refusals.append(
            f"window covers {window_over_tau:.2f} time constants; "
            f"at least {spec.min_window_over_tau:g} required"
        )
    if cond > spec.max_condition_number:
        refusals.append(f"scaled Jacobian condition number {cond:.2e} exceeds limit")
    if corr is not None and abs(corr) > spec.max_correlation:
        refusals.append(f"vmax/tau correlation {corr:.4f} exceeds limit")
    if not result.success:
        refusals.append("optimizer did not converge")
    for i, name in enumerate(names[:2]):
        if np.isfinite(se[i]) and se[i] / p[i] > spec.max_relative_se:
            refusals.append(f"{name} relative standard error {se[i] / p[i]:.2f} exceeds limit")
    identifiable = not refusals
    vmax, tau = float(p[0]), float(p[1])
    f0 = spec.mass_kg * vmax / tau
    return FvFit(
        kind,
        vmax if identifiable else None,
        tau if identifiable else None,
        float(p[2]) if spec.fit_t0 else 0.0,
        f0 if identifiable else None,
        f0 * vmax / 4 if identifiable else None,
        -f0 / vmax if identifiable else None,
        rmse,
        r2,
        len(t),
        {n: float(se[i]) for i, n in enumerate(names)},
        corr,
        cond,
        window_over_tau,
        identifiable,
        refusals,
        assumptions,
    )


def fit_velocity_trace(
    time: FloatArray,
    speed: FloatArray,
    spec: FvSpec,
    start: StartSpec,
    *,
    source: str,
    noise_sd_m_s: float | None = None,
) -> FvFit:
    instantaneous_velocity(source)
    t0 = start_time(time, speed, start)
    keep = time >= t0
    t, v = time[keep] - t0, speed[keep]
    if spec.window_s is not None:
        mask = (t >= spec.window_s[0]) & (t <= spec.window_s[1])
        t, v = t[mask], v[mask]
    if len(t) < 5:
        raise ValueError("too few samples in the fit window")
    return _fit("velocity", t, v, spec, noise_sd_m_s)


def fit_split_times(distances_m: list[float], times_s: list[float], spec: FvSpec) -> FvFit:
    """Distance-time fit from gate splits (segment averages are never differentiated)."""
    if len(distances_m) != len(times_s) or len(distances_m) < 3:
        raise ValueError("at least three paired distance/time splits are required")
    if any(b <= a for a, b in itertools.pairwise(times_s)):
        raise ValueError("split times must increase")
    return _fit("distance", np.asarray(times_s, float), np.asarray(distances_m, float), spec, None)


def fv_start_sensitivity(
    time: FloatArray,
    speed: FloatArray,
    spec: FvSpec,
    starts: dict[str, StartSpec],
    *,
    source: str,
) -> dict[str, dict[str, float | None]]:
    out: dict[str, dict[str, float | None]] = {}
    for name, start in starts.items():
        try:
            fit = fit_velocity_trace(time, speed, spec, start, source=source)
            out[name] = {
                "vmax_m_s": fit.vmax_m_s,
                "tau_s": fit.tau_s,
                "f0_n": fit.f0_n,
                "pmax_w": fit.pmax_w,
            }
        except ValueError:
            out[name] = {"vmax_m_s": None, "tau_s": None, "f0_n": None, "pmax_w": None}
    return out


def report(
    fit: FvFit,
    *,
    source: str,
    input_record: dict[str, object] | None = None,
    milestones_s: dict[str, float | None] | None = None,
) -> str:
    return (
        json.dumps(
            {
                "engine": ENGINE,
                "source": source_estimand(source),
                "input": input_record,
                "milestones_s": milestones_s,
                "fit": asdict(fit),
                "limitations": list(LIMITATIONS),
                "qualification": "engineering calculations only; no scientific qualification",
            },
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n"
    )
