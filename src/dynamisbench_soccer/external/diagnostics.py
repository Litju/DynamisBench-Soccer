"""Numerical diagnostics for the ``shorts-laser`` source. Every output is pipeline-derived.

They measure **internal consistency and method sensitivity** of manufacturer-processed
signals: timing, quantisation, distance/velocity agreement, smoothing lag and model-fit
sensitivity to documented preprocessing choices. They are not laser accuracy, not
reference truth and not athlete assessment. Vendor outputs are never treated as
physical truth: agreement between two vendor channels only shows that they are
mutually consistent.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np

from dynamisbench_soccer.external.shorts_laser import (
    PACKAGE_DOCUMENTED_ROWS,
    PACKAGE_TRIM_S,
    SourceTable,
)
from dynamisbench_soccer.signals.schema import FloatArray

_GOLDEN = (math.sqrt(5) - 1) / 2
WINDOWS_MS = (1, 5, 25, 50)
STAGE = "pipeline_derived"


def _golden(f: Callable[[float], float], lo: float, hi: float, tol: float) -> tuple[float, float]:
    a, b = lo, hi
    c, d = b - _GOLDEN * (b - a), a + _GOLDEN * (b - a)
    fc, fd = f(c), f(d)
    while b - a > tol:
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - _GOLDEN * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + _GOLDEN * (b - a)
            fd = f(d)
    x = (a + b) / 2
    return x, f(x)


@dataclass(frozen=True)
class MonoExpFit:
    """v(t) = MSS * (1 - exp(-(t - TD) / TAU)) for t > TD, else 0 (shorts' model form)."""

    mss_m_s: float
    tau_s: float
    td_s: float
    mac_m_s2: float
    rmse_m_s: float
    n: int


def fit_mono_exponential(t: FloatArray, v: FloatArray) -> MonoExpFit:
    """Least squares in (MSS, TAU, TD): MSS closed form, nested golden searches."""

    def sse_for(td: float, tau: float) -> tuple[float, float]:
        g = np.where(t > td, 1 - np.exp(-(t - td) / tau), 0.0)
        gg = float(g @ g)
        mss = float(g @ v) / gg if gg > 0 else 0.0
        r = v - mss * g
        return float(r @ r), mss

    def best_tau(td: float) -> tuple[float, float]:
        return _golden(lambda tau: sse_for(td, tau)[0], 0.05, 5.0, 1e-6)

    vmax = float(v.max())
    td_hi = float(t[np.argmax(v >= 0.5 * vmax)])
    grid = np.arange(t[0] - 1.0, td_hi, 0.01)
    coarse = [best_tau(float(td))[1] for td in grid]
    i = int(np.argmin(coarse))
    td, _ = _golden(lambda x: best_tau(x)[1], float(grid[max(i - 1, 0)]),
                    float(grid[min(i + 1, len(grid) - 1)]), 1e-6)  # fmt: skip
    tau, sse = best_tau(td)
    _, mss = sse_for(td, tau)
    return MonoExpFit(mss, tau, td, mss / tau, math.sqrt(sse / len(t)), len(t))


def _central_diff(t: FloatArray, x: FloatArray, k: int) -> tuple[FloatArray, slice]:
    inner = slice(k, len(x) - k)
    return (x[2 * k :] - x[: -2 * k]) / (t[2 * k :] - t[: -2 * k]), inner


def _stats(residual: FloatArray) -> dict[str, float]:
    return {
        "rms": float(np.sqrt(np.mean(residual**2))),
        "mean": float(np.mean(residual)),
        "max_abs": float(np.max(np.abs(residual))),
    }


def compute(table: SourceTable) -> dict[str, object]:
    c = table.columns
    t, d, v, vu = c["time"], c["distance"], c["velocity"], c["raw_velocity"]
    dt = float(np.median(np.diff(t)))
    resolution = float(np.min(np.diff(d)[np.diff(d) > 0]))
    out: dict[str, object] = {"stage": STAGE}

    # Package preprocessing, reproduced as arithmetic only (the R script is never run).
    keep375 = t > 0.375
    out["package_preprocessing"] = {
        "rows_after_trim_0_414_s": int(np.sum(t > PACKAGE_TRIM_S)),
        "package_documented_rows": PACKAGE_DOCUMENTED_ROWS,
        "reproduces_documented_rows": int(np.sum(t > PACKAGE_TRIM_S)) == PACKAGE_DOCUMENTED_ROWS,
        "velocity_min_m_s": float(v.min()),
        "time_of_velocity_min_s": float(t[np.argmin(v)]),
        "offset_subtracted_by_package_m_s": float(v[keep375].min()),
        "pre_movement_velocity_median_first_100ms_m_s": float(np.median(v[: round(0.1 / dt)])),
    }

    # Distance derivative vs vendor velocities, by window; quantisation floor q/(sqrt6 * 2h).
    rows = []
    for ms in WINDOWS_MS:
        k = max(1, round(ms / 1000 / dt))
        dv, inner = _central_diff(t, d, k)
        rows.append(
            {
                "half_window_ms": ms,
                "quantisation_floor_rms_m_s": resolution / math.sqrt(6) / (2 * k * dt),
                "vs_velocity": _stats(dv - v[inner]),
                "vs_velocity_vendor_unsmoothed": _stats(dv - vu[inner]),
            }
        )
    out["distance_derivative_vs_vendor_velocity"] = rows

    # Time-base agreement: the shift that best aligns d(distance)/dt with vendor velocity.
    k25 = max(1, round(0.025 / dt))
    dv25, inner25 = _central_diff(t, d, k25)
    v_in = v[inner25]

    def shift_rms(lag: int) -> float:
        a, b = (v_in[lag:], dv25[: len(dv25) - lag]) if lag >= 0 else (v_in[:lag], dv25[-lag:])
        return float(np.sqrt(np.mean((a - b) ** 2)))

    max_lag = min(2000, len(v_in) // 2)
    best_lag = min(range(-max_lag, max_lag + 1), key=shift_rms)
    out["distance_velocity_time_base"] = {
        "best_lag_ms": best_lag * dt * 1000,
        "rms_at_zero_lag_m_s": shift_rms(0),
        "rms_at_best_lag_m_s": shift_rms(best_lag),
        "search_range_ms": max_lag * dt * 1000,
        "sign": "positive lag: vendor velocity trails the distance-derived speed",
        "first_time_distance_above_0_5_m_s": float(t[np.argmax(d > 0.5)]),
        "first_time_velocity_above_1_m_s": float(t[np.argmax(v > 1.0)]),
    }

    # Integrating each vendor velocity vs vendor distance.
    def integral(x: FloatArray) -> FloatArray:
        steps = np.cumsum((x[1:] + x[:-1]) / 2 * np.diff(t))
        return np.asarray(d[0] + np.concatenate([[0.0], steps]), dtype=np.float64)

    out["integrated_velocity_vs_distance"] = {
        name: {
            "end_difference_m": float(integral(x)[-1] - d[-1]),
            "max_abs_difference_m": float(np.max(np.abs(integral(x) - d))),
        }
        for name, x in (("velocity", v), ("velocity_vendor_unsmoothed", vu))
    }

    # Smoothed vs unsmoothed vendor velocity, including the lag that best aligns them.
    lags = range(-200, 201)

    def rms_at(lag: int) -> float:
        a, b = (vu[lag:], v[: len(v) - lag]) if lag >= 0 else (vu[:lag], v[-lag:])
        return float(np.sqrt(np.mean((a - b) ** 2)))

    best = min(lags, key=rms_at)
    out["smoothed_vs_unsmoothed_velocity"] = {
        **_stats(vu - v),
        "best_alignment_lag_ms": best * dt * 1000,
        "rms_at_best_lag": rms_at(best),
        "sign": "positive lag: unsmoothed trails smoothed",
    }

    # Pipeline-derived acceleration: sensitivity to channel and derivative window.
    accel = {}
    for name, x in (("velocity", v), ("velocity_vendor_unsmoothed", vu)):
        for ms in (1, 25):
            k = max(1, round(ms / 1000 / dt))
            a, _ = _central_diff(t, x, k)
            accel[f"{name}_half_window_{ms}ms_max_m_s2"] = float(a.max())
    out["derived_acceleration_peak"] = accel

    # Model-fit sensitivity to documented preprocessing choices.
    reach30 = t <= t[np.argmax(d >= 30.0)] if d.max() >= 30.0 else np.ones_like(t, dtype=bool)
    trim = t > PACKAGE_TRIM_S
    variants = {
        "velocity_all_rows": (t, v),
        "velocity_package_trim_and_offset": (t[trim], v[trim] - v[keep375].min()),
        "velocity_vendor_unsmoothed_all_rows": (t, vu),
        "velocity_to_30m": (t[reach30], v[reach30]),
    }
    fits = {k: asdict(fit_mono_exponential(*xy)) for k, xy in variants.items()}
    out["mono_exponential_fit_sensitivity"] = {
        "variants": fits,
        "spread": {
            key: max(f[key] for f in fits.values()) - min(f[key] for f in fits.values())
            for key in ("mss_m_s", "tau_s", "td_s", "mac_m_s2")
        },
        "note": "model-fit diagnostic of vendor-processed velocity; not reference truth",
    }
    return out
