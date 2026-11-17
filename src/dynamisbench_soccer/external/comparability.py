"""Physical and temporal comparability checks for paired planar positions (RES-721).

These run **before** any error metric, and they never alter the data. Lag, transform
and motion-dependent biases are *measured*; none is applied. If any gate fails, the two
systems are not shown to be comparable, and no "error" between them may be reported:
the difference would mix device error with time, frame, scale or reference-point
mismatch.

Gates (all must pass, per session):

* **temporal (integer lag)**: the integer row lag minimising the RMS residual of a 2D
  similarity fit must be 0. There are no timestamps, so row order is the only time
  relation. A non-zero lag means simultaneity is *not established*. It does not prove
  a delay: a lever-arm offset along the direction of motion can produce the same
  signature at constant speed.
* **frame and scale**: on the lag-aligned pairs, the similarity transform
  ``device ≈ s·R·reference + t`` must be the identity within tolerance.
* **motion-dependent bias**: on the pairs as published (lag 0), the mean residual
  resolved along and across the reference direction of motion must be within
  tolerance. A non-zero along-track bias indicates a fractional time lag *or* a
  mounting (lever-arm) offset; a non-zero cross-track bias indicates a mounting offset.
  The two cannot be separated without timestamps and declared mounting geometry, and
  neither is ever corrected.

Flags (reported, not gating):

* **range sentinels**: negative device ranges.
* **anchor geometry**: fixed anchors are fitted to reference positions and device
  ranges, ``d² = |p - a|² + h²``. The geometry is consistent only if ``h² ≥ 0`` and at
  most 1 % of ranges are shorter than the fitted vertical separation ``sqrt(h²)``, which
  is physically impossible.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from dynamisbench_soccer.signals.schema import FloatArray

IMPOSSIBLE_RANGE_TOLERANCE = 0.05  # source units
MAX_IMPOSSIBLE_FRACTION = 0.01


@dataclass(frozen=True)
class Thresholds:
    max_row_lag: int = 0
    lag_search_rows: int = 30
    max_scale_deviation: float = 0.01
    max_rotation_deg: float = 1.0
    max_translation: float = 0.05  # source units (undeclared; consistent with metres)
    max_motion_bias: float = 0.02  # source units, along- or cross-track mean residual
    min_rows: int = 50


@dataclass(frozen=True)
class SessionCheck:
    scenario: str
    session_id: int
    n_rows: int
    best_row_lag: int
    median_offset_at_zero_lag: float
    median_offset_at_best_lag: float
    scale: float
    rotation_deg: float
    translation: tuple[float, float]
    similarity_residual_rms: float
    along_track_bias: float
    cross_track_bias: float
    range_sentinels: int
    temporal_ok: bool
    frame_ok: bool
    motion_ok: bool
    reasons: list[str] = field(default_factory=list)

    @property
    def comparable(self) -> bool:
        return self.temporal_ok and self.frame_ok and self.motion_ok


def _pairs(u: FloatArray, o: FloatArray, lag: int) -> tuple[FloatArray, FloatArray]:
    """Device row i+lag paired with reference row i (positive lag: device trails)."""
    return (u[lag:], o[: len(o) - lag]) if lag >= 0 else (u[:lag], o[-lag:])


def _offsets(u: FloatArray, o: FloatArray, lag: int) -> FloatArray:
    a, b = _pairs(u, o, lag)
    return np.asarray(np.linalg.norm(a - b, axis=1), dtype=np.float64)


def similarity_fit(
    reference: FloatArray, device: FloatArray
) -> tuple[float, float, FloatArray, float]:
    """Least-squares ``device ≈ s·R·reference + t`` (Umeyama). Returns s, deg, t, RMS residual."""
    mu_r, mu_d = reference.mean(0), device.mean(0)
    rc, dc = reference - mu_r, device - mu_d
    u_, sv, vt = np.linalg.svd(dc.T @ rc)
    d = np.diag([1.0, np.sign(np.linalg.det(u_ @ vt))])
    rot = u_ @ d @ vt
    scale = float(np.trace(np.diag(sv) @ d) / np.sum(rc**2))
    t = mu_d - scale * rot @ mu_r
    resid = device - (scale * reference @ rot.T + t)
    rms = float(np.sqrt(np.mean(np.sum(resid**2, axis=1))))
    return scale, float(np.degrees(np.arctan2(rot[1, 0], rot[0, 0]))), t, rms


def best_row_lag(u: FloatArray, o: FloatArray, search: int) -> int:
    """Integer lag minimising the similarity-fit RMS residual (robust to a fixed frame offset).

    A heading-dependent mounting offset at constant speed can still appear as a lag; the
    motion-bias gate reports that ambiguity rather than resolving it.
    """
    span = min(search, len(u) // 4)

    def residual(lag: int) -> float:
        dev, ref = _pairs(u, o, lag)
        return similarity_fit(ref, dev)[3]

    return min(range(-span, span + 1), key=residual)


def motion_bias(device: FloatArray, reference: FloatArray) -> tuple[float, float]:
    """Mean (device - reference) along and across the reference direction of motion."""
    vel = np.gradient(reference, axis=0)
    speed = np.linalg.norm(vel, axis=1)
    moving = speed > 1e-9
    if moving.sum() < 2:
        return 0.0, 0.0
    tan = vel[moving] / speed[moving, None]
    nrm = np.c_[-tan[:, 1], tan[:, 0]]
    r = (device - reference)[moving]
    return float(np.mean(np.sum(r * tan, axis=1))), float(np.mean(np.sum(r * nrm, axis=1)))


def check_session(
    scenario: str,
    session_id: int,
    device: FloatArray,
    reference: FloatArray,
    ranges: FloatArray,
    th: Thresholds | None = None,
) -> SessionCheck:
    th = th or Thresholds()
    n = len(device)
    sentinels = int(np.sum(ranges < 0))
    if n < th.min_rows:
        return SessionCheck(
            scenario,
            session_id,
            n,
            0,
            0.0,
            0.0,
            1.0,
            0.0,
            (0.0, 0.0),
            0.0,
            0.0,
            0.0,
            sentinels,
            False,
            False,
            False,
            [f"only {n} rows (< {th.min_rows}); comparability undetermined"],
        )
    reasons: list[str] = []
    lag = best_row_lag(device, reference, th.lag_search_rows)
    temporal_ok = abs(lag) <= th.max_row_lag
    if not temporal_ok:
        reasons.append(
            f"best integer row lag {lag} != 0: simultaneity of same-row samples not "
            "established (a delay or an along-track mounting offset)"
        )
    dev_l, ref_l = _pairs(device, reference, lag)
    scale, deg, t, rms = similarity_fit(ref_l, dev_l)
    frame_ok = True
    if abs(scale - 1) > th.max_scale_deviation:
        frame_ok = False
        reasons.append(f"scale {scale:.4f} differs from 1 by > {th.max_scale_deviation:g}")
    if abs(deg) > th.max_rotation_deg:
        frame_ok = False
        reasons.append(f"rotation {deg:.2f} deg exceeds {th.max_rotation_deg:g} deg")
    if float(np.hypot(*t)) > th.max_translation:
        frame_ok = False
        reasons.append(f"translation |t| {float(np.hypot(*t)):.3f} exceeds {th.max_translation:g}")
    along, cross = motion_bias(device, reference)
    motion_ok = abs(along) <= th.max_motion_bias and abs(cross) <= th.max_motion_bias
    if not motion_ok:
        reasons.append(
            f"residual depends on direction of motion (along {along:+.3f}, cross {cross:+.3f}; "
            f"tolerance {th.max_motion_bias:g}): fractional lag or mounting offset, not "
            "separable without timestamps and mounting geometry"
        )
    return SessionCheck(
        scenario,
        session_id,
        n,
        lag,
        float(np.median(_offsets(device, reference, 0))),
        float(np.median(_offsets(device, reference, lag))),
        scale,
        deg,
        (float(t[0]), float(t[1])),
        rms,
        along,
        cross,
        sentinels,
        temporal_ok,
        frame_ok,
        motion_ok,
        reasons,
    )


def anchor_geometry(reference: FloatArray, ranges: FloatArray) -> list[dict[str, float | bool]]:
    """Per anchor: fit ``d² = |p - a|² + h²`` to reference positions (sentinels excluded)."""
    out: list[dict[str, float | bool]] = []
    for i in range(ranges.shape[1]):
        d = ranges[:, i]
        ok = d >= 0
        p = reference[ok]
        a_mat = np.c_[-2 * p[:, 0], -2 * p[:, 1], np.ones(len(p))]
        b = d[ok] ** 2 - (p**2).sum(1)
        (ax, ay, c), *_ = np.linalg.lstsq(a_mat, b, rcond=None)
        h2 = float(c - ax**2 - ay**2)
        floor = np.sqrt(max(h2, 0.0))
        impossible = int(np.sum(d[ok] < floor - IMPOSSIBLE_RANGE_TOLERANCE))
        pred = np.sqrt((p[:, 0] - ax) ** 2 + (p[:, 1] - ay) ** 2 + max(h2, 0.0))
        out.append(
            {
                "anchor": float(i + 1),
                "x": float(ax),
                "y": float(ay),
                "h2": h2,
                "range_residual_median_abs": float(np.median(np.abs(d[ok] - pred))),
                "rows_used": float(ok.sum()),
                "impossibly_short_ranges": float(impossible),
                "geometrically_consistent": h2 >= 0
                and impossible <= MAX_IMPOSSIBLE_FRACTION * float(ok.sum()),
            }
        )
    return out


def planar_disagreement(device: FloatArray, reference: FloatArray) -> dict[str, float]:
    """Same-row planar difference. Call ONLY after every gating check has passed."""
    diff = device - reference
    e = np.linalg.norm(diff, axis=1)
    return {
        "n": float(len(e)),
        "median": float(np.median(e)),
        "p95": float(np.percentile(e, 95)),
        "rmse": float(np.sqrt(np.mean(e**2))),
        "bias_x": float(diff[:, 0].mean()),
        "bias_y": float(diff[:, 1].mean()),
    }


def as_dict(check: SessionCheck) -> dict[str, Any]:
    return {**asdict(check), "comparable": check.comparable}
