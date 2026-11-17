"""Physical and temporal comparability checks for paired planar positions (RES-721).

These run **before** any error metric, and they never alter the data. The row lag and
the similarity transform are *measured*; they are not applied. If a check fails, the
two systems are not shown to be comparable, and no "error" between them may be
reported, because the difference would mix device error with frame, time and
reference-point mismatches.

Gating checks (all must pass, per session):

* **temporal**: the frame-invariant row lag (the lag minimising the similarity-fit
  residual) must be 0. There are no timestamps, so row order is the only time
  relation, and a non-zero lag shows that same-row samples are not simultaneous.
* **frame and scale**: on the lag-aligned pairs, the 2D similarity transform
  ``device ≈ s·R·reference + t`` must be the identity within tolerance (scale,
  rotation, translation). Otherwise the systems do not share a demonstrated frame,
  unit scale or reference point.

Flags (reported, not gating for position comparability):

* **range sentinels**: negative or impossible device ranges.
* **anchor geometry**: fitting fixed anchors to reference positions and device ranges,
  ``d² = |p - a|² + h²``, must give ``h² ≥ 0``. If it doesn't, the ranges are not
  Euclidean distances to the reference positions under any fixed anchor geometry.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from dynamisbench_soccer.signals.schema import FloatArray


@dataclass(frozen=True)
class Thresholds:
    max_row_lag: int = 0
    lag_search_rows: int = 30
    max_scale_deviation: float = 0.01
    max_rotation_deg: float = 1.0
    max_translation: float = 0.05  # source units (undeclared; consistent with metres)
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
    similarity_residual_median: float
    range_sentinels: int
    temporal_ok: bool
    frame_ok: bool
    reasons: list[str] = field(default_factory=list)

    @property
    def comparable(self) -> bool:
        return self.temporal_ok and self.frame_ok


def _offsets(u: FloatArray, o: FloatArray, lag: int) -> FloatArray:
    a, b = _pairs(u, o, lag)
    return np.asarray(np.linalg.norm(a - b, axis=1), dtype=np.float64)


def _pairs(u: FloatArray, o: FloatArray, lag: int) -> tuple[FloatArray, FloatArray]:
    """Device row i+lag paired with reference row i (positive lag: device trails)."""
    return (u[lag:], o[: len(o) - lag]) if lag >= 0 else (u[:lag], o[-lag:])


def best_row_lag(u: FloatArray, o: FloatArray, search: int) -> int:
    """Frame-invariant lag: minimise the *similarity-fit residual* over candidate lags.

    Minimising the raw offset instead would let a spatial frame offset along the path be
    absorbed as a spurious time shift, so time and frame mismatch would be confounded.
    """
    span = min(search, len(u) // 4)

    def residual(lag: int) -> float:
        dev, ref = _pairs(u, o, lag)
        return similarity_fit(ref, dev)[3]

    return min(range(-span, span + 1), key=residual)


def similarity_fit(
    reference: FloatArray, device: FloatArray
) -> tuple[float, float, FloatArray, float]:
    """Least-squares ``device ≈ s·R·reference + t`` (Umeyama). Returns s, deg, t, residual."""
    mu_r, mu_d = reference.mean(0), device.mean(0)
    rc, dc = reference - mu_r, device - mu_d
    u_, sv, vt = np.linalg.svd(dc.T @ rc)
    d = np.diag([1.0, np.sign(np.linalg.det(u_ @ vt))])
    rot = u_ @ d @ vt
    scale = float(np.trace(np.diag(sv) @ d) / np.sum(rc**2))
    t = mu_d - scale * rot @ mu_r
    pred = scale * reference @ rot.T + t
    resid = np.linalg.norm(device - pred, axis=1)
    return scale, float(np.degrees(np.arctan2(rot[1, 0], rot[0, 0]))), t, float(np.median(resid))


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
    reasons: list[str] = []
    if n < th.min_rows:
        reasons.append(f"only {n} rows (< {th.min_rows}); comparability undetermined")
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
            int(np.sum(ranges < 0)),
            False,
            False,
            reasons,
        )
    lag = best_row_lag(device, reference, th.lag_search_rows)
    # frame assessed on the lag-aligned pairs, so a time shift is not read as a frame error
    dev_l, ref_l = _pairs(device, reference, lag)
    scale, deg, t, resid = similarity_fit(ref_l, dev_l)
    temporal_ok = abs(lag) <= th.max_row_lag
    if not temporal_ok:
        reasons.append(f"best row lag {lag} != 0: same-row samples are not simultaneous")
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
        resid,
        int(np.sum(ranges < 0)),
        temporal_ok,
        frame_ok,
        reasons,
    )


def anchor_geometry(reference: FloatArray, ranges: FloatArray) -> list[dict[str, float | bool]]:
    """Per anchor: fit ``d² = |p - a|² + h²`` to reference positions; h² < 0 is impossible."""
    out: list[dict[str, float | bool]] = []
    for i in range(ranges.shape[1]):
        d = ranges[:, i]
        ok = d >= 0
        p = reference[ok]
        a_mat = np.c_[-2 * p[:, 0], -2 * p[:, 1], np.ones(len(p))]
        b = d[ok] ** 2 - (p**2).sum(1)
        (ax, ay, c), *_ = np.linalg.lstsq(a_mat, b, rcond=None)
        h2 = float(c - ax**2 - ay**2)
        pred = np.sqrt(np.maximum((p[:, 0] - ax) ** 2 + (p[:, 1] - ay) ** 2 + max(h2, 0.0), 0))
        out.append(
            {
                "anchor": float(i + 1),
                "x": float(ax),
                "y": float(ay),
                "h2": h2,
                "range_residual_median_abs": float(np.median(np.abs(d[ok] - pred))),
                "rows_used": float(ok.sum()),
                "geometrically_consistent": h2 >= 0,
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
