"""Time-base agreement between vendor distance and vendor velocity (RES-720).

For ``shorts-laser``, d(distance)/dt matches vendor velocity only after a shift of about
1 s. The cause cannot be determined from manufacturer-processed exports, so the shift
is **measured and reported, never applied**. Source timestamps are never corrected.
Whenever the best alignment lag exceeds :data:`TOLERANCE_MS`, any output that combines
the two channels (derivative residuals, integration drift, distance-defined windows) is
labelled *temporally unqualified*.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from dynamisbench_soccer.external.shorts_laser import SourceTable

#: Largest |lag| at which the channels count as sharing one time base (5 samples at 1 kHz).
TOLERANCE_MS = 5.0
HALF_WINDOW_S = 0.025
MAX_LAG_SAMPLES = 2000
Qualification = Literal["qualified", "unqualified"]


@dataclass(frozen=True)
class TimeBaseAgreement:
    best_lag_ms: float
    rms_at_zero_lag_m_s: float
    rms_at_best_lag_m_s: float
    search_range_ms: float
    tolerance_ms: float
    first_time_distance_above_0_5_m_s: float
    first_time_velocity_above_1_m_s: float

    @property
    def qualification(self) -> Qualification:
        return "qualified" if abs(self.best_lag_ms) <= self.tolerance_ms else "unqualified"

    def statement(self) -> str:
        if self.qualification == "qualified":
            return (
                f"joint distance/velocity reconstruction qualified: best lag "
                f"{self.best_lag_ms:.0f} ms within ±{self.tolerance_ms:g} ms"
            )
        return (
            "TEMPORALLY UNQUALIFIED joint distance/velocity reconstruction: d(distance)/dt "
            f"aligns with vendor velocity only after a {self.best_lag_ms:.0f} ms shift (RMS "
            f"{self.rms_at_zero_lag_m_s:.3f} -> {self.rms_at_best_lag_m_s:.3f} m/s; tolerance "
            f"±{self.tolerance_ms:g} ms). Cause undetermined; source timestamps unmodified and "
            "no shift applied."
        )


def assess(table: SourceTable) -> TimeBaseAgreement:
    """Search integer-sample lags; positive lag = vendor velocity trails distance speed."""
    c = table.columns
    t, d, v = c["time"], c["distance"], c["velocity"]
    dt = float(np.median(np.diff(t)))
    k = max(1, round(HALF_WINDOW_S / dt))
    dv = (d[2 * k :] - d[: -2 * k]) / (t[2 * k :] - t[: -2 * k])
    v_in = v[k : len(v) - k]

    def shift_rms(lag: int) -> float:
        a, b = (v_in[lag:], dv[: len(dv) - lag]) if lag >= 0 else (v_in[:lag], dv[-lag:])
        return float(np.sqrt(np.mean((a - b) ** 2)))

    max_lag = min(MAX_LAG_SAMPLES, len(v_in) // 2)
    best = min(range(-max_lag, max_lag + 1), key=shift_rms)
    return TimeBaseAgreement(
        best_lag_ms=best * dt * 1000,
        rms_at_zero_lag_m_s=shift_rms(0),
        rms_at_best_lag_m_s=shift_rms(best),
        search_range_ms=max_lag * dt * 1000,
        tolerance_ms=TOLERANCE_MS,
        first_time_distance_above_0_5_m_s=float(t[np.argmax(d > 0.5)]),
        first_time_velocity_above_1_m_s=float(t[np.argmax(v > 1.0)]),
    )
