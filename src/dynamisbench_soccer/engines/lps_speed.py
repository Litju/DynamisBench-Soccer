"""One interior horizontal tag-speed estimate on the reviewed synthetic grid."""

from __future__ import annotations

import math

import numpy as np

from dynamisbench_soccer.data.manifest import DataCategory
from dynamisbench_soccer.engines.gnss_lps import ProcessingOptions, process
from dynamisbench_soccer.engines.gnss_lps_acquisition import (
    AcquisitionDeclaration,
    AcquisitionRefusal,
)
from dynamisbench_soccer.signals.schema import DeviceClass, SignalBundle


def horizontal_speed(bundle: SignalBundle, declaration: AcquisitionDeclaration, row: int) -> float:
    """Reconstruct the horizontal speed of the tracked tag, in m/s (DEV-005)."""
    if (
        bundle.meta.device.device_class is not DeviceClass.LPS
        or bundle.meta.data_category is not DataCategory.SYNTHETIC
        or declaration.native_hz != 25
        or declaration.timestamp_basis != "measurement"
        or declaration.latency_s != 0
        or declaration.latency_uncertainty_s != 0
        or not 0 < row < bundle.meta.n_rows - 1
    ):
        raise AcquisitionRefusal("unsupported LPS positional-speed slice")
    result = process(bundle, declaration, ProcessingOptions(thresholds=()))
    clock = result.acquisition.clock
    if (
        clock.drift_ppm != 0
        or clock.alignment_uncertainty_s != 0
        or not np.allclose(np.diff(clock.elapsed_s), 0.04, rtol=0, atol=1e-12)
    ):
        raise AcquisitionRefusal("requires exact aligned 25 Hz measurement timestamps")
    return float(result.kinematics.horizontal_positional_speed_m_s[row])


def position_error_radius(
    before_s: float, center_s: float, after_s: float, bound_m: float
) -> float:
    """Deterministic speed-error bound for independent per-row horizontal error balls."""
    left_step, right_step = center_s - before_s, after_s - center_s
    if not all(math.isfinite(v) for v in (left_step, right_step, bound_m)) or (
        min(left_step, right_step) <= 0 or bound_m < 0
    ):
        raise AcquisitionRefusal("invalid position bound or derivative support")
    span = left_step + right_step
    left_weight = right_step / span / left_step
    right_weight = left_step / span / right_step
    radius = 2 * max(left_weight, right_weight) * bound_m
    if not math.isfinite(radius):
        raise AcquisitionRefusal("position-error radius exceeds numerical precision")
    return radius
