"""One interior horizontal tag-speed estimate on the reviewed synthetic grid."""

from __future__ import annotations

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
