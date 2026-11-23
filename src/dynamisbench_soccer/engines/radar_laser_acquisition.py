"""RES-699 native radar/laser acquisition and clock admission.

This layer identifies a *measurand*, never a sprint F-V estimate. Upstream vendor
processing is a declared input stage, not an algorithm we can reverse-engineer.
The RES-669 reference generator is deliberately not imported here.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

import numpy as np

from dynamisbench_soccer.data.manifest import DataCategory
from dynamisbench_soccer.signals.schema import (
    BundleKind,
    DeviceClass,
    FloatArray,
    Quantity,
    SignalBundle,
)


class AcquisitionRefusal(ValueError):
    """A claim needed for interpretable acquisition has not been established."""


class ChannelStage(StrEnum):
    NATIVE = "native"
    MANUFACTURER_PROCESSED = "manufacturer_processed"


@dataclass(frozen=True)
class AdmittedTrace:
    device: DeviceClass
    stage: ChannelStage
    channel: str
    quantity: Quantity
    source_id: str
    source_clock: str
    device_time_s: FloatArray
    reference_time_s: FloatArray
    values: FloatArray
    native_nominal_hz: float
    measured_hz: float
    max_clock_jitter_fraction: float | None
    cosine_angle_rad: float | None
    mounting_height_m: float
    aiming_point: str
    warnings: tuple[str, ...]

    @property
    def n_samples(self) -> int:
        return len(self.values)


def admit(
    bundle: SignalBundle,
    *,
    stage: ChannelStage,
    maximum_rate_error_fraction: float = 0.03,
    maximum_jitter_fraction: float = 0.10,
) -> AdmittedTrace:
    """Admit exactly one v2, single-source radar velocity or laser range channel.

    Source clock is mapped to the declared reference clock. This *does not*
    remove acquisition/trigger latency or infer hardware synchronization quality.
    Missing measured timestamps can only validate the nominal grid contract.
    """
    if not isinstance(stage, ChannelStage):
        raise AcquisitionRefusal("channel processing stage must be explicitly declared")
    if (
        not math.isfinite(maximum_rate_error_fraction)
        or not 0 <= maximum_rate_error_fraction < 1
        or not math.isfinite(maximum_jitter_fraction)
        or not 0 <= maximum_jitter_fraction < 1
    ):
        raise AcquisitionRefusal("rate and jitter thresholds must be finite fractions")
    meta = bundle.meta
    if meta.data_category not in {DataCategory.SYNTHETIC, DataCategory.EMPIRICAL}:
        raise AcquisitionRefusal("analytical truth must never enter a device engine")
    cls = meta.device.device_class
    if cls not in {DeviceClass.RADAR, DeviceClass.LASER}:
        raise AcquisitionRefusal(
            "radar/laser only: GNSS speed and gate splits are different estimands"
        )
    if meta.kind is not BundleKind.TIMESERIES or meta.sampling is None:
        raise AcquisitionRefusal("native radar/laser requires a sampled timeseries")
    if meta.acquisition is None or len(meta.acquisition.sources) != 1:
        raise AcquisitionRefusal("one explicit acquisition source and clock are required")
    source = meta.acquisition.sources[0]
    if source.device.device_class is not cls or source.device != meta.device:
        raise AcquisitionRefusal("source and trial device identity must agree")
    if source.geometry != meta.geometry:
        raise AcquisitionRefusal("source and trial geometry declarations disagree")
    geometry = source.geometry
    if (
        geometry.mounting_height_m is None
        or geometry.aiming_point is None
        or not geometry.aiming_point.strip()
        or geometry.aiming_point.strip().lower() == "unknown"
        or "mounting_height_m" in geometry.unknown
        or "aiming_point" in geometry.unknown
    ):
        raise AcquisitionRefusal("mounting height and aiming point must be declared")
    if meta.test_id not in {"sprint_fv", "sprint_gates"}:
        raise AcquisitionRefusal("radar/laser processing requires a declared sprint context")
    processed_markers = ("manufacturer_processed", "vendor_processed", "resampled", "smoothed")
    source_format = source.raw_format.lower().replace("-", "_")
    declared_processed = any(token in source_format for token in processed_markers)
    transforms = [
        "".join(c for c in step.casefold() if c.isalnum()) for step in meta.provenance.transforms
    ]
    resampled = any(
        marker in step
        for step in transforms
        for marker in ("resampl", "decimat", "upsampl", "downsampl")
    )
    if stage is ChannelStage.NATIVE and (declared_processed or resampled):
        raise AcquisitionRefusal(
            "upstream processed/resampled acquisition cannot be labeled native"
        )
    if stage is ChannelStage.MANUFACTURER_PROCESSED and not declared_processed:
        raise AcquisitionRefusal(
            "manufacturer-processed stage requires matching source processing provenance"
        )
    quantity = Quantity.VELOCITY if cls is DeviceClass.RADAR else Quantity.DISTANCE
    candidates = [
        c.name for c in meta.channels if c.quantity is quantity and c.name in source.channels
    ]
    if len(candidates) != 1:
        raise AcquisitionRefusal("exactly one owned native-quantity channel is required")
    channel = candidates[0]
    if any(c.quantity not in {quantity, Quantity.TIME} for c in meta.channels):
        raise AcquisitionRefusal(
            "mixed derived and native channels require separate declared streams"
        )
    raw_time = np.asarray(bundle.time(), dtype=np.float64)
    values = np.asarray(bundle[channel], dtype=np.float64)
    if len(raw_time) < 7:
        raise AcquisitionRefusal("at least seven native observations are required")
    if not np.all(np.isfinite(raw_time)) or not np.all(np.isfinite(values)):
        raise AcquisitionRefusal("non-finite acquisition values")
    intervals = np.diff(raw_time)
    if np.any(intervals <= 0):
        raise AcquisitionRefusal("clock is non-monotonic or has duplicate timestamps")
    median_interval = float(np.median(intervals))
    observed_rate = 1 / median_interval
    nominal = meta.sampling.nominal_hz
    if abs(observed_rate / nominal - 1) > maximum_rate_error_fraction:
        raise AcquisitionRefusal("observed acquisition rate contradicts nominal rate")
    if meta.sampling.measured_hz is not None and (
        abs(observed_rate / meta.sampling.measured_hz - 1) > maximum_rate_error_fraction
    ):
        raise AcquisitionRefusal("observed rate contradicts declared measured rate")
    jitter: float | None = None
    warnings: list[str] = []
    if "time" in bundle.data:
        jitter = float(np.max(np.abs(intervals / median_interval - 1)))
        if jitter > maximum_jitter_fraction:
            raise AcquisitionRefusal("timestamp gap, dropout, or excessive clock jitter")
    else:
        warnings.append("Implicit nominal timestamps; native jitter cannot be checked")
    reference = raw_time.copy()
    acquisition = meta.acquisition
    if source.clock != acquisition.reference_clock:
        matching = [a for a in acquisition.alignments if a.clock == source.clock]
        if len(matching) != 1:
            raise AcquisitionRefusal("unresolved source-to-reference clock alignment")
        a = matching[0]
        reference = raw_time + a.offset_s + (raw_time - a.origin_s) * a.drift_ppm * 1e-6
        if a.uncertainty_s > median_interval:
            raise AcquisitionRefusal(
                "clock synchronization uncertainty exceeds native sample interval"
            )
        if a.uncertainty_s > 0:
            warnings.append(
                "Declared synchronization uncertainty is not empirical trigger metrology"
            )
    if np.any(np.diff(reference) <= 0):
        raise AcquisitionRefusal("clock transformation reverses time")
    angles = [c.cosine_angle_rad for c in source.calibration if c.cosine_angle_rad is not None]
    if len(angles) > 1 and not np.allclose(angles, angles[0], atol=1e-12, rtol=0):
        raise AcquisitionRefusal("contradictory source cosine calibrations")
    angle = angles[0] if angles else None
    if angle is not None and (not math.isfinite(angle) or abs(angle) >= math.pi / 2):
        raise AcquisitionRefusal("declared cosine angle is nonprojectable")
    if cls is DeviceClass.RADAR and np.min(values) < -0.05:
        raise AcquisitionRefusal("negative radial radar velocity outside declared tolerance")
    if cls is DeviceClass.LASER and np.min(values) < 0:
        raise AcquisitionRefusal("negative laser range is physically unsupported")
    if meta.vendor_reported:
        warnings.append("Vendor-reported values are not reference velocity or physical truth")
    if stage is ChannelStage.MANUFACTURER_PROCESSED:
        warnings.append("Manufacturer processing is opaque; no native raw reconstruction claimed")
    warnings.append(
        "Mount distance behind start and body-point offset are not encoded in schema v2"
    )
    warnings.append("Physical acquisition latency and absolute sprint start remain unverified")
    return AdmittedTrace(
        device=cls,
        stage=stage,
        channel=channel,
        quantity=quantity,
        source_id=source.id,
        source_clock=source.clock,
        device_time_s=raw_time.copy(),
        reference_time_s=reference,
        values=values.copy(),
        native_nominal_hz=nominal,
        measured_hz=observed_rate,
        max_clock_jitter_fraction=jitter,
        cosine_angle_rad=angle,
        mounting_height_m=geometry.mounting_height_m,
        aiming_point=geometry.aiming_point,
        warnings=tuple(warnings),
    )
