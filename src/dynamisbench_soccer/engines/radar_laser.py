"""RES-699 Doppler radar and laser range-to-velocity reference reconstruction.

This module never fits sprint F-V models, substitutes gate speed, or imports
analytical truth. The numerical laser derivative is explicitly not a radar
Doppler measurement. Unknown aiming/body-point motion is not corrected.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from dynamisbench_soccer.engines.filtering import FilterMethod, smooth_samples
from dynamisbench_soccer.engines.radar_laser_acquisition import (
    AcquisitionRefusal,
    AdmittedTrace,
    ChannelStage,
    admit,
)
from dynamisbench_soccer.signals.schema import DeviceClass, FloatArray, SignalBundle

DerivativeMethod = Literal["central", "five_point"]
StartMethod = Literal["record_origin", "velocity_threshold", "range_change", "external_trigger"]
CorrectionMethod = Literal["none", "declared_fixed_angle"]
_MIN_COSINE = 0.25


@dataclass(frozen=True)
class ProcessingOptions:
    channel_stage: ChannelStage
    filter_method: FilterMethod = "none"
    derivative_method: DerivativeMethod = "five_point"
    start_method: StartMethod = "record_origin"
    correction_method: CorrectionMethod = "none"
    threshold_m_s: float = 0.30
    range_change_m: float = 0.05
    external_trigger_reference_s: float | None = None
    timing_perturbation_s: float = 0.02
    cosine_perturbation_deg: float = 1.0
    acceleration_limit_m_s2: float = 35.0
    speed_limit_m_s: float = 20.0
    mounting_distance_behind_start_m: float | None = None


@dataclass(frozen=True)
class VelocityResult:
    source_id: str
    source_stage: str
    native_quantity: str
    native_channel: str
    estimand: str
    filter_method: str
    derivative_method: str | None
    correction_method: str
    start_method: str
    start_reference_s: float
    time_reference_s: tuple[float, ...]
    time_from_start_s: tuple[float, ...]
    velocity_m_s: tuple[float, ...]
    peak_velocity_m_s: float
    nominal_hz: float
    observed_hz: float
    aiming_point: str
    mounting_height_m: float
    mounting_distance_behind_start_m: float | None
    sensitivity: dict[str, float | str]
    limitations: tuple[str, ...]


def _laser_derivative(
    distance: FloatArray, time: FloatArray, method: DerivativeMethod
) -> FloatArray:
    if method == "central":
        return np.asarray(np.gradient(distance, time, edge_order=2), dtype=np.float64)
    if method != "five_point":
        raise AcquisitionRefusal("unsupported named laser differentiation method")
    intervals = np.diff(time)
    step = float(np.median(intervals))
    if float(np.max(np.abs(intervals / step - 1))) > 1e-5:
        raise AcquisitionRefusal("five_point requires a uniform native time grid")
    v = np.asarray(np.gradient(distance, time, edge_order=2), dtype=np.float64)
    v[2:-2] = (distance[:-4] - 8 * distance[1:-3] + 8 * distance[3:-1] - distance[4:]) / (12 * step)
    return v


def _start(
    time: FloatArray,
    velocity: FloatArray,
    raw_distance: FloatArray | None,
    options: ProcessingOptions,
) -> float:
    if options.start_method == "record_origin":
        return float(time[0])
    if options.start_method == "external_trigger":
        candidate = options.external_trigger_reference_s
        if candidate is None or not math.isfinite(candidate):
            raise AcquisitionRefusal("external trigger requires a finite declared reference time")
        if not float(time[0]) <= candidate <= float(time[-1]):
            raise AcquisitionRefusal("external trigger lies outside acquisition")
        return candidate
    if options.start_method == "velocity_threshold":
        mask = velocity >= options.threshold_m_s
    elif options.start_method == "range_change":
        if raw_distance is None:
            raise AcquisitionRefusal("range-change start is laser-only")
        mask = raw_distance - raw_distance[0] >= options.range_change_m
    else:
        raise AcquisitionRefusal("unsupported start detection method")
    # Three consecutive observations: explicit sample-level threshold crossing,
    # not an inference of the athlete's first mechanical movement.
    for index in range(len(mask) - 2):
        if bool(np.all(mask[index : index + 3])):
            return float(time[index])
    raise AcquisitionRefusal("no sustained start threshold; no start event manufactured")


def _quality(trace: AdmittedTrace, velocity: FloatArray, options: ProcessingOptions) -> None:
    if not np.all(np.isfinite(velocity)):
        raise AcquisitionRefusal("differentiation produced non-finite velocity")
    if float(np.min(velocity)) < -0.05:
        raise AcquisitionRefusal("negative reconstructed speed; sign/beam integrity unresolved")
    acceleration = np.diff(velocity) / np.diff(trace.reference_time_s)
    if float(np.max(np.abs(acceleration))) > options.acceleration_limit_m_s2:
        raise AcquisitionRefusal(
            "discontinuity: spike, reflection, target switch or aiming drift is unresolved"
        )
    if float(np.max(velocity)) > options.speed_limit_m_s:
        raise AcquisitionRefusal("implausible speed or foreign target / optical reflection")
    # A pre-sprint standing zero is not a sensor saturation plateau. Only
    # sustained equality to the observed ceiling is a clipping suspicion.
    # Quantization can still cause plateaus; device rail telemetry is needed
    # for a definitive clipping attribution.
    ceiling = float(np.max(trace.values))
    flat = (np.abs(np.diff(trace.values)) < 1e-14) & (trace.values[1:] >= ceiling - 1e-14)
    longest, run = 0, 0
    for equal in flat:
        run = run + 1 if equal else 0
        longest = max(longest, run)
    if longest >= 9 and float(np.ptp(trace.values)) > 0.5:
        raise AcquisitionRefusal("sustained flat rail: clipping or acquisition loss suspected")


def _sensitivity(
    trace: AdmittedTrace,
    line_velocity: FloatArray,
    start_s: float,
    options: ProcessingOptions,
) -> dict[str, float | str]:
    result: dict[str, float | str] = {
        "start_shift_minus_s": start_s - options.timing_perturbation_s,
        "start_shift_plus_s": start_s + options.timing_perturbation_s,
        "body_point_sensitivity": "UNSUPPORTED: no measured body-point-to-COM mapping",
        "aiming_drift_correction": (
            "UNSUPPORTED: a single line-of-sight stream cannot identify drift"
        ),
        "mount_offset_correction": "UNSUPPORTED: source schema has no surveyed start-relative pose",
    }
    # Start sensitivity at a common 1 s elapsed-time evaluation point.
    # Missing support windows refuse *extrapolation of this diagnostic*, while
    # preserving the valid native velocity trace and its other sensitivities.
    evaluation_times = (
        start_s + 1.0 - options.timing_perturbation_s,
        start_s + 1.0,
        start_s + 1.0 + options.timing_perturbation_s,
    )
    if trace.reference_time_s[0] <= min(evaluation_times) and (
        max(evaluation_times) <= trace.reference_time_s[-1]
    ):
        for key, t_s in zip(
            ("start_minus", "baseline", "start_plus"), evaluation_times, strict=True
        ):
            result[f"los_velocity_at_1s_{key}_m_s"] = float(
                np.interp(t_s, trace.reference_time_s, line_velocity)
            )
    else:
        result["los_velocity_at_1s"] = "UNSUPPORTED: start perturbation outside observed clock"
    if options.correction_method == "declared_fixed_angle":
        angle = trace.cosine_angle_rad
        if angle is None:
            raise AcquisitionRefusal("declared cosine calibration missing")
        perturbation = math.radians(options.cosine_perturbation_deg)
        if abs(angle) + perturbation > math.acos(_MIN_COSINE):
            result["cosine_sensitivity"] = "UNSUPPORTED: perturbed angle crosses projection limit"
        else:
            lower = float(np.max(line_velocity / math.cos(max(0.0, abs(angle) - perturbation))))
            upper = float(np.max(line_velocity / math.cos(abs(angle) + perturbation)))
            result["cosine_peak_low_m_s"] = min(lower, upper)
            result["cosine_peak_high_m_s"] = max(lower, upper)
    return result


def process(bundle: SignalBundle, options: ProcessingOptions) -> VelocityResult:
    """Reconstruct a trace under explicitly admitted measurement assumptions.

    No gap filling, automatic time-shifting, proprietary vendor inversion, or
    body-point correction is available. Invalid observations refuse outright.
    """
    numeric = (
        options.threshold_m_s,
        options.range_change_m,
        options.timing_perturbation_s,
        options.cosine_perturbation_deg,
        options.acceleration_limit_m_s2,
        options.speed_limit_m_s,
    )
    if any(not math.isfinite(v) for v in numeric) or min(numeric) < 0:
        raise AcquisitionRefusal("processing thresholds and sensitivity bounds must be finite")
    if options.acceleration_limit_m_s2 == 0 or options.speed_limit_m_s == 0:
        raise AcquisitionRefusal("positive integrity bounds required")
    distance = options.mounting_distance_behind_start_m
    if distance is not None and (not math.isfinite(distance) or distance < 0):
        raise AcquisitionRefusal("mount distance behind start must be nonnegative and declared")
    trace = admit(bundle, stage=options.channel_stage)
    if trace.stage is ChannelStage.MANUFACTURER_PROCESSED and options.correction_method != "none":
        raise AcquisitionRefusal(
            "opaque manufacturer-processed channels cannot be cosine-corrected"
        )
    if options.correction_method not in {"none", "declared_fixed_angle"}:
        raise AcquisitionRefusal("unsupported cosine correction")
    if options.filter_method not in {"none", "boxcar5", "binomial5"}:
        raise AcquisitionRefusal("unsupported named filter")
    line = smooth_samples(trace.values, options.filter_method)
    derivative_method: str | None = None
    if trace.device is DeviceClass.LASER:
        line = _laser_derivative(line, trace.reference_time_s, options.derivative_method)
        derivative_method = options.derivative_method
    if options.correction_method == "declared_fixed_angle":
        angle = trace.cosine_angle_rad
        if angle is None:
            raise AcquisitionRefusal("declared cosine calibration missing; no angle inferred")
        cosine = math.cos(angle)
        if cosine < _MIN_COSINE:
            raise AcquisitionRefusal("near-perpendicular cosine correction is ill-conditioned")
        velocity = line / cosine
        estimand = (
            "radar_declared_axis_projection_velocity"
            if trace.device is DeviceClass.RADAR
            else "laser_declared_axis_projection_range_rate"
        )
    else:
        velocity = line
        if trace.stage is ChannelStage.MANUFACTURER_PROCESSED:
            estimand = (
                "manufacturer_processed_radar_velocity_as_supplied"
                if trace.device is DeviceClass.RADAR
                else "derivative_of_manufacturer_processed_laser_range"
            )
        else:
            estimand = (
                "radar_radial_doppler_velocity"
                if trace.device is DeviceClass.RADAR
                else "laser_line_of_sight_range_rate"
            )
    _quality(trace, velocity, options)
    start = _start(
        trace.reference_time_s,
        velocity,
        trace.values if trace.device is DeviceClass.LASER else None,
        options,
    )
    sensitivity = _sensitivity(trace, line, start, options)
    alternate_peaks: list[float] = []
    for variant in ("none", "boxcar5", "binomial5"):
        alternative = smooth_samples(trace.values, variant)
        if trace.device is DeviceClass.LASER:
            alternative = _laser_derivative(
                alternative, trace.reference_time_s, options.derivative_method
            )
        if options.correction_method == "declared_fixed_angle":
            angle = trace.cosine_angle_rad
            if angle is None:
                raise AcquisitionRefusal("missing cosine calibration")
            alternative /= math.cos(angle)
        alternate_peaks.append(float(np.max(alternative)))
    sensitivity["filter_peak_spread_m_s"] = max(alternate_peaks) - min(alternate_peaks)
    limitations = (
        *trace.warnings,
        "Start thresholds identify algorithmic crossings, not mechanically validated onset",
        "Named filters are offline analysis variants, never claimed as manufacturer algorithms",
        "Clipping, out-of-beam intervals and interference need raw validity flags/rail data",
        "No target tracking: smooth reflection or aiming-point drift may be undetectable",
        "No anatomical COM, GNSS-speed, gate-segment-velocity or sprint F-V equivalence",
        "No empirical agreement, hardware timing metrology or assessment qualification",
    )
    return VelocityResult(
        source_id=trace.source_id,
        source_stage=trace.stage.value,
        native_quantity=trace.quantity.value,
        native_channel=trace.channel,
        estimand=estimand,
        filter_method=options.filter_method,
        derivative_method=derivative_method,
        correction_method=options.correction_method,
        start_method=options.start_method,
        start_reference_s=start,
        time_reference_s=tuple(map(float, trace.reference_time_s)),
        time_from_start_s=tuple(map(float, trace.reference_time_s - start)),
        velocity_m_s=tuple(map(float, velocity)),
        peak_velocity_m_s=float(np.max(velocity)),
        nominal_hz=trace.native_nominal_hz,
        observed_hz=trace.measured_hz,
        aiming_point=trace.aiming_point,
        mounting_height_m=trace.mounting_height_m,
        mounting_distance_behind_start_m=distance,
        sensitivity=sensitivity,
        limitations=limitations,
    )
