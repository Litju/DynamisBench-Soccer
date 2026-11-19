"""Offline position-track kinematics with separate GNSS Doppler estimands."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from dynamisbench_soccer.engines.filtering import FilterMethod, smooth_samples
from dynamisbench_soccer.engines.gnss_lps_acquisition import (
    AcquisitionDeclaration,
    AcquisitionRefusal,
    AdmittedTrack,
    admit,
)
from dynamisbench_soccer.signals.schema import FloatArray, SignalBundle

DerivativeMethod = Literal["three_point", "five_point"]


class SpeedThreshold(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False, frozen=True, strict=True)

    threshold_m_s: float = Field(gt=0)
    dwell_s: float = Field(ge=0)


class ProcessingOptions(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    position_filter: FilterMethod = "none"
    doppler_filter: FilterMethod = "none"
    derivative: DerivativeMethod = "three_point"
    thresholds: tuple[SpeedThreshold, ...] = (
        SpeedThreshold(threshold_m_s=3, dwell_s=0),
        SpeedThreshold(threshold_m_s=3, dwell_s=0.5),
        SpeedThreshold(threshold_m_s=5, dwell_s=0.5),
    )


@dataclass(frozen=True)
class Kinematics:
    position_m: FloatArray
    positional_velocity_m_s: FloatArray
    horizontal_positional_speed_m_s: FloatArray
    positional_acceleration_m_s2: FloatArray
    horizontal_positional_acceleration_m_s2: FloatArray
    accumulated_horizontal_distance_m: FloatArray
    doppler_speed_m_s: FloatArray | None
    doppler_speed_change_m_s2: FloatArray | None


@dataclass(frozen=True)
class ThresholdResult:
    estimand: Literal["horizontal_positional_speed", "doppler_speed"]
    threshold_m_s: float
    dwell_s: float
    runs: int
    duration_s: float
    horizontal_path_distance_m: float | None
    integrated_doppler_distance_m: float | None


@dataclass(frozen=True)
class Peaks:
    reference_rate_hz: float
    position_filter: FilterMethod
    doppler_filter: FilterMethod
    horizontal_positional_speed_m_s: float
    horizontal_positional_acceleration_m_s2: float
    doppler_speed_m_s: float | None
    absolute_doppler_speed_change_m_s2: float | None


@dataclass(frozen=True)
class Sensitivity:
    comparison_duration_s: float
    native: Peaks
    alternate_filter: Peaks
    every_second_sample: Peaks | None


@dataclass(frozen=True)
class TrackResult:
    acquisition: AdmittedTrack
    options: ProcessingOptions
    kinematics: Kinematics
    thresholds: tuple[ThresholdResult, ...]
    sensitivity: Sensitivity
    limitations: tuple[str, ...]


def _differentiate(values: FloatArray, time: FloatArray, method: DerivativeMethod) -> FloatArray:
    # Subtract the first value to minimize cancellation in large-offset coordinates.
    values = values - values[0]
    result = np.asarray(np.gradient(values, time, edge_order=2, axis=0))
    if method == "five_point":
        intervals = np.diff(time)
        step = float(np.median(intervals))
        if np.max(np.abs(intervals / step - 1)) > 1e-5:
            raise AcquisitionRefusal("five_point differentiation requires a uniform time grid")
        result[2:-2] = (values[:-4] - 8 * values[1:-3] + 8 * values[3:-1] - values[4:]) / (
            12 * step
        )
    return result


def _kinematics(
    position: FloatArray, time: FloatArray, doppler: FloatArray | None, options: ProcessingOptions
) -> Kinematics:
    position = np.column_stack(
        [smooth_samples(axis, options.position_filter) for axis in position.T]
    )
    velocity = _differentiate(position, time, options.derivative)
    acceleration = _differentiate(velocity, time, options.derivative)
    distance = np.r_[0.0, np.cumsum(np.linalg.norm(np.diff(position[:, :2], axis=0), axis=1))]
    doppler_change = None
    if doppler is not None:
        doppler = smooth_samples(doppler, options.doppler_filter, extend_slope=False)
        doppler_change = _differentiate(doppler, time, options.derivative)
    horizontal_speed = np.hypot(velocity[:, 0], velocity[:, 1])
    horizontal_acceleration = np.hypot(acceleration[:, 0], acceleration[:, 1])
    derived: tuple[FloatArray, ...] = (
        velocity,
        acceleration,
        distance,
        horizontal_speed,
        horizontal_acceleration,
    )
    if doppler_change is not None:
        derived += (doppler_change,)
    if any(not np.all(np.isfinite(values)) for values in derived):
        raise AcquisitionRefusal("kinematics exceed numerical precision")
    return Kinematics(
        position,
        velocity,
        horizontal_speed,
        acceleration,
        horizontal_acceleration,
        distance,
        doppler,
        doppler_change,
    )


def _thresholds(
    kinematics: Kinematics, time: FloatArray, options: ProcessingOptions
) -> tuple[ThresholdResult, ...]:
    results = []
    for rule in options.thresholds:
        speed_traces: tuple[
            tuple[Literal["horizontal_positional_speed", "doppler_speed"], FloatArray | None], ...
        ] = (
            ("horizontal_positional_speed", kinematics.horizontal_positional_speed_m_s),
            ("doppler_speed", kinematics.doppler_speed_m_s),
        )
        for estimand, speed in speed_traces:
            if speed is None:
                continue
            above = speed >= rule.threshold_m_s
            intervals = above[:-1] & above[1:]
            edges = np.flatnonzero(np.diff(np.r_[False, intervals, False]))
            duration = distance = 0.0
            runs = 0
            for start, stop in edges.reshape(-1, 2):
                span = float(time[stop] - time[start])
                if span < rule.dwell_s and not math.isclose(span, rule.dwell_s, abs_tol=1e-12):
                    continue
                runs += 1
                duration += span
                if estimand == "doppler_speed":
                    distance += float(np.trapezoid(speed[start : stop + 1], time[start : stop + 1]))
                else:
                    path = kinematics.accumulated_horizontal_distance_m
                    distance += float(path[stop] - path[start])
            results.append(
                ThresholdResult(
                    estimand,
                    rule.threshold_m_s,
                    rule.dwell_s,
                    runs,
                    duration,
                    distance if estimand == "horizontal_positional_speed" else None,
                    distance if estimand == "doppler_speed" else None,
                )
            )
    return tuple(results)


def _peaks(kinematics: Kinematics, time: FloatArray, options: ProcessingOptions) -> Peaks:
    doppler = kinematics.doppler_speed_m_s
    change = kinematics.doppler_speed_change_m_s2
    return Peaks(
        1 / float(np.median(np.diff(time))),
        options.position_filter,
        options.doppler_filter,
        float(kinematics.horizontal_positional_speed_m_s.max()),
        float(kinematics.horizontal_positional_acceleration_m_s2.max()),
        float(doppler.max()) if doppler is not None else None,
        float(np.max(np.abs(change))) if change is not None else None,
    )


def _sensitivity(
    track: AdmittedTrack, options: ProcessingOptions, native: Kinematics
) -> Sensitivity:
    time, position, doppler = track.clock.elapsed_s, track.position_m, track.doppler_speed_m_s
    # Equal start and end timestamps keep the comparison on the same part of the trial.
    if len(time) >= 9 and len(time) % 2 == 0:
        time, position = time[:-1], position[:-1]
        doppler = None if doppler is None else doppler[:-1]
        native = _kinematics(position, time, doppler, options)
    alternate = options.model_copy(
        update={
            "position_filter": "binomial5" if options.position_filter == "none" else "none",
            "doppler_filter": "binomial5"
            if doppler is not None and options.doppler_filter == "none"
            else "none",
        }
    )
    decimated = None
    if len(time) >= 9:
        half_time, half_position = time[::2], position[::2]
        half_doppler = None if doppler is None else doppler[::2]
        decimated = _peaks(
            _kinematics(half_position, half_time, half_doppler, options), half_time, options
        )
    return Sensitivity(
        float(time[-1]),
        _peaks(native, time, options),
        _peaks(_kinematics(position, time, doppler, alternate), time, alternate),
        decimated,
    )


def process(
    bundle: SignalBundle,
    declaration: AcquisitionDeclaration,
    options: ProcessingOptions | None = None,
) -> TrackResult:
    """Produce acquisition-linked kinematics from an admitted, contiguous observation."""
    options = (
        ProcessingOptions()
        if options is None
        else ProcessingOptions.model_validate(options.model_dump())
    )
    track = admit(bundle, declaration)
    if track.doppler_speed_m_s is None and options.doppler_filter != "none":
        raise AcquisitionRefusal("a Doppler filter requires native Doppler measurements")
    time = track.clock.elapsed_s
    kinematics = _kinematics(track.position_m, time, track.doppler_speed_m_s, options)
    return TrackResult(
        track,
        options,
        kinematics,
        _thresholds(kinematics, time, options),
        _sensitivity(track, options, kinematics),
        (
            "Receiver position describes the antenna or tag point.",
            "LPS positions are supplied position solutions. Anchor visibility checks provide "
            "no ranging residuals, per-sample covariance, or validation of the anchor survey.",
            "Declared clock and latency bounds exclude variable latency and undocumented "
            "processing. Native status depends on producer metadata and sampled timestamps.",
            "Threshold distance uses whole sampled intervals. Acceleration and distance "
            "depend on sampling, noise, filter choice and boundary support.",
            "Sensitivity uses alternate offline smoothing and unfiltered stride-two "
            "decimation of the supplied record, with no independent lower-rate measurement.",
        ),
    )
