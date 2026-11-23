"""Conditional aligned-clock bounds for admitted position and Doppler traces."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from dynamisbench_soccer.engines.filtering import smooth_samples
from dynamisbench_soccer.engines.gnss_lps import TrackResult, _differentiate
from dynamisbench_soccer.engines.gnss_lps_acquisition import AcquisitionRefusal
from dynamisbench_soccer.signals.schema import FloatArray


@dataclass(frozen=True)
class Bounds:
    low: FloatArray
    high: FloatArray


@dataclass(frozen=True)
class ClockKinematicBounds:
    measurement_time_s: Bounds
    horizontal_positional_speed_m_s: Bounds
    horizontal_positional_acceleration_m_s2: Bounds
    doppler_speed_change_m_s2: Bounds | None
    common_time_offset_bound_s: float


def clock_kinematic_bounds(
    result: TrackResult, relative_scale_error: float
) -> ClockKinematicBounds:
    """Bound time and rates for a common time offset and affine stretch of the admitted grid.

    Positions, Doppler speed, filter weights and row membership are held fixed. The
    offset budget must apply to the entire record. Variable latency and threshold
    eligibility are outside this conditional kinematic calculation.
    """
    if not math.isfinite(relative_scale_error) or not 0 <= relative_scale_error < 1:
        raise AcquisitionRefusal("clock scale error must be finite and in [0, 1)")
    clock = result.acquisition.clock
    declaration = result.acquisition.declaration
    low_scale, high_scale = 1 - relative_scale_error, 1 + relative_scale_error
    nominal_scale = 1 + clock.drift_ppm / 1e6
    max_drift = declaration.maximum_drift_ppm / 1e6
    if nominal_scale * low_scale < 1 - max_drift or nominal_scale * high_scale > 1 + max_drift:
        raise AcquisitionRefusal("perturbed clock exceeds declared drift limit")
    interval = np.diff(clock.elapsed_s) * low_scale
    step = np.linalg.norm(np.diff(result.acquisition.position_m, axis=0), axis=1)
    if np.any(step > interval * declaration.maximum_jump_speed_m_s):
        raise AcquisitionRefusal("perturbed clock fails the position jump limit")
    offset = clock.alignment_uncertainty_s + declaration.latency_uncertainty_s
    if offset >= float(np.median(interval)):
        raise AcquisitionRefusal("perturbed clock uncertainty reaches the sample interval")

    def rate(values: FloatArray, power: int) -> Bounds:
        first, second = values / low_scale**power, values / high_scale**power
        if not np.all(np.isfinite(first) & np.isfinite(second)):
            raise AcquisitionRefusal("perturbed kinematics exceed numerical precision")
        return Bounds(np.minimum(first, second), np.maximum(first, second))

    kinematics = result.kinematics
    origin = clock.measurement_time_s[0]
    change = kinematics.doppler_speed_change_m_s2
    return ClockKinematicBounds(
        measurement_time_s=Bounds(
            origin - offset + clock.elapsed_s * low_scale,
            origin + offset + clock.elapsed_s * high_scale,
        ),
        horizontal_positional_speed_m_s=rate(kinematics.horizontal_positional_speed_m_s, 1),
        horizontal_positional_acceleration_m_s2=rate(
            kinematics.horizontal_positional_acceleration_m_s2,
            2,
        ),
        doppler_speed_change_m_s2=rate(change, 1) if change is not None else None,
        common_time_offset_bound_s=offset,
    )


@dataclass(frozen=True)
class InputErrorRadii:
    horizontal_positional_speed_m_s: float
    horizontal_positional_acceleration_m_s2: float
    accumulated_horizontal_distance_m: FloatArray
    doppler_speed_m_s: float | None
    doppler_speed_change_m_s2: float | None


def input_error_radii(
    result: TrackResult, horizontal_position_m: float, doppler_m_s: float
) -> InputErrorRadii:
    """Conservative radii on the admitted grid with frozen rows and exact time.

    The user supplies horizontal error-ball radii in the already admitted
    Cartesian frame and optional Doppler speed bounds. This is a deterministic
    conditional bound, with no model of receiver covariance or threshold status.
    """
    if not all(math.isfinite(x) and x >= 0 for x in (horizontal_position_m, doppler_m_s)):
        raise AcquisitionRefusal("input error radii must be finite and nonnegative")
    has_doppler = result.kinematics.doppler_speed_m_s is not None
    if not has_doppler and doppler_m_s:
        raise AcquisitionRefusal("Doppler error requires a native Doppler speed trace")
    delay = float(np.min(np.diff(result.acquisition.clock.elapsed_s)))
    # Quadratic boundary rows have absolute weight sum <= 4 / min step.
    # Centered three- and five-point rows have smaller absolute sums.
    derivative_gain = 4 / delay
    smoothing_gain = {"none": 1.0, "boxcar5": 1.8, "binomial5": 1.25}
    position = horizontal_position_m * smoothing_gain[result.options.position_filter]
    count = len(result.acquisition.clock.elapsed_s)
    # Compute in ascending powers so a zero input error remains exactly zero.
    velocity = position * derivative_gain
    acceleration = velocity * derivative_gain
    doppler_change = doppler_m_s * derivative_gain
    distance = position * (2 * (count - 1))
    if not all(math.isfinite(x) for x in (velocity, acceleration, doppler_change, distance)):
        raise AcquisitionRefusal("input error bounds exceed numerical precision")
    return InputErrorRadii(
        horizontal_positional_speed_m_s=velocity,
        horizontal_positional_acceleration_m_s2=acceleration,
        accumulated_horizontal_distance_m=2
        * position
        * np.arange(len(result.acquisition.clock.elapsed_s), dtype=np.float64),
        doppler_speed_m_s=doppler_m_s if has_doppler else None,
        doppler_speed_change_m_s2=doppler_change if has_doppler else None,
    )


@dataclass(frozen=True)
class DerivativeNoise:
    horizontal_velocity_component_sd_m_s: FloatArray
    horizontal_acceleration_component_sd_m_s2: FloatArray
    horizontal_speed_rms_upper_m_s: FloatArray
    horizontal_acceleration_rms_upper_m_s2: FloatArray


def iid_position_noise(result: TrackResult, position_component_sd_m: float) -> DerivativeNoise:
    """Propagate independent zero-mean Cartesian coordinate noise on the fixed admitted grid."""
    if not math.isfinite(position_component_sd_m) or position_component_sd_m < 0:
        raise AcquisitionRefusal("position standard deviation must be finite and nonnegative")
    time = result.acquisition.clock.elapsed_s
    # A row can see at most 13 contiguous input rows after a five-point
    # smoother and two five-point derivatives. Each row's support has unique
    # residues modulo 13, so these impulses recover its weights in O(n) space.
    impulses = np.eye(13)[np.arange(len(time)) % 13]
    filtered = np.column_stack(
        [smooth_samples(column, result.options.position_filter) for column in impulses.T]
    )
    velocity = _differentiate(filtered, time, result.options.derivative)
    acceleration = _differentiate(velocity, time, result.options.derivative)
    sd_velocity = np.linalg.norm(velocity, axis=1) * position_component_sd_m
    sd_acceleration = np.linalg.norm(acceleration, axis=1) * position_component_sd_m
    rms_velocity = math.sqrt(2) * sd_velocity
    rms_acceleration = math.sqrt(2) * sd_acceleration
    if not np.all(np.isfinite(rms_velocity) & np.isfinite(rms_acceleration)):
        raise AcquisitionRefusal("noise propagation exceeds numerical precision")
    # The reverse triangle inequality bounds each magnitude error by its
    # two-component vector error. RMS of the latter is sqrt(2) times the SD.
    return DerivativeNoise(sd_velocity, sd_acceleration, rms_velocity, rms_acceleration)
