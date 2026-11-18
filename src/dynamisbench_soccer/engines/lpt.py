"""Motion and conditional vertical mass-model power from v2 LPT/encoder acquisitions."""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from scipy.interpolate import CubicSpline
from scipy.signal import butter, sosfiltfilt

from dynamisbench_soccer.engines.lpt_acquisition import (
    AcquisitionResult,
    CableSpec,
    GeometrySpec,
    acquire,
    vertical,
)
from dynamisbench_soccer.signals.schema import FloatArray, SignalBundle

ENGINE = "lpt-encoder@1"
GRAVITY_M_S2 = 9.80665
type Method = Literal["finite_difference", "butterworth_then_difference", "cubic_spline"]


class MethodSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    name: Method = "finite_difference"
    cutoff_hz: float = Field(default=12, gt=0)
    timebase_scale_ppm: float = Field(default=0, gt=-1e6)
    summary_trim_s: float = Field(default=0, ge=0)


class PowerSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    system_mass_kg: float = Field(gt=0)
    rigid_vertical_translation: Literal[True]


class SensitivitySpec(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    angle_delta_rad: float = Field(default=0.035, gt=0)
    geometry_delta_m: float = Field(default=0.01, gt=0)
    clock_delta_ppm: float = Field(default=1000, gt=0)
    cutoff_fraction: float = Field(default=0.25, gt=0, lt=1)


@dataclass(frozen=True)
class Motion:
    time_s: FloatArray
    displacement_m: FloatArray
    velocity_m_s: FloatArray
    acceleration_m_s2: FloatArray
    vertical_mass_model_power_w: FloatArray | None

    def series(self) -> dict[str, FloatArray]:
        result = {
            "displacement_m": self.displacement_m,
            "velocity_m_s": self.velocity_m_s,
            "acceleration_m_s2": self.acceleration_m_s2,
        }
        if self.vertical_mass_model_power_w is not None:
            result["vertical_mass_model_power_w"] = self.vertical_mass_model_power_w
        return result


def _motion(
    time: FloatArray, displacement: FloatArray, method: MethodSpec, power: PowerSpec | None
) -> Motion:
    time = (time - time[0]) * (1 + method.timebase_scale_ppm * 1e-6)
    if len(time) < 4 or not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
        raise ValueError("kinematics require four or more valid samples on an increasing timebase")
    if method.name == "cubic_spline":
        spline = CubicSpline(time, displacement, bc_type="not-a-knot")
        velocity, acceleration = spline(time, 1), spline(time, 2)
    else:
        grid = time
        if method.name == "butterworth_then_difference":
            grid = np.linspace(0, time[-1], len(time))
            frequency = (len(grid) - 1) / time[-1]
            if method.cutoff_hz >= 0.5 / np.max(np.diff(time)):
                raise ValueError("cutoff must be below the sampling limit of the longest interval")
            if len(grid) <= 15:
                raise ValueError("fourth-order zero-phase Butterworth requires at least 16 samples")
            sos = butter(4, method.cutoff_hz, fs=frequency, output="sos")
            pad = min(len(grid) - 1, max(15, ceil(3 * frequency / method.cutoff_hz)))
            displacement = sosfiltfilt(sos, np.interp(grid, time, displacement), padlen=pad)
        velocity = np.gradient(displacement, grid, edge_order=2)
        acceleration = np.gradient(velocity, grid, edge_order=2)
        if method.name == "butterworth_then_difference":
            displacement = np.interp(time, grid, displacement)
            velocity = np.interp(time, grid, velocity)
            acceleration = np.interp(time, grid, acceleration)
    model_power = None
    if power is not None:
        model_power = power.system_mass_kg * (GRAVITY_M_S2 + acceleration) * velocity
    motion = Motion(time, displacement - displacement[0], velocity, acceleration, model_power)
    if any(not np.isfinite(values).all() for values in motion.series().values()):
        raise ValueError("nonfinite kinematics or power")
    return motion


def _window(time: FloatArray, trim_s: float) -> np.ndarray[tuple[int], np.dtype[np.bool_]]:
    keep: np.ndarray[tuple[int], np.dtype[np.bool_]] = (time >= trim_s) & (
        time <= time[-1] - trim_s
    )
    if not keep.any():
        raise ValueError("summary trim leaves no samples")
    return keep


def _sensitivity(
    acquisition: AcquisitionResult,
    displacement: FloatArray,
    geometry: GeometrySpec,
    method: MethodSpec,
    power: PowerSpec | None,
    options: SensitivitySpec,
    baseline: Motion,
    keep: np.ndarray[tuple[int], np.dtype[np.bool_]],
) -> dict[str, object]:
    time = acquisition.time_s
    variants: list[tuple[str, dict[str, float | str], float, float]] = [
        (name, dict(name=name), 0.0, 0.0)
        for name in ("finite_difference", "cubic_spline", "butterworth_then_difference")
        if name != method.name
    ]
    for sign, label in ((-1, "lower"), (1, "upper")):
        variants.extend(
            [
                (
                    f"cutoff_{label}",
                    dict(
                        name="butterworth_then_difference",
                        cutoff_hz=method.cutoff_hz * (1 + sign * options.cutoff_fraction),
                    ),
                    0.0,
                    0.0,
                ),
                (
                    f"timebase_scale_{label}",
                    dict(
                        timebase_scale_ppm=method.timebase_scale_ppm
                        + sign * options.clock_delta_ppm,
                    ),
                    0.0,
                    0.0,
                ),
            ]
        )
        if acquisition.geometry_method == "constant_cable_angle_vertical_motion":
            variants.append((f"cable_angle_{label}", {}, sign * options.angle_delta_rad, 0.0))
        elif acquisition.vertical_displacement_m is not None:
            variants.append((f"geometry_{label}", {}, 0.0, sign * options.geometry_delta_m))

    result: dict[str, object] = {}
    baseline_series = baseline.series()
    for label, overrides, angle, geometry_error in variants:
        parameters = {**method.model_dump(), **overrides}
        record: dict[str, object] = {
            "method": parameters,
            "cable_angle_delta_rad": angle,
            "geometry_delta_m": geometry_error,
        }
        try:
            trial_disp = displacement
            if angle or geometry_error:
                _, changed = vertical(acquisition.cables, geometry, angle, geometry_error)
                assert changed is not None
                trial_disp = changed
            variant = _motion(time, trial_disp, MethodSpec.model_validate(parameters), power)
            differences = {}
            for key, series in variant.series().items():
                alternate, original = series[keep], baseline_series[key][keep]
                differences[key] = {
                    "rmse": float(np.linalg.norm(alternate - original) / np.sqrt(len(original))),
                    "peak_delta": float(np.max(alternate) - np.max(original)),
                }
            record["differences"] = differences
        except ValueError as exc:
            record["error"] = str(exc)
        result[label] = record
    return result


def process(
    bundle: SignalBundle,
    cables: tuple[CableSpec, ...],
    geometry: GeometrySpec,
    method: MethodSpec | None = None,
    power: PowerSpec | None = None,
    sensitivity: SensitivitySpec | None = None,
) -> dict[str, object]:
    method = method or MethodSpec()
    result = acquire(bundle, cables, geometry)
    vertical_disp = result.vertical_displacement_m
    rejected = any(c.faults for c in result.cables)
    estimand = None
    motion = None
    summaries = None
    comparisons = None
    limitations = list(dict.fromkeys(result.limitations))
    if not rejected:
        if vertical_disp is None:
            estimand = "cable_displacement"
            displacement = result.cables[0].length_m
            if power is not None:
                limitations.append(
                    "power withheld because vertical attachment motion is unavailable"
                )
        else:
            assert result.attachment_point
            label = result.attachment_point.casefold()
            estimand = (
                "bar_vertical_displacement"
                if label in {"bar", "barbell"}
                else "attachment_point_vertical_displacement"
            )
            displacement = vertical_disp
        active_power = power if vertical_disp is not None else None
        motion = _motion(result.time_s, displacement, method, active_power)
        keep = _window(motion.time_s, method.summary_trim_s)
        summaries = {
            name: {
                "min": float(np.min(values[keep])),
                "max": float(np.max(values[keep])),
            }
            for name, values in motion.series().items()
        }
        if sensitivity is not None:
            comparisons = _sensitivity(
                result, displacement, geometry, method, active_power, sensitivity, motion, keep
            )

    acquisition = bundle.meta.acquisition
    assert acquisition is not None and bundle.meta.sampling is not None
    clocks = {c.source.clock for c in result.cables}
    return {
        "engine": ENGINE,
        "verdict": "reject" if rejected else "processed",
        "bundle_id": bundle.meta.bundle_id,
        "data_sha256": bundle.meta.data_sha256,
        "attachment_point": result.attachment_point,
        "estimand": estimand,
        "geometry_method": result.geometry_method,
        "geometry": geometry.model_dump(),
        "method": method.model_dump(),
        "power_model": power.model_dump() if power else None,
        "limitations": limitations,
        "analysis_clock": (
            result.cables[0].source.clock if len(clocks) == 1 else acquisition.reference_clock
        ),
        "nominal_hz": bundle.meta.sampling.nominal_hz,
        "dual_clock_alignments": [
            link.model_dump()
            for link in acquisition.alignments
            if len(clocks) > 1 and link.clock in clocks
        ],
        "cables": [
            {
                "spec": c.spec.model_dump(),
                "source_sha256": c.source.source_sha256,
                "calibration": [record.model_dump() for record in c.source.calibration],
                "recorded_geometry": c.source.geometry.model_dump(),
                "clock": c.source.clock,
                "native_time_s": c.time_s.tolist(),
                "median_native_hz": float(1 / np.median(np.diff(c.time_s))),
                "resolution_m": c.resolution_m,
                "cable_length_m": c.length_m.tolist() if c.absolute else None,
                "cable_displacement_m": (c.length_m - c.length_m[0]).tolist(),
                "faults": c.faults,
            }
            for c in result.cables
        ],
        "motion": (
            {
                "elapsed_s": motion.time_s.tolist(),
                **{name: values.tolist() for name, values in motion.series().items()},
            }
            if motion
            else None
        ),
        "summary": summaries,
        "sensitivity": comparisons,
    }
