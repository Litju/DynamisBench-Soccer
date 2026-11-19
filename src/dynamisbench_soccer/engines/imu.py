"""Inertial measurement unit acquisition-to-kinematics engine (RES-701).

Specific force (what an accelerometer senses, ``R^T (a - g)``) and free acceleration
(``a`` in the field frame) are different quantities and are never interchanged: free
acceleration is only produced after an explicit orientation estimate and gravity
compensation, and it carries the orientation variant that produced it. Orientation needs
an initial orientation or a static window; yaw is never observable from accelerometer and
gyroscope alone and is reported as such. Velocity and position are conditional on declared
initial conditions and drift without bound unless a zero-velocity update is declared.
No analytical generator is imported: expected values belong in independent tests.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field, model_validator

from dynamisbench_soccer.qc import spec as qc
from dynamisbench_soccer.signals.schema import (
    BundleKind,
    DeviceClass,
    FloatArray,
    Quantity,
    SignalBundle,
)

ENGINE = "imu.v1"
GRAVITY_M_S2 = 9.80665
Matrix3 = tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]]
Vec3 = tuple[float, float, float]
IDENTITY: Matrix3 = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
Orientation = Literal["gyro_integration", "complementary"]
Integration = Literal["none", "declared_initial_velocity", "zupt_linear_drift"]

LIMITATIONS: tuple[str, ...] = (
    "Specific force is not free acceleration. Free acceleration needs an orientation estimate "
    "and gravity compensation and inherits every orientation error (a tilt error e leaks "
    "about g*e into the horizontal channels and g*(1-cos e) into the vertical channel).",
    "Accelerometer and gyroscope alone cannot observe heading (yaw). Field-frame horizontal "
    "axes are only meaningful relative to the declared initial yaw, and yaw drifts with the "
    "uncorrected gyroscope bias. No magnetometer is admitted.",
    "Gyroscope integration drifts with bias; the complementary variant bounds roll/pitch only "
    "while the sensed magnitude stays within the declared gravity tolerance, which sustained "
    "acceleration violates.",
    "Velocity and position integrate free acceleration: an uncorrected accelerometer bias b "
    "gives a velocity error b*t and a position error b*t^2/2. Declared initial conditions and "
    "zero-velocity windows are required; nothing is assumed.",
    "Sensor-to-segment placement is a declared rotation. Soft-tissue artefact, strap slip and "
    "mounting-frame error are not modelled; slip is only flagged when static gravity direction "
    "changes inconsistently with the integrated rotation.",
    "Flight events from the specific-force magnitude assume free flight with the sensor "
    "rigidly attached; arm swing, soft-tissue motion and threshold choice shift take-off and "
    "landing, so variants and a one-sample resolution bound are reported.",
    "Gyroscope rate and accelerometer range are declared rails; clipping cannot be undone and "
    "saturated intervals invalidate orientation and integrated quantities.",
    "Time-synchronisation uncertainty is a declaration, not a measurement.",
    "Engineering calculations only: no device, placement or physiological qualification.",
)


def _vec(value: Vec3) -> FloatArray:
    return np.asarray(value, dtype=np.float64)


class ImuSpec(BaseModel):
    """Caller declarations; everything defaults to the visible, flagged identity."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    accel_bias_m_s2: Vec3 = (0, 0, 0)
    accel_scale: Vec3 = (1, 1, 1)
    accel_misalignment: Matrix3 = IDENTITY
    gyro_bias_rad_s: Vec3 = (0, 0, 0)
    gyro_scale: Vec3 = (1, 1, 1)
    calibration_declared: bool = False
    mounting_rotation: Matrix3 | None = None  # segment-from-sensor
    placement: str | None = None
    sensor_offset_m: Vec3 | None = None
    accel_range_m_s2: float | None = Field(default=None, gt=0)
    gyro_range_rad_s: float | None = Field(default=None, gt=0)
    initial_orientation: Matrix3 | None = None  # field-from-sensor at the first sample
    initial_yaw_rad: float = 0.0
    static_window_s: tuple[float, float] | None = None
    end_static_window_s: tuple[float, float] | None = None
    sync_uncertainty_s: float = Field(default=0.0, ge=0)
    initial_velocity_m_s: Vec3 | None = None
    initial_position_m: Vec3 | None = None
    zupt_windows_s: tuple[tuple[float, float], ...] = ()

    @model_validator(mode="after")
    def valid(self) -> ImuSpec:
        for name in ("accel_misalignment", "mounting_rotation", "initial_orientation"):
            matrix = getattr(self, name)
            if matrix is None:
                continue
            m = np.asarray(matrix, dtype=np.float64)
            if name != "accel_misalignment" and (
                not np.allclose(m.T @ m, np.eye(3), atol=1e-9)
                or not math.isclose(float(np.linalg.det(m)), 1, abs_tol=1e-9)
            ):
                raise ValueError(f"{name} must be a proper rotation")
            if name == "accel_misalignment" and abs(float(np.linalg.det(m))) < 1e-6:
                raise ValueError("accel_misalignment must be invertible")
        if any(s <= 0 for s in (*self.accel_scale, *self.gyro_scale)):
            raise ValueError("axis scale factors must be positive")
        for window in (self.static_window_s, self.end_static_window_s, *self.zupt_windows_s):
            if window is not None and window[1] <= window[0]:
                raise ValueError("windows must increase")
        return self


class MethodSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    version: Literal["imu.v1"] = "imu.v1"
    orientation: Orientation = "gyro_integration"
    complementary_gain_rad_s: float = Field(default=1.0, gt=0)
    gravity_tolerance_m_s2: float = Field(default=0.5, gt=0)
    integration: Integration = "none"
    flight_threshold_m_s2: float = Field(default=3.0, gt=0)
    flight_min_s: float = Field(default=0.1, gt=0)
    impact_window_s: float = Field(default=0.1, gt=0)
    spike_limit_m_s2: float = Field(default=157.0, gt=0)
    rate_mismatch_limit: float = Field(default=0.01, ge=0)
    gap_factor: float = Field(default=1.5, gt=1)
    gyro_static_limit_rad_s: float = Field(default=0.02, ge=0)
    slip_limit_rad: float = Field(default=0.05, gt=0)
    sync_limit_s: float = Field(default=0.005, ge=0)
    rail_margin_fraction: float = Field(default=0.001, ge=0, lt=0.5)
    sensitivity_thresholds_m_s2: tuple[float, ...] = (2.0, 3.0, 5.0)
    sensitivity_accel_bias_m_s2: float = Field(default=0.05, ge=0)


@dataclass(frozen=True)
class Flight:
    takeoff_s: float
    landing_s: float
    duration_s: float
    resolution_s: float
    landing_peak_m_s2: float | None


@dataclass(frozen=True)
class Processed:
    time_s: FloatArray
    specific_force_m_s2: FloatArray
    angular_velocity_rad_s: FloatArray
    rotation_field_from_sensor: FloatArray | None
    free_acceleration_m_s2: FloatArray | None
    velocity_m_s: FloatArray | None
    position_m: FloatArray | None
    flights: tuple[Flight, ...]
    summary: dict[str, object]
    refusals: dict[str, str]
    qc: tuple[qc.Result, ...]
    uncertainty: dict[str, object]
    limitations: tuple[str, ...]
    method_spec: dict[str, object]
    declarations: dict[str, object]

    @property
    def verdict(self) -> str:
        return qc.verdict(list(self.qc))

    def report(self) -> dict[str, object]:
        return {
            "engine": ENGINE,
            "verdict": self.verdict,
            "diagnostic_only": self.verdict == "reject",
            "estimands": {
                "specific_force": "sensed R^T(a-g) in the sensor frame, m/s^2",
                "free_acceleration": "a in the field frame (gravity compensated), m/s^2"
                if self.free_acceleration_m_s2 is not None
                else None,
            },
            "summary": self.summary,
            "flights": [asdict(f) for f in self.flights],
            "refusals": self.refusals,
            "declarations": self.declarations,
            "uncertainty": self.uncertainty,
            "qc": [asdict(r) for r in self.qc],
            "limitations": list(self.limitations),
            "method_spec": self.method_spec,
            "qualification": "engineering calculations only; no scientific qualification",
        }

    def report_json(self) -> str:
        return json.dumps(self.report(), indent=2, sort_keys=True, allow_nan=False) + "\n"


def _skew(w: FloatArray) -> FloatArray:
    return np.array([[0, -w[2], w[1]], [w[2], 0, -w[0]], [-w[1], w[0], 0]], dtype=np.float64)


def exp_rotation(vector: FloatArray) -> FloatArray:
    """Rodrigues rotation matrix for a rotation vector (exact, orthonormal)."""
    angle = float(np.linalg.norm(vector))
    if angle < 1e-12:
        return np.eye(3) + _skew(vector)
    k = _skew(vector / angle)
    return np.asarray(np.eye(3) + math.sin(angle) * k + (1 - math.cos(angle)) * (k @ k))


def tilt_orientation(up_in_sensor: FloatArray, yaw_rad: float) -> FloatArray:
    """Field-from-sensor rotation that maps the sensed up direction to +z, yaw as declared."""
    u = up_in_sensor / np.linalg.norm(up_in_sensor)
    z = np.array([0.0, 0.0, 1.0])
    axis = np.cross(u, z)
    sin = float(np.linalg.norm(axis))
    cos = float(u @ z)
    if sin < 1e-12:
        tilt = np.eye(3) if cos > 0 else np.diag([1.0, -1.0, -1.0])
    else:
        tilt = exp_rotation(axis / sin * math.atan2(sin, cos))
    return exp_rotation(np.array([0, 0, yaw_rad])) @ tilt


def _channels(bundle: SignalBundle) -> tuple[FloatArray, FloatArray, FloatArray, float]:
    meta = bundle.meta
    if meta.kind is not BundleKind.TIMESERIES or meta.sampling is None:
        raise ValueError("IMU engine accepts timeseries bundles only")
    if meta.device.device_class is not DeviceClass.IMU:
        raise ValueError(
            f"IMU engine does not accept device class {meta.device.device_class.value!r}"
        )
    by_name = {c.name: c for c in meta.channels}
    force, gyro = [], []
    for axis in "xyz":
        f, w = by_name.get(f"specific_force_{axis}"), by_name.get(f"angular_velocity_{axis}")
        if f is None or f.quantity is not Quantity.SPECIFIC_FORCE or f.unit != "m/s^2":
            raise ValueError(f"missing specific_force_{axis} (m/s^2)")
        if w is None or w.quantity is not Quantity.ANGULAR_VELOCITY or w.unit != "rad/s":
            raise ValueError(f"missing angular_velocity_{axis} (rad/s)")
        force.append(bundle[f.name])
        gyro.append(bundle[w.name])
    if "time" in bundle.data:
        time = np.asarray(bundle["time"], dtype=np.float64)
    else:
        time = meta.sampling.t0_s + np.arange(meta.n_rows) / meta.sampling.nominal_hz
    return time, np.column_stack(force), np.column_stack(gyro), meta.sampling.nominal_hz


def _window(time: FloatArray, window: tuple[float, float]) -> npt.NDArray[np.bool_]:
    mask = (time >= window[0] - 1e-10) & (time < window[1] - 1e-10)
    if int(mask.sum()) < 3:
        raise ValueError("window contains fewer than three samples")
    return mask


def _propagate(
    time: FloatArray,
    gyro: FloatArray,
    force: FloatArray,
    r0: FloatArray,
    method: MethodSpec,
) -> FloatArray:
    rotations = np.empty((len(time), 3, 3))
    rotations[0] = r0
    for k in range(len(time) - 1):
        dt = time[k + 1] - time[k]
        omega = 0.5 * (gyro[k] + gyro[k + 1])
        r = rotations[k]
        if method.orientation == "complementary":
            f = force[k]
            norm = float(np.linalg.norm(f))
            if abs(norm - GRAVITY_M_S2) <= method.gravity_tolerance_m_s2 and norm > 0:
                predicted_up = r.T @ np.array([0.0, 0.0, 1.0])
                omega = omega + method.complementary_gain_rad_s * np.cross(f / norm, predicted_up)
        rotations[k + 1] = r @ exp_rotation(omega * dt)
    return rotations


def _check(
    identifier: str,
    severity: qc.Severity,
    ok: bool,
    detail_ok: str,
    detail_bad: str,
    **values: float,
) -> qc.Result:
    return qc.Result(
        identifier,
        identifier.lower().replace("-", "_"),
        severity,
        qc.Status.PASS if ok else qc.Status.FAIL,
        detail_ok if ok else detail_bad,
        values,
    )


def _flights(
    time: FloatArray, magnitude: FloatArray, threshold: float, minimum_s: float, window_s: float
) -> tuple[Flight, ...]:
    below = magnitude < threshold
    found: list[Flight] = []
    i, n = 0, len(time)
    while i < n:
        if not below[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and below[j + 1]:
            j += 1
        if i > 0 and j < n - 1:
            takeoff, landing = float(time[i - 1]), float(time[j + 1])
            if landing - takeoff >= minimum_s:
                after = (time > landing) & (time <= landing + window_s)
                found.append(
                    Flight(
                        takeoff,
                        landing,
                        landing - takeoff,
                        float(max(time[i] - time[i - 1], time[j + 1] - time[j])),
                        float(np.max(magnitude[after])) if after.any() else None,
                    )
                )
        i = j + 1
    return tuple(found)


def process(
    bundle: SignalBundle, spec: ImuSpec | None = None, method: MethodSpec | None = None
) -> Processed:
    spec = ImuSpec() if spec is None else ImuSpec.model_validate(spec.model_dump())
    method = MethodSpec() if method is None else MethodSpec.model_validate(method.model_dump())
    time, raw_force, raw_gyro, nominal_hz = _channels(bundle)
    n = len(time)
    if n < 4 or not np.all(np.diff(time) > 0):
        raise ValueError("at least four strictly increasing timestamps are required")

    # Calibration: f = M^-1 ((raw / scale) - bias), gyro likewise without misalignment.
    scale = _vec(spec.accel_scale)
    misalignment = np.asarray(spec.accel_misalignment, dtype=np.float64)
    force = (raw_force / scale - _vec(spec.accel_bias_m_s2)) @ np.linalg.inv(misalignment).T
    gyro = raw_gyro / _vec(spec.gyro_scale) - _vec(spec.gyro_bias_rad_s)
    mount = (
        np.asarray(spec.mounting_rotation, dtype=np.float64)
        if spec.mounting_rotation is not None
        else np.eye(3)
    )
    if spec.mounting_rotation is not None and not (spec.placement and spec.placement.strip()):
        raise ValueError("a mounting rotation needs a declared placement description")
    force_seg = force @ mount.T
    gyro_seg = gyro @ mount.T

    dt = np.diff(time)
    median_dt = float(np.median(dt))
    measured_hz = 1 / median_dt
    magnitude = np.linalg.norm(force, axis=1)
    refusals: dict[str, str] = {}
    checks: list[qc.Result] = [
        _check(
            "IMU-RATE",
            qc.Severity.FLAG,
            abs(measured_hz - nominal_hz) / nominal_hz <= method.rate_mismatch_limit,
            "measured rate agrees with nominal rate",
            "measured rate differs from nominal rate: clock or rate error",
            measured_hz=measured_hz,
            nominal_hz=nominal_hz,
        ),
        _check(
            "IMU-GAPS",
            qc.Severity.REJECT,
            float(dt.max()) <= method.gap_factor * median_dt,
            "no missing samples or timing gaps",
            "missing samples or timing gaps: orientation and integrals are invalid across them",
            max_gap_s=float(dt.max()),
        ),
        _check(
            "IMU-SYNC",
            qc.Severity.FLAG,
            spec.sync_uncertainty_s <= method.sync_limit_s,
            "declared synchronisation uncertainty within limit",
            "declared synchronisation uncertainty exceeds limit",
            sync_uncertainty_s=spec.sync_uncertainty_s,
        ),
        _check(
            "IMU-SPIKE",
            qc.Severity.FLAG,
            float(magnitude.max()) <= method.spike_limit_m_s2,
            "no impact artefact above the declared spike limit",
            "specific-force magnitude exceeds the spike limit: impact artefact or saturation",
            peak_m_s2=float(magnitude.max()),
        ),
    ]
    rails: list[tuple[str, int | None]] = []
    for label, values, rail in (
        ("accelerometer", raw_force, spec.accel_range_m_s2),
        ("gyroscope", raw_gyro, spec.gyro_range_rad_s),
    ):
        if rail is None:
            rails.append((label, None))
            continue
        near = np.abs(values) >= rail * (1 - method.rail_margin_fraction)
        rails.append((label, int(near.sum())))
    for label, count in rails:
        checks.append(
            qc.Result(
                f"IMU-SAT-{label[:4].upper()}",
                f"imu_sat_{label}",
                qc.Severity.REJECT,
                qc.Status.NOT_APPLICABLE
                if count is None
                else qc.Status.FAIL
                if count
                else qc.Status.PASS,
                "no range declared: saturation cannot be assessed"
                if count is None
                else f"{count} samples at the {label} rail: clipped"
                if count
                else "no clipping",
                {} if count is None else {"samples_at_rail": float(count)},
            )
        )

    # Orientation: needs an initial orientation or a static window; yaw is never observable.
    rotations: FloatArray | None = None
    initial_source = None
    r0: FloatArray | None = None
    static_mask = None
    if spec.static_window_s is not None:
        static_mask = _window(time, spec.static_window_s)
        gyro_static = float(np.linalg.norm(gyro[static_mask].mean(axis=0)))
        checks.append(
            _check(
                "IMU-GYROBIAS",
                qc.Severity.FLAG,
                gyro_static <= method.gyro_static_limit_rad_s,
                "gyroscope mean over the static window is within the limit",
                "gyroscope mean over the static window exceeds the limit: uncorrected bias "
                "or the window is not static",
                static_gyro_rad_s=gyro_static,
            )
        )
    if spec.initial_orientation is not None:
        r0 = np.asarray(spec.initial_orientation, dtype=np.float64)
        initial_source = "declared initial orientation"
    elif static_mask is not None:
        r0 = tilt_orientation(force[static_mask].mean(axis=0), spec.initial_yaw_rad)
        initial_source = "static tilt (roll/pitch from gravity; yaw is the declared initial yaw)"
    if r0 is None:
        refusals["orientation"] = (
            "no initial orientation and no static window: orientation is unobservable"
        )
        free = velocity = position = None
    else:
        rotations = _propagate(time, gyro, force, r0, method)
        free = np.einsum("nij,nj->ni", rotations, force) + np.array([0, 0, -GRAVITY_M_S2])
        velocity = position = None
        if spec.end_static_window_s is not None and static_mask is not None:
            end = _window(time, spec.end_static_window_s)
            start_up = rotations[static_mask].mean(axis=0) @ force[static_mask].mean(axis=0)
            end_up = rotations[end].mean(axis=0) @ force[end].mean(axis=0)
            cos = float(start_up @ end_up / (np.linalg.norm(start_up) * np.linalg.norm(end_up)))
            tilt = math.acos(min(1.0, max(-1.0, cos)))
            checks.append(
                _check(
                    "IMU-SLIP",
                    qc.Severity.FLAG,
                    tilt <= method.slip_limit_rad,
                    "static gravity direction is consistent with the integrated rotation",
                    "static gravity direction changed inconsistently with the integrated "
                    "rotation: sensor slip, gyroscope drift or sustained acceleration",
                    inconsistency_rad=tilt,
                )
            )
        if method.integration == "none":
            refusals["integration"] = "integration not requested"
        elif spec.initial_velocity_m_s is None:
            refusals["integration"] = "refused: no declared initial velocity"
        else:
            v = np.column_stack(
                [
                    _vec(spec.initial_velocity_m_s)[i]
                    + np.concatenate(([0.0], np.cumsum(0.5 * (free[1:, i] + free[:-1, i]) * dt)))
                    for i in range(3)
                ]
            )
            if method.integration == "zupt_linear_drift":
                if not spec.zupt_windows_s:
                    raise ValueError("zupt_linear_drift needs declared zero-velocity windows")
                last = _window(time, spec.zupt_windows_s[-1])
                residual = v[last].mean(axis=0)
                t_last = float(time[last].mean() - time[0])
                v = v - np.outer(time - time[0], residual / t_last)
            velocity = v
            if spec.initial_position_m is None:
                refusals["position"] = "refused: no declared initial position"
            else:
                position = np.column_stack(
                    [
                        _vec(spec.initial_position_m)[i]
                        + np.concatenate(([0.0], np.cumsum(0.5 * (v[1:, i] + v[:-1, i]) * dt)))
                        for i in range(3)
                    ]
                )

    flights = _flights(
        time, magnitude, method.flight_threshold_m_s2, method.flight_min_s, method.impact_window_s
    )
    sensitivity: dict[str, object] = {
        f"flight_threshold_{t:g}": [
            f.duration_s
            for f in _flights(time, magnitude, t, method.flight_min_s, method.impact_window_s)
        ]
        for t in method.sensitivity_thresholds_m_s2
    }
    duration = float(time[-1] - time[0])
    bias = method.sensitivity_accel_bias_m_s2
    uncertainty: dict[str, object] = {
        "flight_time_by_threshold_s": sensitivity,
        "flight_time_resolution_s": [f.resolution_s for f in flights],
        "velocity_error_bound_m_s": bias * duration,
        "position_error_bound_m": 0.5 * bias * duration**2,
        "bias_assumed_m_s2": bias,
        "tilt_leakage_per_rad_m_s2": GRAVITY_M_S2,
        "scope": "conditional sensitivity, not a complete uncertainty budget",
    }
    return Processed(
        time_s=time,
        specific_force_m_s2=force_seg,
        angular_velocity_rad_s=gyro_seg,
        rotation_field_from_sensor=rotations,
        free_acceleration_m_s2=free,
        velocity_m_s=velocity,
        position_m=position,
        flights=flights,
        summary={
            "samples": n,
            "measured_hz": measured_hz,
            "peak_specific_force_m_s2": float(magnitude.max()),
            "peak_angular_rate_rad_s": float(np.linalg.norm(gyro, axis=1).max()),
            "orientation_source": initial_source,
            "orientation_variant": method.orientation if r0 is not None else None,
            "yaw_observable": False,
            "flight_count": len(flights),
        },
        refusals=refusals,
        qc=tuple(checks),
        uncertainty=uncertainty,
        limitations=LIMITATIONS,
        method_spec=method.model_dump(mode="json"),
        declarations={
            "calibration_declared": spec.calibration_declared,
            "placement": spec.placement,
            "mounting_declared": spec.mounting_rotation is not None,
            "sensor_offset_m": spec.sensor_offset_m,
            "sync_uncertainty_s": spec.sync_uncertainty_s,
        },
    )
