"""Raw load-cell acquisition: declared units, declared calibration, timing and integrity.

Nothing here is guessed. Raw units are explicit, calibrations come only from declared
records (schema ``Calibration``) or declared static calibration points, and every
integrity check that cannot be evaluated from the declarations says so. No analytical
generator is imported; expected values belong in independent tests.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Literal, get_args

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from dynamisbench_soccer.qc import spec as qc
from dynamisbench_soccer.signals.schema import (
    BundleKind,
    BundleMeta,
    Calibration,
    DeviceClass,
    FloatArray,
    Geometry,
    Quantity,
    SignalBundle,
)

#: Exact conversion (CGPM 1901 standard gravity); kgf is never approximated.
KGF_TO_N = 9.80665

RawUnit = Literal["V", "mV", "count", "N", "kgf"]
RAW_UNITS: frozenset[str] = frozenset(get_args(RawUnit))
ELECTRICAL_UNITS: frozenset[str] = frozenset({"V", "mV", "count"})
FORCE_UNITS: frozenset[str] = frozenset({"N", "kgf"})
LOAD_CELL_CLASSES = frozenset({DeviceClass.LOAD_CELL, DeviceClass.ISOMETRIC_DYNAMOMETER})
_QUANTITY_UNIT: dict[Quantity, RawUnit] = {
    Quantity.VOLTAGE: "V",
    Quantity.FORCE: "N",
    Quantity.COUNT: "count",
}


def _to_newton(unit: str) -> float:
    if unit == "N":
        return 1.0
    if unit == "kgf":
        return KGF_TO_N
    raise ValueError(f"{unit!r} is not a declared force unit (N or kgf)")


def require_unit(unit: str) -> RawUnit:
    if unit not in RAW_UNITS:
        raise ValueError(
            f"unknown raw unit {unit!r}; declare one of {sorted(RAW_UNITS)} (nothing is guessed)"
        )
    return unit  # type: ignore[return-value]


class CalibrationPoints(BaseModel):
    """Declared static calibration: known applied loads and the raw readings they produced."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    method: str = Field(min_length=1)
    reference: str | None = None
    raw_unit: Literal["V", "mV", "count"]
    load_unit: Literal["N", "kgf"]
    applied_loads: tuple[float, ...] = Field(min_length=2)
    readings: tuple[float, ...] = Field(min_length=2)

    @model_validator(mode="after")
    def valid_points(self) -> CalibrationPoints:
        if len(self.applied_loads) != len(self.readings):
            raise ValueError("each applied load needs exactly one raw reading")
        if len(set(self.readings)) < 2 or len(set(self.applied_loads)) < 2:
            raise ValueError("calibration needs at least two distinct loads and readings")
        return self


@dataclass(frozen=True)
class CalibrationFit:
    """Linear map force_N = gain_n_per_unit * raw + offset_n, with fit diagnostics."""

    source: Literal["record", "points", "passthrough", "kgf_exact"]
    raw_unit: RawUnit
    gain_n_per_unit: float
    offset_n: float
    n_points: int | None
    residuals_n: tuple[float, ...]
    rms_residual_n: float | None
    max_abs_residual_n: float | None
    r_squared: float | None
    gain_standard_error: float | None
    full_scale_n: float | None
    record: Calibration | None
    description: str

    def apply(self, raw: FloatArray, gain_error: float = 0.0) -> FloatArray:
        """Gain error is a relative sensitivity perturbation, never a correction."""
        return np.asarray(
            self.gain_n_per_unit * (1 + gain_error) * raw + self.offset_n, dtype=np.float64
        )

    def summary(self) -> dict[str, object]:
        return {
            "source": self.source,
            "raw_unit": self.raw_unit,
            "gain_n_per_unit": self.gain_n_per_unit,
            "offset_n": self.offset_n,
            "n_points": self.n_points,
            "residuals_n": list(self.residuals_n),
            "rms_residual_n": self.rms_residual_n,
            "max_abs_residual_n": self.max_abs_residual_n,
            "r_squared": self.r_squared,
            "gain_standard_error": self.gain_standard_error,
            "full_scale_n": self.full_scale_n,
            "record": self.record.model_dump(mode="json") if self.record is not None else None,
            "description": self.description,
        }


def fit_points(points: CalibrationPoints) -> CalibrationFit:
    """Ordinary least squares of load (N) on reading; two points are an exact solve."""
    points = CalibrationPoints.model_validate(points.model_dump())
    loads = np.array(points.applied_loads, dtype=np.float64) * _to_newton(points.load_unit)
    readings = np.array(points.readings, dtype=np.float64)
    design = np.column_stack([readings, np.ones_like(readings)])
    (gain, offset), *_ = np.linalg.lstsq(design, loads, rcond=None)
    if not np.isfinite([gain, offset]).all() or gain == 0:
        raise ValueError("calibration points do not define a finite nonzero gain")
    residuals = loads - (gain * readings + offset)
    n = len(loads)
    dof = n - 2
    total = float(np.sum((loads - loads.mean()) ** 2))
    sse = float(np.sum(residuals**2))
    se = None
    if dof > 0:
        sxx = float(np.sum((readings - readings.mean()) ** 2))
        se = float(np.sqrt(sse / dof / sxx))
    record = Calibration(
        date=points.date,
        method=f"least-squares {n}-point fit: {points.method}",
        channel=None,
        reference=points.reference,
        uncertainty=float(np.sqrt(sse / dof)) if dof > 0 else None,
        notes=f"force_N = {gain!r} * raw_{points.raw_unit} + {offset!r}",
    )
    return CalibrationFit(
        "points",
        points.raw_unit,
        float(gain),
        float(offset),
        n,
        tuple(float(r) for r in residuals),
        float(np.sqrt(sse / n)),
        float(np.max(np.abs(residuals))),
        1 - sse / total if total > 0 else None,
        se,
        float(np.max(np.abs(loads))),
        record,
        "two-point exact solve; residuals carry no linearity information"
        if n == 2
        else f"ordinary least squares over {n} declared points ({dof} residual dof)",
    )


def fit_record(record: Calibration, raw_unit: RawUnit) -> CalibrationFit:
    """Interpret a declared linear record in either direction, with units checked exactly."""
    record = Calibration.model_validate(record.model_dump())
    if record.gain is None or record.offset is None:
        raise ValueError("calibration record must declare both gain and offset (use 0 explicitly)")
    if record.bias is not None or record.scale is not None or record.cosine_angle_rad is not None:
        raise ValueError("vector/cosine calibration terms do not apply to a uniaxial load cell")
    i, o = record.input_unit, record.output_unit
    if i in ELECTRICAL_UNITS and o in FORCE_UNITS:
        # Readout form: force = gain * raw + offset.
        if i != raw_unit:
            raise ValueError(f"calibration input unit {i!r} does not match raw unit {raw_unit!r}")
        k = _to_newton(o)
        gain, offset = record.gain * k, record.offset * k
        description = f"declared readout record: force_{o} = gain*raw_{i} + offset"
    elif i in FORCE_UNITS and o in ELECTRICAL_UNITS:
        # Sensitivity form: raw = gain * force + offset, inverted exactly.
        if o != raw_unit:
            raise ValueError(f"calibration output unit {o!r} does not match raw unit {raw_unit!r}")
        k = _to_newton(i)
        gain, offset = k / record.gain, -k * record.offset / record.gain
        description = f"declared sensitivity record raw_{o} = gain*force_{i} + offset, inverted"
    else:
        raise ValueError(f"calibration units {i!r}->{o!r} do not map raw electrical units to force")
    return CalibrationFit(
        "record",
        raw_unit,
        gain,
        offset,
        None,
        (),
        None,
        None,
        None,
        None,
        None,
        record,
        description,
    )


@dataclass(frozen=True)
class RawRecord:
    """A single uniaxial raw channel with declared unit and sampling.

    Use this for inputs the canonical signal schema cannot hold (mV, kgf) or for raw
    acquisition files; bundles are converted with :func:`from_bundle`.
    """

    values: FloatArray
    unit: str
    nominal_hz: float
    time_s: FloatArray | None = None
    t0_s: float = 0.0
    label: str = "raw"
    device_class: DeviceClass = DeviceClass.LOAD_CELL
    calibration: tuple[Calibration, ...] = ()
    geometry: Geometry | None = None
    bundle: SignalBundle | None = None
    sync_uncertainty_s: float = 0.0
    clock_identity: str | None = None


def input_digest(record: RawRecord) -> str:
    """Hash of declared unit/rate/label/calibration plus little-endian samples."""
    if record.bundle is not None:
        raw = record.bundle.meta.model_dump(mode="json")
        raw["data_sha256"] = None
        digest = hashlib.sha256(
            json.dumps(raw, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        )
        for values in record.bundle.data.values():
            digest.update(values.astype("<f8", copy=False).tobytes())
        return digest.hexdigest()
    head = {
        "unit": record.unit,
        "nominal_hz": record.nominal_hz,
        "t0_s": record.t0_s,
        "label": record.label,
        "device_class": record.device_class.value,
        "calibration": [c.model_dump(mode="json") for c in record.calibration],
        "geometry": record.geometry.model_dump(mode="json") if record.geometry else None,
    }
    digest = hashlib.sha256(json.dumps(head, sort_keys=True, allow_nan=True).encode())
    digest.update(np.asarray(record.values, dtype="<f8").tobytes())
    if record.time_s is not None:
        digest.update(np.asarray(record.time_s, dtype="<f8").tobytes())
    return digest.hexdigest()


def from_bundle(bundle: SignalBundle, channel: str | None = None) -> RawRecord:
    """Select one raw load-cell channel, its declared calibration records and geometry."""
    meta = BundleMeta.model_validate(bundle.meta.model_dump())
    SignalBundle(meta, bundle.data)  # Revalidate (including finiteness) at the trust boundary.
    if meta.kind is not BundleKind.TIMESERIES or meta.sampling is None:
        raise ValueError("load-cell engine accepts timeseries bundles only")
    candidates = [c for c in meta.channels if c.quantity in _QUANTITY_UNIT]
    if channel is not None:
        candidates = [c for c in candidates if c.name == channel]
    if len(candidates) != 1:
        raise ValueError("declare exactly one raw voltage, count or force channel")
    spec = candidates[0]
    source = None
    if meta.acquisition is not None:
        source = next(s for s in meta.acquisition.sources if spec.name in s.channels)
    device = source.device.device_class if source is not None else meta.device.device_class
    if device not in LOAD_CELL_CLASSES:
        raise ValueError(f"load-cell engine does not accept device class {device.value!r}")
    records = [*meta.calibration, *(source.calibration if source is not None else [])]
    unique: list[Calibration] = []
    for r in records:
        if r.channel in {None, spec.name} and r not in unique:
            unique.append(r)
    native = (
        bundle["time"].copy()
        if "time" in bundle.data
        else meta.sampling.t0_s + np.arange(meta.n_rows) / meta.sampling.nominal_hz
    )
    time, sync, identity = _align(meta, spec.name, native)
    return RawRecord(
        values=bundle[spec.name].copy(),
        unit=_QUANTITY_UNIT[spec.quantity],
        nominal_hz=meta.sampling.nominal_hz,
        time_s=time,
        t0_s=float(time[0]),
        label=f"{meta.bundle_id}:{spec.name}",
        device_class=device,
        calibration=tuple(unique),
        geometry=source.geometry if source is not None else meta.geometry,
        bundle=bundle,
        sync_uncertainty_s=sync,
        clock_identity=identity,
    )


def _align(
    meta: BundleMeta, channel: str, native: FloatArray
) -> tuple[FloatArray, float, str | None]:
    acquisition = meta.acquisition
    if acquisition is None:
        return native, 0.0, None
    source = next(s for s in acquisition.sources if channel in s.channels)
    reference = next(c for c in acquisition.clocks if c.id == acquisition.reference_clock)
    identity = f"{reference.id}:{reference.epoch}"
    if source.clock == acquisition.reference_clock:
        return native, 0.0, identity
    alignment = next(a for a in acquisition.alignments if a.clock == source.clock)
    time = native + alignment.offset_s + (native - alignment.origin_s) * alignment.drift_ppm * 1e-6
    return time, alignment.uncertainty_s, identity


class CellSpec(BaseModel):
    """Instrument and rig declarations. Absent declarations disable, never default, checks."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    channel: str | None = None
    calibration: Calibration | None = None
    calibration_points: CalibrationPoints | None = None
    capacity_n: float | None = Field(default=None, gt=0)
    raw_range: tuple[float, float] | None = None
    rail_margin_fraction: float = Field(default=0.001, ge=0, lt=0.5)
    lever_arm_m: float | None = Field(default=None, gt=0, lt=2.0)
    moment_arm_definition: str | None = None
    gravity_torque_n_m: float | None = None
    temperature_c: tuple[float, float] | None = None
    span_temperature_coefficient_per_c: float | None = Field(default=None, ge=0)
    anti_aliasing_hz: float | None = Field(default=None, gt=0)
    expected_peak_n: tuple[float, float] | None = None

    @model_validator(mode="after")
    def valid_declarations(self) -> CellSpec:
        if self.calibration is not None and self.calibration_points is not None:
            raise ValueError("declare one calibration source: a record or calibration points")
        for name in ("raw_range", "expected_peak_n"):
            bounds = getattr(self, name)
            if bounds is not None and not bounds[0] < bounds[1]:
                raise ValueError(f"{name} must increase")
        if (self.temperature_c is None) != (self.span_temperature_coefficient_per_c is None):
            raise ValueError("temperature drift needs both declared temperatures and coefficient")
        return self


@dataclass(frozen=True)
class Timing:
    time_s: FloatArray
    uniform: bool
    sample_interval_s: float
    measured_hz: float
    nominal_hz: float
    rate_mismatch: float
    timestamps_declared: bool
    sync_uncertainty_s: float
    clock_identity: str | None


@dataclass(frozen=True)
class Acquired:
    record: RawRecord
    raw: FloatArray
    force_n: FloatArray
    calibration: CalibrationFit
    timing: Timing
    cell: CellSpec

    def with_gain_error(self, error: float) -> Acquired:
        return Acquired(
            self.record,
            self.raw,
            self.calibration.apply(self.raw, error),
            self.calibration,
            self.timing,
            self.cell,
        )


def _timing(record: RawRecord, n: int, tolerance: float) -> Timing:
    if not np.isfinite(record.nominal_hz) or record.nominal_hz <= 0:
        raise ValueError("declared positive finite sampling rate required")
    declared = record.time_s is not None
    if record.time_s is not None:
        time = np.asarray(record.time_s, dtype=np.float64).copy()
        if time.shape != (n,):
            raise ValueError("timestamps must match sample count")
        if not np.isfinite(time).all():
            raise ValueError("non-finite timestamps are rejected")
    else:
        time = record.t0_s + np.arange(n, dtype=np.float64) / record.nominal_hz
    if n < 3 or np.any(np.diff(time) <= 0):
        raise ValueError("at least three strictly increasing timestamps required")
    dt = np.diff(time)
    step = float(np.median(dt))
    uniform = bool(np.allclose(dt, step, rtol=tolerance, atol=1e-12))
    measured = 1 / step
    return Timing(
        time,
        uniform,
        step,
        measured,
        record.nominal_hz,
        abs(measured - record.nominal_hz) / record.nominal_hz,
        declared,
        record.sync_uncertainty_s,
        record.clock_identity,
    )


def _exact_fit(
    source: Literal["passthrough", "kgf_exact"], unit: RawUnit, gain: float, description: str
) -> CalibrationFit:
    return CalibrationFit(
        source, unit, gain, 0.0, None, (), None, None, None, None, None, None, description
    )


def resolve_calibration(record: RawRecord, cell: CellSpec) -> CalibrationFit:
    unit = require_unit(record.unit)
    declared = cell.calibration is not None or cell.calibration_points is not None
    if unit in FORCE_UNITS:
        if declared:
            raise ValueError(
                "raw data are already in force units; explicit calibration would apply twice"
            )
        # Records on force-unit channels are upstream provenance, never re-applied.
        if unit == "N":
            return _exact_fit(
                "passthrough",
                unit,
                1.0,
                "declared newton data passed through; upstream records kept as provenance",
            )
        return _exact_fit(
            "kgf_exact", unit, KGF_TO_N, "declared kgf converted by the exact factor 9.80665 N/kgf"
        )
        return CalibrationFit(
            "kgf_exact", unit, KGF_TO_N, 0.0, None, (), None, None, None, None, None, None,
            "declared kgf converted by the exact factor 9.80665 N/kgf",
        )  # fmt: skip
    if cell.calibration_points is not None:
        if cell.calibration_points.raw_unit != unit:
            raise ValueError(
                f"calibration points use {cell.calibration_points.raw_unit!r}, raw unit is {unit!r}"
            )
        return fit_points(cell.calibration_points)
    if cell.calibration is not None:
        return fit_record(cell.calibration, unit)
    numeric = [r for r in record.calibration if r.gain is not None]
    if not numeric:
        raise ValueError(
            f"raw {unit} data need a declared calibration record or points; none is invented"
        )
    if len(numeric) > 1:
        raise ValueError("several numeric calibration records for one channel; select one")
    return fit_record(numeric[0], unit)


def acquire(
    record: RawRecord | SignalBundle,
    cell: CellSpec | None = None,
    *,
    timing_tolerance: float = 1e-6,
) -> Acquired:
    cell = CellSpec() if cell is None else CellSpec.model_validate(cell.model_dump())
    raw_record = from_bundle(record, cell.channel) if isinstance(record, SignalBundle) else record
    values = np.asarray(raw_record.values)
    if values.ndim != 1 or not np.issubdtype(values.dtype, np.number):
        raise ValueError("raw samples must be a one-dimensional numeric array")
    values = values.astype(np.float64, copy=True)
    if not np.isfinite(values).all():
        bad = int(np.count_nonzero(~np.isfinite(values)))
        raise ValueError(f"{bad} non-finite raw samples rejected; no gap filling is performed")
    unit = require_unit(raw_record.unit)
    if unit == "count" and np.any(values != np.floor(values)):
        raise ValueError("ADC counts must be integers")
    timing = _timing(raw_record, len(values), timing_tolerance)
    fit = resolve_calibration(raw_record, cell)
    force = fit.apply(values)
    if not np.isfinite(force).all():
        raise ValueError("calibrated force is not finite")
    return Acquired(raw_record, values, force, fit, timing, cell)


# --------------------------------------------------------------------------------------
# Integrity checks. Each returns a qc.Result; heuristics say so in their detail text.
# --------------------------------------------------------------------------------------


def finding(
    identifier: str, severity: qc.Severity, status: qc.Status, detail: str, **values: float
) -> qc.Result:
    return qc.Result(
        identifier, identifier.lower().replace("-", "_"), severity, status, detail, values
    )


def window_mask(time: FloatArray, window: tuple[float, float]) -> np.ndarray:
    start, end = window
    if not np.isfinite([start, end]).all() or end <= start:
        raise ValueError("window must have positive finite duration")
    dt = float(np.median(np.diff(time)))
    if start < time[0] - 1e-9 or end > time[-1] + dt + 1e-9:
        raise ValueError("window outside sampled interval")
    mask = (time >= start - 1e-10) & (time < end - 1e-10)
    if int(mask.sum()) < 2:
        raise ValueError("window contains fewer than two samples")
    return mask


def check_timing(timing: Timing, rfd_min_hz: float, mismatch_limit: float) -> list[qc.Result]:
    out = [
        finding(
            "LC-TIMING",
            qc.Severity.FLAG,
            qc.Status.PASS if timing.uniform else qc.Status.FAIL,
            "uniform timestamps"
            if timing.uniform
            else "non-uniform timestamps: RFD/impulse windows and filtering are refused",
            sample_interval_s=timing.sample_interval_s,
        ),
        finding(
            "LC-RATE",
            qc.Severity.FLAG,
            qc.Status.PASS
            if timing.rate_mismatch <= mismatch_limit and timing.measured_hz >= rfd_min_hz
            else qc.Status.FAIL,
            "timestamp rate vs declared rate and RFD minimum rate"
            + (
                ""
                if timing.timestamps_declared
                else " (no timestamps: rate taken from declared metadata only)"
            ),
            measured_hz=timing.measured_hz,
            nominal_hz=timing.nominal_hz,
            rate_mismatch=timing.rate_mismatch,
            rfd_min_hz=rfd_min_hz,
        ),
    ]
    return out


def check_anti_aliasing(cell: CellSpec, timing: Timing) -> qc.Result:
    nyquist = timing.measured_hz / 2
    if cell.anti_aliasing_hz is None:
        return finding(
            "LC-ANTIALIAS",
            qc.Severity.NOTE,
            qc.Status.NOT_APPLICABLE,
            "no declared anti-aliasing filter; analog band-limiting below Nyquist is ASSUMED, "
            "aliasing cannot be detected from the samples",
            nyquist_hz=nyquist,
        )
    return finding(
        "LC-ANTIALIAS",
        qc.Severity.FLAG,
        qc.Status.PASS if cell.anti_aliasing_hz < nyquist else qc.Status.FAIL,
        "declared anti-aliasing cutoff versus Nyquist",
        anti_aliasing_hz=cell.anti_aliasing_hz,
        nyquist_hz=nyquist,
    )


def check_saturation(acquired: Acquired) -> list[qc.Result]:
    cell = acquired.cell
    out: list[qc.Result] = []
    if cell.raw_range is None:
        out.append(
            finding(
                "LC-RAILS",
                qc.Severity.REJECT,
                qc.Status.NOT_APPLICABLE,
                "no declared ADC/voltage range: clipping is undetectable (a clipped plateau "
                "is indistinguishable from a held plateau)",
            )
        )
    else:
        lo, hi = cell.raw_range
        margin = cell.rail_margin_fraction * (hi - lo)
        hits = int(np.count_nonzero((acquired.raw <= lo + margin) | (acquired.raw >= hi - margin)))
        out.append(
            finding(
                "LC-RAILS",
                qc.Severity.REJECT,
                qc.Status.FAIL if hits else qc.Status.PASS,
                f"{hits} raw samples at or beyond the declared range (clipping/saturation)"
                if hits
                else "raw samples inside the declared range",
                samples_at_rail=float(hits),
                raw_low=lo,
                raw_high=hi,
            )
        )
    if cell.capacity_n is None:
        out.append(
            finding(
                "LC-CAPACITY",
                qc.Severity.REJECT,
                qc.Status.NOT_APPLICABLE,
                "no declared rated capacity: overload undetectable",
            )
        )
    else:
        peak = float(np.max(np.abs(acquired.force_n)))
        out.append(
            finding(
                "LC-CAPACITY",
                qc.Severity.REJECT,
                qc.Status.FAIL if peak > cell.capacity_n else qc.Status.PASS,
                "calibrated force exceeds declared rated capacity (overload, saturation or "
                "unit mislabel)"
                if peak > cell.capacity_n
                else "calibrated force within declared rated capacity",
                peak_abs_force_n=peak,
                capacity_n=cell.capacity_n,
            )
        )
    return out


def check_units(acquired: Acquired, peak_n: float) -> qc.Result:
    """Heuristic unit-mislabel screen against declared plausibility bounds only."""
    bounds = acquired.cell.expected_peak_n
    if bounds is None:
        return finding(
            "LC-UNITS",
            qc.Severity.FLAG,
            qc.Status.NOT_APPLICABLE,
            "no declared peak plausibility bounds: a constant-factor unit mislabel (e.g. kgf "
            "recorded as N) is undetectable from the record alone",
        )
    lo, hi = bounds
    if lo <= peak_n <= hi:
        return finding(
            "LC-UNITS",
            qc.Severity.REJECT,
            qc.Status.PASS,
            "peak force inside declared plausibility bounds (heuristic, not proof of units)",
            peak_n=peak_n,
            low_n=lo,
            high_n=hi,
        )
    hint = ""
    if lo <= peak_n * KGF_TO_N <= hi:
        hint = "; x9.80665 would be plausible: possible kgf data declared as N"
    elif lo <= peak_n / KGF_TO_N <= hi:
        hint = "; /9.80665 would be plausible: possible N data declared as kgf"
    return finding(
        "LC-UNITS",
        qc.Severity.REJECT,
        qc.Status.FAIL,
        f"peak force {peak_n:.6g} N outside declared plausibility [{lo:g}, {hi:g}] N{hint}",
        peak_n=peak_n,
        low_n=lo,
        high_n=hi,
    )


def check_calibration_fit(fit: CalibrationFit, limit_fraction: float) -> qc.Result:
    if fit.source != "points":
        return finding(
            "LC-CALFIT",
            qc.Severity.NOTE,
            qc.Status.NOT_APPLICABLE,
            f"{fit.description}; no residuals available",
        )
    if fit.n_points == 2 or fit.rms_residual_n is None or fit.full_scale_n is None:
        return finding(
            "LC-CALFIT",
            qc.Severity.NOTE,
            qc.Status.NOT_APPLICABLE,
            "two-point calibration: exact solve, linearity not assessable",
        )
    fraction = fit.rms_residual_n / fit.full_scale_n
    return finding(
        "LC-CALFIT",
        qc.Severity.FLAG,
        qc.Status.PASS if fraction <= limit_fraction else qc.Status.FAIL,
        "multi-point least-squares RMS residual relative to calibrated full scale",
        rms_residual_n=fit.rms_residual_n,
        max_abs_residual_n=fit.max_abs_residual_n or 0.0,
        fraction_of_full_scale=fraction,
        limit_fraction=limit_fraction,
    )


def check_zero(baseline_n: float, tolerance_n: float, zeroing_declared: bool) -> qc.Result:
    fail = abs(baseline_n) > tolerance_n
    return finding(
        "LC-ZERO",
        qc.Severity.FLAG,
        qc.Status.FAIL if fail else qc.Status.PASS,
        (
            "pre-onset baseline exceeds zero tolerance: unzeroed cell or deliberate "
            "pre-tension (the record cannot distinguish them; protocol must declare which)"
            + ("" if zeroing_declared else "; no zeroing record declared")
        )
        if fail
        else "pre-onset baseline within zero tolerance",
        baseline_n=baseline_n,
        tolerance_n=tolerance_n,
    )


def check_drift(
    time: FloatArray,
    force: FloatArray,
    start: tuple[float, float],
    end: tuple[float, float] | None,
    tolerance_n: float,
) -> qc.Result:
    if end is None:
        return finding(
            "LC-DRIFT",
            qc.Severity.FLAG,
            qc.Status.NOT_APPLICABLE,
            "no declared post-effort baseline window: zero drift not assessed",
        )
    a = float(np.mean(force[window_mask(time, start)]))
    b = float(np.mean(force[window_mask(time, end)]))
    return finding(
        "LC-DRIFT",
        qc.Severity.FLAG,
        qc.Status.FAIL if abs(b - a) > tolerance_n else qc.Status.PASS,
        "baseline change between declared start and end windows",
        drift_n=b - a,
        tolerance_n=tolerance_n,
    )


def check_temperature(cell: CellSpec, limit_fraction: float) -> qc.Result:
    if cell.temperature_c is None or cell.span_temperature_coefficient_per_c is None:
        return finding(
            "LC-TEMPERATURE",
            qc.Severity.FLAG,
            qc.Status.NOT_APPLICABLE,
            "no declared temperatures/coefficient: temperature drift not assessed",
        )
    change = abs(cell.temperature_c[1] - cell.temperature_c[0])
    fraction = change * cell.span_temperature_coefficient_per_c
    return finding(
        "LC-TEMPERATURE",
        qc.Severity.FLAG,
        qc.Status.FAIL if fraction > limit_fraction else qc.Status.PASS,
        "declared span temperature coefficient times declared temperature change",
        temperature_change_c=change,
        span_change_fraction=fraction,
        limit_fraction=limit_fraction,
    )


def check_off_axis() -> qc.Result:
    return finding(
        "LC-OFFAXIS",
        qc.Severity.NOTE,
        qc.Status.NOT_APPLICABLE,
        "a uniaxial cell cannot detect off-axis or side loading; this is not checked",
    )


def check_creep(
    time: FloatArray,
    force: FloatArray,
    baseline_n: float,
    hold_fraction: float,
    min_hold_s: float,
    limit_fraction: float,
) -> qc.Result:
    """Heuristic: linear trend over the contiguous near-peak hold around the maximum."""
    net = force - baseline_n
    peak = int(np.argmax(net))
    if net[peak] <= 0:
        return finding("LC-CREEP", qc.Severity.FLAG, qc.Status.NOT_APPLICABLE, "no positive effort")
    above = net >= hold_fraction * net[peak]
    lo = peak
    while lo > 0 and above[lo - 1]:
        lo -= 1
    hi = peak
    while hi < len(net) - 1 and above[hi + 1]:
        hi += 1
    duration = float(time[hi] - time[lo])
    if duration < min_hold_s or hi - lo < 2:
        return finding(
            "LC-CREEP",
            qc.Severity.FLAG,
            qc.Status.NOT_APPLICABLE,
            "near-peak hold shorter than the minimum: creep not assessed",
            hold_s=duration,
        )
    slope = float(np.polyfit(time[lo : hi + 1] - time[lo], net[lo : hi + 1], 1)[0])
    fraction = slope * duration / float(net[peak])
    return finding(
        "LC-CREEP",
        qc.Severity.FLAG,
        qc.Status.FAIL if abs(fraction) > limit_fraction else qc.Status.PASS,
        "heuristic: linear force change across the near-peak hold; cell creep and voluntary "
        "force decline are indistinguishable in a single record",
        hold_s=duration,
        change_fraction=fraction,
        limit_fraction=limit_fraction,
    )
