"""Production load-cell / isometric-dynamometer acquisition-to-metric engine.

Force at the cell and joint torque are distinct estimands. Torque is only produced from a
declared lever arm and moment-arm definition; gravity correction is a named variant, never
a default. Named methods are conditional processing alternatives, not qualification. No
analytical generator is imported: expected values belong in independent tests.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from importlib import metadata
from typing import Literal, get_args

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy import signal

from dynamisbench_soccer.engines.load_cell_acquisition import (
    Acquired,
    CellSpec,
    RawRecord,
    acquire,
    check_anti_aliasing,
    check_calibration_fit,
    check_creep,
    check_drift,
    check_off_axis,
    check_saturation,
    check_temperature,
    check_timing,
    check_units,
    check_zero,
    finding,
    input_digest,
    window_mask,
)
from dynamisbench_soccer.qc import spec as qc
from dynamisbench_soccer.signals.schema import FloatArray, SignalBundle

ENGINE = "load-cell.v1"
#: Same literal names as the force-platform engine, implemented for an isometric baseline.
OnsetMethod = Literal["5sd", "reverse_scan", "first_derivative", "absolute_N"]
ONSET_METHODS: tuple[str, ...] = get_args(OnsetMethod)
GravityCorrection = Literal["none", "declared_gravity_torque"]

LIMITATIONS: tuple[str, ...] = (
    "IMTP (or any isometric test) on a load cell or isometric dynamometer is not "
    "interchangeable with force-plate IMTP without an agreement study (RES-703 open).",
    "Rig compliance is a stated limitation and is NOT modelled; a declared "
    "rig_compliance_m_per_n is recorded only. Compliance delays and attenuates early RFD.",
    "A uniaxial cell cannot detect off-axis or side loading; misalignment biases force low.",
    "Anti-aliasing: analog band-limiting below Nyquist before digitisation is assumed unless "
    "declared; aliasing cannot be detected from the samples.",
    "Unit-mislabel detection is a heuristic against declared capacity/plausibility bounds; "
    "without declarations a constant-factor mislabel (e.g. kgf as N) is undetectable.",
    "Pre-onset baseline offset cannot distinguish an unzeroed cell from deliberate pre-tension.",
    "Creep detection is heuristic: cell creep and voluntary force decline are "
    "indistinguishable in a single record.",
    "Temperature drift is assessed only from declared temperatures and coefficients.",
    "Joint torque = cell force x declared lever arm assumes a perpendicular line of action at "
    "the declared point and a fixed joint centre; joint rotation under load is not modelled.",
    "Uncertainty output is a conditional sensitivity analysis (calibration gain, filter, "
    "onset variants), not a complete measurement-uncertainty budget.",
    "Engineering calculations only: no device or physiological qualification is claimed.",
)


class FilterSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    kind: Literal["none", "butterworth_lowpass"] = "none"
    cutoff_hz: float = Field(default=50, gt=0)
    order: int = Field(default=4, ge=1, le=8)


class MethodSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    version: Literal["load-cell.v1"] = "load-cell.v1"
    peak_filter: FilterSpec = Field(default_factory=FilterSpec)
    rfd_filter: FilterSpec = Field(default_factory=FilterSpec)
    onset: OnsetMethod = "5sd"
    onset_sd_multiple: float = Field(default=5, gt=0)
    onset_floor_n: float = Field(default=1, gt=0)
    onset_absolute_n: float = Field(default=20, gt=0)
    onset_sustain_s: float = Field(default=0.01, ge=0)
    derivative_cutoff_hz: float = Field(default=50, gt=0)
    derivative_floor_n_s: float = Field(default=50, gt=0)
    baseline_window_s: tuple[float, float] = (0.0, 0.3)
    end_baseline_window_s: tuple[float, float] | None = None
    windows_ms: tuple[int, ...] = (50, 100, 200)
    gravity_correction: GravityCorrection = "none"
    zero_tolerance_n: float = Field(default=20, ge=0)
    drift_tolerance_n: float = Field(default=10, ge=0)
    hold_fraction: float = Field(default=0.95, gt=0, lt=1)
    min_hold_s: float = Field(default=0.25, gt=0)
    creep_limit_fraction: float = Field(default=0.02, gt=0)
    calibration_residual_limit_fraction: float = Field(default=0.005, gt=0)
    temperature_limit_fraction: float = Field(default=0.005, gt=0)
    rfd_min_hz: float = Field(default=1000, gt=0)
    rate_mismatch_limit: float = Field(default=0.01, ge=0)
    timing_tolerance: float = Field(default=1e-6, gt=0)
    sensitivity_gain_errors: tuple[float, ...] = (-0.01, 0.01)
    sensitivity_cutoffs_hz: tuple[float, ...] = (10.0, 20.0, 50.0)

    @model_validator(mode="after")
    def valid(self) -> MethodSpec:
        if self.baseline_window_s[1] <= self.baseline_window_s[0]:
            raise ValueError("baseline window must increase")
        if (
            self.end_baseline_window_s is not None
            and self.end_baseline_window_s[1] <= self.end_baseline_window_s[0]
        ):
            raise ValueError("end baseline window must increase")
        if (
            not self.windows_ms
            or any(w <= 0 for w in self.windows_ms)
            or len(set(self.windows_ms)) != len(self.windows_ms)
        ):
            raise ValueError("RFD windows must be unique positive milliseconds")
        if any(abs(e) >= 0.5 for e in self.sensitivity_gain_errors):
            raise ValueError("sensitivity gain errors must be small relative perturbations")
        return self


@dataclass(frozen=True)
class Onset:
    index: int | None
    time_s: float | None
    threshold: float
    threshold_unit: str
    method: str
    status: str


@dataclass(frozen=True)
class Torque:
    available: bool
    lever_arm_m: float | None
    moment_arm_definition: str | None
    gravity_torque_n_m: float
    reason: str | None


@dataclass(frozen=True)
class Processed:
    time_s: FloatArray
    calibrated_force_n: FloatArray
    peak_force_signal_n: FloatArray
    rfd_force_signal_n: FloatArray
    onset: Onset
    torque: Torque
    metrics: dict[str, float | None]
    refusals: dict[str, str]
    qc: tuple[qc.Result, ...]
    calibration: dict[str, object]
    uncertainty: dict[str, object]
    limitations: tuple[str, ...]
    method_spec: dict[str, object]

    @property
    def verdict(self) -> str:
        return qc.verdict(list(self.qc))

    def report(self) -> dict[str, object]:
        return {
            "engine": ENGINE,
            "verdict": self.verdict,
            "diagnostic_only": self.verdict == "reject",
            "estimands": {
                "force": "force along the cell axis at the cell, N",
                "torque": "cell force x declared lever arm about the declared joint centre, N*m"
                if self.torque.available
                else None,
            },
            "onset": asdict(self.onset),
            "torque": asdict(self.torque),
            "metrics": self.metrics,
            "refusals": self.refusals,
            "calibration": self.calibration,
            "uncertainty": self.uncertainty,
            "qc": [asdict(r) for r in self.qc],
            "limitations": list(self.limitations),
            "method_spec": self.method_spec,
            "qualification": "engineering calculations only; no scientific qualification",
        }

    def report_json(self) -> str:
        return json.dumps(self.report(), indent=2, sort_keys=True, allow_nan=False) + "\n"


def lowpass(force: FloatArray, time: FloatArray, spec: FilterSpec, *, uniform: bool) -> FloatArray:
    """Zero-lag (forward-backward) Butterworth; the stated cutoff is the single-pass cutoff."""
    if spec.kind == "none":
        return force.copy()
    if not uniform:
        raise ValueError("digital filtering requires uniform sampling; refused")
    hz = float(1 / np.median(np.diff(time)))
    if spec.cutoff_hz >= hz / 2:
        raise ValueError("filter cutoff must be below Nyquist")
    sos = signal.butter(spec.order, spec.cutoff_hz, fs=hz, output="sos")
    filtered = np.asarray(signal.sosfiltfilt(sos, force, padtype="odd"), dtype=np.float64)
    if not np.isfinite(filtered).all():
        raise ValueError("filter produced non-finite values")
    return filtered


def identify_onset(
    force: FloatArray,
    time: FloatArray,
    baseline: float,
    sd: float,
    baseline_window: tuple[float, float],
    method: MethodSpec,
    *,
    uniform: bool = True,
) -> Onset:
    """Isometric onset: rise above a resting baseline, searched after the baseline window.

    Unlike the force-platform engine no body weight is involved and the excursion is the
    signed force increase (an isometric pull/push raises cell force). One-sample backdating
    matches the force-platform convention and is recorded in the method specification.
    """
    name = method.onset
    search = np.flatnonzero(time >= baseline_window[1] - 1e-10)
    if len(search) == 0:
        return Onset(None, None, 0.0, "N", name, "no samples after baseline window")
    base = max(method.onset_sd_multiple * sd, method.onset_floor_n)
    values = force - baseline
    threshold = base
    unit = "N"
    if name == "absolute_N":
        threshold = method.onset_absolute_n
    if name == "first_derivative":
        smooth = lowpass(
            force,
            time,
            FilterSpec(kind="butterworth_lowpass", cutoff_hz=method.derivative_cutoff_hz),
            uniform=uniform,
        )
        derivative = np.gradient(smooth, time)
        mask = window_mask(time, baseline_window)
        threshold = max(
            method.onset_sd_multiple * float(np.std(derivative[mask], ddof=1)),
            method.derivative_floor_n_s,
        )
        values = derivative
        unit = "N/s"
    if name == "reverse_scan":
        high = max(2 * base, method.onset_absolute_n)
        duration = max(1, int(np.ceil(method.onset_sustain_s / float(np.median(np.diff(time))))))
        above = values > high
        candidates = [
            int(i) for i in search if i + duration <= len(force) and above[i : i + duration].all()
        ]
        if not candidates:
            return Onset(None, None, base, unit, name, "no sustained excursion")
        i = candidates[0]
        while i > search[0] and values[i - 1] > base:
            i -= 1
    else:
        hits = search[values[search] > threshold]
        if len(hits) == 0:
            return Onset(None, None, threshold, unit, name, "no excursion")
        i = int(hits[0])
    if i == search[0]:
        return Onset(
            None, None, threshold, unit, name, "force already above threshold at baseline end"
        )
    i -= 1
    return Onset(i, float(time[i]), threshold, unit, name, "detected")


def _interp(time: FloatArray, values: FloatArray, at: float) -> float:
    return float(np.interp(at, time, values))


def _integral(time: FloatArray, values: FloatArray, start: float, end: float) -> float:
    inside = (time > start) & (time < end)
    t = np.r_[start, time[inside], end]
    y = np.r_[_interp(time, values, start), values[inside], _interp(time, values, end)]
    return float(np.trapezoid(y, t))


def _torque_key(key: str) -> str:
    if key in {"peak_force_n", "net_peak_force_n"}:
        return key.replace("force_n", "torque_n_m")
    if key.startswith("rfd_"):
        return "rtd_" + key.removeprefix("rfd_").removesuffix("_n_s") + "_n_m_s"
    return "net_angular_impulse_" + key.removeprefix("net_impulse_").removesuffix("_n_s") + "_n_m_s"


def resolve_torque(acquired: Acquired, method: MethodSpec) -> Torque:
    cell = acquired.cell
    geometry = acquired.record.geometry
    gravity = 0.0
    if method.gravity_correction == "declared_gravity_torque":
        if cell.gravity_torque_n_m is None:
            raise ValueError("declared_gravity_torque variant needs cell.gravity_torque_n_m")
        gravity = cell.gravity_torque_n_m
    lever = cell.lever_arm_m
    if lever is None and geometry is not None and "lever_arm_m" not in geometry.unknown:
        lever = geometry.lever_arm_m
    definition = cell.moment_arm_definition or (
        geometry.attachment_point if geometry is not None else None
    )
    reason = None
    if lever is None:
        reason = "torque refused: no declared lever arm"
    elif not definition or not definition.strip():
        reason = "torque refused: no declared moment-arm definition (joint centre to cell)"
    return Torque(reason is None, lever, definition, gravity, reason)


def _analyse(acquired: Acquired, method: MethodSpec) -> Processed:
    timing = acquired.timing
    time = timing.time_s
    cell = acquired.cell
    force = acquired.force_n
    checks: list[qc.Result] = [
        *check_timing(timing, method.rfd_min_hz, method.rate_mismatch_limit),
        check_anti_aliasing(cell, timing),
        check_calibration_fit(acquired.calibration, method.calibration_residual_limit_fraction),
        *check_saturation(acquired),
        check_temperature(cell, method.temperature_limit_fraction),
        check_off_axis(),
    ]
    peak_signal = lowpass(force, time, method.peak_filter, uniform=timing.uniform)
    rfd_signal = lowpass(force, time, method.rfd_filter, uniform=timing.uniform)
    mask = window_mask(time, method.baseline_window_s)
    baseline = float(np.mean(force[mask]))
    baseline_sd = float(np.std(force[mask], ddof=1))
    rfd_baseline = float(np.mean(rfd_signal[mask]))
    rfd_sd = float(np.std(rfd_signal[mask], ddof=1))
    peak_baseline = float(np.mean(peak_signal[mask]))
    onset = identify_onset(
        rfd_signal,
        time,
        rfd_baseline,
        rfd_sd,
        method.baseline_window_s,
        method,
        uniform=timing.uniform,
    )
    peak_index = int(np.argmax(peak_signal))
    peak = float(peak_signal[peak_index])
    zeroing = acquired.record.bundle.meta.zeroing if acquired.record.bundle else None
    checks += [
        check_zero(baseline, method.zero_tolerance_n, zeroing is not None and zeroing.performed),
        check_drift(
            time,
            force,
            method.baseline_window_s,
            method.end_baseline_window_s,
            method.drift_tolerance_n,
        ),
        check_creep(
            time,
            peak_signal,
            peak_baseline,
            method.hold_fraction,
            method.min_hold_s,
            method.creep_limit_fraction,
        ),
        check_units(acquired, peak),
    ]
    if onset.index is None:
        checks.append(
            finding("LC-ONSET", qc.Severity.FLAG, qc.Status.FAIL, f"onset: {onset.status}")
        )
    metrics: dict[str, float | None] = {
        "baseline_n": baseline,
        "baseline_sd_n": baseline_sd,
        "peak_force_n": peak,
        "net_peak_force_n": peak - peak_baseline,
        "peak_time_s": float(time[peak_index]),
        "onset_time_s": onset.time_s,
        "time_to_peak_s": None,
        "gross_impulse_record_n_s": None,
    }
    refusals: dict[str, str] = {}
    if onset.time_s is not None and peak_index > (onset.index or 0):
        metrics["time_to_peak_s"] = float(time[peak_index]) - onset.time_s
    else:
        refusals["time_to_peak_s"] = f"onset {onset.status}; or peak precedes onset"
    window_reason = None
    if not timing.uniform:
        window_reason = "non-uniform timestamps: RFD/impulse windows refused"
    elif timing.measured_hz < method.rfd_min_hz * (1 - 1e-9):
        window_reason = f"sampling {timing.measured_hz:.6g} Hz below RFD minimum"
    elif onset.time_s is None:
        window_reason = f"no onset ({onset.status})"
    if timing.uniform:
        metrics["gross_impulse_record_n_s"] = float(np.trapezoid(force, time))
    else:
        refusals["gross_impulse_record_n_s"] = "non-uniform timestamps"
    for w in method.windows_ms:
        rfd_key, imp_key = f"rfd_0_{w}ms_n_s", f"net_impulse_0_{w}ms_n_s"
        reason = window_reason
        end = None if onset.time_s is None else onset.time_s + w / 1000
        if reason is None and end is not None and end > time[-1] + 1e-9:
            reason = "window extends beyond the record"
        if reason is not None or end is None or onset.time_s is None or onset.index is None:
            metrics[rfd_key] = metrics[imp_key] = None
            refusals[rfd_key] = refusals[imp_key] = reason or "no onset"
            continue
        start_force = float(rfd_signal[onset.index])
        metrics[rfd_key] = (_interp(time, rfd_signal, end) - start_force) / (w / 1000)
        metrics[imp_key] = _integral(time, rfd_signal - rfd_baseline, onset.time_s, end)
    torque = resolve_torque(acquired, method)
    force_keys = [k for k in metrics if k.startswith(("peak_force", "net_peak", "rfd_", "net_imp"))]
    for key in force_keys:
        torque_key = _torque_key(key)
        value = metrics[key]
        if not torque.available:
            metrics[torque_key] = None
            refusals[torque_key] = torque.reason or "torque refused"
        elif value is None:
            metrics[torque_key] = None
            refusals[torque_key] = refusals.get(key, "force metric unavailable")
        else:
            assert torque.lever_arm_m is not None
            offset = torque.gravity_torque_n_m if key == "peak_force_n" else 0.0
            metrics[torque_key] = value * torque.lever_arm_m + offset
    geometry = acquired.record.geometry
    uncertainty: dict[str, object] = {
        "complete_measurement_uncertainty": False,
        "sample_interval_s": timing.sample_interval_s,
        "onset_resolution_s": timing.sample_interval_s + 2 * timing.sync_uncertainty_s,
        "sync_uncertainty_s": timing.sync_uncertainty_s,
        "calibration_rms_residual_n": acquired.calibration.rms_residual_n,
        "calibration_gain_standard_error": acquired.calibration.gain_standard_error,
        "declared_calibration_uncertainty": acquired.calibration.record.uncertainty
        if acquired.calibration.record is not None
        else None,
        "rig_compliance": "not modelled (stated limitation); declared value "
        + (
            repr(geometry.rig_compliance_m_per_n)
            if geometry is not None and geometry.rig_compliance_m_per_n is not None
            else "absent"
        ),
        "scope": "use sensitivity() for gain, filter-cutoff and onset-method variants",
    }
    record = acquired.record
    resolved: dict[str, object] = {
        **method.model_dump(mode="json"),
        "cell": cell.model_dump(mode="json"),
        "raw_unit": record.unit,
        "device_class": record.device_class.value,
        "input_label": record.label,
        "input_digest": input_digest(record),
        "input_hash_scheme": "sorted metadata JSON (no data_sha256) + little-endian samples",
        "source_file_sha256": record.bundle.meta.data_sha256 if record.bundle else None,
        "source_provenance": record.bundle.meta.provenance.model_dump(mode="json")
        if record.bundle
        else None,
        "declared_calibration_records": [c.model_dump(mode="json") for c in record.calibration],
        "geometry": geometry.model_dump(mode="json") if geometry is not None else None,
        "clock_identity": timing.clock_identity,
        "timestamps_declared": timing.timestamps_declared,
        "measured_hz": timing.measured_hz,
        "onset_signal": "rfd_filter output",
        "onset_backdating_samples": 1,
        "rfd_definition": "(F(onset+w) - F(onset)) / w on the rfd_filter signal",
        "impulse_definition": "trapezoid of (F - pre-onset baseline) from onset to onset+w",
        "filter_realization": "Butterworth SOS forward-backward (zero lag), odd padding",
        "versions": {"engine": "1", "numpy": np.__version__, "scipy": metadata.version("scipy")},
    }
    limitations = list(LIMITATIONS)
    if torque.reason:
        limitations.append(torque.reason)
    return Processed(
        time.copy(),
        force.copy(),
        peak_signal,
        rfd_signal,
        onset,
        torque,
        metrics,
        refusals,
        tuple(checks),
        acquired.calibration.summary(),
        uncertainty,
        tuple(limitations),
        resolved,
    )


def process(
    source: RawRecord | SignalBundle | Acquired,
    cell: CellSpec | None = None,
    method: MethodSpec | None = None,
) -> Processed:
    method = MethodSpec() if method is None else MethodSpec.model_validate(method.model_dump())
    if isinstance(source, Acquired):
        if cell is not None:
            raise ValueError("an acquired record already carries its cell declarations")
        acquired = source
    else:
        acquired = acquire(source, cell, timing_tolerance=method.timing_tolerance)
    return _analyse(acquired, method)


def _delta(base: dict[str, float | None], other: dict[str, float | None]) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, value in base.items():
        changed = other.get(key)
        if value is None or changed is None:
            out[key] = {"value": changed, "delta": None, "relative": None}
            continue
        delta = changed - value
        out[key] = {
            "value": changed,
            "delta": delta,
            "relative": delta / value if value != 0 else None,
        }
    return out


def sensitivity(
    source: RawRecord | SignalBundle,
    cell: CellSpec | None = None,
    method: MethodSpec | None = None,
) -> dict[str, object]:
    """Metric changes under calibration gain error, filter-cutoff and onset-method variants.

    Conditional sensitivity only; rig compliance is a stated limitation and is not modelled.
    """
    method = MethodSpec() if method is None else MethodSpec.model_validate(method.model_dump())
    acquired = acquire(source, cell, timing_tolerance=method.timing_tolerance)
    base = _analyse(acquired, method)
    variants: dict[str, object] = {}
    skipped: dict[str, str] = {}
    for error in method.sensitivity_gain_errors:
        result = _analyse(acquired.with_gain_error(error), method)
        variants[f"gain_{error:+.4g}"] = _delta(base.metrics, result.metrics)
    nyquist = acquired.timing.measured_hz / 2
    for cutoff in method.sensitivity_cutoffs_hz:
        label = f"butterworth_{cutoff:g}hz"
        if not acquired.timing.uniform or cutoff >= nyquist:
            skipped[label] = "non-uniform timing or cutoff at/above Nyquist"
            continue
        spec = FilterSpec(kind="butterworth_lowpass", cutoff_hz=cutoff)
        variant = method.model_copy(update={"peak_filter": spec, "rfd_filter": spec})
        variants[label] = _delta(base.metrics, _analyse(acquired, variant).metrics)
    for name in ONSET_METHODS:
        if name == method.onset:
            continue
        try:
            result = _analyse(acquired, method.model_copy(update={"onset": name}))
        except ValueError as exc:
            skipped[f"onset_{name}"] = str(exc)
            continue
        variants[f"onset_{name}"] = _delta(base.metrics, result.metrics)
    return {
        "engine": ENGINE,
        "base_method": {"onset": method.onset, "peak_filter": method.peak_filter.model_dump()},
        "base": base.metrics,
        "variants": variants,
        "skipped": skipped,
        "not_modelled": ["rig compliance", "off-axis loading", "temperature (unless declared)"],
        "scope": "conditional one-at-a-time sensitivity; not a measurement-uncertainty budget",
    }
