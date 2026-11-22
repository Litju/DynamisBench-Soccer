"""Nordic-hamstring and hip-strength processing built on the load-cell core (RES-697).

Strap or cuff force at the device is the only measured estimand. It is neither muscle force
nor joint moment unless a lever arm and its definition are declared, and then only the
named torque estimand ``strap_force x lever_arm`` is produced. Side identity, force basis
(raw calibrated or vendor corrected) and protocol geometry are mandatory declarations; a
missing or contradictory declaration is refused, never repaired. Joint angle is only ever
used when a measured angle channel exists: it is never recovered from force or from an
assumed constant descent rate. No analytical generator is imported (RES-669 stays in tests).
"""

from __future__ import annotations

import itertools
import json
import math
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy import signal

from dynamisbench_soccer.engines.load_cell import FilterSpec, identify_onset, lowpass
from dynamisbench_soccer.engines.load_cell import MethodSpec as CellMethod
from dynamisbench_soccer.engines.load_cell_acquisition import (
    Acquired,
    CellSpec,
    acquire,
    check_anti_aliasing,
    check_calibration_fit,
    check_saturation,
    check_timing,
    finding,
    from_bundle,
    window_mask,
)
from dynamisbench_soccer.qc import spec as qc
from dynamisbench_soccer.signals.schema import (
    BundleKind,
    DeviceClass,
    FloatArray,
    Quantity,
    Side,
    SignalBundle,
)

ENGINE = "nordic-hip.v1"
ADMITTED = frozenset({DeviceClass.NORDIC_DYNAMOMETER, DeviceClass.HIP_DYNAMOMETER})
ForceBasis = Literal["raw_calibrated", "vendor_corrected"]
Protocol = Literal["nordic_eccentric", "hip_squeeze"]
OnsetVariant = Literal["first_exceedance", "5sd", "reverse_scan", "first_derivative", "absolute_N"]
BreakVariant = Literal["peak_force_angle", "angular_acceleration"]
Asymmetry = Literal["weaker_over_stronger_percent", "difference_over_mean_percent"]

LIMITATIONS: tuple[str, ...] = (
    "Measured force is strap/cuff force at the device. It is not hamstring or hip-muscle "
    "force: ankle-strap force is also set by lever length, segment weight, hip position and "
    "rig compliance, none of which are corrected unless declared.",
    "Torque is strap force x the declared lever arm assuming a perpendicular line of action; "
    "lever-arm and gravity-of-segment errors are not modelled beyond the declared uncertainty.",
    "Vendor-corrected and raw-calibrated force are different estimands; they are labelled and "
    "never pooled, compared or converted by this engine (RES-703 owns any agreement study).",
    "Knee/hip angle is used only from a measured angle channel. It is never inferred from "
    "force, time or an assumed descent rate.",
    "A kinematic break point needs a measured angle at a rate fine enough to resolve the "
    "loss of control; without it the break point is refused, not estimated.",
    "Bilateral asymmetry is only meaningful for identical force basis, lever arm and protocol "
    "on both sides; side identity must be declared and agree with any bundle side metadata.",
    "Repetition detection is a declared peak-finding policy, not an athlete-effort judgement.",
    "Uncertainty output is a conditional sensitivity analysis (gain, filter, declared lever "
    "arm), not a complete measurement-uncertainty budget.",
    "Engineering calculations only: no device, athlete-risk or physiological qualification.",
)


class SideDeclaration(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    channel: str = Field(min_length=1)
    side: Literal["left", "right", "unilateral"]


class NordicSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    protocol: Protocol
    force_basis: ForceBasis
    sides: tuple[SideDeclaration, ...] = Field(min_length=1)
    cells: dict[str, CellSpec] = Field(default_factory=dict)
    lever_arm_m: float | None = Field(default=None, gt=0, lt=2.0)
    moment_arm_definition: str | None = None
    lever_arm_uncertainty_m: float | None = Field(default=None, ge=0)
    angle_channel: str | None = None

    @model_validator(mode="after")
    def valid(self) -> NordicSpec:
        channels = [s.channel for s in self.sides]
        if len(set(channels)) != len(channels):
            raise ValueError("a channel may be declared for one side only")
        if set(self.cells) - set(channels):
            raise ValueError("cell declarations for undeclared channels")
        names = [s.side for s in self.sides]
        if len(set(names)) != len(names):
            raise ValueError("side identity is ambiguous: two channels declare the same side")
        if "unilateral" in names and len(names) > 1:
            raise ValueError("'unilateral' cannot be combined with left/right channels")
        return self


class MethodSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    version: Literal["nordic-hip.v1"] = "nordic-hip.v1"
    filter: FilterSpec = Field(default_factory=FilterSpec)
    onset: OnsetVariant = "first_exceedance"
    onset_threshold_n: float = Field(default=20, gt=0)
    onset_sd_multiple: float = Field(default=5, gt=0)
    baseline_window_s: tuple[float, float] = (0.0, 0.3)
    repetition: Literal["single_trial", "detect"] = "single_trial"
    rep_min_peak_n: float = Field(default=50, gt=0)
    rep_min_separation_s: float = Field(default=1.0, gt=0)
    rep_prominence_n: float = Field(default=25, gt=0)
    max_mean_window_s: float = Field(default=0.5, gt=0)
    break_point: BreakVariant | None = None
    break_velocity_factor: float = Field(default=2.0, gt=1)
    break_controlled_fraction: float = Field(default=0.5, gt=0, lt=1)
    break_sustain_s: float = Field(default=0.05, ge=0)
    asymmetry: Asymmetry = "weaker_over_stronger_percent"
    rfd_min_hz: float = Field(default=100, gt=0)
    rate_mismatch_limit: float = Field(default=0.01, ge=0)
    timing_tolerance: float = Field(default=1e-6, gt=0)
    sensitivity_gain_errors: tuple[float, ...] = (-0.01, 0.01)
    sensitivity_cutoffs_hz: tuple[float, ...] = (10.0, 20.0, 50.0)

    @model_validator(mode="after")
    def valid(self) -> MethodSpec:
        if self.baseline_window_s[1] <= self.baseline_window_s[0]:
            raise ValueError("baseline window must increase")
        if any(abs(e) >= 0.5 for e in self.sensitivity_gain_errors):
            raise ValueError("sensitivity gain errors must be small relative perturbations")
        return self


@dataclass(frozen=True)
class SideResult:
    channel: str
    side: str
    onset: dict[str, object]
    metrics: dict[str, float | None]
    repetitions: list[dict[str, float]]
    refusals: dict[str, str]


@dataclass(frozen=True)
class Processed:
    protocol: str
    force_basis: str
    sides: dict[str, SideResult]
    bilateral: dict[str, object]
    qc: tuple[qc.Result, ...]
    uncertainty: dict[str, object]
    limitations: tuple[str, ...]
    method_spec: dict[str, object]
    athlete_declarations: dict[str, object]

    @property
    def verdict(self) -> str:
        return qc.verdict(list(self.qc))

    def report(self) -> dict[str, object]:
        return {
            "engine": ENGINE,
            "protocol": self.protocol,
            "force_basis": self.force_basis,
            "verdict": self.verdict,
            "diagnostic_only": self.verdict == "reject",
            "estimands": {
                "force": "strap/cuff force at the device, N (not muscle force)",
                "torque": "strap force x declared lever arm, N*m",
            },
            "sides": {k: asdict(v) for k, v in self.sides.items()},
            "bilateral": self.bilateral,
            "declarations": self.athlete_declarations,
            "uncertainty": self.uncertainty,
            "qc": [asdict(r) for r in self.qc],
            "limitations": list(self.limitations),
            "method_spec": self.method_spec,
            "qualification": "engineering calculations only; no scientific qualification",
        }

    def report_json(self) -> str:
        return json.dumps(self.report(), indent=2, sort_keys=True, allow_nan=False) + "\n"


def _admit(bundle: SignalBundle, spec: NordicSpec) -> tuple[object, list[str]]:
    meta = bundle.meta
    if meta.kind is not BundleKind.TIMESERIES or meta.sampling is None:
        raise ValueError("nordic/hip engine accepts timeseries bundles only")
    device = meta.device.device_class
    if device not in ADMITTED:
        raise ValueError(f"nordic/hip engine does not accept device class {device.value!r}")
    expected = (
        DeviceClass.NORDIC_DYNAMOMETER
        if spec.protocol == "nordic_eccentric"
        else DeviceClass.HIP_DYNAMOMETER
    )
    if device is not expected:
        raise ValueError(f"protocol {spec.protocol!r} contradicts device class {device.value!r}")
    names = [s.side for s in spec.sides]
    if spec.protocol == "nordic_eccentric" and sorted(names) != ["left", "right"]:
        raise ValueError("nordic eccentric protocol needs exactly one left and one right channel")
    by_name = {c.name: c for c in meta.channels}
    notes: list[str] = []
    for decl in spec.sides:
        channel = by_name.get(decl.channel)
        if channel is None or channel.quantity not in {Quantity.FORCE, Quantity.VOLTAGE}:
            raise ValueError(f"channel {decl.channel!r} is missing or not a force/voltage channel")
        if channel.side is not Side.NONE and channel.side.value != decl.side:
            raise ValueError(
                f"side contradiction on {decl.channel!r}: bundle says {channel.side.value}, "
                f"declaration says {decl.side}"
            )
        if channel.side is Side.NONE:
            notes.append(f"{decl.channel}: side {decl.side} declared by the caller only")
    return meta, notes


def _check_source_geometry(bundle: SignalBundle, spec: NordicSpec) -> None:
    trial = bundle.meta.geometry
    acquisition = bundle.meta.acquisition
    if acquisition is None or trial is None:
        return
    channels = {decl.channel for decl in spec.sides}
    angle = "hip_flexion_deg" if spec.protocol == "hip_squeeze" else "knee_angle_deg"
    for source in acquisition.sources:
        if channels.isdisjoint(source.channels):
            continue
        for field in ("lever_arm_m", angle):
            recorded = getattr(source.geometry, field)
            if recorded is None or field in source.geometry.unknown:
                continue
            expected = getattr(trial, field) if field not in trial.unknown else None
            if field == "lever_arm_m" and spec.lever_arm_m is not None:
                expected = spec.lever_arm_m
            if expected is not None and not math.isclose(
                expected, recorded, rel_tol=0, abs_tol=1e-9
            ):
                raise ValueError(f"{source.id}: declared {field} contradicts trial geometry")


def _lever(spec: NordicSpec, meta: object) -> tuple[float | None, str | None, str | None]:
    geometry = getattr(meta, "geometry", None)
    recorded = (
        geometry.lever_arm_m
        if geometry is not None and "lever_arm_m" not in geometry.unknown
        else None
    )
    lever = spec.lever_arm_m
    if lever is None:
        lever = recorded
    elif recorded is not None and not math.isclose(lever, recorded, rel_tol=0, abs_tol=1e-9):
        raise ValueError("configured lever arm contradicts bundle geometry")
    definition = spec.moment_arm_definition
    if lever is None:
        return None, definition, "torque refused: no declared lever arm"
    if not definition or not definition.strip():
        return None, None, "torque refused: no declared moment-arm definition"
    return lever, definition, None


def _angle(bundle: SignalBundle, spec: NordicSpec) -> tuple[FloatArray | None, str | None]:
    if spec.angle_channel is None:
        return None, "no measured angle channel: angle metrics refused (never recovered from force)"
    channel = next((c for c in bundle.meta.channels if c.name == spec.angle_channel), None)
    if channel is None or channel.quantity is not Quantity.ANGLE:
        raise ValueError(f"angle channel {spec.angle_channel!r} missing or not an angle")
    if channel.unit != "deg":
        raise ValueError("angle channel unit must be the canonical 'deg'")
    return np.radians(np.asarray(bundle[spec.angle_channel], dtype=np.float64)), None


def _segments(filtered: FloatArray, time: FloatArray, method: MethodSpec) -> list[slice]:
    if method.repetition == "single_trial":
        return [slice(0, len(filtered))]
    dt = float(np.median(np.diff(time)))
    peaks, _ = signal.find_peaks(
        filtered,
        height=method.rep_min_peak_n,
        distance=max(1, round(method.rep_min_separation_s / dt)),
        prominence=method.rep_prominence_n,
    )
    if len(peaks) == 0:
        return []
    cuts = [0]
    for a, b in itertools.pairwise(peaks):
        cuts.append(int(a + np.argmin(filtered[a:b])))
    cuts.append(len(filtered))
    return [slice(cuts[i], cuts[i + 1]) for i in range(len(peaks))]


def _max_mean(values: FloatArray, n: int) -> float | None:
    if n < 1 or n > len(values):
        return None
    csum = np.concatenate(([0.0], np.cumsum(values)))
    return float(np.max(csum[n:] - csum[:-n]) / n)


def _break_point(
    angle: FloatArray, time: FloatArray, filtered: FloatArray, peak_i: int, method: MethodSpec
) -> tuple[float | None, str]:
    if method.break_point == "peak_force_angle":
        return float(np.degrees(angle[peak_i])), "angle at peak force"
    omega = np.gradient(angle, time)
    n_ctrl = max(3, int(method.break_controlled_fraction * peak_i))
    controlled = float(np.median(omega[:n_ctrl]))
    if controlled <= 0:
        return None, "break refused: no positive controlled descent rate before the peak"
    dt = float(np.median(np.diff(time)))
    sustain = max(1, int(np.ceil(method.break_sustain_s / dt)))
    above = omega > method.break_velocity_factor * controlled
    for i in range(n_ctrl, len(above) - sustain + 1):
        if above[i : i + sustain].all():
            return float(np.degrees(angle[i])), "angular velocity exceeded controlled rate"
    return None, "break refused: angular velocity never exceeded the declared factor"


def _side(
    acquired: Acquired,
    decl: SideDeclaration,
    method: MethodSpec,
    lever: float | None,
    lever_refusal: str | None,
    angle: FloatArray | None,
    angle_refusal: str | None,
) -> tuple[SideResult, FloatArray, FloatArray]:
    time = acquired.timing.time_s
    force = acquired.force_n
    filtered = lowpass(force, time, method.filter, uniform=acquired.timing.uniform)
    refusals: dict[str, str] = {}
    if lever is None:
        refusals["torque"] = lever_refusal or "torque refused"
    if angle is None:
        refusals["angle_metrics"] = angle_refusal or "angle refused"
    if method.break_point is not None and angle is None:
        refusals["break_point"] = "break point needs a measured angle channel"

    dt = float(np.median(np.diff(time)))
    onset: dict[str, object]
    if method.onset == "first_exceedance":
        hits = np.flatnonzero(filtered > method.onset_threshold_n)
        onset = {
            "index": int(hits[0]) if len(hits) else None,
            "time_s": float(time[hits[0]]) if len(hits) else None,
            "threshold": method.onset_threshold_n,
            "method": "first_exceedance",
            "status": "detected" if len(hits) else "no excursion",
        }
    else:
        mask = window_mask(time, method.baseline_window_s)
        base = float(np.mean(filtered[mask]))
        sd = float(np.std(filtered[mask], ddof=1))
        found = identify_onset(
            filtered,
            time,
            base,
            sd,
            method.baseline_window_s,
            CellMethod(
                onset=method.onset,
                onset_sd_multiple=method.onset_sd_multiple,
                onset_absolute_n=method.onset_threshold_n,
                baseline_window_s=method.baseline_window_s,
            ),
            uniform=acquired.timing.uniform,
        )
        onset = {
            "index": found.index,
            "time_s": found.time_s,
            "threshold": found.threshold,
            "method": found.method,
            "status": found.status,
        }

    segments = _segments(filtered, time, method)
    reps: list[dict[str, float]] = []
    for seg in segments:
        sub = filtered[seg]
        i = seg.start + int(np.argmax(sub))
        reps.append({"peak_force_n": float(filtered[i]), "peak_time_s": float(time[i])})
    metrics: dict[str, float | None] = {}
    if not reps:
        refusals["repetitions"] = "no repetition met the declared peak criteria"
        return SideResult(decl.channel, decl.side, onset, metrics, reps, refusals), force, filtered
    best = max(range(len(reps)), key=lambda k: reps[k]["peak_force_n"])
    seg = segments[best]
    peak_i = int(np.argmax(filtered[seg])) + seg.start
    metrics["peak_force_n"] = float(filtered[peak_i])
    metrics["peak_time_s"] = float(time[peak_i])
    metrics["mean_peak_force_n"] = float(np.mean([r["peak_force_n"] for r in reps]))
    metrics["repetitions"] = float(len(reps))
    metrics["time_to_peak_from_record_start_s"] = float(time[peak_i] - time[0])
    onset_time = onset["time_s"]
    if isinstance(onset_time, float):
        metrics["time_to_peak_from_onset_s"] = float(time[peak_i]) - onset_time
    metrics["impulse_to_peak_n_s"] = float(np.trapezoid(filtered[: peak_i + 1], time[: peak_i + 1]))
    metrics["impulse_record_n_s"] = float(np.trapezoid(filtered, time))
    window = max(1, round(method.max_mean_window_s / dt))
    metrics["max_mean_force_n"] = _max_mean(filtered[seg], window)
    if lever is not None:
        metrics["peak_torque_n_m"] = float(filtered[peak_i]) * lever
    if angle is not None:
        metrics["angle_at_peak_force_deg"] = float(np.degrees(angle[peak_i]))
        torque = (filtered * lever) if lever is not None else None
        if torque is not None:
            metrics["eccentric_work_j"] = float(np.trapezoid(torque, angle))
        else:
            refusals["eccentric_work"] = lever_refusal or "work refused: no lever arm"
        if method.break_point is not None:
            value, note = _break_point(angle, time, filtered, peak_i, method)
            metrics["break_angle_deg"] = value
            if value is None:
                refusals["break_point"] = note
    return SideResult(decl.channel, decl.side, onset, metrics, reps, refusals), force, filtered


def _bilateral(sides: dict[str, SideResult], method: MethodSpec, basis: str) -> dict[str, object]:
    if set(sides) != {"left", "right"}:
        return {"available": False, "reason": "needs one left and one right channel"}
    left = sides["left"].metrics.get("peak_force_n")
    right = sides["right"].metrics.get("peak_force_n")
    if left is None or right is None:
        return {"available": False, "reason": "a side has no peak force"}
    strong, weak = max(left, right), min(left, right)
    if strong <= 0:
        return {"available": False, "reason": "no positive peak force"}
    value = (
        100 * weak / strong
        if method.asymmetry == "weaker_over_stronger_percent"
        else 100 * (strong - weak) / ((strong + weak) / 2)
    )
    return {
        "available": True,
        "force_basis": basis,
        "sum_peak_force_n": left + right,
        "weaker_side": "left" if left < right else "right" if right < left else "equal",
        "asymmetry": method.asymmetry,
        "asymmetry_value_percent": value,
    }


def process(bundle: SignalBundle, spec: NordicSpec, method: MethodSpec | None = None) -> Processed:
    method = MethodSpec() if method is None else MethodSpec.model_validate(method.model_dump())
    spec = NordicSpec.model_validate(spec.model_dump())
    meta, notes = _admit(bundle, spec)
    lever, definition, lever_refusal = _lever(spec, meta)
    _check_source_geometry(bundle, spec)
    geometry = bundle.meta.geometry
    checks: list[qc.Result] = []
    if spec.protocol == "hip_squeeze" and (
        geometry is None
        or "hip_flexion_deg" in geometry.unknown
        or geometry.hip_flexion_deg is None
    ):
        raise ValueError("hip squeeze refused: hip flexion angle of the protocol is undeclared")
    if spec.protocol == "nordic_eccentric" and (
        geometry is None or "knee_angle_deg" in geometry.unknown or geometry.knee_angle_deg is None
    ):
        raise ValueError("nordic protocol refused: knee/start angle is undeclared")
    angle, angle_refusal = _angle(bundle, spec)
    results: dict[str, SideResult] = {}
    calibration: dict[str, object] = {}
    base_acquired: dict[str, Acquired] = {}
    for decl in spec.sides:
        cell = (spec.cells.get(decl.channel) or CellSpec()).model_copy(
            update={"channel": decl.channel}
        )
        raw = from_bundle(bundle, decl.channel, admitted_classes=ADMITTED)
        acquired = acquire(raw, cell, timing_tolerance=method.timing_tolerance)
        base_acquired[decl.channel] = acquired
        checks += [
            *check_timing(acquired.timing, method.rfd_min_hz, method.rate_mismatch_limit),
            check_anti_aliasing(acquired.cell, acquired.timing),
            check_calibration_fit(acquired.calibration, 0.005),
            *check_saturation(acquired),
        ]
        result, _, _ = _side(acquired, decl, method, lever, lever_refusal, angle, angle_refusal)
        results[decl.side] = result
        calibration[decl.channel] = acquired.calibration.summary()
    checks.append(
        finding(
            "NH-SIDE",
            qc.Severity.FLAG,
            qc.Status.FAIL if notes else qc.Status.PASS,
            "side identity declared by the caller only; bundle carries no side metadata"
            if notes
            else "side declarations agree with bundle metadata",
        )
    )
    sensitivity: dict[str, object] = {}
    base = {k: r.metrics.get("peak_force_n") for k, r in results.items()}
    for error in method.sensitivity_gain_errors:
        sensitivity[f"gain_{error:+.3f}"] = {
            k: (None if v is None else v * (1 + error)) for k, v in base.items()
        }
    for cutoff in method.sensitivity_cutoffs_hz:
        alt: dict[str, float | None] = {}
        for decl in spec.sides:
            acq = base_acquired[decl.channel]
            if cutoff >= acq.timing.measured_hz / 2 or not acq.timing.uniform:
                alt[decl.side] = None
                continue
            smooth = lowpass(
                acq.force_n,
                acq.timing.time_s,
                FilterSpec(kind="butterworth_lowpass", cutoff_hz=cutoff),
                uniform=True,
            )
            alt[decl.side] = float(np.max(smooth))
        sensitivity[f"cutoff_{cutoff:g}_hz"] = alt
    torque_u: dict[str, float | None] = {}
    for k, r in results.items():
        t = r.metrics.get("peak_torque_n_m")
        torque_u[k] = (
            None
            if t is None or lever is None or spec.lever_arm_uncertainty_m is None
            else t * spec.lever_arm_uncertainty_m / lever
        )
    return Processed(
        protocol=spec.protocol,
        force_basis=spec.force_basis,
        sides=results,
        bilateral=_bilateral(results, method, spec.force_basis),
        qc=tuple(checks),
        uncertainty={
            "peak_force_variants_n": sensitivity,
            "peak_torque_lever_uncertainty_n_m": torque_u,
            "scope": "conditional sensitivity, not a complete uncertainty budget",
        },
        limitations=LIMITATIONS,
        method_spec=method.model_dump(mode="json"),
        athlete_declarations={
            "lever_arm_m": lever,
            "moment_arm_definition": definition,
            "calibration": calibration,
            "angle_channel": spec.angle_channel,
            "side_notes": notes,
        },
    )
