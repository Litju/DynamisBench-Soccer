"""Production force-platform reconstruction and reusable event/metric primitives.

No analytical generator is imported. Expected results belong in independent tests.
Named methods are conditional processing alternatives, never scientific qualification.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from importlib import metadata, resources
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from scipy import integrate, signal

from dynamisbench_soccer.engines.force_platform_acquisition import (
    Acquired,
    PlateSpec,
    acquire,
    cop,
    input_digest,
    window_mask,
)
from dynamisbench_soccer.qc import spec as qc
from dynamisbench_soccer.signals.schema import (
    Acquisition,
    AcquisitionSource,
    BundleMeta,
    ChannelSpec,
    Clock,
    Device,
    DeviceClass,
    FloatArray,
    Frame,
    Geometry,
    Parent,
    Provenance,
    Quantity,
    Sampling,
    Side,
    SignalBundle,
)


class MethodSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    version: Literal["force-platform.v1"] = "force-platform.v1"
    filter: Literal["raw", "butterworth", "chebyshev"] = "raw"
    filter_order: int = Field(default=4, ge=1, le=12)
    cutoff_hz: float = Field(default=20, gt=0)
    ripple_db: float = Field(default=0.05, gt=0)
    onset: Literal["5sd", "reverse_scan", "first_derivative", "absolute_N"] = "5sd"
    onset_sd_multiple: float = Field(default=5, gt=0)
    onset_floor_n: float = Field(default=1, gt=0)
    onset_absolute_n: float = Field(default=20, gt=0)
    onset_sustain_s: float = Field(default=0.01, ge=0)
    derivative_cutoff_hz: float = Field(default=10, gt=0)
    derivative_floor_n_s: float = Field(default=50, gt=0)
    contact: Literal["fixed_N", "system_weight_drop", "k_sd_unloaded"] = "fixed_N"
    contact_signal: Literal["raw", "filtered"] = "raw"
    contact_threshold_n: float = Field(default=20, gt=0)
    weight_drop_fraction: float = Field(default=0.97, gt=0, lt=1)
    unloaded_sd_multiple: float = Field(default=5, gt=0)
    unloaded_floor_n: float = Field(default=1, gt=0)
    unloaded_window_s: tuple[float, float] | None = None
    min_flight_s: float = Field(default=0.02, gt=0)
    integration: Literal["trapezoid", "simpson"] = "trapezoid"
    integration_start: Literal["quiet_start", "onset", "record_start"] = "quiet_start"
    initial_velocity_m_s: float | None = None
    quiet_window_s: tuple[float, float] = (0, 1)
    maximum_quiet_sd_n: float = Field(default=5, gt=0)
    gravity_m_s2: float = Field(default=9.80665, gt=0)
    alignment: Literal["strict", "linear_overlap"] = "strict"
    assume_shared_clock: bool = False
    cop_threshold_n: float = Field(default=20, gt=0)
    calibration_force_bound_n: float | None = Field(default=None, ge=0)
    body_weight_bound_n: float | None = Field(default=None, ge=0)
    initial_velocity_bound_m_s: float = Field(default=0, ge=0)
    cross_talk_limit_fraction: float = Field(default=0.02, ge=0, lt=1)

    @model_validator(mode="after")
    def valid_windows(self) -> MethodSpec:
        if self.quiet_window_s[1] - self.quiet_window_s[0] < 1:
            raise ValueError("bodyweight requires at least one second of quiet stance")
        if (
            self.unloaded_window_s is not None
            and self.unloaded_window_s[1] <= self.unloaded_window_s[0]
        ):
            raise ValueError("unloaded window must increase")
        return self


@dataclass(frozen=True)
class Onset:
    index: int | None
    time_s: float | None
    threshold: float
    threshold_unit: str
    status: str


@dataclass(frozen=True)
class Flight:
    takeoff_s: float
    landing_s: float | None
    takeoff_velocity_m_s: float | None
    height_velocity_m: float | None
    height_flight_time_m: float | None
    timing_resolution_s: float


@dataclass(frozen=True)
class Processed:
    time_s: FloatArray
    raw_force_n: FloatArray
    force_n: FloatArray
    acceleration_m_s2: FloatArray
    velocity_m_s: FloatArray
    displacement_m: FloatArray
    grf_power_w: FloatArray
    net_power_w: FloatArray
    contact_state: np.ndarray
    body_weight_n: float
    body_weight_sd_n: float
    mass_kg: float
    onset: Onset
    flights: tuple[Flight, ...]
    metrics: dict[str, float]
    qc: tuple[qc.Result, ...]
    uncertainty: dict[str, float | bool | str | None]
    method_spec: dict[str, object]
    cop_m: FloatArray | None
    plate_cop_m: dict[int, FloatArray | None]
    plate_force_n: dict[int, FloatArray]

    @property
    def verdict(self) -> str:
        return qc.verdict(list(self.qc))

    def report(self) -> dict[str, object]:
        return {
            "engine": "force-platform.v1",
            "verdict": self.verdict,
            "diagnostic_only": self.verdict == "reject",
            "body_weight_n": self.body_weight_n,
            "body_weight_sd_n": self.body_weight_sd_n,
            "mass_kg": self.mass_kg,
            "onset": asdict(self.onset),
            "flights": [asdict(f) for f in self.flights],
            "metrics": self.metrics,
            "uncertainty": self.uncertainty,
            "qc": [asdict(r) for r in self.qc],
            "method_spec": self.method_spec,
            "cop_available": self.cop_m is not None,
            "plate_cop_available": {str(k): v is not None for k, v in self.plate_cop_m.items()},
            "qualification": "engineering calculations only; no scientific qualification",
        }

    def report_json(self) -> str:
        return json.dumps(self.report(), indent=2, sort_keys=True, allow_nan=False) + "\n"


def filter_force(
    force: FloatArray, time: FloatArray, method: MethodSpec, *, derivative: bool = False
) -> FloatArray:
    if method.filter == "raw" and not derivative:
        return force.copy()
    hz = float(1 / np.median(np.diff(time)))
    cutoff = method.derivative_cutoff_hz if derivative else method.cutoff_hz
    if cutoff >= hz / 2:
        raise ValueError("filter cutoff must be below reference-time Nyquist")
    if derivative or method.filter == "butterworth":
        sos = signal.butter(method.filter_order, cutoff, fs=hz, output="sos")
    else:
        sos = signal.cheby1(method.filter_order, method.ripple_db, cutoff, fs=hz, output="sos")
    # Preserve the calibration/BW DC level, including even-order Chebyshev passband ripple.
    dc = abs(signal.sosfreqz(sos, worN=[0], fs=hz)[1][0])
    sos[0, :3] /= dc
    filtered = np.asarray(signal.sosfiltfilt(sos, force, padtype="odd"), dtype=np.float64)
    if not np.isfinite(filtered).all():
        raise ValueError("filter produced nonfinite values")
    return filtered


def cumulative(
    values: FloatArray, time: FloatArray, method: Literal["trapezoid", "simpson"]
) -> FloatArray:
    if (
        values.ndim != 1
        or time.ndim != 1
        or values.shape != time.shape
        or len(time) < 3
        or not np.isfinite(values).all()
        or not np.isfinite(time).all()
        or np.any(np.diff(time) <= 0)
    ):
        raise ValueError("finite equal-length traces and increasing time required")
    if method == "trapezoid":
        return np.asarray(integrate.cumulative_trapezoid(values, time, initial=0), dtype=np.float64)
    if method == "simpson":
        return np.asarray(integrate.cumulative_simpson(values, x=time, initial=0), dtype=np.float64)
    raise ValueError("unsupported integration method")


def identify_onset(
    force: FloatArray,
    time: FloatArray,
    bw: float,
    sd: float,
    quiet: tuple[float, float],
    method: MethodSpec,
) -> Onset:
    # Search starts halfway through the declared pre-movement weighing interval.
    search = np.flatnonzero(time >= quiet[0] + (quiet[1] - quiet[0]) / 2)
    base = max(method.onset_sd_multiple * sd, method.onset_floor_n)
    values = np.abs(force - bw)
    threshold = base
    unit = "N"
    if method.onset == "absolute_N":
        threshold = method.onset_absolute_n
    if method.onset == "first_derivative":
        smooth = filter_force(force, time, method, derivative=True)
        derivative = np.gradient(smooth, time)
        mask = window_mask(time, quiet)
        threshold = max(
            method.onset_sd_multiple * float(np.std(derivative[mask], ddof=1)),
            method.derivative_floor_n_s,
        )
        values = np.abs(derivative)
        unit = "N/s"
    if method.onset == "reverse_scan":
        high = max(2 * base, method.onset_absolute_n)
        duration = max(1, int(np.ceil(method.onset_sustain_s / float(np.median(np.diff(time))))))
        candidates = [
            int(i)
            for i in search
            if i + duration <= len(force) and np.all(values[i : i + duration] > high)
        ]
        if not candidates:
            return Onset(None, None, base, unit, "no sustained excursion")
        i = candidates[0]
        while i > 0 and values[i - 1] > base:
            i -= 1
    else:
        threshold_indices = search[values[search] > threshold]
        if len(threshold_indices) == 0:
            return Onset(None, None, threshold, unit, "no excursion")
        i = int(threshold_indices[0])
    if i == 0:
        return Onset(None, None, threshold, unit, "onset predates recording")
    i -= 1  # Named one-sample backdating, recorded in the resolved method specification.
    return Onset(i, float(time[i]), threshold, unit, "detected")


def _default_rules() -> qc.RuleSet:
    resource = resources.files("dynamisbench_soccer.qc").joinpath("rules.toml")
    if resource.is_file():
        import tomllib

        return qc.RuleSet.model_validate(tomllib.loads(resource.read_text(encoding="utf-8")))
    return qc.load(Path(__file__).resolve().parents[3] / "qc/rules.toml")


def _validated_rules(rules: qc.RuleSet | None) -> qc.RuleSet:
    baseline = _default_rules()
    active = baseline if rules is None else qc.RuleSet.model_validate(rules.model_dump())
    problems = qc.validate(active)
    if problems:
        raise ValueError("invalid QC rules: " + "; ".join(problems))
    for rule in active.rule:
        required = set(baseline.by_id(rule.id).params)
        if set(rule.params) != required:
            raise ValueError(f"{rule.id}: QC parameter names must match {sorted(required)}")
        if any(not np.isfinite(v) or v < 0 for v in rule.params.values()):
            raise ValueError(f"{rule.id}: finite nonnegative QC parameters required")
        if "min_run" in rule.params and (
            rule.params["min_run"] < 1 or not rule.params["min_run"].is_integer()
        ):
            raise ValueError(f"{rule.id}: integer min_run >=1 required")
    return active


def _qc_bundle(acquired: Acquired) -> SignalBundle:
    original = acquired.plates[0].source
    raw = original.meta.model_dump(mode="json")
    channels = [ChannelSpec(name="time", quantity=Quantity.TIME, unit="s")]
    data = {"time": acquired.time_s.copy()}
    sources: list[AcquisitionSource] = []
    for p in acquired.plates:
        name = f"fz_plate{p.plate}"
        channels.append(
            ChannelSpec(
                name=name,
                quantity=Quantity.FORCE,
                unit="N",
                axis="z",
                plate=p.plate,
                side=p.side,
                frame="field",
            )
        )
        data[name] = p.wrench[:, 2].copy()
        sources.append(
            AcquisitionSource(
                id=f"plate{p.plate}",
                device=Device(device_class=DeviceClass.FORCE_PLATE),
                clock="reference",
                channels=(["time"] if len(sources) == 0 else []) + [name],
                geometry=Geometry(plate_layout="single"),
                calibration_unknown_reason="Calibration and source records in method_spec",
                raw_format="RES-670 calibrated aligned SI measurements",
            )
        )
    geometry = Geometry(
        plate_layout="single" if len(sources) == 1 else "dual_side_by_side",
        summed_force_policy=None if len(sources) == 1 else "sum_fz_all_plates",
    )
    hz = float(1 / np.median(np.diff(acquired.time_s)))
    raw.update(
        bundle_id="force-platform-calibrated",
        channels=[c.model_dump(mode="json") for c in channels],
        n_rows=len(acquired.time_s),
        sampling=Sampling(
            nominal_hz=round(hz, 9), measured_hz=hz, t0_s=float(acquired.time_s[0])
        ).model_dump(),
        geometry=geometry.model_dump(),
        calibration=[],
        zeroing=None,
        acquisition_unknown_reason=None,
        acquisition=Acquisition(
            clocks=[Clock(id="reference", source="host", epoch="aligned source reference time")],
            reference_clock="reference",
            sources=sources,
            frames=[Frame(id="field", kind="field", definition="common right-handed axes, z up")],
        ).model_dump(),
        vendor_reported={},
        data_sha256=None,
        provenance=Provenance(
            created_by="force-platform@1",
            parents=[
                Parent(bundle_id=p.source.meta.bundle_id, sha256=input_digest(p.source))
                for p in acquired.plates
            ],
            transforms=[
                "calibration -> unloaded zero -> axes/origin -> clock/latency -> explicit alignment"
            ],
        ).model_dump(),
    )
    return SignalBundle(BundleMeta.model_validate(raw), data)


def _finding(
    identifier: str, severity: qc.Severity, status: qc.Status, detail: str, **values: float
) -> qc.Result:
    return qc.Result(
        identifier, identifier.lower().replace("-", "_"), severity, status, detail, values
    )


def _reuse_zero_offsets(
    acquired: Acquired, selected: tuple[PlateSpec, ...]
) -> tuple[PlateSpec, ...]:
    lookup = {p.plate: p for p in selected}
    result = []
    for trace in acquired.plates:
        original = lookup.get(trace.plate, PlateSpec(plate=trace.plate))
        result.append(
            original.model_copy(
                update={
                    "zero_window_s": None,
                    "zero_offsets": {
                        c: float(trace.offset[j])
                        for j, c in enumerate(("fx", "fy", "fz", "mx", "my", "mz"))
                        if c in trace.components
                    },
                }
            )
        )
    return tuple(result)


def process(
    bundles: SignalBundle | tuple[SignalBundle, ...],
    *,
    method: MethodSpec | None = None,
    plates: tuple[PlateSpec, ...] = (),
    weighing: SignalBundle | None = None,
    rules: qc.RuleSet | None = None,
    assignment_reference: dict[int, Side] | None = None,
    isolated_loads: dict[int, SignalBundle] | None = None,
) -> Processed:
    method = MethodSpec() if method is None else MethodSpec.model_validate(method.model_dump())
    active_rules = _validated_rules(rules)
    if assignment_reference is not None and any(
        not isinstance(v, Side) for v in assignment_reference.values()
    ):
        raise ValueError("assignment_reference values must be explicit Side values")
    selected = tuple(PlateSpec.model_validate(p.model_dump()) for p in plates)
    acquired = acquire(
        bundles,
        selected,
        alignment=method.alignment,
        assume_shared_clock=method.assume_shared_clock,
    )
    time = acquired.time_s
    raw_force = acquired.vertical_force_n
    dt = float(np.median(np.diff(time)))
    if 1 / dt < 500 - 1e-6:
        raise ValueError("force-platform processing requires at least 500 reference samples/s")
    for configured_plate in selected:
        if (
            weighing is None
            and configured_plate.zero_window_s is not None
            and max(configured_plate.zero_window_s[0], method.quiet_window_s[0])
            < min(configured_plate.zero_window_s[1], method.quiet_window_s[1])
        ):
            raise ValueError("unloaded zeroing cannot overlap loaded weighing")
    weight_data = acquired
    if weighing is not None:
        weighing_context = acquired.plates[0].source.meta
        if (
            weighing.meta.subject != weighing_context.subject
            or weighing.meta.data_category != weighing_context.data_category
            or weighing.meta.session.session_id != weighing_context.session.session_id
            or weighing.meta.session.date != weighing_context.session.date
        ):
            raise ValueError("weighing record must match subject/session/date/category")
        if method.integration_start == "quiet_start":
            raise ValueError(
                "separate weighing requires record_start/onset and explicit initial velocity"
            )
        if method.initial_velocity_m_s is None:
            raise ValueError(
                "separate weighing needs explicit initial velocity at integration start"
            )
        weight_data = acquire(
            weighing,
            _reuse_zero_offsets(acquired, selected),
            alignment=method.alignment,
            assume_shared_clock=method.assume_shared_clock,
        )
    mask = window_mask(weight_data.time_s, method.quiet_window_s, 1.0)
    weights = weight_data.vertical_force_n[mask]
    bw = float(np.mean(weights))
    sd = float(np.std(weights, ddof=1))
    if not np.isfinite([bw, sd]).all() or bw <= 0:
        raise ValueError("positive finite body weight required")
    if sd > method.maximum_quiet_sd_n:
        raise ValueError(
            "weighing window is not stable quiet stance; supply a separate weighing record"
        )
    mass = bw / method.gravity_m_s2
    force = filter_force(raw_force, time, method)
    # A separate weighing interval is not falsely asserted to occur inside the motion record.
    search_quiet = (
        method.quiet_window_s if weighing is None else (float(time[0]) - 1, float(time[0]))
    )
    if method.onset == "first_derivative" and weighing is not None:
        onset = Onset(
            None,
            None,
            method.derivative_floor_n_s,
            "N/s",
            "separate weighing: no pre-movement derivative baseline",
        )
    else:
        onset = identify_onset(force, time, bw, sd, search_quiet, method)
    if method.contact == "fixed_N":
        threshold = method.contact_threshold_n
    elif method.contact == "system_weight_drop":
        threshold = bw * (1 - method.weight_drop_fraction)
    else:
        if method.unloaded_window_s is None:
            raise ValueError("k_sd_unloaded requires an explicitly recorded unloaded window")
        unloaded = raw_force[window_mask(time, method.unloaded_window_s)]
        threshold = max(
            float(np.mean(unloaded))
            + method.unloaded_sd_multiple * float(np.std(unloaded, ddof=1)),
            method.unloaded_floor_n,
        )
    if not 0 < threshold < bw:
        raise ValueError("contact threshold must lie between zero and body weight")
    contact = (raw_force if method.contact_signal == "raw" else force) > threshold
    if method.integration_start == "onset":
        if onset.index is None:
            raise ValueError("cannot integrate from an unidentified onset")
        start = onset.index
    elif method.integration_start == "quiet_start":
        start = int(np.flatnonzero(window_mask(time, method.quiet_window_s, 1))[0])
    else:
        start = 0
    if len(time) - start < 3:
        raise ValueError("integration start leaves fewer than three samples")
    v0 = method.initial_velocity_m_s
    if v0 is None:
        if weighing is not None or (
            method.integration_start == "record_start" and time[0] < method.quiet_window_s[0] - dt
        ):
            raise ValueError("initial velocity is not established by quiet stance")
        v0 = 0.0
    acceleration = (force - bw) / mass
    velocity = np.full(len(time), np.nan, dtype=np.float64)
    displacement = velocity.copy()
    velocity[start:] = v0 + cumulative(acceleration[start:], time[start:], method.integration)
    displacement[start:] = cumulative(velocity[start:], time[start:], method.integration)
    flight_records = []
    # A flight must follow loaded contact; initial falling intervals are not rebound flights.
    edges = np.diff(np.r_[contact[0], contact].astype(np.int8))
    takeoffs = np.flatnonzero(edges == -1)
    landings = np.flatnonzero(edges == 1)
    sync = max(p.sync_uncertainty_s for p in acquired.plates)
    for off in takeoffs:
        next_on = landings[landings > off]
        land = int(next_on[0]) if len(next_on) else None
        end = float(time[land]) if land is not None else float(time[-1] + dt)
        if end - float(time[off]) < method.min_flight_s:
            continue
        v = float(velocity[off]) if np.isfinite(velocity[off]) else None
        flight_records.append(
            Flight(
                float(time[off]),
                float(time[land]) if land is not None else None,
                v,
                v * v / (2 * method.gravity_m_s2) if v is not None and v > 0 else None,
                method.gravity_m_s2 * (end - float(time[off])) ** 2 / 8
                if land is not None
                else None,
                dt + 2 * sync,
            )
        )
    calibrated = _qc_bundle(acquired)
    checks = list(qc.run(calibrated, active_rules))
    checks.append(
        _finding(
            "FP-BW",
            qc.Severity.REJECT,
            qc.Status.PASS if sd <= method.maximum_quiet_sd_n else qc.Status.FAIL,
            "declared weighing-window stability",
            sd_n=sd,
            body_weight_n=bw,
        )
    )
    checks.append(
        _finding(
            "FP-RAILS",
            qc.Severity.REJECT,
            qc.Status.FAIL
            if any(p.saturated for p in acquired.plates)
            else qc.Status.NOT_APPLICABLE,
            "declared sensor rails exceeded"
            if any(p.saturated for p in acquired.plates)
            else "no declared rail exceedance; missing rail metadata leaves clipping uncertain",
        )
    )
    checks.append(
        _finding(
            "FP-SYNC",
            qc.Severity.FLAG,
            qc.Status.FAIL if acquired.assumed_clock else qc.Status.PASS,
            "explicit unverified shared-clock assumption"
            if acquired.assumed_clock
            else "declared clocks aligned; no interpolation"
            if not acquired.resampled
            else "explicit linear overlap interpolation",
        )
    )
    checks.append(
        _finding(
            "FP-AXES",
            qc.Severity.FLAG,
            qc.Status.FAIL if any(p.assumed_axes for p in acquired.plates) else qc.Status.PASS,
            "explicit field-axis assumption used"
            if any(p.assumed_axes for p in acquired.plates)
            else "declared static frames resolved or explicit rotation applied",
        )
    )
    if assignment_reference is None:
        checks.append(
            _finding(
                "FP-ASSIGN",
                qc.Severity.FLAG,
                qc.Status.NOT_APPLICABLE,
                "requires independent plate-to-side reference",
            )
        )
    else:
        if set(assignment_reference) != {p.plate for p in acquired.plates}:
            raise ValueError("assignment reference must cover every plate")
        ok = all(
            assignment_reference[p.plate] == p.side
            and p.side in {Side.LEFT, Side.RIGHT, Side.BILATERAL}
            for p in acquired.plates
        )
        checks.append(
            _finding(
                "FP-ASSIGN",
                qc.Severity.REJECT,
                qc.Status.PASS if ok else qc.Status.FAIL,
                "independent plate-side assignment comparison",
            )
        )
    if not isolated_loads:
        checks.append(
            _finding(
                "FP-CROSSTALK",
                qc.Severity.FLAG,
                qc.Status.NOT_APPLICABLE,
                "needs isolated-load trials; bilateral correlation does not prove cross-talk",
            )
        )
    for loaded, trial in (isolated_loads or {}).items():
        primary_device = acquired.plates[0].source.meta.device
        for name in ("serial_hash", "manufacturer", "model", "firmware"):
            known = getattr(primary_device, name)
            observed = getattr(trial.meta.device, name)
            if known is not None and observed is not None and known != observed:
                raise ValueError("isolated-load record refers to a different instrument")
        if len(acquired.plates) != 2 or loaded not in {p.plate for p in acquired.plates}:
            raise ValueError("cross-talk trial needs two plates and a known loaded plate")
        isolated = acquire(
            trial,
            _reuse_zero_offsets(acquired, selected),
            alignment=method.alignment,
            assume_shared_clock=method.assume_shared_clock,
        )
        load = next(p for p in isolated.plates if p.plate == loaded)
        denominator = float(np.sqrt(np.mean(load.wrench[:, 2] ** 2)))
        if denominator <= 1:
            raise ValueError("isolated-load reference is not loaded")
        fraction = (
            sum(
                float(np.sqrt(np.mean(p.wrench[:, 2] ** 2)))
                for p in isolated.plates
                if p.plate != loaded
            )
            / denominator
        )
        checks.append(
            _finding(
                f"FP-CROSSTALK-{loaded}",
                qc.Severity.REJECT,
                qc.Status.PASS if fraction <= method.cross_talk_limit_fraction else qc.Status.FAIL,
                "independent isolated-load RMS leakage ratio",
                fraction=fraction,
            )
        )
    if not np.isfinite(velocity[start:]).all() or not np.isfinite(displacement[start:]).all():
        raise ValueError("numerical reconstruction is not finite")
    grf_power = force * velocity
    net_power = (force - bw) * velocity
    metrics = {
        "peak_force_n": float(np.max(force)),
        "net_impulse_n_s": float(
            cumulative((force - bw)[start:], time[start:], method.integration)[-1]
        ),
        "peak_velocity_m_s": float(np.nanmax(velocity)),
        "minimum_displacement_m": float(np.nanmin(displacement)),
        "peak_grf_power_w": float(np.nanmax(grf_power)),
        "peak_net_power_w": float(np.nanmax(net_power)),
    }
    duration = float(time[-1] - time[start])
    force_bound = method.calibration_force_bound_n
    weight_bound = method.body_weight_bound_n
    uncertainty: dict[str, float | bool | str | None] = {
        "complete_measurement_uncertainty": False,
        "body_weight_standard_error_n": sd / np.sqrt(len(weights)),
        "body_weight_se_assumption": "independent quiet samples; systematic error excluded",
        "sample_interval_s": dt,
        "sync_uncertainty_s": sync,
        "event_interval_resolution_s": dt + 2 * sync,
        "velocity_input_error_envelope_m_s": None,
        "displacement_input_error_envelope_m": None,
        "scope": "conditional fixed-grid sensitivity only; incomplete measurement uncertainty",
    }
    if force_bound is not None and weight_bound is not None and method.filter == "raw":
        if weight_bound >= bw:
            raise ValueError("bodyweight error bound must be smaller than bodyweight")
        acceleration_bound = (
            method.gravity_m_s2
            * (force_bound * bw + float(np.max(np.abs(force))) * weight_bound)
            / (bw * (bw - weight_bound))
        )
        factor = 1.0 if method.integration == "trapezoid" else 1.5
        velocity_bound = method.initial_velocity_bound_m_s + factor * duration * acceleration_bound
        uncertainty.update(
            velocity_input_error_envelope_m_s=velocity_bound,
            displacement_input_error_envelope_m=factor * duration * velocity_bound,
        )
    if method.filter != "raw":
        uncertainty["input_envelope_unavailable"] = (
            "filtered envelope unavailable without filter-operator propagation"
        )
    primary = acquired.plates[0].source
    resolved: dict[str, object] = {
        **method.model_dump(mode="json"),
        "plates": [p.model_dump(mode="json") for p in selected],
        "resolved_plate_ids": [p.plate for p in acquired.plates],
        "resolved_components": {str(p.plate): p.components for p in acquired.plates},
        "cop_signal": "calibrated unfiltered wrench; actual surface plane, field XYZ",
        "surface_planes": {
            str(p.plate): {"normal": p.normal, "origin_m": p.origin} for p in acquired.plates
        },
        "zero_offsets_output_units": {str(p.plate): p.offset.tolist() for p in acquired.plates},
        "actual_reference_hz": 1 / dt,
        "resolved_start_s": float(time[start]),
        "initial_velocity_used_m_s": v0,
        "resolved_contact_threshold_n": threshold,
        "onset_backdating_samples": 1,
        "filter_realization": "DC-normalized SOS forward/backward, odd padding; one-pass cutoff",
        "integration_realization": "cumulative Simpson with quadratic neighbor/tail"
        if method.integration == "simpson"
        else "cumulative trapezoid",
        "input_digests": [input_digest(p.source) for p in acquired.plates],
        "weighing_input_digest": input_digest(weighing)
        if weighing is not None
        else input_digest(primary),
        "input_hash_scheme": "sorted metadata JSON (no data_sha256) + little-endian samples",
        "source_file_sha256": [p.source.meta.data_sha256 for p in acquired.plates],
        "qc_rules": active_rules.model_dump(mode="json"),
        "qc_rules_sha256": hashlib.sha256(
            json.dumps(
                active_rules.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
            ).encode()
        ).hexdigest(),
        "weighing_context": weighing.meta.model_dump(mode="json") if weighing is not None else None,
        "upstream_calibration": [
            [c.model_dump(mode="json") for c in p.source.meta.calibration] for p in acquired.plates
        ],
        "upstream_zeroing": [
            p.source.meta.zeroing.model_dump(mode="json")
            if p.source.meta.zeroing is not None
            else None
            for p in acquired.plates
        ],
        "input_identity": {
            "subject": primary.meta.subject.pseudonym,
            "session": primary.meta.session.model_dump(mode="json"),
            "test_id": primary.meta.test_id,
            "protocol": primary.meta.protocol.model_dump(mode="json"),
            "category": primary.meta.data_category.value,
        },
        "source_acquisition": [
            p.source.meta.acquisition.model_dump(mode="json")
            if p.source.meta.acquisition is not None
            else None
            for p in acquired.plates
        ],
        "zero_window_sd_output_units": {
            str(p.plate): p.offset_sd.tolist() for p in acquired.plates
        },
        "cross_talk_inputs": {str(k): input_digest(v) for k, v in (isolated_loads or {}).items()},
        "assignment_reference": {str(k): v.value for k, v in (assignment_reference or {}).items()},
        "source_provenance": [
            p.source.meta.provenance.model_dump(mode="json") for p in acquired.plates
        ],
        "versions": {"engine": "1", "numpy": np.__version__, "scipy": metadata.version("scipy")},
    }
    combined_cop, plate_cop = cop(acquired, method.cop_threshold_n)
    return Processed(
        time.copy(),
        raw_force.copy(),
        force,
        acceleration,
        velocity,
        displacement,
        grf_power,
        net_power,
        contact.copy(),
        bw,
        sd,
        mass,
        onset,
        tuple(flight_records),
        metrics,
        tuple(checks),
        uncertainty,
        resolved,
        combined_cop,
        plate_cop,
        {p.plate: p.wrench[:, 2].copy() for p in acquired.plates},
    )
