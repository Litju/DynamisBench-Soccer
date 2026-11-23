"""Jump assessment variants on force-platform observations (RES-671).

Every estimand is a named method with its own definition. Impulse-momentum, flight-time and
(where available) contact-system, LPT or IMU heights are kept apart: this layer only ever
labels the force-platform ones and reports their numerical difference as analytical model
agreement. It never states that two devices are interchangeable (RES-703 owns that).
Protocol eligibility is checked from the signal and refused rather than assumed.
No analytical generator is imported: expected values belong in independent tests.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from dynamisbench_soccer.engines.force_platform import Processed as ForceProcessed
from dynamisbench_soccer.signals.schema import FloatArray

ENGINE = "jump-assessment.v1"
GRAVITY_M_S2 = 9.80665
Protocol = Literal["cmj", "sj", "slcmj", "dj", "repeated_hops"]
HeightBasis = Literal["flight_time", "impulse_momentum"]

LIMITATIONS: tuple[str, ...] = (
    "Impulse-momentum and flight-time heights estimate different things (centre-of-mass rise "
    "after take-off versus time airborne, with equal take-off and landing posture assumed). "
    "Their numerical difference is reported, never merged.",
    "Heights from contact mats, linear position transducers and IMUs are separate estimands "
    "with their own device engines; analytical agreement does not establish empirical "
    "interchangeability (RES-703).",
    "Phase boundaries are signal-derived definitions (velocity extrema and zero crossings) "
    "and move with the declared onset threshold; they are not anatomical events.",
    "Take-off and landing are the first sample on each side of the contact threshold; the "
    "timing resolution is one sample interval.",
    "Drop-jump rebound heights and RSI depend on the declared drop height and contact "
    "definition; RSI with a flight-time basis and an impulse-momentum basis are different "
    "numbers.",
    "Single-leg eligibility is judged only from the declared unloaded-plate leakage; it does "
    "not verify which limb was tested.",
    "Uncertainty output is conditional sensitivity (timing resolution, body-weight error, "
    "contact threshold), not a complete measurement-uncertainty budget.",
    "Engineering calculations only: no scientific qualification of any jump metric.",
)


class JumpSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    version: Literal["jump-assessment.v1"] = "jump-assessment.v1"
    protocol: Protocol
    contact_threshold_n: float = Field(default=20, gt=0)
    quiet_s: float = Field(default=0.5, gt=0)
    onset_fraction_of_bw: float = Field(default=0.02, gt=0)
    unweight_min_fraction: float = Field(default=0.03, gt=0)
    sj_max_unweight_fraction: float = Field(default=0.02, gt=0)
    min_flight_s: float = Field(default=0.04, gt=0)
    rsi_height_basis: HeightBasis = "flight_time"
    drop_height_m: float | None = Field(default=None, gt=0)
    drop_speed_tolerance_m_s: float = Field(default=0.3, gt=0)
    plate_sides: dict[int, Literal["left", "right"]] = Field(default_factory=dict)
    unloaded_leakage_fraction: float = Field(default=0.05, ge=0, lt=1)
    min_hops: int = Field(default=3, ge=2)
    sensitivity_thresholds_n: tuple[float, ...] = (10.0, 20.0, 50.0)
    body_weight_error_fraction: float = Field(default=0.01, ge=0, lt=0.5)


class DropEvidence(BaseModel):
    """Touchdown velocity established independently of the declared drop height."""

    model_config = ConfigDict(
        extra="forbid", frozen=True, allow_inf_nan=False, str_strip_whitespace=True
    )
    touchdown_velocity_m_s: float = Field(lt=0)
    source: str = Field(min_length=1)


@dataclass(frozen=True)
class JumpSignal:
    """Force-platform quantities the assessment needs; build one from the device engine."""

    time_s: FloatArray
    force_n: FloatArray
    velocity_m_s: FloatArray
    displacement_m: FloatArray
    body_weight_n: float
    mass_kg: float
    plate_force_n: dict[int, FloatArray] = field(default_factory=dict)
    device_verdict: str = "unknown"
    raw_force_n: FloatArray | None = (
        None  # unfiltered force: contact events never use filtered data
    )

    @classmethod
    def from_engine(cls, result: ForceProcessed) -> JumpSignal:
        return cls(
            result.time_s,
            result.force_n,
            result.velocity_m_s,
            result.displacement_m,
            result.body_weight_n,
            result.mass_kg,
            dict(result.plate_force_n),
            result.verdict,
            result.raw_force_n,
        )


@dataclass(frozen=True)
class Assessment:
    protocol: str
    eligible: bool
    device_verdict: str
    eligibility: list[dict[str, object]]
    events: dict[str, float | None]
    phases: dict[str, float | None]
    estimands: dict[str, float]
    contacts: list[dict[str, float]]
    bilateral: dict[str, object]
    refusals: dict[str, str]
    uncertainty: dict[str, object]
    limitations: tuple[str, ...]
    method_spec: dict[str, object]
    diagnostics: dict[str, object] = field(default_factory=dict)

    def report(self) -> dict[str, object]:
        return {
            "engine": ENGINE,
            "protocol": self.protocol,
            "eligible": self.eligible,
            "device_verdict": self.device_verdict,
            "diagnostic_only": self.device_verdict == "reject" or not self.eligible,
            "eligibility": self.eligibility,
            "events": self.events,
            "phases": self.phases,
            "estimands": self.estimands,
            "diagnostics": self.diagnostics,
            "contacts": self.contacts,
            "bilateral": self.bilateral,
            "refusals": self.refusals,
            "uncertainty": self.uncertainty,
            "limitations": list(self.limitations),
            "method_spec": self.method_spec,
            "qualification": "engineering calculations only; no scientific qualification",
        }

    def report_json(self) -> str:
        return json.dumps(self.report(), indent=2, sort_keys=True, allow_nan=False) + "\n"


def _segments(contact: np.ndarray) -> list[tuple[int, int]]:
    """Index ranges [start, end) of consecutive contact samples."""
    edges = np.flatnonzero(np.diff(np.concatenate(([0], contact.astype(int), [0]))))
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2], strict=True)]


def _flights(contact: np.ndarray, time: FloatArray, minimum_s: float) -> list[tuple[int, int]]:
    """Complete flights as (first airborne index, first following contact index)."""
    found = []
    for (_, end), (start, _) in zip(_segments(contact), _segments(contact)[1:], strict=False):
        if time[start] - time[end] >= minimum_s - 1e-12:
            found.append((end, start))
    return found


def _height_flight(flight_s: float) -> float:
    return GRAVITY_M_S2 * flight_s**2 / 8


def _height_velocity(velocity: float) -> float:
    return velocity**2 / (2 * GRAVITY_M_S2) if velocity > 0 else 0.0


def _check(name: str, ok: bool, detail: str) -> dict[str, object]:
    return {"check": name, "passed": bool(ok), "detail": detail}


def _unweight(sig: JumpSignal, onset: int, takeoff: int) -> tuple[float, int]:
    window = slice(onset, takeoff)
    index = onset + int(np.argmin(sig.velocity_m_s[window]))
    return float(sig.velocity_m_s[index]), index


def assess(
    signal: JumpSignal, spec: JumpSpec, *, drop_evidence: DropEvidence | None = None
) -> Assessment:
    time, force = signal.time_s, signal.force_n
    dt = float(np.median(np.diff(time)))
    event_force = signal.raw_force_n if signal.raw_force_n is not None else force
    contact = event_force > spec.contact_threshold_n
    flights = _flights(contact, time, spec.min_flight_s)
    segments = _segments(contact)
    bw = signal.body_weight_n
    checks: list[dict[str, object]] = []
    refusals: dict[str, str] = {}
    events: dict[str, float | None] = {}
    phases: dict[str, float | None] = {}
    estimands: dict[str, float] = {}
    contacts: list[dict[str, float]] = []

    checks.append(
        _check("uniform_sampling", bool(np.allclose(np.diff(time), dt, rtol=1e-6)), "uniform grid")
    )
    needs_flight = len(flights) >= 1
    checks.append(
        _check("flight_present", needs_flight, f"{len(flights)} complete flight(s) found")
    )
    sd = float(np.std(force[time < time[0] + spec.quiet_s], ddof=1)) if len(time) > 3 else 0.0

    def quiet_ok() -> bool:
        window = time < time[0] + spec.quiet_s
        return bool(window.sum() >= 3 and np.all(np.abs(force[window] - bw) <= 0.05 * bw))

    def movement_onset() -> int | None:
        threshold = max(spec.onset_fraction_of_bw * bw, 5 * sd)
        hits = np.flatnonzero(np.abs(force - bw) > threshold)
        return int(hits[0]) - 1 if len(hits) and hits[0] > 0 else None

    if spec.protocol in {"cmj", "sj", "slcmj"} and flights:
        takeoff_i, landing_i = flights[0]
        onset = movement_onset()
        checks.append(_check("quiet_standing_before_movement", quiet_ok(), "force within 5% of BW"))
        checks.append(_check("onset_found", onset is not None, "movement onset located"))
        if onset is not None:
            min_v, min_i = _unweight(signal, onset, takeoff_i)
            depth = float((bw - force[onset:takeoff_i].min()) / bw)
            countermovement = depth >= spec.unweight_min_fraction
            if spec.protocol == "sj":
                checks.append(
                    _check(
                        "no_countermovement",
                        depth <= spec.sj_max_unweight_fraction,
                        f"unweighting depth {depth:.3f} of BW vs limit "
                        f"{spec.sj_max_unweight_fraction}",
                    )
                )
            else:
                checks.append(
                    _check(
                        "countermovement_present",
                        countermovement,
                        f"unweighting depth {depth:.3f} of BW vs minimum "
                        f"{spec.unweight_min_fraction}",
                    )
                )
            zero = np.flatnonzero(signal.velocity_m_s[min_i:takeoff_i] >= 0)
            zero_i = min_i + int(zero[0]) if len(zero) else None
            events.update(
                onset_s=float(time[onset]),
                minimum_velocity_s=float(time[min_i]),
                zero_velocity_s=float(time[zero_i]) if zero_i is not None else None,
            )
            phases["unweighting_s"] = float(time[min_i] - time[onset])
            phases["braking_s"] = float(time[zero_i] - time[min_i]) if zero_i is not None else None
            phases["propulsion_s"] = float(time[takeoff_i] - time[zero_i]) if zero_i else None
            phases["time_to_takeoff_s"] = float(time[takeoff_i] - time[onset])
            estimands["unweighting_depth_fraction_of_bw"] = depth
            estimands["peak_negative_velocity_m_s"] = min_v
            if zero_i is not None:
                net = force[zero_i:takeoff_i] - bw
                estimands["propulsive_net_impulse_n_s"] = float(
                    np.trapezoid(net, time[zero_i:takeoff_i])
                )
            estimands["peak_force_n"] = float(force[onset:takeoff_i].max())
            power = force[onset:takeoff_i] * signal.velocity_m_s[onset:takeoff_i]
            estimands["peak_power_w"] = float(power.max())
            estimands["peak_power_w_per_kg"] = float(power.max() / signal.mass_kg)
        events["takeoff_s"] = float(time[takeoff_i])
        events["landing_s"] = float(time[landing_i])
        v_to = float(signal.velocity_m_s[takeoff_i])
        flight_s = float(time[landing_i] - time[takeoff_i])
        estimands["flight_time_s"] = flight_s
        estimands["takeoff_velocity_m_s"] = v_to
        estimands["height_flight_time_m"] = _height_flight(flight_s)
        estimands["height_impulse_momentum_m"] = _height_velocity(v_to)
        estimands["height_difference_flight_minus_impulse_m"] = (
            estimands["height_flight_time_m"] - estimands["height_impulse_momentum_m"]
        )
        to_takeoff = phases.get("time_to_takeoff_s")
        if to_takeoff:
            estimands["rsi_modified_m_s"] = estimands["height_impulse_momentum_m"] / to_takeoff
        if spec.protocol == "slcmj":
            if len(signal.plate_force_n) != 2 or len(spec.plate_sides) != 2:
                checks.append(
                    _check("single_leg_plates", False, "two plates and declared sides required")
                )
            else:
                loaded = max(
                    signal.plate_force_n, key=lambda k: float(signal.plate_force_n[k].max())
                )
                other = next(k for k in signal.plate_force_n if k != loaded)
                leak = float(
                    np.mean(signal.plate_force_n[other][onset or 0 : takeoff_i])
                    / max(np.mean(signal.plate_force_n[loaded][onset or 0 : takeoff_i]), 1e-9)
                )
                checks.append(
                    _check(
                        "contralateral_unloaded",
                        leak <= spec.unloaded_leakage_fraction,
                        f"unloaded-plate leakage {leak:.3f} vs limit "
                        f"{spec.unloaded_leakage_fraction}",
                    )
                )
                events["loaded_side_plate"] = float(loaded)
    elif spec.protocol in {"cmj", "sj", "slcmj"}:
        refusals["jump"] = "no complete flight: take-off, landing and jump heights refused"

    if spec.protocol == "dj":
        if not segments or segments[0][1] == len(time):
            checks.append(_check("complete_rebound", False, "no complete initial contact"))
            refusals["dj"] = "no complete initial contact"
        else:
            first_start, first_end = segments[0]
            gct = float(time[first_end] - time[first_start])
            contacts.append({"start_s": float(time[first_start]), "contact_time_s": gct})
            estimands["ground_contact_time_s"] = gct
            rebound = bool(
                flights and flights[0][0] == first_end and signal.velocity_m_s[first_end] > 0
            )
            checks.append(
                _check(
                    "complete_rebound",
                    rebound,
                    "initial contact followed by upward take-off and landing",
                )
            )
            if rebound:
                takeoff_i, landing_i = flights[0]
                flight_s = float(time[landing_i] - time[takeoff_i])
                v_to = float(signal.velocity_m_s[takeoff_i])
                estimands["flight_time_s"] = flight_s
                estimands["height_flight_time_m"] = _height_flight(flight_s)
                estimands["takeoff_velocity_m_s"] = v_to
                estimands["height_impulse_momentum_m"] = _height_velocity(v_to)
                basis = (
                    estimands["height_flight_time_m"]
                    if spec.rsi_height_basis == "flight_time"
                    else estimands["height_impulse_momentum_m"]
                )
                estimands["rsi_m_s"] = basis / gct
                events.update(
                    contact_start_s=float(time[first_start]),
                    takeoff_s=float(time[takeoff_i]),
                    landing_s=float(time[landing_i]),
                )
            else:
                refusals["rsi"] = "no complete rebound flight after the first contact"
            quiet_contact = np.abs(event_force[first_start:first_end] - bw) <= 0.05 * bw
            standing = any(
                (b - a) * dt >= spec.quiet_s - dt / 2 for a, b in _segments(quiet_contact)
            )
            checks.append(
                _check(
                    "drop_landing",
                    signal.velocity_m_s[first_start] < 0 and first_start > 0 and not standing,
                    "observed airborne-to-contact transition, descending velocity and no stance",
                )
            )
            checks.append(
                _check(
                    "drop_evidence",
                    spec.drop_height_m is not None and drop_evidence is not None,
                    "requires declared drop height and independently sourced touchdown velocity",
                )
            )
            if drop_evidence is None:
                refusals["touchdown_velocity"] = "no independently sourced touchdown velocity"
            if spec.drop_height_m is None:
                refusals["drop_height"] = "no declared drop height: drop-speed check refused"
            elif drop_evidence is not None:
                expected = math.sqrt(2 * GRAVITY_M_S2 * spec.drop_height_m)
                measured = -drop_evidence.touchdown_velocity_m_s
                integrated = float(-signal.velocity_m_s[first_start])
                checks.append(
                    _check(
                        "impact_speed_matches_drop_height",
                        abs(measured - expected) <= spec.drop_speed_tolerance_m_s,
                        f"independent touchdown {measured:.3f} vs declared-drop {expected:.3f} m/s",
                    )
                )
                checks.append(
                    _check(
                        "integration_matches_touchdown",
                        abs(integrated - measured) <= spec.drop_speed_tolerance_m_s,
                        f"integrated {integrated:.3f} vs independent touchdown {measured:.3f} m/s",
                    )
                )
                estimands["drop_height_declared_m"] = spec.drop_height_m

    if spec.protocol == "repeated_hops":
        usable = [(a, b) for a, b in flights]
        for i, (end, start) in enumerate(usable):
            prior = [s for s in segments if s[1] == end]
            if not prior:
                continue
            gct = float(time[prior[0][1]] - time[prior[0][0]])
            flight_s = float(time[start] - time[end])
            contacts.append(
                {
                    "index": float(i),
                    "contact_time_s": gct,
                    "flight_time_s": flight_s,
                    "height_flight_time_m": _height_flight(flight_s),
                    "rsi_m_s": _height_flight(flight_s) / gct,
                }
            )
        checks.append(
            _check(
                "enough_hops",
                len(contacts) >= spec.min_hops,
                f"{len(contacts)} hops vs minimum {spec.min_hops}",
            )
        )
        if contacts:
            for key in ("contact_time_s", "flight_time_s", "height_flight_time_m", "rsi_m_s"):
                values = np.array([c[key] for c in contacts])
                estimands[f"mean_{key}"] = float(values.mean())
            estimands["hop_count"] = float(len(contacts))

    # Bilateral quantities need declared sides; never inferred from plate order.
    bilateral: dict[str, object] = {
        "available": False,
        "reason": "needs two plates with declared sides",
    }
    if len(signal.plate_force_n) == 2 and set(spec.plate_sides) == set(signal.plate_force_n):
        by_side = {spec.plate_sides[k]: v for k, v in signal.plate_force_n.items()}
        if set(by_side) == {"left", "right"}:
            window = slice(None)
            if "takeoff_s" in events and events["takeoff_s"] is not None:
                onset_t = events.get("onset_s", events.get("contact_start_s", time[0]))
                window = (time >= (onset_t or time[0])) & (time <= events["takeoff_s"])
            peaks = {s: float(v[window].max()) for s, v in by_side.items()}
            impulses = {s: float(np.trapezoid(v[window], time[window])) for s, v in by_side.items()}
            strong = max(impulses.values())
            bilateral = {
                "available": True,
                "peak_force_n": peaks,
                "impulse_n_s": impulses,
                "impulse_asymmetry_weaker_over_stronger_percent": 100
                * min(impulses.values())
                / strong
                if strong > 0
                else None,
            }

    # Conditional sensitivity.
    uncertainty: dict[str, object] = {
        "flight_time_resolution_s": dt,
        "height_flight_time_resolution_m": (
            GRAVITY_M_S2 * estimands["flight_time_s"] * dt / 4
            if estimands.get("flight_time_s")
            else None
        ),
        "scope": "conditional sensitivity, not a complete uncertainty budget",
    }
    by_threshold: dict[str, object] = {}
    for threshold in spec.sensitivity_thresholds_n:
        alt = _flights(event_force > threshold, time, spec.min_flight_s)
        by_threshold[f"{threshold:g}_n"] = [float(time[b] - time[a]) for a, b in alt]
    uncertainty["flight_time_by_contact_threshold_s"] = by_threshold
    if estimands.get("takeoff_velocity_m_s") is not None:
        v = estimands["takeoff_velocity_m_s"]
        e = spec.body_weight_error_fraction
        uncertainty["body_weight_error_fraction"] = e
        uncertainty["height_impulse_momentum_bw_sensitivity_m"] = [
            _height_velocity(v * (1 - e)),
            _height_velocity(v * (1 + e)),
        ]

    eligible = all(bool(c["passed"]) for c in checks)
    diagnostics: dict[str, object] = {}
    method_spec = spec.model_dump(mode="json")
    if spec.protocol == "dj":
        method_spec["drop_evidence"] = drop_evidence.model_dump() if drop_evidence else None
        if not eligible or signal.device_verdict == "reject":
            diagnostics = {"estimands": estimands, "contacts": contacts, "bilateral": bilateral}
            estimands, contacts, bilateral = {}, [], {"available": False}
    return Assessment(
        protocol=spec.protocol,
        eligible=eligible,
        device_verdict=signal.device_verdict,
        eligibility=checks,
        events=events,
        phases=phases,
        estimands=estimands,
        contacts=contacts,
        bilateral=bilateral,
        refusals=refusals,
        uncertainty=uncertainty,
        limitations=LIMITATIONS,
        method_spec=method_spec,
        diagnostics=diagnostics,
    )


__all__ = ["Assessment", "JumpSignal", "JumpSpec", "asdict", "assess"]


def rj_10_5_selection(
    assessment: Assessment, *, n_jumps: int = 10, n_best: int = 5
) -> dict[str, object]:
    """10/5 repeated-jump summaries (RJ-003): best-5-of-10 mean RSI and peak single RSI.

    Both are reported as named variants. The protocol needs exactly ``n_jumps`` valid hops
    from an eligible, device-accepted assessment; anything else is refused, never padded or
    truncated, and a rejected source never yields an accepted summary.
    """
    if assessment.protocol != "repeated_hops":
        raise ValueError(
            f"10/5 selection needs a repeated_hops assessment, not {assessment.protocol}"
        )
    if assessment.device_verdict == "reject" or not assessment.eligible:
        raise ValueError("assessment is ineligible or device-rejected: no accepted 10/5 summary")
    hops_ = assessment.contacts
    if len(hops_) != n_jumps:
        raise ValueError(f"10/5 requires exactly {n_jumps} valid hops, found {len(hops_)}")
    order = sorted(range(len(hops_)), key=lambda i: -hops_[i]["rsi_m_s"])
    best = sorted(order[:n_best])
    return {
        "protocol": "rj_10_5",
        "n_hops": len(hops_),
        "best_5_of_10_mean": {
            "rsi_m_s": float(np.mean([hops_[i]["rsi_m_s"] for i in best])),
            "selected_hop_indices": best,
        },
        "peak_single_rsi": {
            "rsi_m_s": float(hops_[order[0]]["rsi_m_s"]),
            "selected_hop_indices": [order[0]],
        },
        "all_hops_mean_rsi_m_s": float(np.mean([h["rsi_m_s"] for h in hops_])),
        "rsi_height_basis": "flight_time",  # per-hop RSI is always flight-time based
        "contact_time_criterion": "not applied (RJ-003 variants rank by RSI only)",
        "device_verdict": assessment.device_verdict,
        "variants_are_not_interchangeable": True,
    }


@dataclass(frozen=True)
class SourceEstimate:
    """One device's flight-time estimand, labelled with its own definition and resolution."""

    source: str
    definition: str
    flight_time_s: float
    flight_time_low_s: float
    flight_time_high_s: float
    height_flight_time_m: float
    height_low_m: float
    height_high_m: float
    # Eligibility and QC context travels with the number (RES-671); the comparison gate
    # refuses rejected or ineligible estimates even if diagnostic_only is cleared.
    qc_verdict: str = "unreported"
    protocol: str | None = None
    protocol_eligible: bool | None = None
    diagnostic_only: bool = False
    basis: str = ""
    uncertainty_assumptions: str = "resolution interval only; not a full uncertainty budget"


def _flight_estimate(
    source: str,
    definition: str,
    value: float,
    low: float,
    high: float,
    *,
    qc_verdict: str = "unreported",
    protocol: str | None = None,
    protocol_eligible: bool | None = None,
    diagnostic_only: bool = False,
    basis: str = "",
) -> SourceEstimate:
    if not all(math.isfinite(v) for v in (value, low, high)) or not 0 < low <= value <= high:
        raise ValueError("flight time and its bounds must be finite, positive and ordered")
    return SourceEstimate(
        source,
        definition,
        value,
        low,
        high,
        _height_flight(value),
        _height_flight(low),
        _height_flight(high),
        qc_verdict=qc_verdict,
        protocol=protocol,
        protocol_eligible=protocol_eligible,
        diagnostic_only=diagnostic_only,
        basis=basis,
    )


def _refuse_rejected(verdict: str, what: str, allow_diagnostic: bool) -> bool:
    """True when the source is rejected and the caller explicitly accepted diagnostics."""
    if verdict != "reject":
        return False
    if not allow_diagnostic:
        raise ValueError(
            f"{what} was rejected by QC: no accepted estimate (diagnostic opt-in only)"
        )
    return True


def contact_system_estimates(
    result: object, *, allow_diagnostic: bool = False
) -> list[SourceEstimate]:
    """Flight-time heights from a contact-system result (switch-state flight time)."""
    flights = getattr(result, "flights", None)
    if flights is None:
        raise ValueError("expected a contact-system result")
    definition = getattr(result, "estimand", "contact_system_switch_flight_time")
    verdict = str(getattr(result, "verdict", "unreported"))
    diagnostic = _refuse_rejected(verdict, "contact-system result", allow_diagnostic)
    if any(getattr(flight, "flight_time_height_m", None) is None for flight in flights):
        raise ValueError("contact-system height requires an eligible ballistic flight estimate")
    return [
        _flight_estimate(
            f"contact_system:{getattr(result, 'system', 'unknown')}",
            definition,
            f.flight_s.value,
            f.flight_s.low,
            f.flight_s.high,
            qc_verdict=verdict,
            diagnostic_only=diagnostic,
            basis=f"trigger: {getattr(result, 'trigger_definition', 'unreported')}",
        )
        for f in flights
    ]


def imu_estimates(result: object, *, allow_diagnostic: bool = False) -> list[SourceEstimate]:
    """Flight-time heights from IMU specific-force-magnitude flights."""
    flights = getattr(result, "flights", None)
    if flights is None:
        raise ValueError("expected an IMU result")
    verdict = str(getattr(result, "verdict", "unreported"))
    diagnostic = _refuse_rejected(verdict, "IMU result", allow_diagnostic)
    return [
        _flight_estimate(
            "imu:specific_force_magnitude",
            "specific-force magnitude below threshold for the minimum duration",
            f.duration_s,
            max(f.duration_s - f.resolution_s, 1e-9),
            f.duration_s + f.resolution_s,
            qc_verdict=verdict,
            diagnostic_only=diagnostic,
            basis="IMU specific-force magnitude; mounting site and threshold dependent",
        )
        for f in flights
    ]


def force_platform_estimate(
    assessment: Assessment, *, allow_diagnostic: bool = False
) -> SourceEstimate:
    """Flight-time estimate that keeps the assessment's QC verdict and protocol eligibility.

    An ineligible or device-rejected assessment is refused unless the caller opts into a
    diagnostic estimate, which is then flagged ``diagnostic_only`` and cannot be compared.
    """
    diagnostic = assessment.device_verdict == "reject" or not assessment.eligible
    if diagnostic and not allow_diagnostic:
        raise ValueError(
            f"{assessment.protocol} assessment is ineligible or device-rejected: "
            "no accepted flight-time estimate (diagnostic opt-in only)"
        )
    held = assessment.diagnostics.get("estimands", {})
    estimands: dict[str, float] = assessment.estimands or (held if isinstance(held, dict) else {})
    value = estimands.get("flight_time_s")
    if value is None:
        raise ValueError("force-platform assessment has no flight time")
    resolution = float(assessment.uncertainty["flight_time_resolution_s"])  # type: ignore[arg-type]
    return _flight_estimate(
        "force_platform:contact_threshold",
        "force below the declared contact threshold",
        value,
        max(value - resolution, 1e-9),
        value + resolution,
        qc_verdict=assessment.device_verdict,
        protocol=assessment.protocol,
        protocol_eligible=assessment.eligible,
        diagnostic_only=diagnostic,
        basis="force-platform vertical force below the declared contact threshold",
    )


def compare_sources(
    estimates: list[SourceEstimate], *, allow_diagnostic: bool = False
) -> dict[str, object]:
    """Pairwise numerical differences; interval overlap is NOT interchangeability.

    Diagnostic-only estimates are refused unless explicitly allowed, and then mark every pair
    they enter. Estimates from different declared protocols are never compared.
    """
    if not allow_diagnostic and any(
        e.diagnostic_only or e.qc_verdict == "reject" or e.protocol_eligible is False
        for e in estimates
    ):
        raise ValueError(
            "diagnostic-only, QC-rejected or protocol-ineligible estimates cannot enter an "
            "accepted comparison"
        )
    declared = {e.protocol for e in estimates if e.protocol is not None}
    if len(declared) > 1:
        raise ValueError(f"estimates come from different protocols: {sorted(declared)}")
    pairs = []
    for i, a in enumerate(estimates):
        for b in estimates[i + 1 :]:
            overlap = (
                a.flight_time_low_s <= b.flight_time_high_s
                and b.flight_time_low_s <= a.flight_time_high_s
            )
            pairs.append(
                {
                    "a": a.source,
                    "b": b.source,
                    "flight_time_difference_s": b.flight_time_s - a.flight_time_s,
                    "height_difference_m": b.height_flight_time_m - a.height_flight_time_m,
                    "resolution_intervals_overlap": overlap,
                    "diagnostic_only": a.diagnostic_only or b.diagnostic_only,
                    "qc_verdicts": {a.source: a.qc_verdict, b.source: b.qc_verdict},
                    "empirical_agreement": "not established (RES-703)",
                    "scope": "single-trial analytical/model comparison; not an agreement study",
                }
            )
    return {
        "estimates": [asdict(e) for e in estimates],
        "pairs": pairs,
        "merged_value": None,
    }


@dataclass(frozen=True)
class LptHeights:
    """LPT attachment-motion quantities; each is its own named estimand."""

    estimand: str
    height_peak_displacement_above_start_m: float
    height_peak_velocity_m: float
    peak_velocity_m_s: float
    definitions: dict[str, str]
    refusal: str | None = None
    qc_verdict: str = "unreported"
    diagnostic_only: bool = False


def lpt_estimates(report: dict[str, object], *, allow_diagnostic: bool = False) -> LptHeights:
    """Attachment-motion heights from an LPT/encoder engine report (RES-695).

    Refused for cable-length-only reports, because cable length is not vertical displacement.
    """
    estimand = report.get("estimand")
    motion = report.get("motion")
    if not isinstance(estimand, str) or not isinstance(motion, dict):
        raise ValueError("expected a processed LPT report with motion series")
    if estimand == "cable_displacement":
        raise ValueError("cable displacement is not vertical displacement: jump heights refused")
    verdict = str(report.get("verdict", "unreported"))
    diagnostic = _refuse_rejected(verdict, "LPT report", allow_diagnostic)
    displacement = np.asarray(motion["displacement_m"], dtype=np.float64)
    velocity = np.asarray(motion["velocity_m_s"], dtype=np.float64)
    peak_v = float(velocity.max())
    return LptHeights(
        estimand=estimand,
        height_peak_displacement_above_start_m=float(displacement.max() - displacement[0]),
        height_peak_velocity_m=_height_velocity(peak_v),
        peak_velocity_m_s=peak_v,
        qc_verdict=verdict,
        diagnostic_only=diagnostic,
        definitions={
            "height_peak_displacement_above_start_m": "peak attachment displacement minus the "
            "first sample (includes any rise during the push-off)",
            "height_peak_velocity_m": "v^2/2g of the peak upward attachment velocity "
            "(an upper bound of take-off velocity while the attachment still accelerates)",
            "comparability": "not comparable with force-platform heights without RES-703 evidence",
        },
    )
