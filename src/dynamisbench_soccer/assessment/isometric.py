"""Isometric and dynamometry assessment protocols (RES-672).

Protocol logic on top of the load-cell (RES-696) and Nordic/hip (RES-697) engines: declared
test eligibility, onset variants, peak force, RFD windows, impulse, torque, repetition
policy, trial consistency and calibration sensitivity. Force is force at the device;
protocol quantities never become muscle force. IMTP on a load-cell rig and IMTP on a force
plate are different measurement systems (RES-703). No analytical generator is imported.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from dynamisbench_soccer.engines.load_cell import ONSET_METHODS, MethodSpec, Processed, process
from dynamisbench_soccer.engines.load_cell_acquisition import CellSpec
from dynamisbench_soccer.engines.nordic_hip import Processed as NordicProcessed
from dynamisbench_soccer.signals.schema import Geometry, SignalBundle

ENGINE = "isometric-assessment.v1"
Protocol = Literal["imtp", "isometric_hamstring", "hip_adduction", "hip_abduction"]
Policy = Literal["best_peak", "mean_peak", "first_valid", "all_valid"]

LIMITATIONS: tuple[str, ...] = (
    "Force is force at the device. It is not muscle force, and torque is device force times a "
    "declared lever arm with the engine's stated assumptions.",
    "IMTP, hamstring and hip tests on a load-cell rig are different measurement systems from "
    "the same test on a force plate; this layer states no interchangeability (RES-703).",
    "RFD and impulse windows depend on the onset definition; every onset variant is reported "
    "and the headline uses only the declared one. Rig compliance delays and attenuates early "
    "RFD and is not corrected.",
    "Joint-angle eligibility uses declared geometry only; the angle actually achieved by the "
    "athlete is not measured by this layer.",
    "Trial-consistency limits are declared protocol thresholds, not validated reliability "
    "(use the RES-674 statistics on repeated sessions).",
    "Nordic eccentric curves reported here are strap force at the device; break-point "
    "detection needs a measured angle channel upstream.",
    "Engineering calculations only: no scientific qualification of any isometric metric.",
)

# Pooled trials must share one protocol geometry (module constant: frozen engine-v0.1 spec).
ANGLE_TOLERANCE_DEG = 1.0

_ANGLE_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "imtp": ("knee_angle_deg",),
    "isometric_hamstring": ("knee_angle_deg",),
    "hip_adduction": ("hip_flexion_deg",),
    "hip_abduction": ("hip_flexion_deg",),
}


class IsoSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    version: Literal["isometric-assessment.v1"] = "isometric-assessment.v1"
    protocol: Protocol
    policy: Policy = "best_peak"
    min_trials: int = Field(default=2, ge=1)
    consistency_limit_fraction: float = Field(default=0.05, gt=0)
    angle_range_deg: tuple[float, float] | None = None
    max_baseline_n: float | None = Field(default=None, ge=0)
    require_peak_before_end_s: float = Field(default=0.05, ge=0)
    body_mass_kg: float | None = Field(default=None, gt=0)
    headline_window_ms: int = 200
    calibration_gain_errors: tuple[float, ...] = (-0.01, 0.01)


@dataclass(frozen=True)
class IsoResult:
    protocol: str
    eligible: bool
    eligibility: list[dict[str, object]]
    trials: list[dict[str, Any]]
    summary: dict[str, float | None]
    consistency: dict[str, object]
    onset_variants: dict[str, dict[str, Any]]
    uncertainty: dict[str, object]
    refusals: dict[str, str]
    limitations: tuple[str, ...]
    method_spec: dict[str, object]

    def report(self) -> dict[str, object]:
        return {
            "engine": ENGINE,
            "protocol": self.protocol,
            "eligible": self.eligible,
            "diagnostic_only": not self.eligible,
            "eligibility": self.eligibility,
            "trials": self.trials,
            "summary": self.summary,
            "consistency": self.consistency,
            "onset_variants": self.onset_variants,
            "uncertainty": self.uncertainty,
            "refusals": self.refusals,
            "limitations": list(self.limitations),
            "method_spec": self.method_spec,
            "qualification": "engineering calculations only; no scientific qualification",
        }

    def report_json(self) -> str:
        return json.dumps(self.report(), indent=2, sort_keys=True, allow_nan=False) + "\n"


def _check(name: str, ok: bool, detail: str) -> dict[str, object]:
    return {"check": name, "passed": ok, "detail": detail}


def _geometry_checks(geometry: Geometry | None, spec: IsoSpec) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for field_name in _ANGLE_REQUIREMENTS[spec.protocol]:
        value = None if geometry is None else getattr(geometry, field_name)
        known = value is not None and (geometry is None or field_name not in geometry.unknown)
        out.append(_check(f"{field_name}_declared", known, f"{field_name}={value}"))
        if known and spec.angle_range_deg is not None:
            low, high = spec.angle_range_deg
            out.append(
                _check(
                    f"{field_name}_in_protocol_range",
                    low <= float(value or 0) <= high,
                    f"{value} deg vs declared range {low}-{high} deg",
                )
            )
    return out


def _trial_geometry_exclusions(
    geometry: Geometry | None, spec: IsoSpec, lever_arm_m: float | None
) -> list[str]:
    """Reasons a single trial's own declared geometry makes it ineligible (empty = eligible)."""
    reasons = [
        f"{c['check']}: {c['detail']}" for c in _geometry_checks(geometry, spec) if not c["passed"]
    ]
    if (
        geometry is not None
        and lever_arm_m is not None
        and geometry.lever_arm_m is not None
        and "lever_arm_m" not in geometry.unknown
        and abs(geometry.lever_arm_m - lever_arm_m) > 1e-9 * max(1.0, lever_arm_m)
    ):
        reasons.append(
            f"lever_arm_conflict: trial geometry {geometry.lever_arm_m} m vs cell {lever_arm_m} m"
        )
    return reasons


def _geometry_consistency(
    geometries: list[Geometry | None], indices: list[int], spec: IsoSpec
) -> list[dict[str, object]]:
    out: list[dict[str, object]] = []
    for field_name in _ANGLE_REQUIREMENTS[spec.protocol]:
        values = [
            float(getattr(geometries[i], field_name))
            for i in indices
            if geometries[i] is not None and getattr(geometries[i], field_name) is not None
        ]
        if len(values) >= 2:
            spread = max(values) - min(values)
            out.append(
                _check(
                    f"{field_name}_consistent_across_pooled_trials",
                    spread <= ANGLE_TOLERANCE_DEG,
                    f"spread {spread:.3f} deg vs tolerance {ANGLE_TOLERANCE_DEG} deg",
                )
            )
    return out


def run_trials(bundles: list[SignalBundle], cell: CellSpec, method: MethodSpec) -> list[Processed]:
    return [process(b, cell, method) for b in bundles]


def _trial(result: Processed, spec: IsoSpec, index: int) -> dict[str, Any]:
    m = result.metrics
    window = spec.headline_window_ms
    passed_end = (
        result.time_s[-1] - (result.onset.time_s or 0) > 0 and m.get("peak_time_s") is not None
    )
    peak_time = m.get("peak_time_s")
    near_end = (
        peak_time is not None
        and float(result.time_s[-1]) - peak_time < spec.require_peak_before_end_s
    )
    return {
        "index": index,
        "verdict": result.verdict,
        "onset_s": result.onset.time_s,
        "onset_method": result.onset.method,
        "peak_force_n": m.get("peak_force_n"),
        "net_peak_force_n": m.get("net_peak_force_n"),
        "time_to_peak_s": m.get("time_to_peak_s"),
        "rfd_headline_n_s": m.get(f"rfd_0_{window}ms_n_s"),
        "impulse_headline_n_s": m.get(f"net_impulse_0_{window}ms_n_s"),
        "rfd_windows_n_s": {k: v for k, v in m.items() if k.startswith("rfd_")},
        "net_impulse_windows_n_s": {k: v for k, v in m.items() if k.startswith("net_impulse_")},
        "peak_torque_n_m": m.get("peak_torque_n_m"),
        "valid": result.verdict != "reject"
        and result.onset.index is not None
        and not near_end
        and passed_end,
        "peak_at_record_end": bool(near_end),
        "baseline_n": m.get("baseline_n"),
    }


def assess(
    results: list[Processed],
    spec: IsoSpec,
    *,
    geometry: Geometry | None = None,
    geometries: list[Geometry | None] | None = None,
    bundles: list[SignalBundle] | None = None,
    cell: CellSpec | None = None,
    method: MethodSpec | None = None,
) -> IsoResult:
    if not results:
        raise ValueError("at least one processed trial is required")
    if bundles is not None and len(bundles) != len(results):
        raise ValueError("bundles must correspond one-to-one with processed trials")
    if geometries is None:
        # Each trial is bound to its own declared geometry. The single ``geometry`` argument
        # applies only to a trial that has no bundle of its own.
        geometries = (
            [b.meta.geometry for b in bundles] if bundles is not None else [geometry] * len(results)
        )
    if len(geometries) != len(results):
        raise ValueError("geometries must correspond one-to-one with processed trials")
    checks: list[dict[str, object]] = []
    trials = [_trial(r, spec, i) for i, r in enumerate(results)]
    refusals: dict[str, str] = {}
    lever = None if cell is None else cell.lever_arm_m
    for t, g in zip(trials, geometries, strict=True):
        reasons = _trial_geometry_exclusions(g, spec, lever)
        t["geometry"] = {
            f: (None if g is None else getattr(g, f)) for f in _ANGLE_REQUIREMENTS[spec.protocol]
        }
        t["geometry_exclusions"] = reasons
        for c in _geometry_checks(g, spec):
            checks.append({**c, "trial": t["index"], "scope": "trial"})
        t["diagnostic_only"] = bool(reasons)
        if reasons:
            t["valid"] = False
    for t in trials:
        if spec.max_baseline_n is not None and t["baseline_n"] is not None:
            checks.append(
                _check(
                    f"baseline_within_limit_trial_{t['index']}",
                    abs(float(t["baseline_n"])) <= spec.max_baseline_n,
                    f"baseline {t['baseline_n']} N vs limit {spec.max_baseline_n} N",
                )
            )
            if abs(float(t["baseline_n"])) > spec.max_baseline_n:
                t["valid"] = False
    chosen: list[dict[str, Any]] = []
    valid = [t for t in trials if t["valid"]]
    checks.extend(_geometry_consistency(geometries, [int(t["index"]) for t in valid], spec))
    checks.append(
        _check(
            "enough_valid_trials",
            len(valid) >= spec.min_trials,
            f"{len(valid)} valid of {len(trials)}",
        )
    )
    summary: dict[str, float | None] = {"valid_trials": float(len(valid))}
    consistency: dict[str, object] = {"evaluated": False}
    if valid:
        peaks = np.array([float(t["peak_force_n"]) for t in valid])
        best = int(np.argmax(peaks))
        if spec.policy == "best_peak":
            chosen = [valid[best]]
        elif spec.policy == "first_valid":
            chosen = [valid[0]]
        else:
            chosen = valid
        for key in (
            "peak_force_n",
            "time_to_peak_s",
            "rfd_headline_n_s",
            "impulse_headline_n_s",
            "peak_torque_n_m",
        ):
            values = [t[key] for t in chosen if t[key] is not None]
            summary[key] = float(np.mean(values)) if values else None
        if spec.policy == "all_valid":
            summary["peak_force_n"] = float(np.mean(peaks))
        if len(valid) >= 2:
            spread = float((peaks.max() - peaks.min()) / peaks.max())
            consistency = {
                "evaluated": True,
                "peak_spread_fraction_of_best": spread,
                "limit": spec.consistency_limit_fraction,
                "consistent": spread <= spec.consistency_limit_fraction,
            }
            checks.append(
                _check(
                    "trial_consistency",
                    spread <= spec.consistency_limit_fraction,
                    f"spread {spread:.4f} vs limit {spec.consistency_limit_fraction}",
                )
            )
        peak_summary = summary.get("peak_force_n")
        if spec.body_mass_kg is not None and peak_summary is not None:
            summary["peak_force_n_per_kg"] = peak_summary / spec.body_mass_kg
        else:
            refusals["relative_force"] = "no declared body mass: relative force refused"
    else:
        refusals["summary"] = "no valid trial: protocol quantities refused"
    # Onset-variant sensitivity is computed on the trial(s) the headline actually uses.
    onset_variants: dict[str, dict[str, Any]] = {}
    if valid and bundles is not None and cell is not None and method is not None:
        source = [int(t["index"]) for t in chosen]
        window = spec.headline_window_ms
        for name in ONSET_METHODS:
            try:
                alts = [
                    process(bundles[i], cell, method.model_copy(update={"onset": name}))
                    for i in source
                ]
            except ValueError as exc:
                onset_variants[name] = {"error_refused": None, "source_trial_indices": source}
                refusals[f"onset_{name}"] = str(exc)
                continue

            def mean(values: list[float | None]) -> float | None:
                real = [v for v in values if v is not None]
                return float(np.mean(real)) if len(real) == len(values) and real else None

            onset_variants[name] = {
                "source_trial_indices": source,
                "onset_s": mean([a.onset.time_s for a in alts]),
                "rfd_headline_n_s": mean([a.metrics.get(f"rfd_0_{window}ms_n_s") for a in alts]),
                "impulse_headline_n_s": mean(
                    [a.metrics.get(f"net_impulse_0_{window}ms_n_s") for a in alts]
                ),
            }
    uncertainty: dict[str, object] = {
        "calibration_gain_sensitivity": {
            f"{e:+.3f}": {
                k: (None if (v := summary.get(k)) is None else v * (1 + e))
                for k in ("peak_force_n", "rfd_headline_n_s", "impulse_headline_n_s")
            }
            for e in spec.calibration_gain_errors
        },
        "rig_compliance": "not modelled; declared rig compliance is recorded only",
        "scope": "conditional sensitivity, not a complete uncertainty budget",
    }
    return IsoResult(
        protocol=spec.protocol,
        eligible=all(bool(c["passed"]) for c in checks if c.get("scope") != "trial"),
        eligibility=checks,
        trials=trials,
        summary=summary,
        consistency=consistency,
        onset_variants=onset_variants,
        uncertainty=uncertainty,
        refusals=refusals,
        limitations=LIMITATIONS,
        method_spec=spec.model_dump(mode="json"),
    )


class NordicSpec(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    policy: Literal["best_peak", "mean_peak"] = "best_peak"
    require_break_point: bool = False
    min_trials: int = Field(default=1, ge=1)
    consistency_limit_fraction: float = Field(default=0.10, gt=0)


def assess_nordic(results: list[NordicProcessed], spec: NordicSpec) -> dict[str, object]:
    """Eccentric Nordic protocol: bilateral peak, torque, asymmetry, optional break angle."""
    if not results:
        raise ValueError("at least one Nordic trial is required")
    bases = {r.force_basis for r in results}
    if len(bases) != 1:
        raise ValueError("raw-calibrated and vendor-corrected trials cannot be combined")
    checks: list[dict[str, object]] = []
    rows: list[dict[str, Any]] = []
    for i, r in enumerate(results):
        sides = r.sides
        bilateral = r.bilateral
        ok = set(sides) == {"left", "right"} and bool(bilateral.get("available"))
        break_ok = all(s.metrics.get("break_angle_deg") is not None for s in sides.values())
        rows.append(
            {
                "trial": i,
                "verdict": r.verdict,
                "peak_force_n": {k: v.metrics.get("peak_force_n") for k, v in sides.items()},
                "peak_torque_n_m": {k: v.metrics.get("peak_torque_n_m") for k, v in sides.items()},
                "break_angle_deg": {k: v.metrics.get("break_angle_deg") for k, v in sides.items()},
                "asymmetry_percent": bilateral.get("asymmetry_value_percent"),
                "valid": ok
                and r.verdict != "reject"
                and (break_ok or not spec.require_break_point),
            }
        )
    valid = [r for r in rows if r["valid"]]
    checks.append(
        _check("enough_valid_trials", len(valid) >= spec.min_trials, f"{len(valid)} valid")
    )
    if spec.require_break_point:
        checks.append(_check("break_point_available", len(valid) > 0, "measured angle needed"))
    summary: dict[str, object] = {"valid_trials": len(valid), "force_basis": next(iter(bases))}
    if valid:
        totals = [sum(float(v) for v in t["peak_force_n"].values()) for t in valid]
        best = int(np.argmax(totals))
        chosen = [valid[best]] if spec.policy == "best_peak" else valid
        for side in ("left", "right"):
            summary[f"peak_force_n_{side}"] = float(
                np.mean([float(t["peak_force_n"][side]) for t in chosen])
            )
        if len(valid) >= 2:
            spread = float((max(totals) - min(totals)) / max(totals))
            checks.append(
                _check(
                    "trial_consistency",
                    spread <= spec.consistency_limit_fraction,
                    f"spread {spread:.4f}",
                )
            )
    return {
        "engine": ENGINE,
        "protocol": "nordic_eccentric",
        "eligible": all(bool(c["passed"]) for c in checks),
        "eligibility": checks,
        "trials": rows,
        "summary": summary,
        "limitations": list(LIMITATIONS),
        "qualification": "engineering calculations only; no scientific qualification",
    }
