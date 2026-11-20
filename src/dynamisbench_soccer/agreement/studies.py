"""Analytical cross-device studies for every declared non-equivalence (RES-703).

Each study drives real device engines (or an explicit closed form) with RES-669 analytical
scenarios and reports the numerical difference with paired-agreement statistics. All
studies are *analytical*: idealised observation models of the same truth. They quantify how
definitions, resolution, latency, geometry and differentiation separate the estimands;
none supports empirical interchangeability, and every result says so.
"""

from __future__ import annotations

import itertools
import json
import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from dynamisbench_soccer.agreement.stats import (
    Classification,
    classify,
    paired_agreement,
)
from dynamisbench_soccer.assessment import jumps
from dynamisbench_soccer.engines import imu as imu_engine
from dynamisbench_soccer.engines import load_cell, radar_laser
from dynamisbench_soccer.engines.contact import ContactPolicy, DurationRange, contact
from dynamisbench_soccer.engines.event_time import ClockTiming, EdgeTiming, Timing
from dynamisbench_soccer.engines.force_platform import process as force_process
from dynamisbench_soccer.engines.force_platform_acquisition import PlateSpec
from dynamisbench_soccer.engines.load_cell_acquisition import CellSpec
from dynamisbench_soccer.engines.radar_laser_acquisition import ChannelStage
from dynamisbench_soccer.reference.clocks import DeviceClock
from dynamisbench_soccer.reference.exporters import export_case
from dynamisbench_soccer.reference.kinematics import G, Phase, Trajectory, _flight_and_land
from dynamisbench_soccer.reference.motion_models import IsometricRamp, MonoExponentialSprint
from dynamisbench_soccer.signals.schema import SignalBundle

ENGINE = "cross-device-agreement.v1"
EVIDENCE_CLASSES = (
    "ENGINE_TO_ENGINE",
    "ENGINE_TO_INDEPENDENT_CLOSED_FORM",
    "CLOSED_FORM_DEMONSTRATION",
    "DIAGNOSTIC_ONLY",
    "UNSUPPORTED",
    "EMPIRICAL_MEASURED",  # reserved: only qualified measured paired data may earn it
)
TAKEOFF_SPEEDS = (2.0, 2.3, 2.6, 2.9, 3.2)
NOT_EMPIRICAL = (
    "analytical study only: idealised observation models of one truth; no measured paired "
    "data exist, so empirical agreement and interchangeability are not established"
)


@dataclass(frozen=True)
class Study:
    id: str
    left: str
    right: str
    quantity: str
    classification: Classification
    interchangeable: str
    reason: str
    definitions: dict[str, str]
    pairs: list[tuple[float, float]]
    statistics: dict[str, Any] | None
    assumptions: list[str]
    sensitivities: dict[str, Any] = field(default_factory=dict)
    execution_basis: tuple[str, str] = ("unavailable", "unavailable")
    empirical_status: str = "missing_evidence: " + NOT_EMPIRICAL
    evidence_class: str = "UNSUPPORTED"
    source_evidence: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _jump(v: float) -> Trajectory:
    return Trajectory(
        75.0,
        0.0,
        (Phase("quiet", 1.0, 0.0, True), Phase("propel", 0.3, v / 0.3, True), *_flight_and_land(v)),
        assessment_id="sj",
    )


def _contact_policy(latency: float = 0.0) -> ContactPolicy:
    clock = ClockTiming(resolution_s=0.001, sync_latency_s=0, sync_latency_error_s=0)
    edge = EdgeTiming(latency_s=latency, latency_error_s=0.002)
    return ContactPolicy.model_validate(
        {
            "timing": Timing(clock_model="wired", clocks={"device": clock}, authority="RES-669"),
            "system": "contact_mat",
            "initial_event": "contact_snapshot",
            "switch_on": edge,
            "switch_off": edge,
            "debounce_s": 0.0,
            "expected_flights": 1,
            "flight": DurationRange(min_s=0.15, max_s=0.9),
            "ground_contact": DurationRange(min_s=0.04, max_s=0.5),
            "ballistic_equal_com_height": True,
        }
    )


def _force_flights(*, accepted: bool = True) -> tuple[list[float], list[jumps.Assessment]]:
    """Force-platform squat-jump assessments of the analytical jumps (no countermovement).

    The exported force has a flat maximum, which the clipping rule (QC-S03) rejects. A
    sub-newton ripple on loaded samples, far below the contact threshold, gives device-accepted
    trials. ``accepted=False`` keeps the rejected fixture as the diagnostic negative control.
    """
    truth, out = [], []
    for v in TAKEOFF_SPEEDS:
        motion = _jump(v)
        truth.append(motion.flight_time_s())
        observation = export_case("force_platform.single", motion, hz=1000).observation
        if accepted:
            force = observation["fz"]
            ripple = (force > 0) * np.sin(np.arange(len(force))) * 0.1
            observation = SignalBundle(observation.meta, {**observation.data, "fz": force + ripple})
        device = force_process(observation, plates=(PlateSpec(plate=1, assume_field_axes=True),))
        out.append(
            jumps.assess(jumps.JumpSignal.from_engine(device), jumps.JumpSpec(protocol="sj"))
        )
    return truth, out


def _force_evidence(index: int, a: jumps.Assessment) -> dict[str, Any]:
    return {
        "source": "force_platform.single",
        "trial": index,
        "protocol": a.protocol,
        "protocol_eligible": a.eligible,
        "qc_verdict": a.device_verdict,
        "diagnostic_only": a.device_verdict == "reject" or not a.eligible,
        "estimand": "flight_time (force below contact threshold)",
    }


def _imu_evidence(index: int, result: imu_engine.Processed) -> dict[str, Any]:
    return {
        "source": "imu.inertial",
        "trial": index,
        "protocol": "sj",
        "protocol_eligible": None,
        "qc_verdict": result.verdict,
        "diagnostic_only": result.verdict == "reject",
        "estimand": "flight_time (specific-force magnitude below threshold)",
    }


def _contact_evidence(index: int) -> dict[str, Any]:
    return {
        "source": "timing_contact.contact",
        "trial": index,
        "protocol": "sj",
        "protocol_eligible": None,
        "qc_verdict": "unreported",
        "diagnostic_only": False,
        "estimand": "flight_time (switch-state)",
    }


def _native_evidence(
    source: str, trial: int, verdict: str, estimand: str, protocol: str
) -> dict[str, Any]:
    return {
        "source": source,
        "trial": trial,
        "protocol": protocol,
        "protocol_eligible": None,
        "qc_verdict": verdict,
        "diagnostic_only": verdict == "reject",
        "estimand": estimand,
    }


def evidence_class(
    execution_basis: tuple[str, str], source_evidence: list[dict[str, Any]], unsupported: bool
) -> str:
    """What the executed inputs can support; never EMPIRICAL_MEASURED from synthetic data."""
    if unsupported or "unavailable" in execution_basis:
        return "UNSUPPORTED"
    if any(e["diagnostic_only"] for e in source_evidence):
        return "DIAGNOSTIC_ONLY"
    if execution_basis == ("engine", "engine"):
        return "ENGINE_TO_ENGINE"
    if "engine" in execution_basis:
        return "ENGINE_TO_INDEPENDENT_CLOSED_FORM"
    return "CLOSED_FORM_DEMONSTRATION"


def _study(
    id_: str,
    left: str,
    right: str,
    quantity: str,
    *,
    definitions: dict[str, str],
    execution_basis: tuple[str, str],
    definitions_match: bool,
    pairs: list[tuple[float, float]],
    assumptions: list[str],
    tolerance: float | None = None,
    sensitivities: dict[str, Any] | None = None,
    unsupported_reason: str | None = None,
    source_evidence: list[dict[str, Any]] | None = None,
) -> Study:
    evidence = source_evidence or []
    klass = evidence_class(execution_basis, evidence, unsupported_reason is not None)
    rejected = klass == "DIAGNOSTIC_ONLY"
    verdict = classify(
        origin="analytical" if pairs and not rejected else "none",
        definitions_match=definitions_match,
        unsupported_reason=unsupported_reason,
    )
    stats = (
        paired_agreement(
            [p[0] for p in pairs], [p[1] for p in pairs], tolerance=tolerance
        ).to_dict()
        if len(pairs) >= 3 and not rejected
        else None
    )
    return Study(
        id_,
        left,
        right,
        quantity,
        verdict.classification,
        verdict.interchangeable,
        verdict.reason,
        definitions,
        pairs,
        stats,
        assumptions,
        sensitivities or {},
        execution_basis,
        evidence_class=klass,
        source_evidence=evidence,
    )


def force_vs_contact_flight_time(*, accepted_force: bool = True) -> Study:
    truth, forces = _force_flights(accepted=accepted_force)
    pairs, latency_runs = [], {}
    evidence = [_force_evidence(i, fp) for i, fp in enumerate(forces)] + [
        _contact_evidence(i) for i in range(len(forces))
    ]
    for v, fp in zip(TAKEOFF_SPEEDS, forces, strict=True):
        edge_case = export_case("timing_contact.contact", _jump(v), clock=DeviceClock(0.0, 0.0))
        c = contact(edge_case.observation, _contact_policy())
        pairs.append((fp.estimands["flight_time_s"], c.flights[0].flight_s.value))
    for latency in (0.0, 0.01, 0.02):
        case = export_case(
            "timing_contact.contact", _jump(2.6), clock=DeviceClock(0.0, 0.0), latency_s=latency
        )
        undeclared = contact(case.observation, _contact_policy()).flights[0].flight_s.value
        latency_runs[f"symmetric_latency_{latency:g}_s_undeclared"] = undeclared - truth[2]
    return _study(
        "force_vs_contact_flight_time",
        "force_platform.single",
        "timing_contact.contact",
        "flight_time",
        execution_basis=("engine", "engine"),
        definitions={
            "force_platform": "time between force dropping below and rising above the contact threshold",  # noqa: E501
            "contact_system": "time between switch-off and switch-on, with declared latency model",
        },
        definitions_match=True,
        pairs=pairs,
        tolerance=0.004,
        assumptions=["rigid ideal switch", "declared symmetric latency", "1 kHz force sampling"],
        sensitivities={"error_vs_truth_s": latency_runs, "truth_flight_time_s": truth},
        source_evidence=evidence,
    )


def force_vs_imu_flight_time(*, accepted_force: bool = True) -> Study:
    truth, forces = _force_flights(accepted=accepted_force)
    pairs = []
    evidence = [_force_evidence(i, fp) for i, fp in enumerate(forces)]
    for i, (fp, v) in enumerate(zip(forces, TAKEOFF_SPEEDS, strict=True)):
        result = imu_engine.process(
            export_case("imu.inertial", _jump(v), hz=500).observation,
            imu_engine.ImuSpec(initial_orientation=((1, 0, 0), (0, 1, 0), (0, 0, 1))),
        )
        evidence.append(_imu_evidence(i, result))
        pairs.append((fp.estimands["flight_time_s"], result.flights[0].duration_s))
    thresholds = {}
    for t in (1.0, 3.0, 6.0):
        r = imu_engine.process(
            export_case("imu.inertial", _jump(2.6), hz=500).observation,
            imu_engine.ImuSpec(initial_orientation=((1, 0, 0), (0, 1, 0), (0, 0, 1))),
            imu_engine.MethodSpec(flight_threshold_m_s2=t),
        )
        thresholds[f"threshold_{t:g}"] = r.flights[0].duration_s - truth[2]
    return _study(
        "force_vs_imu_flight_time",
        "force_platform.single",
        "imu.inertial",
        "flight_time",
        execution_basis=("engine", "engine"),
        definitions={
            "force_platform": "force below the contact threshold",
            "imu": "specific-force magnitude below a declared threshold for a minimum duration",
        },
        definitions_match=True,
        pairs=pairs,
        tolerance=0.006,
        assumptions=[
            "sensor rigidly at COM",
            "no soft-tissue artefact",
            "ideal flight specific force of zero",
        ],
        sensitivities={"error_vs_truth_by_threshold_s": thresholds},
        source_evidence=evidence,
    )


def force_vs_load_cell_isometric_force() -> Study:
    pairs = []
    evidence: list[dict[str, Any]] = []
    for peak in (1500.0, 2000.0, 2500.0, 3000.0):
        ramp = IsometricRamp(peak_n=peak)
        # The plate-side comparator is the ideal vertical force from the closed form.
        cell = load_cell.process(
            export_case("load_cell_isometric.load_cell", ramp, hz=1000).observation,
            CellSpec(),
            load_cell.MethodSpec(),
        )
        plate_equiv = ramp.peak_n  # an ideal plate reads the ramp force exactly
        measured = cell.metrics["peak_force_n"]
        assert measured is not None
        evidence.append(
            _native_evidence(
                "load_cell_isometric.load_cell",
                len(evidence),
                cell.verdict,
                "peak force at the cell",
                "isometric_ramp",
            )
        )
        pairs.append((plate_equiv, measured))
    return _study(
        "force_vs_load_cell_isometric_force",
        "force_platform.single",
        "load_cell_isometric.load_cell",
        "isometric_force",
        execution_basis=("closed_form", "engine"),
        definitions={
            "force_platform": "vertical ground reaction force of the whole system on the plate",
            "load_cell": "uniaxial force at the cell, with rig compliance and line of action unmodelled",  # noqa: E501
        },
        definitions_match=True,
        pairs=pairs,
        tolerance=0.01,
        assumptions=["rigid zero-compliance rig", "perfectly aligned axes", "ideal calibration"],
        sensitivities={"rig_compliance": "not modelled; delays and attenuates early RFD"},
        source_evidence=evidence,
    )


def radar_vs_laser_velocity() -> Study:
    sprint = MonoExponentialSprint()
    pairs = []
    evidence: list[dict[str, Any]] = []
    for hz in (47, 100, 256):
        radar = radar_laser.process(
            export_case("radar_laser.radar", sprint, hz=hz).observation,
            radar_laser.ProcessingOptions(
                channel_stage=ChannelStage.NATIVE,
                correction_method="declared_fixed_angle",
            ),
        )
        laser = radar_laser.process(
            export_case("radar_laser.laser", sprint, hz=hz).observation,
            radar_laser.ProcessingOptions(
                channel_stage=ChannelStage.NATIVE, derivative_method="five_point"
            ),
        )
        pairs.append((radar.peak_velocity_m_s, laser.peak_velocity_m_s))
        for name, result in (("radar_laser.radar", radar), ("radar_laser.laser", laser)):
            # The engine refuses unacceptable traces outright, so a returned result was admitted.
            evidence.append(
                _native_evidence(
                    name,
                    len(pairs) - 1,
                    "admitted_by_engine_refusal_gate",
                    result.estimand,
                    "sprint",
                )
            )
    sd = 0.01
    noise_sd = 0.95 * sd * 256  # five-point derivative of white noise: sqrt(130)/12 * sd / dt
    return _study(
        "radar_vs_laser_velocity",
        "radar_laser.radar",
        "radar_laser.laser",
        "doppler_vs_differentiated_velocity",
        execution_basis=("engine", "engine"),
        definitions={
            "radar": "Doppler radial speed (direct measurement of velocity)",
            "laser": "derivative of measured distance (noise amplified by 1/dt)",
        },
        definitions_match=False,
        pairs=pairs,
        assumptions=[
            "ideal noise-free observations",
            "declared fixed angle for radar",
            "aiming point on the laser axis",
        ],
        source_evidence=evidence,
        sensitivities={
            "five_point_derivative_noise_sd_m_s_at_256hz_for_0.01m_range_noise": noise_sd,
            "note": "range noise of 1 cm at 256 Hz gives ~2.4 m/s of velocity noise before filtering",  # noqa: E501
        },
    )


def radar_vs_gates_velocity() -> Study:
    sprint = MonoExponentialSprint()
    gates = (0.0, 5.0, 10.0, 20.0, 30.0, 40.0)
    pairs = []
    for a, b in itertools.pairwise(gates):
        ta, tb = sprint.gate_time_s(a), sprint.gate_time_s(b)
        segment_average = (b - a) / (tb - ta)
        midpoint_speed = sprint.velocity_m_s((ta + tb) / 2)
        pairs.append((midpoint_speed, segment_average))
    return _study(
        "radar_vs_gates_velocity",
        "radar_laser.radar",
        "timing_contact.gates",
        "sprint_velocity",
        execution_basis=("closed_form", "closed_form"),
        definitions={
            "radar": "instantaneous Doppler speed at a time",
            "gates": "segment-average velocity over a distance interval",
        },
        definitions_match=False,
        pairs=pairs,
        assumptions=["exact crossing times", "radar compared at the segment time midpoint"],
        sensitivities={
            "note": "average over distance differs from speed at the time midpoint by a curvature term"  # noqa: E501
        },
    )


def radar_vs_positional_speed(
    latency_s: float = 0.1, right: str = "gnss_lps.gnss", id_: str = "radar_vs_positional_speed"
) -> Study:
    sprint = MonoExponentialSprint()
    times = np.arange(1.0, 6.0, 0.5)
    pairs = [
        (sprint.velocity_m_s(float(t)), sprint.velocity_m_s(float(t) - latency_s)) for t in times
    ]
    return _study(
        id_,
        "radar_laser.radar",
        right,
        "running_speed",
        execution_basis=("closed_form", "closed_form"),
        definitions={
            "radar": "Doppler radial speed along the beam",
            "gnss_lps": "horizontal speed differentiated from reported positions, with processing latency",  # noqa: E501
        },
        definitions_match=False,
        pairs=pairs,
        assumptions=[
            "declared latency of 0.1 s modelled as a pure delay",
            "no smoothing",
            "no multipath",
        ],
        sensitivities={
            "latency_bias_at_t_s": {
                f"{t:g}": float(
                    sprint.velocity_m_s(float(t)) - sprint.velocity_m_s(float(t) - latency_s)
                )
                for t in (1.0, 2.0, 4.0)
            },
            "closed_form": "bias ~ acceleration x latency while accelerating",
        },
    )


def cable_vs_vertical_displacement() -> Study:
    heights = np.linspace(0.1, 0.6, 6)
    pairs = []
    for h in heights:
        for anchor_x in (0.3,):
            cable = math.hypot(anchor_x, float(h)) - anchor_x
            pairs.append((float(h), cable))
    sweep = {
        f"anchor_offset_{x:g}_m": float(math.hypot(x, 0.5) - x - 0.5) for x in (0.0, 0.1, 0.3, 0.5)
    }
    return _study(
        "cable_vs_vertical_displacement",
        "force_platform.single",
        "lpt_encoder.lpt",
        "cable_vs_vertical_displacement",
        execution_basis=("closed_form", "closed_form"),
        definitions={
            "force_platform": "vertical centre-of-mass displacement",
            "lpt": "change of cable length, equal to vertical displacement only for a vertical cable",  # noqa: E501
        },
        definitions_match=False,
        pairs=pairs,
        assumptions=["anchor at the same height as the start point", "rigid attachment point"],
        sensitivities={"error_at_h_0.5_m_by_anchor_offset": sweep},
    )


def attachment_vs_com_power() -> Study:
    mass_body, bar = 75.0, 20.0
    pairs = []
    for v in (1.0, 2.0, 3.0):
        force_plate_power = mass_body * (G + 2.0) * v
        lpt_power = (mass_body + bar) * (G + 2.0) * v  # system mass assumption without bar
        pairs.append((force_plate_power, lpt_power))
    return _study(
        "attachment_vs_com_power",
        "force_platform.single",
        "lpt_encoder.lpt",
        "attachment_vs_com_power",
        execution_basis=("closed_form", "closed_form"),
        definitions={
            "force_platform": "power of ground reaction force on the whole-body COM velocity",
            "lpt": "vertical mass-model power of the attachment point with declared system mass",
        },
        definitions_match=False,
        pairs=pairs,
        assumptions=["rigid vertical translation", "declared 20 kg bar added to 75 kg body"],
        sensitivities={
            "note": "attachment velocity differs from COM velocity under any joint motion"
        },
    )


def nordic_raw_vs_vendor_corrected() -> Study:
    return _study(
        "nordic_raw_vs_vendor_corrected",
        "nordic_hip.nordic",
        "nordic_hip.nordic",
        "force_basis",
        execution_basis=("unavailable", "unavailable"),
        definitions={
            "raw": "calibrated force at the strap",
            "vendor_corrected": "vendor-defined correction (undisclosed algorithm)",
        },
        definitions_match=False,
        pairs=[],
        assumptions=[],
        unsupported_reason=(
            "no vendor-corrected data or algorithm is available; the engine refuses to pool or "
            "convert the two bases"
        ),
    )


def gates_vs_radar_laser_timing() -> Study:
    sprint = MonoExponentialSprint()
    distances = (10.0, 20.0, 30.0)
    radar_hz = 100
    radar = radar_laser.process(
        export_case("radar_laser.radar", sprint, hz=radar_hz).observation,
        radar_laser.ProcessingOptions(
            channel_stage=ChannelStage.NATIVE, correction_method="declared_fixed_angle"
        ),
    )
    t = np.asarray(radar.time_from_start_s)
    v = np.asarray(radar.velocity_m_s)
    x = np.concatenate(([0.0], np.cumsum(0.5 * (v[1:] + v[:-1]) * np.diff(t))))
    pairs = [(float(np.interp(d, x, t)), sprint.gate_time_s(d)) for d in distances]
    return _study(
        "gates_vs_radar_laser_timing",
        "radar_laser.radar",
        "timing_contact.gates",
        "sprint_split_time",
        execution_basis=("engine", "closed_form"),
        definitions={
            "gates": "time of a beam interruption at a fixed position",
            "radar": "time at which integrated Doppler speed reaches the distance",
        },
        definitions_match=True,
        pairs=pairs,
        tolerance=0.01,
        assumptions=[
            "radar start equals gate start",
            "integration of a synthetic radar speed trace",
            "no start-definition bias",
        ],
        sensitivities={"start_offset_effect": "a start offset adds directly to every split"},
    )


def radar_vs_lps_speed() -> Study:
    return radar_vs_positional_speed(0.05, "gnss_lps.lps", "radar_vs_lps_speed")


def force_vs_lpt_jump_height() -> Study:
    pairs = []
    start_offsets = {}
    for v in TAKEOFF_SPEEDS:
        motion = _jump(v)
        takeoff = motion.takeoff_s()
        takeoff_height = motion.state(takeoff).displacement_m
        peak = takeoff_height + v * v / (2 * G)
        impulse_momentum = v * v / (2 * G)
        lpt_peak_minus_start = peak - motion.state(0.0).displacement_m
        pairs.append((impulse_momentum, lpt_peak_minus_start))
        start_offsets[f"takeoff_speed_{v:g}"] = takeoff_height
    return _study(
        "force_vs_lpt_jump_height",
        "force_platform.single",
        "lpt_encoder.lpt",
        "jump_height",
        execution_basis=("closed_form", "closed_form"),
        definitions={
            "force_platform": "centre-of-mass rise after take-off, v^2 / 2g from the impulse",
            "lpt": "peak attachment displacement above the declared standing start",
        },
        definitions_match=False,
        pairs=pairs,
        assumptions=["vertical cable", "attachment moves with the COM", "standing start declared"],
        sensitivities={"take_off_height_above_start_m": start_offsets},
    )


STUDIES = (
    force_vs_contact_flight_time,
    force_vs_imu_flight_time,
    force_vs_load_cell_isometric_force,
    radar_vs_laser_velocity,
    radar_vs_positional_speed,
    radar_vs_lps_speed,
    radar_vs_gates_velocity,
    force_vs_lpt_jump_height,
    cable_vs_vertical_displacement,
    attachment_vs_com_power,
    nordic_raw_vs_vendor_corrected,
    gates_vs_radar_laser_timing,
)


def run_all() -> dict[str, Any]:
    studies = [s().to_dict() for s in STUDIES]
    counts: dict[str, int] = {}
    for s in studies:
        counts[s["classification"]] = counts.get(s["classification"], 0) + 1
    evidence_counts: dict[str, int] = {}
    for s in studies:
        evidence_counts[s["evidence_class"]] = evidence_counts.get(s["evidence_class"], 0) + 1
    return {
        "engine": ENGINE,
        "studies": studies,
        "classification_counts": counts,
        "evidence_class_counts": evidence_counts,
        "interchangeable_claims": 0,
        "empirical_device_agreement_studies": 0,
        "statement": NOT_EMPIRICAL,
    }


def report_json() -> str:
    return json.dumps(run_all(), indent=2, sort_keys=True, allow_nan=False) + "\n"
