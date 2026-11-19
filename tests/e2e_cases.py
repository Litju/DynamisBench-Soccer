"""End-to-end verification matrix for all eight device families (RES-675).

Each case drives a real device engine (and, where one exists, the assessment layer) from a
RES-669 native observation, compares with an independent closed form, and records the axes it
exercises: native rate, filter, event/onset method, differentiation/integration, calibration or
geometry condition, corruption class, protocol eligibility and refusal boundary. Tolerances are
derived from sample interval, resolution or the stated discretisation bound of the method, never
tuned to an observed result. Used by ``test_end_to_end_qualification.py`` and
``scripts/build_verification.py``.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from test_gnss_lps_acquisition import track_case
from test_lpt_acquisition import LINE, LPT, MOTION, _modify
from test_timing_gates import DISTANCES, gate_policy

from dynamisbench_soccer.assessment import cod, isometric, jumps, sprint
from dynamisbench_soccer.engines import imu as imu_engine
from dynamisbench_soccer.engines import load_cell, radar_laser
from dynamisbench_soccer.engines import nordic_hip as nordic
from dynamisbench_soccer.engines.contact import ContactPolicy, DurationRange, contact
from dynamisbench_soccer.engines.event_time import (
    ClockTiming,
    EdgeTiming,
    EventRefusal,
    Timing,
)
from dynamisbench_soccer.engines.force_platform import MethodSpec as ForceMethod
from dynamisbench_soccer.engines.force_platform import process as force_process
from dynamisbench_soccer.engines.force_platform_acquisition import PlateSpec
from dynamisbench_soccer.engines.gnss_lps import ProcessingOptions as TrackOptions
from dynamisbench_soccer.engines.gnss_lps import process as track_process
from dynamisbench_soccer.engines.load_cell_acquisition import KGF_TO_N, CellSpec, RawRecord
from dynamisbench_soccer.engines.lpt import MethodSpec as LptMethod
from dynamisbench_soccer.engines.lpt import process as lpt_process
from dynamisbench_soccer.engines.lpt_acquisition import GeometrySpec
from dynamisbench_soccer.engines.radar_laser_acquisition import AcquisitionRefusal, ChannelStage
from dynamisbench_soccer.engines.timing_gates import sprint as gate_sprint
from dynamisbench_soccer.reference.clocks import DeviceClock
from dynamisbench_soccer.reference.exporters import export_case
from dynamisbench_soccer.reference.faults import (
    ContinuousFault,
    ContinuousKind,
    EventFault,
    EventKind,
    InertialFault,
    InertialKind,
    PositionFault,
    PositionKind,
    inject,
)
from dynamisbench_soccer.reference.kinematics import cmj, sj
from dynamisbench_soccer.reference.motion_models import IsometricRamp, MonoExponentialSprint
from dynamisbench_soccer.reference.spatial import PlanarPhase, PlanarTrajectory, RigidBody
from dynamisbench_soccer.signals.schema import SignalBundle

G = 9.80665
SPRINT = MonoExponentialSprint()
PLATE = (PlateSpec(plate=1, assume_field_axes=True),)
EYE = ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
MARKER_RADIUS_M = 0.1  # declared trigger_method of the RES-669 gate observation
AXES = (
    "rate_hz",
    "filter",
    "event_method",
    "diff_integ",
    "calibration_geometry",
    "corruption",
    "eligibility",
    "refusal",
)


@dataclass
class Outcome:
    passed: bool
    observed: Any
    expected: Any
    tolerance: Any
    detail: str = ""


@dataclass
class Case:
    family: str
    subfamily: str
    name: str
    chain: str
    axes: dict[str, str]
    run: Callable[[], Outcome] = field(repr=False)


CASES: list[Case] = []


def case(family: str, subfamily: str, name: str, chain: str, **axes: object):
    def register(fn: Callable[[], Outcome]) -> Callable[[], Outcome]:
        CASES.append(Case(family, subfamily, name, chain, {k: str(v) for k, v in axes.items()}, fn))
        return fn

    return register


def close(observed: float, expected: float, tol: float) -> Outcome:
    return Outcome(abs(observed - expected) <= tol, observed, expected, tol)


def holds(condition: bool, observed: Any, expected: Any, detail: str = "") -> Outcome:
    return Outcome(bool(condition), observed, expected, None, detail)


def refuses(fn: Callable[[], object], match: str) -> Outcome:
    try:
        fn()
    except (ValueError, EventRefusal, AcquisitionRefusal) as exc:
        return Outcome(
            match.lower() in str(exc).lower(), str(exc)[:160], f"refusal matching {match!r}", None
        )
    return Outcome(False, "no refusal", f"refusal matching {match!r}", None)


# ---------------------------------------------------------------- force platforms

FP_TRUTH_HEIGHT = 2.7**2 / (2 * G)


def _fp_height(hz: int, **method: object) -> float:
    obs = export_case("force_platform.single", cmj(), hz=hz).observation
    device = force_process(obs, plates=PLATE, method=ForceMethod(**method))  # type: ignore[arg-type]
    return jumps.assess(
        jumps.JumpSignal.from_engine(device), jumps.JumpSpec(protocol="cmj")
    ).estimands["height_impulse_momentum_m"]


for _hz in (500, 1000, 2000):

    def _make(hz: int = _hz) -> Callable[[], Outcome]:
        return lambda: close(_fp_height(hz), FP_TRUTH_HEIGHT, 6e-3)

    case(
        "force_platform",
        "force_platform.single",
        f"cmj_height_{_hz}hz",
        "native force -> engine -> jump assessment",
        rate_hz=_hz,
        diff_integ="trapezoid",
        calibration_geometry="single plate, declared field axes",
    )(_make())

case(
    "force_platform",
    "force_platform.single",
    "cmj_height_simpson",
    "native force -> engine -> jump assessment",
    rate_hz=1000,
    diff_integ="simpson",
)(lambda: close(_fp_height(1000, integration="simpson"), FP_TRUTH_HEIGHT, 6e-3))
case(
    "force_platform",
    "force_platform.single",
    "cmj_height_butterworth_50hz",
    "native force -> engine -> jump assessment",
    rate_hz=1000,
    filter="butterworth_50hz",
    diff_integ="trapezoid",
)(lambda: close(_fp_height(1000, filter="butterworth", cutoff_hz=50), FP_TRUTH_HEIGHT, 0.02))

for _onset in ("5sd", "reverse_scan", "first_derivative", "absolute_N"):

    def _onset_case(onset: str = _onset) -> Callable[[], Outcome]:
        def run() -> Outcome:
            obs = export_case("force_platform.single", cmj(), hz=1000).observation
            device = force_process(obs, plates=PLATE, method=ForceMethod(onset=onset))  # type: ignore[arg-type]
            assert device.onset.time_s is not None
            # A zero-phase 10 Hz derivative filter spreads a step over ~1/(2 pi 10) s per side.
            tol = 2 / (2 * math.pi * 10) if onset == "first_derivative" else 0.01
            return close(device.onset.time_s, 1.0, tol)

        return run

    case(
        "force_platform",
        "force_platform.single",
        f"onset_{_onset}",
        "native force -> engine onset",
        rate_hz=1000,
        event_method=_onset,
    )(_onset_case())


@case(
    "force_platform",
    "force_platform.single",
    "clipped_record_is_rejected",
    "corruption -> engine verdict",
    rate_hz=1000,
    corruption="clipping at 1000 N",
)
def _fp_clip() -> Outcome:
    obs = export_case("force_platform.single", cmj(), hz=1000).observation
    clipped = inject(obs, ContinuousFault(ContinuousKind.CLIPPING, ("fz",), 1000.0))
    return holds(force_process(clipped, plates=PLATE).verdict == "reject", "reject", "reject")


@case(
    "force_platform",
    "force_platform.single",
    "baseline_offset_shifts_body_weight",
    "corruption -> engine body weight",
    rate_hz=1000,
    corruption="constant offset 30 N",
)
def _fp_offset() -> Outcome:
    obs = export_case("force_platform.single", sj(), hz=1000).observation
    shifted = inject(obs, ContinuousFault(ContinuousKind.OFFSET, ("fz",), 30.0))
    return close(force_process(shifted, plates=PLATE).body_weight_n, 75 * G + 30.0, 0.5)


@case(
    "force_platform",
    "force_platform.single",
    "squat_record_is_not_a_countermovement_jump",
    "assessment eligibility",
    rate_hz=1000,
    eligibility="SJ record under CMJ protocol",
)
def _fp_elig() -> Outcome:
    obs = export_case("force_platform.single", sj(), hz=1000).observation
    result = jumps.assess(
        jumps.JumpSignal.from_engine(force_process(obs, plates=PLATE)),
        jumps.JumpSpec(protocol="cmj"),
    )
    return holds(not result.eligible, result.eligible, False)


@case(
    "force_platform",
    "force_platform.single",
    "sampling_below_500hz_is_refused",
    "native force -> engine refusal",
    rate_hz=250,
    refusal="rate floor 500 Hz",
)
def _fp_rate() -> Outcome:
    obs = export_case("force_platform.single", cmj(), hz=250).observation
    return refuses(lambda: force_process(obs, plates=PLATE), "at least 500")


@case(
    "force_platform",
    "force_platform.dual",
    "dual_plate_body_weight_and_bilateral_split",
    "native dual force -> engine -> bilateral assessment",
    rate_hz=1000,
    calibration_geometry="two plates, declared sides",
    diff_integ="trapezoid",
)
def _fp_dual() -> Outcome:
    obs = export_case("force_platform.dual", cmj(), hz=1000).observation
    device = force_process(
        obs,
        plates=(
            PlateSpec(plate=1, assume_field_axes=True),
            PlateSpec(plate=2, assume_field_axes=True),
        ),
    )
    result = jumps.assess(
        jumps.JumpSignal.from_engine(device),
        jumps.JumpSpec(protocol="cmj", plate_sides={1: "left", 2: "right"}),
    )
    asym = result.bilateral["impulse_asymmetry_weaker_over_stronger_percent"]
    return close(float(asym), 100 * 0.4 / 0.6, 0.5)  # type: ignore[arg-type]


@case(
    "force_platform",
    "force_platform.dual",
    "dual_swapped_sides_swap_the_labels_not_the_asymmetry",
    "assessment side declaration",
    rate_hz=1000,
    eligibility="side declaration reversed",
)
def _fp_dual_swap() -> Outcome:
    obs = export_case("force_platform.dual", cmj(), hz=1000).observation
    device = force_process(
        obs,
        plates=(
            PlateSpec(plate=1, assume_field_axes=True),
            PlateSpec(plate=2, assume_field_axes=True),
        ),
    )
    signal = jumps.JumpSignal.from_engine(device)
    a = jumps.assess(signal, jumps.JumpSpec(protocol="cmj", plate_sides={1: "left", 2: "right"}))
    b = jumps.assess(signal, jumps.JumpSpec(protocol="cmj", plate_sides={1: "right", 2: "left"}))
    return close(a.bilateral["impulse_n_s"]["left"], b.bilateral["impulse_n_s"]["right"], 1e-9)  # type: ignore[index]


# ---------------------------------------------------------------- LPT and encoders


def _lpt(hz: int, method: str, geometry: GeometrySpec = LINE) -> dict[str, Any]:
    case_ = export_case("lpt_encoder.lpt", MOTION, test_id="cmj", hz=hz)
    return lpt_process(case_.observation, (LPT,), geometry, LptMethod(name=method))  # type: ignore[arg-type]


for _method in ("finite_difference", "butterworth_then_difference", "cubic_spline"):

    def _lpt_case(method: str = _method) -> Callable[[], Outcome]:
        def run() -> Outcome:
            report = _lpt(200, method)
            t = np.asarray(report["motion"]["elapsed_s"])
            velocity = np.asarray(report["motion"]["velocity_m_s"])
            keep = (t > 0.1) & (t < t[-1] - 0.1)
            truth = 0.4 + 2 * t  # d/dt (0.4 t + t^2)
            # The truth is quadratic in time (exact for central differences and splines); the
            # low-pass variant adds a bounded edge effect after trimming 0.1 s.
            return close(float(np.max(np.abs(velocity[keep] - truth[keep]))), 0.0, 0.02)

        return run

    case(
        "lpt_encoder",
        "lpt_encoder.lpt",
        f"velocity_{_method}",
        "native cable -> engine -> velocity",
        rate_hz=200,
        diff_integ=_method,
        calibration_geometry="vertical line above anchor declared",
    )(_lpt_case())


@case(
    "lpt_encoder",
    "lpt_encoder.lpt",
    "cable_only_geometry_refuses_vertical_claims",
    "native cable -> engine refusal",
    rate_hz=200,
    calibration_geometry="no vertical line declared",
    refusal="cable is not vertical displacement",
)
def _lpt_cable_only() -> Outcome:
    report = _lpt(200, "cubic_spline", GeometrySpec(attachment_point="COM"))
    return holds(
        report["estimand"] == "cable_displacement", report["estimand"], "cable_displacement"
    )


@case(
    "lpt_encoder",
    "lpt_encoder.lpt",
    "jump_height_is_an_attachment_estimand",
    "engine -> assessment",
    rate_hz=200,
    eligibility="attachment heights separate from force-plate heights",
)
def _lpt_height() -> Outcome:
    motion = cmj()
    report = lpt_process(
        export_case("lpt_encoder.lpt", motion, test_id="cmj", hz=200).observation,
        (LPT,),
        LINE,
        LptMethod(name="cubic_spline"),
    )
    heights = jumps.lpt_estimates(report)
    apex = motion.state(motion.takeoff_s() + motion.takeoff_velocity_m_s() / G).displacement_m
    return close(heights.height_peak_displacement_above_start_m, apex, 0.003)


@case(
    "lpt_encoder",
    "lpt_encoder.lpt",
    "time_jump_in_stream_is_rejected",
    "corruption -> engine verdict",
    rate_hz=100,
    corruption="timestamp discontinuity",
    refusal="integrity failure",
)
def _lpt_gap() -> Outcome:
    base = export_case("lpt_encoder.lpt", MOTION, test_id="cmj", hz=100).observation
    data = {k: v.copy() for k, v in base.data.items()}
    data["time"][50:] += 0.1
    report = lpt_process(_modify(base, data=data), (LPT,), LINE, LptMethod())
    return holds(report["verdict"] == "reject", report["verdict"], "reject")


@case(
    "lpt_encoder",
    "lpt_encoder.encoder",
    "encoder_counts_resolution_bound",
    "native counts -> engine",
    rate_hz=100,
    calibration_geometry="counts per metre declared",
    diff_integ="quantized cable length",
)
def _enc_counts() -> Outcome:
    base = export_case("lpt_encoder.encoder", MOTION, test_id="cmj")
    from test_lpt_acquisition import COUNTS

    report = lpt_process(
        base.observation, (COUNTS,), GeometrySpec(attachment_point="COM"), LptMethod()
    )
    cable = report["cables"][0]
    t = np.asarray(cable["native_time_s"])
    z = 0.4 * t + t**2
    displacement = np.asarray(cable["cable_displacement_m"])
    # Start and current readings are each quantised by at most half a count.
    expected = np.hypot(1.0, z) - 1.0
    return close(float(np.max(np.abs(displacement - expected))), 0.0, cable["resolution_m"] + 1e-12)


@case(
    "lpt_encoder",
    "lpt_encoder.encoder",
    "encoder_without_vertical_geometry_withholds_vertical_displacement",
    "native counts -> engine refusal",
    rate_hz=100,
    calibration_geometry="no vertical line declared",
    refusal="cable coordinates only",
)
def _enc_geometry() -> Outcome:
    from test_lpt_acquisition import COUNTS

    base = export_case("lpt_encoder.encoder", MOTION, test_id="cmj")
    report = lpt_process(
        base.observation, (COUNTS,), GeometrySpec(attachment_point="COM"), LptMethod()
    )
    return holds(
        report["estimand"] == "cable_displacement" and report["geometry_method"] == "cable_only",
        report["estimand"],
        "cable_displacement",
    )


@case(
    "lpt_encoder",
    "lpt_encoder.encoder",
    "encoder_time_discontinuity_is_rejected",
    "corruption -> engine verdict",
    rate_hz=100,
    corruption="timestamp discontinuity",
    refusal="integrity failure",
)
def _enc_gap() -> Outcome:
    from test_lpt_acquisition import COUNTS

    base = export_case("lpt_encoder.encoder", MOTION, test_id="cmj").observation
    data = {k: v.copy() for k, v in base.data.items()}
    data["time"][len(data["time"]) // 2 :] += 0.5
    report = lpt_process(_modify(base, data=data), (COUNTS,), LINE, LptMethod())
    return holds(report["verdict"] == "reject", report["verdict"], "reject")


# ---------------------------------------------------------------- load cells and rigs


def _cell(subfamily: str, hz: int, ramp: IsometricRamp | None = None) -> SignalBundle:
    return export_case(subfamily, ramp or IsometricRamp(), hz=hz).observation


for _sub in ("load_cell_isometric.load_cell", "load_cell_isometric.isometric_rig"):
    for _hz in (1000, 2000):

        def _cell_case(sub: str = _sub, hz: int = _hz) -> Callable[[], Outcome]:
            def run() -> Outcome:
                result = load_cell.process(
                    _cell(sub, hz), CellSpec(), load_cell.MethodSpec(windows_ms=(50, 100))
                )
                rfd = result.metrics["rfd_0_100ms_n_s"]
                assert rfd is not None
                return close(rfd, 2500 / 0.4, 1.0)

            return run

        case(
            "load_cell_isometric",
            _sub,
            f"rfd_0_100ms_{_hz}hz",
            "native cell -> engine",
            rate_hz=_hz,
            event_method="5sd",
            calibration_geometry="declared sensitivity record",
            diff_integ="windowed difference",
        )(_cell_case())

for _onset in load_cell.ONSET_METHODS:

    def _cell_onset(onset: str = _onset) -> Callable[[], Outcome]:
        def run() -> Outcome:
            result = load_cell.process(
                _cell("load_cell_isometric.load_cell", 1000),
                CellSpec(),
                load_cell.MethodSpec(onset=onset),  # type: ignore[arg-type]
            )
            assert result.onset.time_s is not None
            return close(result.onset.time_s, 0.5, 20 / 6250 + 0.03)

        return run

    case(
        "load_cell_isometric",
        "load_cell_isometric.load_cell",
        f"onset_{_onset}",
        "native cell -> engine onset",
        rate_hz=1000,
        event_method=_onset,
    )(_cell_onset())


@case(
    "load_cell_isometric",
    "load_cell_isometric.load_cell",
    "butterworth_peak_filter_keeps_peak",
    "native cell -> engine filter",
    rate_hz=1000,
    filter="butterworth_50hz",
)
def _cell_filter() -> Outcome:
    method = load_cell.MethodSpec(
        peak_filter=load_cell.FilterSpec(kind="butterworth_lowpass", cutoff_hz=50)
    )
    result = load_cell.process(_cell("load_cell_isometric.load_cell", 1000), CellSpec(), method)
    return close(result.metrics["peak_force_n"] or 0.0, 2500.0, 25.0)


@case(
    "load_cell_isometric",
    "load_cell_isometric.load_cell",
    "torque_refused_without_lever_arm",
    "engine refusal",
    rate_hz=1000,
    calibration_geometry="no lever arm declared",
    refusal="torque refused",
)
def _cell_torque() -> Outcome:
    result = load_cell.process(
        _cell("load_cell_isometric.load_cell", 1000), CellSpec(), load_cell.MethodSpec()
    )
    return holds(not result.torque.available, result.torque.available, False)


@case(
    "load_cell_isometric",
    "load_cell_isometric.load_cell",
    "clipped_record_is_rejected",
    "corruption -> engine verdict",
    rate_hz=1000,
    corruption="clipping at 2 V",
)
def _cell_clip() -> Outcome:
    clipped = inject(
        _cell("load_cell_isometric.load_cell", 1000),
        ContinuousFault(ContinuousKind.CLIPPING, ("voltage",), 2.0),
    )
    return holds(
        load_cell.process(clipped, CellSpec(raw_range=(-2.0, 2.0))).verdict == "reject",
        "reject",
        "reject",
    )


@case(
    "load_cell_isometric",
    "load_cell_isometric.load_cell",
    "kgf_mislabelled_as_newton_is_flagged",
    "corruption -> engine verdict",
    rate_hz=1000,
    corruption="unit mislabel",
    calibration_geometry="declared expected peak range",
)
def _cell_units() -> Outcome:
    ramp = IsometricRamp()
    t = np.arange(0, ramp.duration_s, 0.001)
    kgf = np.array([ramp.force_n(float(x)) for x in t]) / KGF_TO_N
    result = load_cell.process(RawRecord(kgf, "N", 1000), CellSpec(expected_peak_n=(1000, 6000)))
    return holds(result.verdict == "reject", result.verdict, "reject")


@case(
    "load_cell_isometric",
    "load_cell_isometric.load_cell",
    "low_rate_refuses_rfd_windows",
    "engine refusal",
    rate_hz=500,
    refusal="RFD needs >= 1000 Hz",
)
def _cell_rate() -> Outcome:
    result = load_cell.process(
        _cell("load_cell_isometric.load_cell", 500), CellSpec(), load_cell.MethodSpec()
    )
    return holds(result.metrics["rfd_0_50ms_n_s"] is None, result.metrics["rfd_0_50ms_n_s"], None)


@case(
    "load_cell_isometric",
    "load_cell_isometric.load_cell",
    "imtp_protocol_eligibility_needs_declared_knee_angle",
    "engine -> isometric assessment",
    rate_hz=1000,
    eligibility="knee angle undeclared",
)
def _cell_elig() -> Outcome:
    bundle = _cell("load_cell_isometric.load_cell", 1000)
    result = isometric.assess(
        [load_cell.process(bundle, CellSpec(), load_cell.MethodSpec())],
        isometric.IsoSpec(protocol="imtp", min_trials=1),
        geometry=bundle.meta.geometry,
    )
    return holds(not result.eligible, result.eligible, False)


@case(
    "load_cell_isometric",
    "load_cell_isometric.isometric_rig",
    "imtp_summary_from_rig_trials",
    "native rig -> engine -> isometric assessment",
    rate_hz=1000,
    eligibility="hip protocol, declared hip flexion",
)
def _rig_summary() -> Outcome:
    bundle = _cell("load_cell_isometric.isometric_rig", 1000)
    results = [load_cell.process(bundle, CellSpec(), load_cell.MethodSpec()) for _ in range(2)]
    result = isometric.assess(
        results,
        isometric.IsoSpec(protocol="hip_adduction", min_trials=2),
        geometry=bundle.meta.geometry,
    )
    return close(result.summary["peak_force_n"] or 0.0, 2500.0, 1e-6)


# ---------------------------------------------------------------- Nordic and hip


def _nordic_spec(**kw: object) -> nordic.NordicSpec:
    base: dict[str, Any] = {
        "protocol": "nordic_eccentric",
        "force_basis": "raw_calibrated",
        "sides": (
            nordic.SideDeclaration(channel="force_left", side="left"),
            nordic.SideDeclaration(channel="force_right", side="right"),
        ),
        "lever_arm_m": 0.4,
        "moment_arm_definition": "knee joint centre to strap",
    }
    return nordic.NordicSpec(**{**base, **kw})


for _hz in (500, 1000, 2000):

    def _nordic_case(hz: int = _hz) -> Callable[[], Outcome]:
        def run() -> Outcome:
            bundle = export_case("nordic_hip.nordic", hz=hz).observation
            result = nordic.process(bundle, _nordic_spec())
            return close(
                result.sides["left"].metrics["impulse_to_peak_n_s"] or 0.0,
                160 * (math.pi / 4 / 0.2) / 2,
                160.0 / hz,  # the peak sample can sit one interval from the true peak time
            )

        return run

    case(
        "nordic_hip",
        "nordic_hip.nordic",
        f"impulse_to_peak_{_hz}hz",
        "native bilateral force -> engine",
        rate_hz=_hz,
        event_method="first_exceedance",
        diff_integ="trapezoid",
        calibration_geometry="lever arm and moment-arm definition declared",
    )(_nordic_case())


@case(
    "nordic_hip",
    "nordic_hip.nordic",
    "butterworth_filter_peak",
    "native bilateral force -> engine filter",
    rate_hz=1000,
    filter="butterworth_20hz",
)
def _nordic_filter() -> Outcome:
    bundle = export_case("nordic_hip.nordic", hz=1000).observation
    method = nordic.MethodSpec(
        filter=load_cell.FilterSpec(kind="butterworth_lowpass", cutoff_hz=20)
    )
    return close(
        nordic.process(bundle, _nordic_spec(), method).sides["left"].metrics["peak_force_n"] or 0.0,
        160.0,
        1.0,
    )


@case(
    "nordic_hip",
    "nordic_hip.nordic",
    "side_contradiction_is_refused",
    "engine refusal",
    rate_hz=1000,
    calibration_geometry="swapped side declaration",
    refusal="side contradiction",
)
def _nordic_side() -> Outcome:
    bundle = export_case("nordic_hip.nordic", hz=1000).observation
    swapped = _nordic_spec(
        sides=(
            nordic.SideDeclaration(channel="force_left", side="right"),
            nordic.SideDeclaration(channel="force_right", side="left"),
        )
    )
    return refuses(lambda: nordic.process(bundle, swapped), "side contradiction")


@case(
    "nordic_hip",
    "nordic_hip.nordic",
    "clipped_side_is_visible_in_peak",
    "corruption -> engine metric",
    rate_hz=1000,
    corruption="clipping at 100 N",
)
def _nordic_clip() -> Outcome:
    bundle = export_case("nordic_hip.nordic", hz=1000).observation
    clipped = inject(bundle, ContinuousFault(ContinuousKind.CLIPPING, ("force_left",), 100.0))
    return close(
        nordic.process(clipped, _nordic_spec()).sides["left"].metrics["peak_force_n"] or 0.0,
        100.0,
        0.05,
    )


@case(
    "nordic_hip",
    "nordic_hip.nordic",
    "angle_is_never_recovered_from_force",
    "engine refusal",
    rate_hz=1000,
    eligibility="no measured angle channel",
    refusal="angle metrics refused",
)
def _nordic_angle() -> Outcome:
    bundle = export_case("nordic_hip.nordic", hz=1000).observation
    result = nordic.process(
        bundle, _nordic_spec(), nordic.MethodSpec(break_point="peak_force_angle")
    )
    return holds(
        "break_point" in result.sides["left"].refusals,
        list(result.sides["left"].refusals),
        "break_point refused",
    )


@case(
    "nordic_hip",
    "nordic_hip.nordic",
    "raw_and_vendor_corrected_trials_cannot_be_pooled",
    "engine -> assessment refusal",
    rate_hz=1000,
    eligibility="mixed force basis",
    refusal="cannot be combined",
)
def _nordic_basis() -> Outcome:
    bundle = export_case("nordic_hip.nordic", hz=1000).observation
    a = nordic.process(bundle, _nordic_spec())
    b = nordic.process(bundle, _nordic_spec(force_basis="vendor_corrected"))
    return refuses(
        lambda: isometric.assess_nordic([a, b], isometric.NordicSpec()), "cannot be combined"
    )


@case(
    "nordic_hip",
    "nordic_hip.hip",
    "hip_squeeze_peak_and_torque",
    "native hip force -> engine",
    rate_hz=1000,
    event_method="first_exceedance",
    calibration_geometry="hip flexion angle declared",
)
def _hip_peak() -> Outcome:
    bundle = export_case("nordic_hip.hip", hz=1000).observation
    spec = nordic.NordicSpec(
        protocol="hip_squeeze",
        force_basis="raw_calibrated",
        sides=(nordic.SideDeclaration(channel="force", side="unilateral"),),
        lever_arm_m=0.4,
        moment_arm_definition="hip joint centre to cuff",
    )
    return close(
        nordic.process(bundle, spec).sides["unilateral"].metrics["peak_torque_n_m"] or 0.0,
        1000.0,
        1e-4,
    )


@case(
    "nordic_hip",
    "nordic_hip.hip",
    "hip_squeeze_refused_without_hip_flexion",
    "engine refusal",
    rate_hz=1000,
    calibration_geometry="hip flexion undeclared",
    refusal="hip flexion angle",
)
def _hip_geometry() -> Outcome:
    bundle = export_case("nordic_hip.hip", hz=1000).observation
    geometry = bundle.meta.geometry.model_copy(update={"hip_flexion_deg": None})
    stripped = SignalBundle(bundle.meta.model_copy(update={"geometry": geometry}), bundle.data)
    spec = nordic.NordicSpec(
        protocol="hip_squeeze",
        force_basis="raw_calibrated",
        sides=(nordic.SideDeclaration(channel="force", side="unilateral"),),
    )
    return refuses(lambda: nordic.process(stripped, spec), "hip flexion angle")


# ---------------------------------------------------------------- timing gates and contact


@case(
    "timing_contact",
    "timing_contact.gates",
    "flying_start_gate_splits_match_exact_position_inverse",
    "native gate events -> engine",
    event_method="first_gate",
    calibration_geometry="gate positions declared",
)
def _gates_splits() -> Outcome:
    obs = export_case("timing_contact.gates", SPRINT).observation
    result = gate_sprint([obs], gate_policy())
    times = [s.cumulative_s.value for s in result.splits]
    # The exporter's marker has radius 0.1 m with the beam at its centre height, so the beam
    # first breaks when the marker front reaches the gate: x(t) = d - 0.1.
    # The clock starts at gate 1.
    first = SPRINT.gate_time_s(DISTANCES[0] - MARKER_RADIUS_M)
    expected = [SPRINT.gate_time_s(d - MARKER_RADIUS_M) - first for d in DISTANCES]
    return close(max(abs(a - b) for a, b in zip(times, expected, strict=True)), 0.0, 2e-4)


def _trigger_splits() -> list[float]:
    from test_timing_gates import gate_stream

    from dynamisbench_soccer.engines.timing_gates import TriggerStart

    case_ = export_case("timing_contact.gates")
    rows = [
        (0.04, 0, 0, 0, 1),
        *[
            (t, d, i, 1, 0)
            for i, (t, d) in enumerate(zip(case_.observation["t_s"], DISTANCES, strict=True), 1)
        ],
    ]
    policy = gate_policy(
        start=TriggerStart(
            mode="pressure_pad", position_m=0, timing=EdgeTiming(latency_s=0.04, latency_error_s=0)
        )
    )
    result = gate_sprint([gate_stream(case_.observation, rows)], policy)
    return [s.cumulative_s.value for s in result.splits]


@case(
    "timing_contact",
    "timing_contact.gates",
    "standing_start_gate_fit_recovers_fv_parameters",
    "gate events + start trigger -> engine -> sprint F-V",
    event_method="pressure_pad start trigger",
    diff_integ="distance-time fit",
)
def _gates_fv() -> Outcome:
    front = [d - MARKER_RADIUS_M for d in DISTANCES]  # distance travelled by the triggering marker
    fit = sprint.fit_split_times(front, _trigger_splits(), sprint.FvSpec(mass_kg=80.0))
    return close(fit.vmax_m_s or 0.0, 9.0, 0.05)


@case(
    "timing_contact",
    "timing_contact.gates",
    "flying_start_splits_are_not_a_standing_start_fit",
    "gate events -> assessment eligibility",
    event_method="first_gate",
    eligibility="flying start under a standing-start model",
    refusal="fit refused or visibly biased",
)
def _gates_flying() -> Outcome:
    obs = export_case("timing_contact.gates", SPRINT).observation
    times = [s.cumulative_s.value for s in gate_sprint([obs], gate_policy()).splits]
    fit = sprint.fit_split_times(list(DISTANCES), times, sprint.FvSpec(mass_kg=80.0))
    wrong = fit.vmax_m_s is None or abs(fit.vmax_m_s - 9.0) > 0.45
    return holds(wrong, fit.vmax_m_s, "not within 5% of the true Vmax")


@case(
    "timing_contact",
    "timing_contact.gates",
    "gates_never_yield_instantaneous_velocity",
    "assessment refusal",
    eligibility="gate source requested as speed trace",
    refusal="segment averages",
)
def _gates_refusal() -> Outcome:
    return refuses(lambda: sprint.instantaneous_velocity("gates"), "segment averages")


for _kind in (EventKind.MISSING, EventKind.DUPLICATE, EventKind.OUT_OF_ORDER):

    def _gate_fault(kind: EventKind = _kind) -> Callable[[], Outcome]:
        def run() -> Outcome:
            obs = export_case("timing_contact.gates", SPRINT).observation
            return refuses(
                lambda: gate_sprint([inject(obs, EventFault(kind, index=1))], gate_policy()), ""
            )

        return run

    case(
        "timing_contact",
        "timing_contact.gates",
        f"adversarial_{_kind.value}",
        "corruption -> engine refusal",
        corruption=_kind.value,
        refusal="event stream refused",
    )(_gate_fault())


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


for _latency in (0.0, 0.02):

    def _contact_case(latency: float = _latency) -> Callable[[], Outcome]:
        def run() -> Outcome:
            obs = export_case(
                "timing_contact.contact", cmj(), clock=DeviceClock(0.0, 0.0), latency_s=latency
            ).observation
            flight = contact(obs, _contact_policy(latency)).flights[0]
            return close(flight.flight_s.value, 2 * 2.7 / G, 2e-3)

        return run

    case(
        "timing_contact",
        "timing_contact.contact",
        f"cmj_flight_time_latency_{_latency:g}s",
        "native switch events -> engine -> jump source",
        event_method="switch edge pairing",
        calibration_geometry=f"declared latency {_latency:g} s",
    )(_contact_case())


@case(
    "timing_contact",
    "timing_contact.contact",
    "duplicate_switch_event_is_refused",
    "corruption -> engine refusal",
    corruption="duplicate event",
    refusal="event stream refused",
)
def _contact_dup() -> Outcome:
    obs = export_case("timing_contact.contact", cmj(), clock=DeviceClock(0.0, 0.0)).observation
    return refuses(
        lambda: contact(inject(obs, EventFault(EventKind.DUPLICATE, index=1)), _contact_policy()),
        "",
    )


# ---------------------------------------------------------------- radar and laser


def _radar(hz: int, **opts: Any) -> radar_laser.VelocityResult:
    obs = export_case("radar_laser.radar", SPRINT, hz=hz).observation
    options = radar_laser.ProcessingOptions(
        channel_stage=ChannelStage.NATIVE, correction_method="declared_fixed_angle", **opts
    )
    return radar_laser.process(obs, options)


for _hz in (47, 100, 256):

    def _radar_case(hz: int = _hz) -> Callable[[], Outcome]:
        def run() -> Outcome:
            result = _radar(hz)
            t = np.asarray(result.time_from_start_s)
            truth = np.array([SPRINT.velocity_m_s(float(x)) for x in t])
            return close(float(np.max(np.abs(np.abs(result.velocity_m_s) - truth))), 0.0, 1e-6)

        return run

    case(
        "radar_laser",
        "radar_laser.radar",
        f"doppler_velocity_{_hz}hz",
        "native Doppler -> engine -> sprint F-V",
        rate_hz=_hz,
        event_method="record_origin",
        calibration_geometry="declared fixed angle",
    )(_radar_case())


@case(
    "radar_laser",
    "radar_laser.radar",
    "fv_fit_from_native_radar",
    "native Doppler -> engine -> sprint F-V",
    rate_hz=100,
    diff_integ="mono-exponential fit",
)
def _radar_fv() -> Outcome:
    result = _radar(100)
    fit = sprint.fit_velocity_trace(
        np.asarray(result.time_from_start_s),
        np.abs(np.asarray(result.velocity_m_s)),
        sprint.FvSpec(mass_kg=80.0),
        sprint.StartSpec(),
        source="radar",
    )
    return close(fit.f0_n or 0.0, 80 * 9 / 1.2, 0.05 * 600)


@case(
    "radar_laser",
    "radar_laser.radar",
    "manufacturer_processed_channel_without_cosine_qualification_refuses",
    "engine refusal",
    rate_hz=100,
    calibration_geometry="vendor-processed stage",
    refusal="stage not qualified",
)
def _radar_stage() -> Outcome:
    obs = export_case("radar_laser.radar", SPRINT, hz=100).observation
    options = radar_laser.ProcessingOptions(
        channel_stage=ChannelStage.MANUFACTURER_PROCESSED, correction_method="declared_fixed_angle"
    )
    return refuses(lambda: radar_laser.process(obs, options), "")


for _method in ("central", "five_point"):

    def _laser_case(method: str = _method) -> Callable[[], Outcome]:
        def run() -> Outcome:
            obs = export_case("radar_laser.laser", SPRINT, hz=256).observation
            result = radar_laser.process(
                obs,
                radar_laser.ProcessingOptions(
                    channel_stage=ChannelStage.NATIVE, derivative_method=method
                ),  # type: ignore[arg-type]
            )
            t = np.asarray(result.time_from_start_s)
            truth = np.array([SPRINT.velocity_m_s(float(x)) for x in t])
            keep = slice(5, -5)
            tol = 5e-3 if method == "central" else 1e-4
            return close(
                float(np.max(np.abs(np.abs(result.velocity_m_s)[keep] - truth[keep]))), 0.0, tol
            )

        return run

    case(
        "radar_laser",
        "radar_laser.laser",
        f"range_rate_{_method}",
        "native range -> engine -> velocity",
        rate_hz=256,
        diff_integ=_method,
        calibration_geometry="aiming point declared",
    )(_laser_case())


# ---------------------------------------------------------------- GNSS and LPS


for _kind, _hz in (("gnss", 18), ("lps", 25)):
    for _deriv in ("three_point", "five_point"):

        def _track_case(
            kind: str = _kind, hz: int = _hz, deriv: str = _deriv
        ) -> Callable[[], Outcome]:
            def run() -> Outcome:
                translation = PlanarTrajectory(
                    (PlanarPhase(3, (2, 0)),), initial_velocity_m_s=(1, 0)
                )
                raw = export_case(
                    f"gnss_lps.{kind}",
                    RigidBody(translation),
                    hz=hz,
                    clock=DeviceClock(0.21, 160),
                    latency_s=0.038,
                )
                built, declaration = track_case(kind, hz=hz, raw=raw)
                result = track_process(
                    built.observation, declaration, TrackOptions(derivative=deriv)
                )  # type: ignore[arg-type]
                t = built.truth["time"]
                error = np.abs(result.kinematics.horizontal_positional_speed_m_s - (1 + 2 * t))
                return close(float(error.max()), 0.0, 1e-6)

            return run

        case(
            "gnss_lps",
            f"gnss_lps.{_kind}",
            f"positional_speed_{_deriv}",
            "native track -> engine -> speed",
            rate_hz=_hz,
            diff_integ=_deriv,
            calibration_geometry="declared datum, latency and clock drift",
        )(_track_case())


@case(
    "gnss_lps",
    "gnss_lps.gnss",
    "doppler_and_positional_speed_stay_separate",
    "native track -> engine",
    rate_hz=18,
    eligibility="Doppler channel declared",
)
def _gnss_doppler() -> Outcome:
    built, declaration = track_case("gnss", hz=18)
    result = track_process(built.observation, declaration)
    return holds(
        result.kinematics.doppler_speed_m_s is not None
        and not np.allclose(
            result.kinematics.doppler_speed_m_s, result.kinematics.horizontal_positional_speed_m_s
        ),
        "separate arrays",
        "Doppler != positional",
    )


for _kind, _hz in (("gnss", 18), ("lps", 25)):

    def _track_jump(kind: str = _kind, hz: int = _hz) -> Callable[[], Outcome]:
        def run() -> Outcome:
            built, declaration = track_case(kind, hz=hz)
            jumped = inject(
                built.observation, PositionFault(PositionKind.JUMP, index=10, shift_m=(5.0, 0, 0))
            )
            try:
                result = track_process(jumped, declaration)
            except (ValueError, AcquisitionRefusal) as exc:
                return Outcome(True, str(exc)[:120], "refusal or reported discontinuity", None)
            peak = result.sensitivity.native.horizontal_positional_speed_m_s
            return holds(peak > 20, peak, "discontinuity visible in peak speed")

        return run

    case(
        "gnss_lps",
        f"gnss_lps.{_kind}",
        "position_jump_is_not_hidden",
        "corruption -> engine",
        rate_hz=_hz,
        corruption="5 m position jump",
    )(_track_jump())


# ---------------------------------------------------------------- IMU

IMU_BODY = RigidBody(
    PlanarTrajectory((PlanarPhase(1, (0, 0)), PlanarPhase(2, (2, 1)), PlanarPhase(1, (-1, 0.5)))),
    sensor_offset_m=(0.1, 0, 0.2),
    angular_velocity_rad_s=(0.2, 0.1, 0.3),
)

for _hz in (100, 200, 500):

    def _imu_case(hz: int = _hz) -> Callable[[], Outcome]:
        def run() -> Outcome:
            obs = export_case("imu.inertial", IMU_BODY, hz=hz).observation
            spec = imu_engine.ImuSpec(initial_orientation=tuple(map(tuple, IMU_BODY.rotation(0))))  # type: ignore[arg-type]
            result = imu_engine.process(obs, spec)
            assert result.rotation_field_from_sensor is not None
            k = len(result.time_s) - 1
            error = np.linalg.norm(
                result.rotation_field_from_sensor[k] - IMU_BODY.rotation(float(result.time_s[k]))
            )
            return close(float(error), 0.0, 1e-8)

        return run

    case(
        "imu",
        "imu.inertial",
        f"orientation_{_hz}hz",
        "native IMU -> engine orientation",
        rate_hz=_hz,
        event_method="gyro_integration",
        diff_integ="rodrigues propagation",
        calibration_geometry="declared initial orientation",
    )(_imu_case())


@case(
    "imu",
    "imu.inertial",
    "free_acceleration_matches_rigid_body",
    "native IMU -> engine gravity compensation",
    rate_hz=200,
    diff_integ="gravity compensation",
)
def _imu_free() -> Outcome:
    obs = export_case("imu.inertial", IMU_BODY, hz=200).observation
    spec = imu_engine.ImuSpec(initial_orientation=tuple(map(tuple, IMU_BODY.rotation(0))))  # type: ignore[arg-type]
    result = imu_engine.process(obs, spec)
    assert result.free_acceleration_m_s2 is not None
    worst = max(
        float(
            np.linalg.norm(
                result.free_acceleration_m_s2[k] - IMU_BODY.sensor_state(float(result.time_s[k]))[2]
            )
        )
        for k in range(0, len(result.time_s), 20)
    )
    return close(worst, 0.0, 1e-7)


@case(
    "imu",
    "imu.inertial",
    "complementary_filter_bounds_gyro_bias_tilt",
    "native IMU -> engine orientation",
    rate_hz=100,
    event_method="complementary",
    corruption="gyro bias 0.05 rad/s",
)
def _imu_comp() -> Outcome:
    still = RigidBody(
        PlanarTrajectory((PlanarPhase(10, (0, 0)),)), angular_velocity_rad_s=(0, 0, 0)
    )
    biased = inject(
        export_case("imu.inertial", still, hz=100).observation,
        InertialFault(InertialKind.BIAS, "angular_velocity", bias=(0.05, 0, 0)),
    )
    result = imu_engine.process(
        biased,
        imu_engine.ImuSpec(initial_orientation=EYE),
        imu_engine.MethodSpec(orientation="complementary"),
    )
    assert result.rotation_field_from_sensor is not None
    tilt = math.acos(min(1.0, float(result.rotation_field_from_sensor[-1][2, 2])))
    return close(tilt, 0.05, 0.01)


@case(
    "imu",
    "imu.inertial",
    "flight_time_from_specific_force",
    "native IMU -> engine event -> jump source",
    rate_hz=500,
    event_method="specific-force magnitude threshold",
)
def _imu_flight() -> Outcome:
    result = imu_engine.process(
        export_case("imu.inertial", cmj(), hz=500).observation,
        imu_engine.ImuSpec(initial_orientation=EYE),
    )
    return close(result.flights[0].duration_s, 2 * 2.7 / G, 4e-3)


@case(
    "imu",
    "imu.inertial",
    "saturated_accelerometer_is_rejected",
    "corruption -> engine verdict",
    rate_hz=200,
    corruption="accelerometer clipping at 5 m/s^2",
)
def _imu_sat() -> Outcome:
    obs = export_case("imu.inertial", IMU_BODY, hz=200).observation
    clipped = inject(obs, InertialFault(InertialKind.SATURATION, "specific_force", limit=5))
    return holds(
        imu_engine.process(clipped, imu_engine.ImuSpec(accel_range_m_s2=5.0)).verdict == "reject",
        "reject",
        "reject",
    )


@case(
    "imu",
    "imu.inertial",
    "orientation_unobservable_without_initial_condition",
    "engine refusal",
    rate_hz=200,
    calibration_geometry="no initial orientation or static window",
    refusal="orientation unobservable",
)
def _imu_refusal() -> Outcome:
    result = imu_engine.process(export_case("imu.inertial", IMU_BODY, hz=200).observation)
    return holds("orientation" in result.refusals, list(result.refusals), "orientation refused")


@case(
    "imu",
    "imu.inertial",
    "integration_refused_without_declared_initial_velocity",
    "engine refusal",
    rate_hz=200,
    diff_integ="velocity integration",
    refusal="integration refused",
)
def _imu_int() -> Outcome:
    obs = export_case("imu.inertial", IMU_BODY, hz=200).observation
    result = imu_engine.process(
        obs,
        imu_engine.ImuSpec(initial_orientation=EYE),
        imu_engine.MethodSpec(integration="declared_initial_velocity"),
    )
    return holds("integration" in result.refusals, list(result.refusals), "integration refused")


# ---------------------------------------------------------------- COD (speed sources)


@case(
    "radar_laser",
    "radar_laser.radar",
    "cod_trace_needs_a_declared_path",
    "engine -> assessment refusal",
    rate_hz=100,
    eligibility="COD path undeclared",
    refusal="declared path required",
)
def _cod_path() -> Outcome:
    return refuses(
        lambda: cod.CodSpec.model_validate(
            {
                "declared_path": "",
                "path_length_m": 20,
                "turn_distance_m": 10,
                "turn_angle_deg": 90,
                "source": "radar",
            }
        ),
        "declared_path",
    )


def run_all() -> list[dict[str, Any]]:
    rows = []
    for c in CASES:
        try:
            out = c.run()
        except Exception as exc:  # a crash is a failed case, recorded verbatim
            out = Outcome(False, f"{type(exc).__name__}: {exc}"[:200], "no exception", None)

        def plain(value: Any, tolerance: Any = None) -> Any:
            if isinstance(value, np.floating | np.integer):
                value = float(value)
            if isinstance(value, float) and isinstance(tolerance, int | float) and tolerance > 0:
                # Report two digits beyond the tolerance so floating-point noise at the 1e-13
                # level cannot make the committed report platform dependent.
                value = round(value, min(12, max(2, math.ceil(-math.log10(tolerance)) + 2)))
            return value

        rows.append(
            {
                "family": c.family,
                "subfamily": c.subfamily,
                "case": c.name,
                "chain": c.chain,
                "axes": c.axes,
                "passed": out.passed,
                "observed": plain(out.observed, out.tolerance),
                "expected": plain(out.expected, out.tolerance),
                "tolerance": plain(out.tolerance),
                "detail": out.detail,
            }
        )
    return rows
