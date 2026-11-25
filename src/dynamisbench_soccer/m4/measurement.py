"""Replay the declared M5 measurement procedures against admitted inputs."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from dynamisbench_soccer.agreement.paired_speed import PairedInput, compare
from dynamisbench_soccer.data.manifest import DataCategory
from dynamisbench_soccer.engines import force_platform_acquisition, imu, lpt, lpt_acquisition
from dynamisbench_soccer.engines.contact import ContactPolicy, contact
from dynamisbench_soccer.engines.event_time import Estimate, EventRefusal
from dynamisbench_soccer.engines.gnss_lps import ProcessingOptions, process
from dynamisbench_soccer.engines.gnss_lps_acquisition import (
    AcquisitionDeclaration,
    AcquisitionRefusal,
)
from dynamisbench_soccer.engines.load_cell import MethodSpec as LoadCellMethod
from dynamisbench_soccer.engines.load_cell import process as process_load_cell
from dynamisbench_soccer.engines.load_cell_acquisition import CellSpec, acquire, check_saturation
from dynamisbench_soccer.engines.position_clock_bounds import input_error_radii
from dynamisbench_soccer.engines.timing_gates import GatePolicy, sprint
from dynamisbench_soccer.m4.admission import (
    CELL_TASK,
    ENCODER_TASK,
    IMU_TASK,
    LPT_TASK,
    RIG_TASK,
    Kind,
)
from dynamisbench_soccer.m4.key_store import digest
from dynamisbench_soccer.m4.task_registry import Method, RegistryTask
from dynamisbench_soccer.qc.spec import Status
from dynamisbench_soccer.signals.schema import (
    BundleMeta,
    Calibration,
    DeviceClass,
    FloatArray,
    Quantity,
    Side,
    SignalBundle,
)
from dynamisbench_soccer.statistics.longitudinal import fixed_session_change

Reason = Literal["eligible", "not_eligible", "insufficient_evidence", "unsupported_claim"]
ENGINE = "m5-measurement.v1"
GRADER = "measurement.v1"
ANSWER_VERSION = "m5.measurement-answer.v1"
IMU_POSITION = "fixed-attitude-sensor-position"
IMU_SIGNATURE = "specific-force-signature-flight"
IMU_OPERATIONS = (IMU_POSITION, IMU_SIGNATURE)
ENCODER_OPERATION = "modular-attachment-change"
LPT_OPERATION = "fixed-angle-attachment-speed"
RIG_OPERATION = "paired-net-peak-difference"
RIG_SPECIFICATION = {
    "operation": "signed_difference_of_separately_baseline_corrected_peaks",
    "diagnostic": "individual_unfiltered_100ms_RFD",
}
CELL_OPERATION = "unfiltered-net-peak-torque"
CELL_SPECIFICATION = {
    "operation": "unfiltered_net_peak_force_times_perpendicular_moment_arm",
    "onset": "absolute_N_one_sample_backdate",
    "window_ms": 100,
    "impulse": "baseline_corrected_trapezoid",
}
CELL_ATTACHMENT = "Perpendicular fixed knee centre to distal shank strap"


@dataclass(frozen=True)
class Procedure:
    source: str
    kind: Kind
    estimand: str
    dimension: str
    unit: str
    claims: tuple[str, ...]
    dependencies: tuple[str, ...]


PROCEDURES = {
    "beam-split": Procedure(
        "timing_contact.gates",
        "protocol_assessment",
        "beam_crossing_segment_average_velocity",
        "speed",
        "m/s",
        ("DEV-004",),
        ("engines/timing_gates.py::sprint",),
    ),
    "switch-flight-height": Procedure(
        "timing_contact.contact",
        "protocol_assessment",
        "contact_flight_time_height",
        "length",
        "m",
        ("DEV-003",),
        ("engines/contact.py::contact",),
    ),
    "enu-positional-speed": Procedure(
        "gnss_lps.gnss",
        "native_signal",
        "horizontal_positional_speed",
        "speed",
        "m/s",
        ("DEV-005", "CX-003"),
        ("engines/gnss_lps.py::process", "engines/position_clock_bounds.py::input_error_radii"),
    ),
    "lps-filter-sensitivity": Procedure(
        "gnss_lps.lps",
        "native_signal",
        "horizontal_positional_acceleration_filter_delta",
        "acceleration",
        "m/s^2",
        ("DEV-005", "CX-003"),
        ("engines/gnss_lps.py::process", "engines/position_clock_bounds.py::input_error_radii"),
    ),
    "paired-speed-bias": Procedure(
        "gnss_lps.lps",
        "cross_device",
        "horizontal_tag_speed_paired_bias",
        "speed",
        "m/s",
        ("DEV-005", "DEV-008", "ST-004"),
        ("agreement/paired_speed.py::compare", "engines/gnss_lps.py::process"),
    ),
    IMU_POSITION: Procedure(
        "imu.inertial",
        "protocol_assessment",
        "field_x_instrument_point_position",
        "length",
        "m",
        ("DEV-002",),
        ("engines/imu.py::process", "m4/measurement.py::_imu_position"),
    ),
    IMU_SIGNATURE: Procedure(
        "imu.inertial",
        "protocol_assessment",
        "sensor_threshold_signature_duration",
        "time",
        "s",
        ("DEV-002",),
        ("engines/imu.py::process", "m4/measurement.py::_imu_signature"),
    ),
    ENCODER_OPERATION: Procedure(
        "lpt_encoder.encoder",
        "native_signal",
        "signed_vertical_attachment_change",
        "length",
        "m",
        ("DEV-001",),
        ("engines/lpt_acquisition.py::acquire", "m4/measurement.py::_encoder"),
    ),
    LPT_OPERATION: Procedure(
        "lpt_encoder.lpt",
        "native_signal",
        "bar_vertical_attachment_speed",
        "speed",
        "m/s",
        ("DEV-001",),
        ("engines/lpt_acquisition.py::acquire", "m4/measurement.py::_lpt"),
    ),
    RIG_OPERATION: Procedure(
        "load_cell_isometric.isometric_rig",
        "rig_plate_pair",
        "rig_minus_plate_net_peak_force",
        "force",
        "N",
        ("DEV-006",),
        (
            "engines/load_cell_acquisition.py::acquire",
            "engines/force_platform_acquisition.py::acquire",
            "m4/measurement.py::_rig_pair",
        ),
    ),
    CELL_OPERATION: Procedure(
        "load_cell_isometric.load_cell",
        "protocol_assessment",
        "peak_external_knee_torque_above_baseline",
        "torque",
        "N*m",
        ("DEV-006",),
        ("engines/load_cell.py::process", "m4/measurement.py::_load_cell"),
    ),
    "fixed-session-change": Procedure(
        "gnss_lps.lps",
        "statistical",
        "horizontal_tag_speed_session_mean_change",
        "speed",
        "m/s",
        ("ST-001", "ST-005", "ST-006"),
        (
            "statistics/longitudinal.py::fixed_session_change",
            "statistics/reliability.py::paired_reliability",
            "statistics/change.py::individual_change",
        ),
    ),
}


@dataclass(frozen=True)
class Reading:
    reason: Reason
    claims: tuple[str, ...]
    value: float | None = None
    radius: float | None = None
    operands: tuple[tuple[str, float], ...] = ()
    selected_trials: tuple[str, ...] = ()


def _estimate(estimate: Estimate, claims: tuple[str, ...], **operands: float) -> Reading:
    return Reading(
        "eligible",
        claims,
        estimate.value,
        max(estimate.value - estimate.low, estimate.high - estimate.value),
        tuple((field, round(value, 12)) for field, value in operands.items()),
    )


def _timing(operation: str, bundle: SignalBundle, policy: object, request: str) -> Reading:
    try:
        if operation == "beam-split":
            gate_policy = GatePolicy.model_validate(policy)
            result = sprint((bundle,), gate_policy)
        else:
            contact_policy = ContactPolicy.model_validate(policy)
            contact_result = contact(bundle, contact_policy)
    except EventRefusal as exc:
        for prefix, claim in (
            ("missing gate ", "missing_station"),
            ("gate 2: duplicate or ambiguous beam trigger", "duplicate_beam"),
            ("unexpected gate ID or position at ", "course_mismatch"),
            ("missing touchdown for the final flight", "missing_touchdown"),
            ("flight duration outside declared protocol limits", "flight_outside_protocol"),
        ):
            if str(exc).startswith(prefix):
                return Reading("not_eligible", (claim,))
        raise ValueError("event refusal outside the reviewed case classes") from exc

    if request == "equivalence":
        return Reading("unsupported_claim", ("different_estimands", "paired_validation_absent"))
    if operation == "beam-split":
        first, last = result.splits
        if last.split_s is None or last.segment_average_velocity_m_s is None:
            raise ValueError("reference requires a resolved segment split")
        return _estimate(
            last.segment_average_velocity_m_s,
            (
                "beam_crossing",
                "segment_average",
                f"start_{result.start_mode}",
                f"{gate_policy.gates[-1].beam_type}_beam",
            ),
            segment_distance_m=last.position_m - first.position_m,
            split_s=last.split_s.value,
            cumulative_s=last.cumulative_s.value,
        )

    flight = contact_result.flights[0]
    if flight.flight_time_height_m is None:
        return Reading("insufficient_evidence", ("equal_com_height_unestablished",))
    if flight.ground_contact_s is None:
        raise ValueError("reference requires a measured preceding touchdown")
    claims: tuple[str, ...] = (
        "switch_events",
        "ballistic_equal_com_height",
        contact_result.trigger_definition.split(":")[1],
    )
    if any(discard.reason == "contact switching bounce" for discard in contact_result.discarded):
        claims += ("bounce_removed",)
    return _estimate(
        flight.flight_time_height_m,
        claims,
        takeoff_s=flight.takeoff.time_s.value,
        touchdown_s=flight.touchdown.time_s.value,
        ground_contact_s=flight.ground_contact_s.value,
        flight_s=flight.flight_s.value,
    )


def _position(
    operation: str, bundle: SignalBundle, data: dict[str, object], request: str
) -> Reading:
    lps = operation == "lps-filter-sensitivity"
    if bundle.meta.device.device_class != (DeviceClass.LPS if lps else DeviceClass.GNSS):
        raise ValueError("position replay source type differs from the declared method")
    row, bound = data["row"], data["horizontal_error_m"]
    if type(row) is not int or isinstance(bound, bool) or not isinstance(bound, (float, int)):
        raise ValueError("invalid reference row or uncertainty")
    support = 4 if lps else 1
    if not support <= row < bundle.meta.n_rows - support:
        raise ValueError("reference row lacks interior stencil support")
    try:
        declaration = AcquisitionDeclaration.model_validate(data["declaration"])
        native = process(bundle, declaration, ProcessingOptions(thresholds=()))
    except AcquisitionRefusal as exc:
        message = str(exc)
        if message in {
            "vendor-processed or resampled samples cannot enter as native",
            "observed or declared rate contradicts native acquisition rate",
        }:
            finding = "native_rate_unverified"
        elif message in {
            "WGS84 fixes and an explicit ENU origin are required",
            "LPS positions require a declared Z-up field frame",
        }:
            finding = "declared_geometry_required"
        elif message == "GNSS multipath flagged" or message.startswith(
            "LPS anchor occlusion or invalid visibility"
        ):
            finding = "failed_signal_qc"
        else:
            raise ValueError("acquisition refusal outside the reviewed case classes") from exc
        return Reading("not_eligible", (finding,))
    if request == "equivalence":
        finding = "filter_sensitivity_not_accuracy" if lps else "doppler_separate_estimand"
        return Reading("unsupported_claim", (finding, "synthetic_evidence_only"))

    clock = native.acquisition.clock
    grid = np.diff(clock.elapsed_s)
    if (
        declaration.native_hz != 25
        or clock.alignment_uncertainty_s
        or declaration.latency_uncertainty_s
        or not np.allclose(grid, grid[0], rtol=0, atol=1e-12)
    ):
        raise ValueError("measurement reference requires a uniform aligned grid with exact timing")
    radii = input_error_radii(native, bound, 0)
    fields: tuple[str, ...]
    if lps:
        alternate = process(
            bundle, declaration, ProcessingOptions(position_filter="binomial5", thresholds=())
        )
        alternate_radii = input_error_radii(alternate, bound, 0)
        value = (
            alternate.kinematics.horizontal_positional_acceleration_m_s2[row]
            - native.kinematics.horizontal_positional_acceleration_m_s2[row]
        )
        radius = (
            radii.horizontal_positional_acceleration_m_s2
            + alternate_radii.horizontal_positional_acceleration_m_s2
        )
        claims = (
            "tag_horizontal_acceleration",
            "filter_sensitivity_not_accuracy",
            "fixed_grid_bound",
        )
        fields = ("time", "position_x", "position_y")
    else:
        value = native.kinematics.horizontal_positional_speed_m_s[row]
        radius = radii.horizontal_positional_speed_m_s
        claims = (
            "antenna_horizontal_positional_speed",
            "doppler_separate_estimand",
            "fixed_grid_bound",
        )
        fields = ("time", "latitude", "longitude", "height")
    operands = tuple(
        (f"{name}.{idx}", float(bundle[name][idx]))
        for name in fields
        for idx in range(row - support, row + support + 1)
    )
    return Reading(
        "eligible", claims, float(value), radius, (*operands, ("horizontal_error_m", bound))
    )


def _paired(data: dict[str, object]) -> Reading:
    result = compare(PairedInput.model_validate_json(json.dumps(data)))
    classification = result.classification
    if classification == "definition_mismatch":
        return Reading("unsupported_claim", (classification, "hardware_interchangeability_unknown"))
    if classification == "repeat_sessions_required":
        return Reading("insufficient_evidence", (classification, "paired_devices_not_retest"))
    if classification in {"independent_pairs_required", "failed_signal_qc"}:
        return Reading("not_eligible", (classification,))
    agreement = result.agreement
    if agreement is None or result.expanded_uncertainty is None:
        raise ValueError("paired computation lacks statistics or uncertainty")
    evidence = [
        (f"{method}.{trial}", round(value, 10))
        for trial, radar, lps in result.pairs
        for method, value in (("radar", radar), ("lps", lps))
    ]
    lower, upper = agreement.limits
    evidence.extend(
        (
            ("n_pairs", float(agreement.n_pairs)),
            ("sd_difference", round(agreement.sd_difference, 10)),
            ("lower_limit", round(lower, 10)),
            ("upper_limit", round(upper, 10)),
        )
    )
    return Reading(
        "eligible",
        (
            classification,
            "hardware_interchangeability_unknown",
            "matched_horizontal_tag_speed",
            "limits_within_tolerance" if result.within_tolerance else "limits_exceed_tolerance",
            "descriptive_limits_not_reliability",
        ),
        agreement.bias,
        result.expanded_uncertainty,
        tuple(evidence),
        tuple(trial for trial, _, _ in result.pairs),
    )


def _longitudinal(data: dict[str, object]) -> Reading:
    study = fixed_session_change(data)
    if isinstance(study, str):
        reason: Reason = (
            "insufficient_evidence"
            if study in {"baseline_design_incomplete", "cohort_error_unavailable"}
            else "not_eligible"
        )
        return Reading(reason, (study,))
    change = study.change
    return Reading(
        "eligible",
        (
            "conditional_trial_model",
            "te_estimation_uncertainty_excluded",
            change.classification,
            "point_reaches_swc" if study.reaches_swc else "point_below_swc",
        ),
        change.change,
        study.half_width,
        (
            ("typical_error", study.reliability.typical_error),
            ("swc", study.reliability.swc),
            ("baseline_mean", change.baseline_mean),
            ("current_mean", change.current_mean),
            ("change_se", change.change_se),
            ("interval_lower", change.interval[0]),
            ("interval_upper", change.interval[1]),
        ),
    )


class LoadCellSetup(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    adc_rails_v: tuple[float, float]
    analysis_calibration: Calibration | None = None
    baseline_samples: int = Field(ge=2)
    eligible_knee_angle_deg: tuple[float, float]
    lever_error_m: float = Field(ge=0)
    net_force_error_n: float = Field(ge=0)
    onset_threshold_n: float = Field(gt=0)
    required_side: Literal["right"]
    window_samples: Literal[100]


def _load_cell(data: dict[str, object]) -> Reading:
    meta = BundleMeta.model_validate(data["meta"])
    setup = LoadCellSetup.model_validate(data["setup"])
    acquisition, sampling = meta.acquisition, meta.sampling
    if (
        meta.data_category is not DataCategory.SYNTHETIC
        or meta.device.device_class is not DeviceClass.LOAD_CELL
        or acquisition is None
        or len(acquisition.sources) != 1
        or len(meta.channels) != 1
    ):
        raise ValueError("reference requires one native synthetic load-cell channel")
    (source,) = acquisition.sources
    (channel,) = meta.channels
    if (
        source.device.device_class is not DeviceClass.LOAD_CELL
        or source.channels != [channel.name]
        or channel.quantity is not Quantity.VOLTAGE
        or source.raw_format != "Prescribed native synthetic array on the uniform shared counter"
    ):
        raise ValueError("reference requires native voltage observations")
    if (
        sampling is None
        or sampling.nominal_hz != 1000
        or sampling.measured_hz != 1000
        or sampling.irregular
        or len(acquisition.clocks) != 1
        or source.clock != acquisition.reference_clock
    ):
        raise ValueError("reference requires exact native 1000 Hz on one counter")

    samples = data["samples"]
    if not isinstance(samples, list) or any(type(x) not in (int, float) for x in samples):
        raise ValueError("expected numeric ADC samples")
    bundle = SignalBundle(meta, {channel.name: np.asarray(samples, dtype=np.float64)})
    calibration = setup.analysis_calibration
    if calibration is None:
        if len(source.calibration) > 1:
            raise ValueError("reference requires an unambiguous source calibration")
        calibration = next(iter(source.calibration), None)
    if calibration is None or calibration.gain is None or calibration.offset is None:
        return Reading("not_eligible", ("missing_calibration",))
    if (
        (calibration.input_unit, calibration.output_unit) not in {("V", "N"), ("N", "V")}
        or calibration.channel not in (None, channel.name)
        or calibration.gain == 0
    ):
        return Reading("not_eligible", ("calibration_unit_mismatch",))

    geometry = source.geometry
    cell = CellSpec(
        calibration=setup.analysis_calibration, raw_range=setup.adc_rails_v, rail_margin_fraction=0
    )
    native = acquire(bundle, cell)
    if any(check.status is Status.FAIL for check in check_saturation(native)):
        return Reading("not_eligible", ("adc_saturation",))
    if channel.side is not Side.RIGHT:
        return Reading("not_eligible", ("wrong_limb",))
    angle = geometry.knee_angle_deg
    lo, hi = setup.eligible_knee_angle_deg
    if (
        meta.protocol is None
        or meta.protocol.protocol_id != "res704.fixed-isometric-knee-flexion.v1"
        or channel.axis != "x"
        or channel.frame != "cell_axis"
        or geometry.rig_compliance_m_per_n != 0
        or angle is None
        or not lo <= angle <= hi
    ):
        return Reading("not_eligible", ("protocol_geometry_ineligible",))
    lever = geometry.lever_arm_m
    if (
        lever is None
        or not geometry.attachment_point
        or {"lever_arm_m", "attachment_point"} & set(geometry.unknown)
    ):
        return Reading("insufficient_evidence", ("moment_arm_unestablished",))
    if geometry.attachment_point != CELL_ATTACHMENT:
        return Reading("not_eligible", ("protocol_geometry_ineligible",))
    if setup.baseline_samples >= meta.n_rows:
        return Reading("not_eligible", ("no_resolved_window",))

    result = process_load_cell(
        native,
        method=LoadCellMethod(
            onset="absolute_N",
            onset_absolute_n=setup.onset_threshold_n,
            baseline_window_s=(sampling.t0_s, sampling.t0_s + setup.baseline_samples / 1000),
            windows_ms=(100,),
        ),
    )
    operands = {
        "baseline_n": result.metrics["baseline_n"],
        "gross_peak_n": result.metrics["peak_force_n"],
        "net_peak_n": result.metrics["net_peak_force_n"],
        "onset_s": result.metrics["onset_time_s"],
        "rfd_100ms_n_s": result.metrics["rfd_0_100ms_n_s"],
        "impulse_100ms_n_s": result.metrics["net_impulse_0_100ms_n_s"],
        "lever_arm_m": lever,
    }
    if operands["rfd_100ms_n_s"] is None or operands["impulse_100ms_n_s"] is None:
        return Reading("not_eligible", ("no_resolved_window",))
    torque, net = result.metrics["net_peak_torque_n_m"], operands["net_peak_n"]
    if result.verdict == "reject" or torque is None or net is None or None in operands.values():
        raise ValueError("ineligible load-cell reconstruction")
    force_error, lever_error = setup.net_force_error_n, setup.lever_error_m
    radius = max(
        abs(force * arm - torque)
        for force in (net - force_error, net + force_error)
        for arm in (lever - lever_error, lever + lever_error)
    )
    return Reading(
        "eligible",
        ("calibrated_cell_force", "baseline_net_force", "perpendicular_external_joint_torque"),
        torque,
        radius,
        tuple((key, round(value, 10)) for key, value in operands.items() if value is not None),
    )


def _imu_position(spec: imu.ImuSpec, native: imu.Processed, request: str) -> Reading:
    if request == "com_position":
        return Reading("unsupported_claim", ("instrument_point_not_com",))
    if spec.initial_orientation is None:
        return Reading(
            "insufficient_evidence", ("initial_attitude_unestablished", "gravity_norm_not_static")
        )
    if spec.initial_velocity_m_s is None:
        return Reading("insufficient_evidence", ("initial_velocity_unestablished",))
    if spec.initial_position_m is None:
        return Reading("insufficient_evidence", ("initial_position_unestablished",))

    gyro = float(np.linalg.norm(native.angular_velocity_rad_s, axis=1).max())
    acceleration = native.free_acceleration_m_s2
    if (
        gyro > 1e-12
        or acceleration is None
        or not np.allclose(acceleration, acceleration[0], atol=1e-12, rtol=0)
    ):
        return Reading("not_eligible", ("fixed_attitude_model_unavailable",))
    position, velocity = native.position_m, native.velocity_m_s
    radius = native.uncertainty["position_error_bound_m"]
    if position is None or velocity is None or not isinstance(radius, float):
        raise ValueError("declared integration did not produce instrument-point position")
    ax, ay, az = map(float, acceleration[0])
    return Reading(
        "eligible",
        ("fixed_attitude", "calibrated_specific_force", "gravity_removed", "instrument_point_only"),
        float(position[-1, 0]),
        radius,
        (
            ("free_ax_m_s2", ax),
            ("free_ay_m_s2", ay),
            ("free_az_m_s2", az),
            ("terminal_vx_m_s", float(velocity[-1, 0])),
            ("elapsed_s", float(native.time_s[-1] - native.time_s[0])),
            ("peak_corrected_gyro_rad_s", gyro),
        ),
    )


def _imu_signature(native: imu.Processed, threshold: float) -> Reading:
    low = np.flatnonzero(np.linalg.norm(native.specific_force_m_s2, axis=1) < threshold)
    time = native.time_s
    if (
        len(native.flights) != 1
        or not len(low)
        or low[0] == 0
        or low[-1] == len(time) - 1
        or np.any(np.diff(low) != 1)
    ):
        return Reading("not_eligible", ("incomplete_signature",))
    (event,) = native.flights
    start, end = int(low[0]), int(low[-1])
    if (event.takeoff_s, event.landing_s) != (time[start - 1], time[end + 1]):
        raise ValueError("native signature disagrees with its complete edge brackets")
    return Reading(
        "eligible",
        (
            "calibrated_specific_force",
            "bracketed_threshold_signature",
            "force_platform_events_unestablished",
            "whole_body_flight_unestablished",
        ),
        event.duration_s,
        float(time[start] - time[start - 1] + time[end + 1] - time[end]),
        (
            ("takeoff_edge_s", event.takeoff_s),
            ("landing_edge_s", event.landing_s),
            ("threshold_m_s2", threshold),
        ),
    )


def _imu(operation: str, data: dict[str, object]) -> Reading:
    signature = operation == IMU_SIGNATURE
    request = data["request"]
    if request not in (
        ("signature_flight_time",) if signature else ("field_x_position", "com_position")
    ):
        raise ValueError("IMU artifact request disagrees with checkpoint method")

    meta = BundleMeta.model_validate(data["meta"])
    acquisition, sampling = meta.acquisition, meta.sampling
    if (
        meta.data_category is not DataCategory.SYNTHETIC
        or meta.device.device_class is not DeviceClass.IMU
        or acquisition is None
        or len(acquisition.sources) != 1
        or sampling is None
    ):
        raise ValueError("reference requires one native synthetic IMU")
    (source,) = acquisition.sources
    raw = data["channels"]
    axes = {
        f"{quantity}_{axis}": axis
        for quantity in ("specific_force", "angular_velocity")
        for axis in "xyz"
    }
    names = {"time", *axes}
    if (
        source.device.device_class is not DeviceClass.IMU
        or source.raw_format != "generated native sample values for independent RES-704 control"
        or not isinstance(raw, dict)
        or raw.keys() != names
        or {channel.name for channel in meta.channels} != names
        or meta.channel("time").quantity is not Quantity.TIME
        or any(
            meta.channel(name).axis != axis or meta.channel(name).frame != "sensor"
            for name, axis in axes.items()
        )
    ):
        raise ValueError("reference requires native timestamps and canonical sensor axes")
    if {source.source_sha256, meta.provenance.source_sha256} != {
        digest((json.dumps(raw, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())
    }:
        raise ValueError("IMU source channels disagree with provenance digests")
    bundle = SignalBundle(
        meta,
        {
            channel.name: np.asarray(raw[channel.name], dtype=np.float64)
            for channel in meta.channels
        },
    )
    if (
        sampling.nominal_hz != 20
        or sampling.measured_hz != 20
        or sampling.irregular
        or len(acquisition.clocks) != 2
        or len(acquisition.alignments) != 1
    ):
        return Reading("not_eligible", ("native_clock_gap",))
    (alignment,) = acquisition.alignments
    if (
        alignment.clock != source.clock
        or alignment.offset_s != 0
        or alignment.drift_ppm != 0
        or alignment.uncertainty_s > 1e-12
        or len(alignment.sync_events) != 2
        or any(left != right for left, right in alignment.sync_events)
        or not alignment.sync_events[0][0]
        <= bundle["time"][0]
        <= bundle["time"][-1]
        <= alignment.sync_events[-1][0]
        or alignment.sync_events[0][0] >= alignment.sync_events[-1][0]
    ):
        return Reading("not_eligible", ("native_clock_gap",))

    try:
        spec = imu.ImuSpec.model_validate(data["spec"])
    except ValueError:
        return Reading("not_eligible", ("invalid_calibration",))
    if (
        not spec.calibration_declared
        or spec.accel_range_m_s2 is None
        or spec.gyro_range_rad_s is None
        or not {"accel_scale", "accel_bias_m_s2", "accel_misalignment", "gyro_bias_rad_s"}
        <= spec.model_fields_set
        or len(source.calibration) != 1
        or (not signature and data["gravity_m_s2"] != imu.GRAVITY_M_S2)
    ):
        return Reading("not_eligible", ("invalid_calibration",))
    if (
        spec.placement != "rigid tag on moving platform"
        or spec.placement != source.geometry.sensor_placement
        or spec.placement != meta.geometry.sensor_placement
        or spec.mounting_rotation is not None
        or spec.sensor_offset_m is not None
        or {frame.id: (frame.kind, frame.parent) for frame in acquisition.frames}
        != {"field": ("field", None), "sensor": ("sensor", "field")}
        or any(frame.rotation_to_parent is not None for frame in acquisition.frames)
        or spec.static_window_s is not None
    ):
        return Reading("not_eligible", ("mounting_or_frame_unestablished",))

    if spec.sync_uncertainty_s > 1e-12 or not np.allclose(
        bundle["time"], sampling.t0_s + np.arange(meta.n_rows) / 20, atol=1e-12, rtol=0
    ):
        return Reading("not_eligible", ("native_clock_gap",))
    method = imu.MethodSpec.model_validate(
        {
            "integration": "none" if signature else "declared_initial_velocity",
            "flight_threshold_m_s2": data["flight_threshold_m_s2"] if signature else 3.0,
            "flight_min_s": data["flight_min_s"] if signature else 0.1,
            "gap_factor": data["flight_max_gap_factor"] if signature else 1.5,
            "sensitivity_accel_bias_m_s2": data["residual_field_bias_bound_m_s2"],
        }
    )
    engine_spec = spec.model_copy(update={"initial_orientation": None}) if signature else spec
    native = imu.process(bundle, engine_spec, method)
    if native.verdict == "reject":
        if any(
            check.rule_id == "IMU-SAT-ACCE" and check.status is Status.FAIL for check in native.qc
        ):
            return Reading("not_eligible", ("accelerometer_rail_clipping",))
        if any(
            check.rule_id == "IMU-SAT-GYRO" and check.status is Status.FAIL for check in native.qc
        ):
            return Reading("not_eligible", ("gyroscope_rail_clipping",))
        raise ValueError("native IMU quality rejected outside the reviewed fault classes")
    if signature:
        return _imu_signature(native, method.flight_threshold_m_s2)
    assert isinstance(request, str)
    return _imu_position(spec, native, request)


def _lpt(data: dict[str, object]) -> Reading:
    meta = BundleMeta.model_validate(data["meta"])
    acquisition, sampling = meta.acquisition, meta.sampling
    cable = lpt_acquisition.CableSpec.model_validate(data["cable"])
    geometry = lpt_acquisition.GeometrySpec.model_validate(data["geometry"])
    raw = data["channels"]
    names = {
        "time": Quantity.TIME,
        "cable_length": Quantity.DISTANCE,
        "slack": Quantity.DIMENSIONLESS,
    }
    if (
        meta.data_category is not DataCategory.SYNTHETIC
        or meta.device.device_class is not DeviceClass.LPT
        or acquisition is None
        or len(acquisition.sources) != 1
        or sampling is None
        or not isinstance(raw, dict)
        or raw.keys() != names.keys()
        or {channel.name: channel.quantity for channel in meta.channels} != names
    ):
        raise ValueError("reference requires canonical native synthetic LPT distance")
    (source,) = acquisition.sources
    if (
        source.device.device_class is not DeviceClass.LPT
        or source.id != cable.source
        or cable.channel != "cable_length"
        or cable.slack_channel != "slack"
        or source.raw_format != "generated native sample values for independent RES-704 control"
        or source.geometry != meta.geometry
        or source.geometry.attachment_point != "barbell"
        or geometry.attachment_point != "barbell"
        or geometry.vertical_line_xy_m is not None
    ):
        raise ValueError("reference requires one declared barbell cable with its source slack")
    if (
        len(source.calibration) != 1
        or source.calibration != meta.calibration
        or (
            source.calibration[0].channel,
            source.calibration[0].input_unit,
            source.calibration[0].output_unit,
            source.calibration[0].gain,
            source.calibration[0].offset,
        )
        != ("cable_length", "m", "m", 1, 0)
    ):
        raise ValueError("reference requires declared metre-to-metre identity calibration")
    raw_sha = digest((json.dumps(raw, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())
    if source.source_sha256 != raw_sha or meta.provenance.source_sha256 != raw_sha:
        raise ValueError("native cable channels disagree with source provenance")
    bundle = SignalBundle(
        meta,
        {
            channel.name: np.asarray(raw[channel.name], dtype=np.float64)
            for channel in meta.channels
        },
    )
    time = bundle["time"]
    if np.any(np.diff(time) <= 0):
        return Reading("not_eligible", ("nonincreasing_native_clock",))
    if (
        sampling.nominal_hz != 50
        or sampling.measured_hz != 50
        or sampling.irregular
        or not np.allclose(time, sampling.t0_s + np.arange(meta.n_rows) / 50, atol=1e-12, rtol=0)
    ):
        return Reading("not_eligible", ("unresolved_native_grid",))

    request = data["request"]
    if request == "force_plate_power":
        return Reading(
            "unsupported_claim", ("mass_model_distinct_from_plate_power", "com_unobserved")
        )
    if request != "vertical_speed":
        raise ValueError("unsupported LPT request")
    native = lpt_acquisition.acquire(bundle, (cable,), geometry)
    (instrument,) = native.cables
    if instrument.faults:
        claim = "cable_slack" if "slack" in instrument.faults else "cable_integrity_failed"
        return Reading("not_eligible", (claim,))
    vertical = native.vertical_displacement_m
    if vertical is None:
        return Reading("insufficient_evidence", ("cable_only", "vertical_geometry_unobserved"))

    power = lpt.PowerSpec.model_validate(data["power"])
    angle = source.geometry.cable_angle_rad
    row, da, ppm = (data["row"], data["angle_bound_rad"], data["clock_scale_bound_ppm"])
    if (
        angle is None
        or type(row) is not int
        or not 1 <= row < len(time) - 1
        or not isinstance(da, (int, float))
        or not (0 <= da <= angle and angle + da < math.acos(1e-3))
        or not isinstance(ppm, (int, float))
        or not 0 <= ppm < 1e6
    ):
        raise ValueError("invalid native stencil or nonsingular angle/clock bounds")
    span = float(native.time_s[row + 1] - native.time_s[row - 1])
    cable_delta = float(instrument.length_m[row + 1] - instrument.length_m[row - 1])
    cable_speed = cable_delta / span
    speed = float(vertical[row + 1] - vertical[row - 1]) / span
    acceleration = (
        float(vertical[row + 1] - 2 * vertical[row] + vertical[row - 1]) / (span / 2) ** 2
    )
    radius = max(
        abs(cable_speed / (math.cos(angle + a) * (1 + scale * 1e-6)) - speed)
        for a in (-da, da)
        for scale in (-ppm, ppm)
    )
    return Reading(
        "eligible",
        ("fixed_projection", "mass_model_distinct_from_plate_power", "attachment_point_only"),
        speed,
        radius,
        (
            ("cable_speed_m_s", round(cable_speed, 10)),
            ("vertical_displacement_m", round(float(vertical[row]), 10)),
            ("vertical_acceleration_m_s2", round(acceleration, 10)),
            (
                "vertical_mass_model_power_w",
                round(power.system_mass_kg * (lpt.GRAVITY_M_S2 + acceleration) * speed, 10),
            ),
        ),
    )


def _encoder(data: dict[str, object]) -> Reading:
    meta = BundleMeta.model_validate(data["meta"])
    acquisition, sampling = meta.acquisition, meta.sampling
    cable = lpt_acquisition.CableSpec.model_validate(data["cable"])
    geometry = lpt_acquisition.GeometrySpec.model_validate(data["geometry"])
    raw = data["channels"]
    names = {"time": Quantity.TIME, "counts": Quantity.COUNT}
    if (
        meta.data_category is not DataCategory.SYNTHETIC
        or meta.device.device_class is not DeviceClass.ENCODER
        or acquisition is None
        or len(acquisition.sources) != 1
        or sampling is None
        or not isinstance(raw, dict)
        or raw.keys() != names.keys()
        or {column.name: column.quantity for column in meta.channels} != names
    ):
        raise ValueError("reference requires one native synthetic encoder with counts and time")
    (source,) = acquisition.sources
    if (
        source.device.device_class is not DeviceClass.ENCODER
        or source.id != cable.source
        or cable.channel != "counts"
        or cable.count_modulus is None
        or cable.slack_channel is not None
        or cable.initial_length_m is not None
        or source.raw_format != "generated native sample values for independent RES-704 control"
        or source.geometry != meta.geometry
        or source.geometry.attachment_point != "barbell"
        or geometry.attachment_point != "barbell"
        or geometry.vertical_line_xy_m is not None
    ):
        raise ValueError("reference requires a declared modular counter on one bar attachment")
    if (
        len(source.calibration) != 1
        or source.calibration != meta.calibration
        or (
            source.calibration[0].channel,
            source.calibration[0].input_unit,
            source.calibration[0].output_unit,
            source.calibration[0].gain,
            source.calibration[0].offset,
        )
        != ("counts", "1", "1", 1, 0)
    ):
        raise ValueError("reference requires declared native count identity calibration")
    raw_sha = digest((json.dumps(raw, indent=2, sort_keys=True, allow_nan=False) + "\n").encode())
    if source.source_sha256 != raw_sha or meta.provenance.source_sha256 != raw_sha:
        raise ValueError("native encoder channels disagree with source provenance")
    counts = raw["counts"]
    if not isinstance(counts, list) or any(type(count) is not int for count in counts):
        raise ValueError("native encoder counts must be JSON integers")
    bundle = SignalBundle(
        meta,
        {column.name: np.asarray(raw[column.name], dtype=np.float64) for column in meta.channels},
    )
    if (
        sampling.nominal_hz != 50
        or sampling.measured_hz != 50
        or sampling.irregular
        or not np.allclose(
            bundle["time"], sampling.t0_s + np.arange(meta.n_rows) / 50, atol=1e-12, rtol=0
        )
    ):
        return Reading("not_eligible", ("unresolved_native_grid",))

    request = data["request"]
    if request == "com_change":
        return Reading("unsupported_claim", ("attachment_point_only", "com_unobserved"))
    if request != "vertical_change":
        raise ValueError("unsupported encoder request")
    try:
        native = lpt_acquisition.acquire(bundle, (cable,), geometry)
    except ValueError as exc:
        if str(exc) == "missing encoder counts_per_m":
            return Reading("insufficient_evidence", ("missing_count_scale",))
        if str(exc) == "encoder rollover is ambiguous at the declared speed and sample intervals":
            return Reading("not_eligible", ("ambiguous_rollover",))
        raise

    (instrument,) = native.cables
    if instrument.faults:
        return Reading("not_eligible", ("implausible_count_jump",))
    vertical = native.vertical_displacement_m
    if vertical is None:
        return Reading("insufficient_evidence", ("cable_only", "vertical_geometry_unobserved"))

    resolution, angle = instrument.resolution_m, source.geometry.cable_angle_rad
    assert resolution is not None and angle is not None
    elapsed = float(native.time_s[-1] - native.time_s[0])
    cable_change = float(instrument.length_m[-1] - instrument.length_m[0])
    change = float(vertical[-1])
    # Native unwrapping produced exact integers before converting them to metres.
    count_change = round(cable_change / resolution)
    return Reading(
        "eligible",
        ("signed_native_rollover", "attachment_point_only", "endpoint_quantization_bound"),
        change,
        resolution / math.cos(angle),
        (
            ("signed_count_change", float(count_change)),
            ("elapsed_s", round(elapsed, 10)),
            ("cable_displacement_m", round(cable_change, 10)),
            ("mean_vertical_velocity_m_s", round(change / elapsed, 10)),
        ),
    )


class RigPairSetup(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    baseline_samples: int = Field(ge=2)
    onset_threshold_n: float = Field(gt=0)
    window_samples: Literal[100]
    load_transfer: str
    net_peak_error_n: tuple[float, float]
    peak_difference_limit_n: float = Field(gt=0)


def _rig_bundle(data: object, device: DeviceClass) -> SignalBundle:
    if not isinstance(data, dict):
        raise ValueError("expected an instrument record")
    meta = BundleMeta.model_validate(data["meta"])
    samples = data["samples"]
    acquisition, sampling = meta.acquisition, meta.sampling
    if (
        meta.data_category is not DataCategory.SYNTHETIC
        or meta.device.device_class is not device
        or acquisition is None
        or sampling is None
        or len(acquisition.sources) != 1
        or len(meta.channels) != 1
        or meta.channels[0].name != "force"
        or meta.channels[0].quantity is not Quantity.FORCE
        or meta.channels[0].plate != (1 if device is DeviceClass.FORCE_PLATE else None)
        or not isinstance(samples, list)
        or any(type(value) not in (int, float) for value in samples)
    ):
        raise ValueError("expected direct synthetic single-channel newton acquisition")
    (source,) = acquisition.sources
    if (
        source.device.device_class is not device
        or source.geometry != meta.geometry
        or source.raw_format != "Prescribed native synthetic array on the uniform shared counter"
        or source.calibration
        != [
            Calibration(
                date="2000-01-01",
                method="Prescribed direct synthetic force, no electrical correction",
            )
        ]
        or meta.calibration
    ):
        raise ValueError("unexpected synthetic arm source or calibration")
    sha = digest((json.dumps(samples, separators=(",", ":"), allow_nan=False) + "\n").encode())
    if source.source_sha256 != sha or meta.provenance.source_sha256 != sha:
        raise ValueError("paired force arm disagrees with its source SHA-256")
    return SignalBundle(meta, {"force": np.asarray(samples, dtype=np.float64)})


def _rig_clock(meta: BundleMeta) -> tuple[str, str] | None:
    acquisition, sampling = meta.acquisition, meta.sampling
    assert acquisition is not None and sampling is not None
    if (
        len(acquisition.clocks) != 1
        or sampling.nominal_hz != 1000
        or sampling.measured_hz != 1000
        or sampling.irregular
    ):
        return None
    (clock,) = acquisition.clocks
    if (
        clock.source != "host"
        or clock.epoch.strip().lower() in {"", "unknown", "unrecorded", "not recorded"}
        or clock.drift_ppm not in (0, None)
    ):
        return None
    return clock.id, clock.epoch


def _rig_vertical(bundle: SignalBundle) -> bool:
    meta = bundle.meta
    assert meta.acquisition is not None
    return (
        meta.channels[0].axis == "z"
        and meta.channels[0].frame == "field"
        and len(meta.acquisition.frames) == 1
        and meta.acquisition.frames[0].id == "field"
        and meta.acquisition.frames[0].kind == "field"
        and meta.acquisition.frames[0].parent is None
        and meta.acquisition.frames[0].definition == "Laboratory vertical field frame, +z up"
    )


def _rig_window(net: FloatArray, setup: RigPairSetup) -> tuple[float, float] | None:
    first = setup.baseline_samples - 1
    crossing = np.flatnonzero(net[first:] > setup.onset_threshold_n)
    if not len(crossing) or crossing[0] == 0:
        return None
    onset = first + int(crossing[0]) - 1
    end = onset + setup.window_samples
    if end >= len(net):
        return None
    return float(net.max()), float(net[end] - net[onset]) / 0.1


def _rig_pair(data: dict[str, object]) -> Reading:
    setup = RigPairSetup.model_validate(data["setup"])
    rig = _rig_bundle(data["rig"], DeviceClass.ISOMETRIC_DYNAMOMETER)
    plate = _rig_bundle(data["plate"], DeviceClass.FORCE_PLATE)
    left, right = rig.meta, plate.meta
    clock = _rig_clock(left)
    if (
        clock is None
        or clock != _rig_clock(right)
        or left.session != right.session
        or left.subject != right.subject
        or left.sampling != right.sampling
        or left.n_rows != right.n_rows
        or left.test_id != right.test_id
        or left.protocol != right.protocol
    ):
        return Reading("not_eligible", ("pair_timing_or_identity_unresolved",))
    if (
        setup.load_transfer != "coaxial_quasistatic"
        or not _rig_vertical(rig)
        or not _rig_vertical(plate)
        or left.geometry.attachment_point != "Vertical bar line of action"
        or left.geometry.rig_compliance_m_per_n != 0
        or right.geometry.plate_layout != "single"
        or left.test_id != "imtp"
        or left.protocol.protocol_id != "res704.fixed-imtp.v1"
    ):
        return Reading("unsupported_claim", ("estimand_mismatch",))
    if setup.baseline_samples >= left.n_rows or min(setup.net_peak_error_n) < 0:
        raise ValueError("invalid baseline length or unsigned net peak bounds")

    native_rig = acquire(rig)
    t0 = float(native_rig.timing.time_s[0])
    native_plate = force_platform_acquisition.acquire(
        plate,
        (
            force_platform_acquisition.PlateSpec(
                plate=1, zero_window_s=(t0, t0 + setup.baseline_samples / 1000)
            ),
        ),
    )
    if (
        not np.array_equal(native_rig.timing.time_s, native_plate.time_s)
        or not native_rig.timing.uniform
        or native_plate.resampled
    ):
        return Reading("not_eligible", ("pair_timing_or_identity_unresolved",))
    rig_force = native_rig.force_n
    rig_metrics = _rig_window(rig_force - rig_force[: setup.baseline_samples].mean(), setup)
    plate_metrics = _rig_window(native_plate.vertical_force_n, setup)
    if rig_metrics is None or plate_metrics is None:
        return Reading("not_eligible", ("unresolved_pair_rfd_window",))
    rig_peak, rig_rfd = rig_metrics
    plate_peak, plate_rfd = plate_metrics
    difference = rig_peak - plate_peak
    classification = (
        "inside_declared_peak_tolerance"
        if abs(difference) <= setup.peak_difference_limit_n
        else "outside_declared_peak_tolerance"
    )
    return Reading(
        "eligible",
        (
            "matched_net_vertical_force",
            "analytical_single_pair",
            classification,
            "paired_variance_unavailable",
            "hardware_interchangeability_unknown",
        ),
        difference,
        sum(setup.net_peak_error_n),
        (
            ("rig_net_peak_n", round(rig_peak, 10)),
            ("plate_net_peak_n", round(plate_peak, 10)),
            ("rig_rfd_100ms_n_s", round(rig_rfd, 10)),
            ("plate_rfd_100ms_n_s", round(plate_rfd, 10)),
        ),
    )


def is_imu_episode(task: RegistryTask) -> bool:
    return (task.task.id, task.task.revision) == IMU_TASK


def checkpoint_methods(task: RegistryTask, checkpoint_id: str) -> tuple[Method, ...]:
    if not is_imu_episode(task):
        return task.methods
    return tuple(
        method
        for method in task.methods
        if isinstance(checkpoints := method.specification.get("checkpoints"), list)
        and checkpoint_id in checkpoints
    )


def is_measurement_task(task: RegistryTask) -> bool:
    if (task.task.id, task.task.revision) == RIG_TASK:
        return task.engine == "m5-advanced-development.v1" and task.methods[0].id == RIG_OPERATION
    if (task.task.id, task.task.revision) == CELL_TASK:
        return task.engine == "m5-advanced-development.v1" and task.methods[0].id == CELL_OPERATION
    if (task.task.id, task.task.revision) == ENCODER_TASK:
        return (
            task.engine == "m5-advanced-development.v1" and task.methods[0].id == ENCODER_OPERATION
        )
    if (task.task.id, task.task.revision) == LPT_TASK:
        return task.engine == "m5-advanced-development.v1" and task.methods[0].id == LPT_OPERATION
    if is_imu_episode(task):
        return (
            task.engine == ENGINE and tuple(method.id for method in task.methods) == IMU_OPERATIONS
        )
    return task.engine == ENGINE and all(
        method.id
        not in (CELL_OPERATION, ENCODER_OPERATION, LPT_OPERATION, RIG_OPERATION, *IMU_OPERATIONS)
        for method in task.methods
    )


def replay(operation: str, data: dict[str, object]) -> Reading:
    if operation == ENCODER_OPERATION:
        return _encoder(data)
    if operation == LPT_OPERATION:
        return _lpt(data)
    if operation == RIG_OPERATION:
        return _rig_pair(data)
    if operation == CELL_OPERATION:
        return _load_cell(data)
    if operation in IMU_OPERATIONS:
        return _imu(operation, data)
    if operation == "paired-speed-bias":
        return _paired(data)
    if operation == "fixed-session-change":
        return _longitudinal(data)
    request = data["request"]
    if not isinstance(request, str) or request not in ("compute", "equivalence"):
        raise ValueError("unknown measurement request")
    meta = BundleMeta.model_validate(data["meta"])
    raw = data["channels"]
    if meta.data_category is not DataCategory.SYNTHETIC or not isinstance(raw, dict):
        raise ValueError("expected native synthetic channels")
    channels = {channel.name for channel in meta.channels}
    if raw.keys() != channels:
        raise ValueError("declared and supplied channels differ")
    bundle = SignalBundle(
        meta,
        {
            channel.name: np.asarray(raw[channel.name], dtype=np.float64)
            for channel in meta.channels
        },
    )
    if operation in ("beam-split", "switch-flight-height"):
        return _timing(operation, bundle, data["policy"], request)
    return _position(operation, bundle, data, request)
