"""Replay the declared M5 measurement procedures against admitted native inputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

import numpy as np

from dynamisbench_soccer.agreement.paired_speed import PairedInput, compare
from dynamisbench_soccer.data.manifest import DataCategory
from dynamisbench_soccer.engines.contact import ContactPolicy, contact
from dynamisbench_soccer.engines.event_time import Estimate, EventRefusal
from dynamisbench_soccer.engines.gnss_lps import ProcessingOptions, process
from dynamisbench_soccer.engines.gnss_lps_acquisition import (
    AcquisitionDeclaration,
    AcquisitionRefusal,
)
from dynamisbench_soccer.engines.position_clock_bounds import input_error_radii
from dynamisbench_soccer.engines.timing_gates import GatePolicy, sprint
from dynamisbench_soccer.m4.admission import Kind
from dynamisbench_soccer.signals.schema import BundleMeta, DeviceClass, SignalBundle

Reason = Literal["eligible", "not_eligible", "insufficient_evidence", "unsupported_claim"]
ENGINE = "m5-measurement.v1"
GRADER = "measurement.v1"
ANSWER_VERSION = "m5.measurement-answer.v1"


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


def replay(operation: str, data: dict[str, object]) -> Reading:
    if operation == "paired-speed-bias":
        return _paired(data)
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
