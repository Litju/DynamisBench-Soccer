"""Solve the declared synthetic load-cell and rig exercises with the standard library."""

import json
import sys
from itertools import pairwise
from statistics import fmean

ANSWER_VERSION = "m5.measurement-answer.v1"
CITATION = "DEV-006"


def response(method, reason, claims, artifact=None, quantity=None, operands=None):
    return {
        "version": ANSWER_VERSION,
        "decision": "compute" if reason == "eligible" else "refuse",
        "method": method,
        "reason": reason,
        "selected_trials": [artifact] if artifact else [],
        "citations": [CITATION],
        "claims": list(claims),
        "quantity": quantity,
        "operands": [
            {"artifact": artifact, "field": field, "value": round(value, 10)}
            for field, value in (operands or {}).items()
        ],
    }


def force_window(values, baseline_count, threshold, samples, hz):
    baseline = fmean(values[:baseline_count])
    crossing = next(
        (
            row
            for row in range(baseline_count - 1, len(values))
            if values[row] - baseline > threshold
        ),
        None,
    )
    if crossing is None or crossing == baseline_count - 1 or crossing - 1 + samples >= len(values):
        return None
    start = crossing - 1
    net = [value - baseline for value in values[start : start + samples + 1]]
    return (
        baseline,
        start / hz,
        (net[-1] - net[0]) * hz / samples,
        sum((x + y) / (2 * hz) for x, y in pairwise(net)),
    )


def cell_response(artifact, source):
    method = "unfiltered-net-peak-torque"
    meta = source["meta"]
    setup = source["setup"]
    native = meta["acquisition"]["sources"][0]
    calibration = setup.get("analysis_calibration") or next(
        iter(native.get("calibration", [])), None
    )
    if calibration is None:
        return response(method, "not_eligible", ("missing_calibration",))
    units = calibration["input_unit"], calibration["output_unit"]
    gain, offset = calibration["gain"], calibration["offset"]
    voltage = source["samples"]
    lo, hi = setup["adc_rails_v"]
    if units not in {("V", "N"), ("N", "V")}:
        return response(method, "not_eligible", ("calibration_unit_mismatch",))
    if any(sample <= lo or sample >= hi for sample in voltage):
        return response(method, "not_eligible", ("adc_saturation",))
    if meta["channels"][0]["side"] != setup["required_side"]:
        return response(method, "not_eligible", ("wrong_limb",))
    angle_lo, angle_hi = setup["eligible_knee_angle_deg"]
    geometry = native["geometry"]
    if not angle_lo <= geometry["knee_angle_deg"] <= angle_hi:
        return response(method, "not_eligible", ("protocol_geometry_ineligible",))
    lever = geometry.get("lever_arm_m")
    if lever is None or not geometry.get("attachment_point"):
        return response(method, "insufficient_evidence", ("moment_arm_unestablished",))

    if units == ("V", "N"):
        force = [gain * sample + offset for sample in voltage]
    else:
        force = [(sample - offset) / gain for sample in voltage]
    hz = meta["sampling"]["nominal_hz"]
    metrics = force_window(
        force, setup["baseline_samples"], setup["onset_threshold_n"], setup["window_samples"], hz
    )
    if metrics is None:
        return response(method, "not_eligible", ("no_resolved_window",))
    baseline, onset, rfd, impulse = metrics
    peak = max(force)
    net = peak - baseline
    force_radius = setup["net_force_error_n"]
    lever_radius = setup["lever_error_m"]
    radius = net * lever_radius + lever * force_radius + force_radius * lever_radius
    return response(
        method,
        "eligible",
        ("calibrated_cell_force", "baseline_net_force", "perpendicular_external_joint_torque"),
        artifact,
        {
            "value": net * lever,
            "unit": "N*m",
            "dimension": "torque",
            "estimand": "peak_external_knee_torque_above_baseline",
            "uncertainty": radius,
        },
        {
            "baseline_n": baseline,
            "gross_peak_n": peak,
            "net_peak_n": net,
            "onset_s": onset,
            "rfd_100ms_n_s": rfd,
            "impulse_100ms_n_s": impulse,
            "lever_arm_m": lever,
        },
    )


def rig_response(artifact, source):
    method = "paired-net-peak-difference"
    rig = source["rig"]
    plate = source["plate"]
    setup = source["setup"]
    if (
        setup["load_transfer"] != "coaxial_quasistatic"
        or plate["meta"]["channels"][0]["axis"] != "z"
    ):
        return response(method, "unsupported_claim", ("estimand_mismatch",))
    left, right = rig["meta"], plate["meta"]
    if (
        not all(
            left[field] == right[field] for field in ("session", "subject", "sampling", "protocol")
        )
        or left["acquisition"]["clocks"] != right["acquisition"]["clocks"]
    ):
        return response(method, "not_eligible", ("pair_timing_or_identity_unresolved",))

    def trace(record):
        values = record["samples"]
        metrics = force_window(
            values,
            setup["baseline_samples"],
            setup["onset_threshold_n"],
            setup["window_samples"],
            record["meta"]["sampling"]["nominal_hz"],
        )
        if metrics is None:
            raise ValueError("no eligible force window")
        baseline, _, rfd, _ = metrics
        return max(values) - baseline, rfd

    rig_peak, rig_rfd = trace(rig)
    plate_peak, plate_rfd = trace(plate)
    difference = rig_peak - plate_peak
    tolerance = setup["peak_difference_limit_n"]
    classification = (
        "inside_declared_peak_tolerance"
        if abs(difference) <= tolerance
        else ("outside_declared_peak_tolerance")
    )
    return response(
        method,
        "eligible",
        (
            "matched_net_vertical_force",
            "analytical_single_pair",
            classification,
            "paired_variance_unavailable",
            "hardware_interchangeability_unknown",
        ),
        artifact,
        {
            "value": difference,
            "unit": "N",
            "dimension": "force",
            "estimand": "rig_minus_plate_net_peak_force",
            "uncertainty": sum(setup["net_peak_error_n"]),
        },
        {
            "rig_net_peak_n": rig_peak,
            "plate_net_peak_n": plate_peak,
            "rig_rfd_100ms_n_s": rig_rfd,
            "plate_rfd_100ms_n_s": plate_rfd,
        },
    )


def main():
    payload = json.load(sys.stdin)
    (artifact,) = payload["checkpoint"]["artifacts"]
    source = json.loads(payload["inputs"][artifact])
    task = payload["task"]["id"]
    solver = {
        "load-cell-net-torque-ramp": cell_response,
        "rig-plate-net-peak-pair": rig_response,
    }[task]
    print(json.dumps(solver(artifact, source), allow_nan=False, sort_keys=True))


if __name__ == "__main__":
    main()
