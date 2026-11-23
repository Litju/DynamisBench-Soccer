"""Independent standard-library solver for the public motion development episodes."""

import itertools
import json
import math
import sys
from statistics import median


def cable_answer(data):
    channels, spec = data["channels"], data["cable"]
    geometry = data["meta"]["acquisition"]["sources"][0]["geometry"]
    if data["request"] == "force_plate_power":
        return "unsupported_claim", ("mass_model_distinct_from_plate_power", "com_unobserved")

    time = channels["time"]
    intervals = [end - start for start, end in itertools.pairwise(time)]
    if min(intervals) <= 0:
        return "not_eligible", ("nonincreasing_native_clock",)
    if any(channels.get("slack", [])):
        return "not_eligible", ("cable_slack",)
    if geometry.get("encoder_counts_per_m") is None and "counts" in channels:
        return "not_eligible", ("missing_count_scale",)

    angle = geometry.get("cable_angle_rad")
    if angle is None or "cable_angle_rad" in geometry.get("unknown", []):
        return "insufficient_evidence", ("cable_only", "vertical_geometry_unobserved")
    projection = math.cos(angle)
    if "counts" in channels:
        modulo = spec["count_modulus"]
        scale = geometry["encoder_counts_per_m"]
        allowances = [spec["max_speed_m_s"] * dt * scale + 1 for dt in intervals]
        if max(allowances) >= modulo / 2:
            return "not_eligible", ("ambiguous_rollover",)
        counts = channels["counts"]
        deltas = [
            ((b - a + modulo // 2) % modulo) - modulo // 2 for a, b in itertools.pairwise(counts)
        ]
        if any(abs(delta) > allowance for delta, allowance in zip(deltas, allowances, strict=True)):
            return "not_eligible", ("implausible_count_jump",)
        signed = math.fsum(deltas) * spec["count_direction"]
        cable = signed / scale
        change = cable / projection
        elapsed = time[-1] - time[0]
        return (
            change,
            1 / (scale * projection),
            dict(
                signed_count_change=signed,
                elapsed_s=elapsed,
                cable_displacement_m=cable,
                mean_vertical_velocity_m_s=change / elapsed,
            ),
            ("signed_native_rollover", "attachment_point_only", "endpoint_quantization_bound"),
        )

    lengths = channels["cable_length"]
    row = data["row"]
    before, now, after = lengths[row - 1 : row + 2]
    elapsed = time[row + 1] - time[row - 1]
    cable_speed = (after - before) / elapsed
    displacement = (now - lengths[0]) / projection
    acceleration = (after - 2 * now + before) / ((elapsed / 2) ** 2 * projection)
    speed = cable_speed / projection
    angle_bound = data["angle_bound_rad"]
    scale_bound = data["clock_scale_bound_ppm"] * 1e-6
    radius = max(
        abs(cable_speed / (math.cos(angle + da) * (1 + ds)) - speed)
        for da in (-angle_bound, angle_bound)
        for ds in (-scale_bound, scale_bound)
    )
    mass = data["power"]["system_mass_kg"]
    return (
        speed,
        radius,
        dict(
            cable_speed_m_s=cable_speed,
            vertical_displacement_m=displacement,
            vertical_acceleration_m_s2=acceleration,
            vertical_mass_model_power_w=mass * (9.80665 + acceleration) * speed,
        ),
        ("fixed_projection", "mass_model_distinct_from_plate_power", "attachment_point_only"),
    )


def calibrated_force(data):
    spec, channels = data["spec"], data["channels"]
    raw = [channels[f"specific_force_{axis}"] for axis in "xyz"]
    limits = [
        (raw, spec["accel_range_m_s2"]),
        ([channels[f"angular_velocity_{axis}"] for axis in "xyz"], spec["gyro_range_rad_s"]),
    ]
    for axes, limit in limits:
        if any(abs(value) >= 0.999 * limit for axis in axes for value in axis):
            return None
    bias, scale = spec["accel_bias_m_s2"], spec["accel_scale"]
    shear = spec["accel_misalignment"][0][1]
    vectors = []
    for sample in zip(*raw, strict=True):
        x, y, z = [value / scale[i] - bias[i] for i, value in enumerate(sample)]
        vectors.append((x - shear * y, y, z))
    return vectors


def imu_answer(data):
    spec, channels = data["spec"], data["channels"]
    request = data["request"]
    if request == "com_position":
        return "unsupported_claim", ("instrument_point_only", "com_unobserved")
    force = calibrated_force(data)
    if force is None:
        return "not_eligible", ("clipped_instrument",)
    time = channels["time"]
    intervals = [b - a for a, b in itertools.pairwise(time)]
    if min(intervals) <= 0 or max(intervals) > 1.5 * median(intervals):
        return "not_eligible", ("native_clock_gap",)

    if request == "signature_flight_time":
        threshold = data["flight_threshold_m_s2"]
        low = [i for i, row in enumerate(force) if math.hypot(*row) < threshold]
        if (
            not low
            or low[0] == 0
            or low[-1] == len(time) - 1
            or low != list(range(low[0], low[-1] + 1))
        ):
            return "not_eligible", ("incomplete_signature",)
        start, end = low[0] - 1, low[-1] + 1
        duration = time[end] - time[start]
        if duration < data["flight_min_s"]:
            return "not_eligible", ("incomplete_signature",)
        return (
            duration,
            intervals[start] + intervals[end - 1],
            dict(takeoff_edge_s=time[start], landing_edge_s=time[end], threshold_m_s2=threshold),
            ("specific_force_threshold_signature", "whole_body_flight_unestablished"),
        )

    if "initial_orientation" not in spec:
        return "insufficient_evidence", ("orientation_unobservable", "gravity_norm_not_static")
    if "initial_velocity_m_s" not in spec:
        return "insufficient_evidence", ("missing_initial_velocity",)
    if "initial_position_m" not in spec:
        return "insufficient_evidence", ("missing_initial_position",)
    rotation = spec["initial_orientation"]
    for axis, bias in zip("xyz", spec["gyro_bias_rad_s"], strict=True):
        if any(
            not math.isclose(raw, bias, abs_tol=1e-12)
            for raw in channels[f"angular_velocity_{axis}"]
        ):
            return "not_eligible", ("fixed_attitude_model_unavailable",)
    ax = math.fsum(coefficient * force[0][i] for i, coefficient in enumerate(rotation[0]))
    az = (
        math.fsum(coefficient * force[0][i] for i, coefficient in enumerate(rotation[2]))
        - data["gravity_m_s2"]
    )
    elapsed = time[-1] - time[0]
    v0 = spec["initial_velocity_m_s"][0]
    p0 = spec["initial_position_m"][0]
    return (
        p0 + v0 * elapsed + 0.5 * ax * elapsed**2,
        0.5 * data["residual_field_bias_bound_m_s2"] * elapsed**2,
        dict(
            free_ax_m_s2=ax, free_az_m_s2=az, terminal_vx_m_s=v0 + ax * elapsed, elapsed_s=elapsed
        ),
        (
            "field_frame",
            "declared_initial_state",
            "instrument_point_only",
            "conditional_bias_bound",
        ),
    )


def main():
    episode = json.load(sys.stdin)
    (artifact,) = episode["checkpoint"]["artifacts"]
    data = json.loads(episode["inputs"][artifact])
    source = episode["task"]["source"]
    if source == "lpt_encoder.lpt":
        method, estimand, dimension, unit, citation = (
            "fixed-angle-attachment-speed",
            "bar_vertical_attachment_speed",
            "speed",
            "m/s",
            "DEV-001",
        )
    elif source == "lpt_encoder.encoder":
        method, estimand, dimension, unit, citation = (
            "modular-attachment-change",
            "signed_vertical_attachment_change",
            "length",
            "m",
            "DEV-001",
        )
    elif data["request"] == "signature_flight_time":
        method, estimand, dimension, unit, citation = (
            "specific-force-signature-flight",
            "sensor_threshold_signature_duration",
            "time",
            "s",
            "DEV-002",
        )
    else:
        method, estimand, dimension, unit, citation = (
            "fixed-attitude-sensor-position",
            "field_x_instrument_point_position",
            "length",
            "m",
            "DEV-002",
        )
    result = imu_answer(data) if source == "imu.inertial" else cable_answer(data)
    response = dict(
        version="m5.measurement-answer.v1", method=method, citations=[citation], corrects=None
    )
    if len(result) == 2:
        reason, claims = result
        response.update(
            decision="refuse",
            reason=reason,
            claims=claims,
            selected_trials=[],
            quantity=None,
            operands=[],
        )
    else:
        value, uncertainty, operands, claims = result
        response.update(
            decision="compute",
            reason="eligible",
            claims=claims,
            selected_trials=[artifact],
            quantity=dict(
                value=value,
                uncertainty=uncertainty,
                estimand=estimand,
                dimension=dimension,
                unit=unit,
            ),
            operands=[
                dict(artifact=artifact, field=field, value=reading)
                for field, reading in operands.items()
            ],
        )
    print(json.dumps(response, allow_nan=False))


if __name__ == "__main__":
    main()
