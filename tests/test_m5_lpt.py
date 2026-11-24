"""Independent frozen and nonlinear reference tests for the LPT task."""

import json
import math
from hashlib import sha256
from pathlib import Path

import numpy as np
import pytest

from dynamisbench_soccer.engines import lpt
from dynamisbench_soccer.engines.lpt_acquisition import CableSpec, GeometrySpec
from dynamisbench_soccer.m4.measurement import LPT_OPERATION, replay
from dynamisbench_soccer.m4.task_registry import load_registry
from dynamisbench_soccer.signals.schema import BundleMeta, SignalBundle

ROOT = Path(__file__).resolve().parents[1]
TASK = "lpt-vertical-attachment"
TOL = 1e-9


def _fixture(index):
    return json.loads((ROOT / f"benchmark/fixtures/m5-motion/{TASK}-{index:02}.json").read_bytes())


def _channel_hash(channels):
    return sha256((json.dumps(channels, indent=2, sort_keys=True) + "\n").encode()).hexdigest()


def _rehash(data):
    native_hash = _channel_hash(data["channels"])
    data["meta"]["provenance"]["source_sha256"] = native_hash
    data["meta"]["acquisition"]["sources"][0]["source_sha256"] = native_hash


def _independent_stencil(coefficients, time, h):
    b, c, d, e = coefficients
    absolute = b * time + c * time**2 + d * time**3 + e * time**4
    velocity = b + 2 * c * time + 3 * d * time**2 + 4 * e * time**3
    acceleration = 2 * c + 6 * d * time + 12 * e * time**2
    # Taylor expansion of the two centered stencils is exact for a quartic.
    return absolute, velocity + h**2 * (d + 4 * e * time), acceleration + 2 * e * h**2


def _speed_interval(speed, angle, delta, scale):
    projection = math.cos(angle)
    slow = speed * projection / (math.cos(angle - delta) * (1 + scale))
    fast = speed * projection / (math.cos(angle + delta) * (1 - scale))
    return max(abs(fast - speed), abs(speed - slow))


def _expected_operands(position, speed, acceleration, cosine, mass):
    return {
        "cable_speed_m_s": round(cosine * speed, 10),
        "vertical_displacement_m": round(position, 10),
        "vertical_acceleration_m_s2": round(acceleration, 10),
        "vertical_mass_model_power_w": round(mass * (9.80665 + acceleration) * speed, 10),
    }


def test_frozen_native_lpt_matches_closed_form_and_engine():
    data = _fixture(1)
    reading = replay(LPT_OPERATION, data)
    position, speed, acceleration = _independent_stencil((0.4, 1, 0, 0), 0.06, 0.02)
    assert reading.reason == "eligible"
    assert reading.value == pytest.approx(speed, abs=TOL, rel=0)
    assert reading.radius == pytest.approx(
        _speed_interval(speed, math.acos(0.8), 0.02, 250e-6), abs=TOL, rel=0
    )
    expected = _expected_operands(position, speed, acceleration, 0.8, 20)
    assert dict(reading.operands) == expected
    meta = BundleMeta.model_validate(data["meta"])
    engine = lpt.process(
        SignalBundle(
            meta, {c.name: np.array(data["channels"][c.name], dtype=float) for c in meta.channels}
        ),
        (CableSpec.model_validate(data["cable"]),),
        GeometrySpec.model_validate(data["geometry"]),
        power=lpt.PowerSpec.model_validate(data["power"]),
    )
    assert engine["geometry_method"] == "constant_cable_angle_vertical_motion"
    assert engine["analysis_clock"] == "device"
    assert engine["cables"][0]["native_time_s"][3] == 0.06
    assert engine["motion"]["displacement_m"][3] == pytest.approx(position, abs=TOL)
    assert engine["motion"]["velocity_m_s"][3] == pytest.approx(speed, abs=TOL)
    assert engine["motion"]["acceleration_m_s2"][3] == pytest.approx(acceleration, abs=TOL)
    power = engine["motion"]["vertical_mass_model_power_w"][3]
    assert power == pytest.approx(expected["vertical_mass_model_power_w"], abs=TOL)


@pytest.mark.parametrize(
    ("coefficients", "row", "cosine", "mass", "origin", "ppm"),
    [
        ((0.4, 1, 5, 20), 3, 0.6, 13, 2.0, 900),
        ((-0.5, 1, 3, -12), 2, 0.8, 27, 0.0, 500),
    ],
)
def test_nonlinear_and_signed_reference_controls(coefficients, row, cosine, mass, origin, ppm):
    from dynamisbench_soccer.m4.grader import ExpectedMethod, _replay_measurement

    data = _fixture(1)
    angle = math.acos(cosine)
    data["meta"]["sampling"]["t0_s"] = origin
    data["meta"]["geometry"]["cable_angle_rad"] = angle
    data["meta"]["acquisition"]["sources"][0]["geometry"]["cable_angle_rad"] = angle
    data["channels"]["time"] = [origin + i / 50 for i in range(7)]
    data["channels"]["cable_length"] = [
        0.7 + cosine * sum(c * (i / 50) ** k for k, c in enumerate(coefficients, 1))
        for i in range(7)
    ]
    data["power"]["system_mass_kg"] = mass
    data["clock_scale_bound_ppm"] = ppm
    data["row"] = row
    _rehash(data)
    expected_motion = _independent_stencil(coefficients, row / 50, 1 / 50)
    _, speed, acceleration = expected_motion
    fields = _expected_operands(*expected_motion, cosine, mass)
    radius = _speed_interval(speed, angle, 0.02, ppm * 1e-6)
    result = replay(LPT_OPERATION, data)
    assert result.reason == "eligible"
    assert result.value == pytest.approx(speed, abs=TOL, rel=0)
    assert result.radius == pytest.approx(radius, abs=TOL, rel=0)
    assert dict(result.operands) == pytest.approx(fields, abs=TOL, rel=0)
    assert math.copysign(1, fields["vertical_mass_model_power_w"]) == math.copysign(1, speed)

    # These positives exercise the same deterministic reference replay as the frozen task.
    task = load_registry(ROOT / "benchmark/tasks")[TASK, 1]
    artifact = "independent-stencil-control"
    claims = ("fixed_projection", "mass_model_distinct_from_plate_power", "attachment_point_only")
    expected = ExpectedMethod.model_validate_json(
        json.dumps(
            {
                "decision": "compute",
                "reasons": ["eligible"],
                "selected_trials": [artifact],
                "required_citations": ["DEV-001"],
                "allowed_citations": ["DEV-001"],
                "required_claims": claims,
                "allowed_claims": claims,
                "quantity": {
                    "value": speed,
                    "uncertainty": radius,
                    "unit": "m/s",
                    "dimension": "speed",
                    "estimand": "bar_vertical_attachment_speed",
                },
                "operands": [
                    {"field": name, "value": value, "artifact": artifact}
                    for name, value in fields.items()
                ],
            }
        )
    )
    payload = json.dumps({"inputs": {artifact: json.dumps(data)}}).encode()
    assert _replay_measurement(expected, task.methods[0], payload, (artifact,))
    # Differentiating the centered velocity again uses the ±2 stencil on an interior row.
    quartic_term = coefficients[3]
    wrong_acceleration = acceleration + 6 * quartic_term * (1 / 50) ** 2
    wrong = list(expected.operands)
    wrong[2] = wrong[2].model_copy(update={"value": wrong_acceleration})
    assert not _replay_measurement(
        expected.model_copy(update={"operands": tuple(wrong)}),
        task.methods[0],
        payload,
        (artifact,),
    )


FROZEN_REFUSALS = (
    (2, "insufficient_evidence", ("cable_only", "vertical_geometry_unobserved")),
    (3, "not_eligible", ("cable_slack",)),
    (4, "not_eligible", ("nonincreasing_native_clock",)),
    (5, "unsupported_claim", ("mass_model_distinct_from_plate_power", "com_unobserved")),
)


@pytest.mark.parametrize(("index", "reason", "claims"), FROZEN_REFUSALS)
def test_frozen_lpt_refusals(index, reason, claims):
    reading = replay(LPT_OPERATION, _fixture(index))
    assert (reading.reason, reading.claims) == (reason, claims)
    assert reading.value is reading.radius is None and not reading.operands


def test_lpt_rejects_stale_channels_calibration_clock_and_invalid_power():
    data = _fixture(1)
    data["channels"]["time"][1] = 0.01
    with pytest.raises(ValueError, match="source provenance"):
        replay(LPT_OPERATION, data)
    _rehash(data)
    assert replay(LPT_OPERATION, data).claims == ("unresolved_native_grid",)
    data = _fixture(1)
    data["power"]["rigid_vertical_translation"] = False
    with pytest.raises(ValueError, match="rigid_vertical_translation"):
        replay(LPT_OPERATION, data)
    data = _fixture(1)
    data["meta"]["acquisition"]["sources"][0]["calibration"][0]["gain"] = -1
    with pytest.raises(ValueError, match="identity calibration"):
        replay(LPT_OPERATION, data)
    data = _fixture(1)
    data["angle_bound_rad"] = 1.0
    with pytest.raises(ValueError, match="nonsingular angle"):
        replay(LPT_OPERATION, data)
