"""Independent signed modular controls for the frozen encoder task."""

import json
import math
from fractions import Fraction
from hashlib import sha256
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from dynamisbench_soccer.engines import lpt
from dynamisbench_soccer.engines.lpt_acquisition import CableSpec, GeometrySpec
from dynamisbench_soccer.m4.measurement import ENCODER_OPERATION, replay
from dynamisbench_soccer.m4.task_registry import load_registry
from dynamisbench_soccer.signals.schema import BundleMeta, SignalBundle

ROOT = Path(__file__).resolve().parents[1]
TASK = "encoder-rollover-attachment"
TOL = 1e-9


def _fixture(index):
    return json.loads((ROOT / f"benchmark/fixtures/m5-motion/{TASK}-{index:02}.json").read_bytes())


def _channel_hash(channels):
    return sha256((json.dumps(channels, indent=2, sort_keys=True) + "\n").encode()).hexdigest()


def _rehash(data):
    sha = _channel_hash(data["channels"])
    data["meta"]["provenance"]["source_sha256"] = sha
    data["meta"]["acquisition"]["sources"][0]["source_sha256"] = sha


def _allowed_lifts(difference, modulus, allowance):
    return [
        difference + k * modulus for k in range(-2, 3) if abs(difference + k * modulus) <= allowance
    ]


def _independent_truth(count_change, scale, projection, duration):
    cable = Fraction(count_change, scale)
    change = cable / projection
    radius = Fraction(1, scale) / projection
    return (
        change,
        radius,
        {
            "signed_count_change": count_change,
            "elapsed_s": round(float(duration), 10),
            "cable_displacement_m": round(float(cable), 10),
            "mean_vertical_velocity_m_s": round(float(change / duration), 10),
        },
    )


def _assert_reading(data, count_change, scale, projection):
    elapsed = Fraction(len(data["channels"]["counts"]) - 1, 50)
    change, radius, operands = _independent_truth(count_change, scale, projection, elapsed)
    reading = replay(ENCODER_OPERATION, data)
    assert reading.reason == "eligible"
    assert reading.value == pytest.approx(float(change), abs=TOL, rel=0)
    assert reading.radius == pytest.approx(float(radius), abs=TOL, rel=0)
    assert dict(reading.operands) == operands


def test_frozen_signed_encoder_matches_independent_forward_counts():
    data = _fixture(1)
    assert data["channels"]["counts"] == [(248 + 4 * row) % 256 for row in range(7)]
    allowance = 2 * Fraction(1, 50) * 400 + 1
    assert allowance == 17 < 128
    assert _allowed_lifts(-252, 256, allowance) == [4]
    _assert_reading(data, -24, 400, Fraction(1))

    meta = BundleMeta.model_validate(data["meta"])
    engine = lpt.process(
        SignalBundle(
            meta,
            {
                column.name: np.array(data["channels"][column.name], dtype=float)
                for column in meta.channels
            },
        ),
        (CableSpec.model_validate(data["cable"]),),
        GeometrySpec.model_validate(data["geometry"]),
    )
    assert engine["geometry_method"] == "constant_cable_angle_vertical_motion"
    assert engine["cables"][0]["faults"] == {}
    assert engine["analysis_clock"] == "device"
    assert engine["cables"][0]["resolution_m"] == 0.0025
    assert engine["motion"]["displacement_m"][-1] == pytest.approx(-0.06, abs=TOL, rel=0)


@pytest.mark.parametrize(
    ("increments", "start", "direction", "scale", "projection", "origin", "speed"),
    [
        ((-4,) * 6, 2, -1, 500, Fraction(3, 5), 2.0, 2),
        ((44, 44, 44, 44, 40, 40), 248, 1, 400, Fraction(4, 5), 0.0, 8),
    ],
)
def test_independent_reverse_boundary_and_full_turn(
    increments, start, direction, scale, projection, origin, speed
):
    from dynamisbench_soccer.m4.grader import ExpectedMethod, _replay_measurement

    data = _fixture(1)
    angle = math.acos(float(projection))
    residues = [(start + sum(increments[:i])) % 256 for i in range(len(increments) + 1)]
    allowance = speed * Fraction(1, 50) * scale + 1
    assert allowance < 128
    assert [
        _allowed_lifts(after - before, 256, allowance) for before, after in pairwise(residues)
    ] == [[step] for step in increments]
    data["channels"] = {"counts": residues, "time": [origin + row / 50 for row in range(7)]}
    data["meta"]["sampling"]["t0_s"] = origin
    for geometry in (
        data["meta"]["geometry"],
        data["meta"]["acquisition"]["sources"][0]["geometry"],
    ):
        geometry.update(encoder_counts_per_m=scale, cable_angle_rad=angle)
    data["cable"].update(count_direction=direction, max_speed_m_s=speed, zero_count=131)
    _rehash(data)
    count_change = direction * sum(increments)
    _assert_reading(data, count_change, scale, projection)

    artifact = "independent-encoder-control"
    task = load_registry(ROOT / "benchmark/tasks")[TASK, 1]
    value, radius, fields = _independent_truth(count_change, scale, projection, Fraction(3, 25))
    claims = ["signed_native_rollover", "attachment_point_only", "endpoint_quantization_bound"]
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
                    "value": float(value),
                    "uncertainty": float(radius),
                    "unit": "m",
                    "dimension": "length",
                    "estimand": "signed_vertical_attachment_change",
                },
                "operands": [
                    {"field": field, "value": number, "artifact": artifact}
                    for field, number in fields.items()
                ],
            }
        )
    )
    payload = json.dumps({"inputs": {artifact: json.dumps(data)}}).encode()
    assert _replay_measurement(expected, task.methods[0], payload, (artifact,))
    wrong_counts = expected.operands[0].model_copy(
        update={"value": float(direction * (residues[-1] - residues[0]))}
    )
    assert not _replay_measurement(
        expected.model_copy(update={"operands": (wrong_counts, *expected.operands[1:])}),
        task.methods[0],
        payload,
        (artifact,),
    )


ENCODER_REFUSALS = (
    (2, "insufficient_evidence", ("missing_count_scale",)),
    (3, "not_eligible", ("ambiguous_rollover",)),
    (4, "insufficient_evidence", ("cable_only", "vertical_geometry_unobserved")),
    (5, "not_eligible", ("implausible_count_jump",)),
)


@pytest.mark.parametrize(("index", "reason", "claims"), ENCODER_REFUSALS)
def test_frozen_encoder_refusals(index, reason, claims):
    result = replay(ENCODER_OPERATION, _fixture(index))
    assert (result.reason, result.claims) == (reason, claims)
    assert result.value is result.radius is None and not result.operands


def test_true_half_turn_tie_and_frozen_global_guard():
    data = _fixture(1)
    data["cable"]["max_speed_m_s"] = 15.875
    data["channels"]["counts"][1] = 120
    _rehash(data)
    assert _allowed_lifts(-128, 256, 128) == [-128, 128]
    assert replay(ENCODER_OPERATION, data).claims == ("ambiguous_rollover",)
    assert _allowed_lifts(4, 256, 161) == [4]
    assert replay(ENCODER_OPERATION, _fixture(3)).claims == ("ambiguous_rollover",)


@pytest.mark.parametrize("invalid", [-1, 256, 0.5, True])
def test_native_counter_refuses_invalid_residues(invalid):
    data = _fixture(1)
    data["channels"]["counts"][2] = invalid
    _rehash(data)
    with pytest.raises(ValueError, match=r"integer|count modulus"):
        replay(ENCODER_OPERATION, data)


@pytest.mark.parametrize("when", [0, 0.03, 0.04])
def test_native_counter_refuses_duplicate_jitter_and_gap(when):
    data = _fixture(1)
    if when == 0.04:
        data["channels"]["time"][1:] = [when + i / 50 for i in range(6)]
    else:
        data["channels"]["time"][1] = when
    with pytest.raises(ValueError, match="source provenance"):
        replay(ENCODER_OPERATION, data)
    _rehash(data)
    result = replay(ENCODER_OPERATION, data)
    assert result.reason == "not_eligible" and result.claims == ("unresolved_native_grid",)


def test_encoder_refuses_whole_athlete_com_interpretation():
    data = _fixture(1)
    data["request"] = "com_change"
    result = replay(ENCODER_OPERATION, data)
    assert result.reason == "unsupported_claim" and "com_unobserved" in result.claims
    assert result.value is result.radius is None
