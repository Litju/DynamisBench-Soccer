import json

import pytest

from dynamisbench_soccer.m4.grader import ExpectedMethod, check_control

ANSWER = {
    "version": "m4.formal-answer.v1",
    "decision": "compute",
    "method": "elapsed",
    "reason": "eligible",
    "explanation": "Subtract ordered endpoints.",
    "selected_trials": ["trial"],
    "citations": ["timestamps"],
    "claims": ["elapsed_only"],
    "quantity": {
        "value": 0.25,
        "unit": "s",
        "dimension": "time",
        "estimand": "elapsed",
        "uncertainty": 0.002,
    },
    "operands": [
        {"artifact": "timestamps", "field": "finish", "value": 2.25},
        {"artifact": "timestamps", "field": "start", "value": 2},
    ],
}

EXPECTED = ExpectedMethod.model_validate_json(
    json.dumps(
        {
            "decision": "compute",
            "reasons": ["eligible"],
            "selected_trials": ANSWER["selected_trials"],
            "required_citations": ANSWER["citations"],
            "allowed_citations": ANSWER["citations"],
            "required_claims": ANSWER["claims"],
            "allowed_claims": ANSWER["claims"],
            "quantity": ANSWER["quantity"],
            "operands": ANSWER["operands"],
        }
    )
)


def test_positive_control_allows_equivalent_explanation() -> None:
    raw = json.dumps({**ANSWER, "explanation": "Difference on the common clock."}).encode()
    assert check_control(raw, EXPECTED, 1e-9) == ()


@pytest.mark.parametrize(
    ("path", "value", "finding"),
    [
        ("quantity.unit", "ms", "unit_or_dimension"),
        ("quantity.value", 250, "value"),
        ("quantity.value", -0.25, "value"),
        ("quantity.estimand", "instantaneous-speed", "estimand"),
        ("quantity.uncertainty", None, "uncertainty"),
        ("citations", ["vendor"], "source_citations"),
        ("claims", ["physical_interchangeability"], "interpretation_boundary"),
        ("selected_trials", ["wrong-trial"], "trial_selection"),
        ("operands", [{"artifact": "vendor", "field": "elapsed", "value": 0.25}], "derivation"),
        ("corrects", "wrong-step", "correction_lineage"),
    ],
)
def test_negative_controls(path: str, value: object, finding: str) -> None:
    raw = json.loads(json.dumps(ANSWER))
    key, _, field = path.partition(".")
    if field:
        raw[key][field] = value
    else:
        raw[key] = value
    assert check_control(json.dumps(raw).encode(), EXPECTED, 1e-9) == (finding,)


def test_duplicate_json_keys_fail() -> None:
    assert check_control(b'{"answer":1,"answer":2}', EXPECTED, 1e-9) == ("invalid_submission",)


def test_excludes_development_tasks_without_reading_private_store(tmp_path) -> None:
    from pathlib import Path

    from dynamisbench_soccer.m4.grader import score_checkpoint
    from dynamisbench_soccer.m4.key_store import digest
    from dynamisbench_soccer.m4.task_registry import load_registry

    root = Path(__file__).resolve().parents[1]
    task = load_registry(root / "benchmark/tasks")["elapsed-demo", 1]
    catalog = (root / "benchmark/reviewed-slices.json").read_bytes()
    outcome = score_checkpoint(
        root,
        task,
        catalog,
        digest(catalog),
        tmp_path / "absent",
        "initial",
        {},
        json.dumps(ANSWER).encode(),
    )
    assert outcome.outcome == "EXCLUDED_FROM_SCORED_BENCHMARK"


@pytest.mark.parametrize(
    "update",
    [
        {"reasons": ["insufficient_evidence"]},
        {"allowed_citations": []},
        {"allowed_claims": []},
        {"quantity": None},
    ],
)
def test_inconsistent_private_reference_is_rejected(update) -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        ExpectedMethod.model_validate_json(json.dumps(EXPECTED.model_dump(mode="json") | update))


def test_native_replay_checks_real_eligibility_and_uncertainty() -> None:
    from math import hypot
    from pathlib import Path

    from dynamisbench_soccer.m4.grader import _replay_lps
    from dynamisbench_soccer.m4.task_registry import load_registry

    root = Path(__file__).resolve().parents[1]
    task = load_registry(root / "benchmark/tasks")["native-lps-speed", 2]
    inputs = {item.name: (root / item.path).read_bytes() for item in task.artifacts}
    artifact = task.checkpoints[0].artifacts[0]
    fixture = json.loads(inputs[artifact])
    row = fixture["row"]
    channels = fixture["channels"]
    delta = {n: channels[n][row + 1] - channels[n][row - 1] for n in channels}
    candidate = {
        "decision": "compute",
        "reasons": ["eligible"],
        "selected_trials": [artifact],
        "required_citations": ["DEV-005"],
        "allowed_citations": ["DEV-005"],
        "required_claims": [],
        "allowed_claims": [],
        "operands": [
            {"artifact": artifact, "field": f"{name}.{index}", "value": channels[name][index]}
            for name in ("time", "position_x", "position_y")
            for index in range(row - 1, row + 2)
        ],
        "quantity": {
            "value": hypot(delta["position_x"], delta["position_y"]) / delta["time"],
            "unit": "m/s",
            "estimand": "horizontal_positional_speed",
            "dimension": "speed",
            "uncertainty": fixture["horizontal_error_m"] * 25,
        },
    }
    expected = ExpectedMethod.model_validate_json(json.dumps(candidate))
    assert _replay_lps(expected, task.candidate_payload("clean", inputs, 0), (artifact,), 1e-12)
    wrong = expected.model_copy(
        update={"quantity": expected.quantity.model_copy(update={"uncertainty": 0})}
    )
    assert not _replay_lps(wrong, task.candidate_payload("clean", inputs, 0), (artifact,), 1e-12)
    refusal = ExpectedMethod.model_validate_json(
        json.dumps(
            candidate
            | {
                "decision": "refuse",
                "reasons": ["not_eligible"],
                "selected_trials": [],
                "quantity": None,
                "operands": [],
            }
        )
    )
    for checkpoint, eligible in zip(task.checkpoints, (False, True), strict=True):
        assert (
            _replay_lps(
                refusal,
                task.candidate_payload(checkpoint.id, inputs, 0),
                checkpoint.artifacts,
                1e-12,
            )
            == eligible
        )
