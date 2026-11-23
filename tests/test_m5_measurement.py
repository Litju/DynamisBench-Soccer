import json
from pathlib import Path

import pytest

from dynamisbench_soccer.m4.admission import evaluate
from dynamisbench_soccer.m4.grader import MeasurementSubmission, _check_submission_json
from dynamisbench_soccer.m4.key_store import digest
from dynamisbench_soccer.m4.measurement import ENGINE, PROCEDURES, replay
from dynamisbench_soccer.m4.task_registry import load_registry

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("procedure", PROCEDURES)
def test_measurement_episode_replays_without_disclosing_reference(procedure):
    tasks = load_registry(ROOT / "benchmark/tasks")
    (task,) = [
        task for task in tasks.values() if task.engine == ENGINE and task.methods[0].id == procedure
    ]
    assert task.task.source == PROCEDURES[procedure].source
    raw = (ROOT / "benchmark/reviewed-slices.json").read_bytes()
    assert evaluate(ROOT, task, raw, digest(raw)).status == "DEVELOPMENT_ONLY_NOT_QUALIFIED"
    inputs = {item.name: (ROOT / item.path).read_bytes() for item in task.artifacts}
    seen = set()
    decisions = set()
    for checkpoint in task.checkpoints:
        (artifact,) = checkpoint.artifacts
        seen.add(artifact)
        payload = json.loads(task.candidate_payload(checkpoint.id, inputs, 17))
        assert set(payload) == {"task", "checkpoint", "inputs", "seed"}
        assert set(payload["inputs"]) == seen
        assert payload["checkpoint"]["artifacts"] == [artifact]
        data = json.loads(payload["inputs"][artifact])
        reading = replay(procedure, data)
        decisions.add(reading.reason)
        if reading.reason == "eligible":
            assert reading.value is not None and reading.radius > 0
            assert reading.operands
        else:
            assert reading.value is None and reading.radius is None and not reading.operands
    assert {"eligible", "not_eligible", "unsupported_claim"} <= decisions
    if procedure == "switch-flight-height":
        assert "insufficient_evidence" in decisions


def test_submission_cannot_attach_extra_inference_or_shadow_fields():
    response = {
        "version": "m5.measurement-answer.v1",
        "decision": "refuse",
        "method": "beam-split",
        "reason": "unsupported_claim",
        "selected_trials": [],
        "citations": ["DEV-004"],
        "claims": ["different_estimands", "paired_validation_absent"],
        "quantity": None,
        "operands": [],
    }
    raw = json.dumps(response).encode()
    assert MeasurementSubmission.model_validate_json(raw).decision == "refuse"
    with pytest.raises(ValueError, match="Extra inputs"):
        MeasurementSubmission.model_validate_json(
            json.dumps({**response, "explanation": "Therefore the instruments are interchangeable"})
        )
    with pytest.raises(ValueError, match="duplicate"):
        _check_submission_json(raw[:-1] + b', "decision": "compute"}')
