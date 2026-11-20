"""DEVELOPMENT_ONLY_NOT_QUALIFIED contract and grader controls."""

import pytest
from pydantic import ValidationError

from dynamisbench_soccer.m4.contracts import STAMP, Answer, Episode, Task, grade


def episode():
    return Episode(
        episode_id="trial-1",
        seed=42,
        task=Task(
            task_id="test",
            evidence="analytical",
            prompt="Compute x.",
            inputs={"x": 1},
            estimand="test",
            unit="m",
            method_spec={},
            provenance="test",
        ),
    )


def test_versioned_and_marked_contracts():
    trial = episode()
    assert Episode.model_validate_json(trial.model_dump_json()) == trial
    assert trial.task.status == trial.status == STAMP
    with pytest.raises(ValidationError):
        Episode.model_validate({**trial.model_dump(), "version": "m4.episode.v2"})
    with pytest.raises(ValidationError):
        Task.model_validate({**trial.task.model_dump(), "evidence": "empirical"})
    with pytest.raises(ValidationError):
        Task.model_validate({**trial.task.model_dump(), "status": "qualified"})


def test_grader_distinguishes_correct_numbers_from_correct_method_and_refusal():
    trial = episode()
    key = Answer(
        decision="compute",
        method="closed_form",
        reason="declared_origin",
        metrics={"x": 2},
        estimand="test",
        unit="m",
    )
    assert grade(trial, key.model_dump_json().encode(), key, atol=1e-6).passed
    for updates, expected in [
        ({"method": "guess"}, (True, False)),
        ({"metrics": {"x": 2.1}}, (False, True)),
        ({"metrics": {"x": 2, "extra": 0}}, (False, True)),
        ({"decision": "refuse"}, (True, False)),
    ]:
        score = grade(
            trial, key.model_copy(update=updates).model_dump_json().encode(), key, atol=1e-6
        )
        assert (score.numerical, score.methodological) == expected
        assert not score.passed and score.status == STAMP
    for invalid in (b"", b"null", b"{}", key.model_dump_json().replace("2.0", "NaN").encode()):
        assert grade(trial, invalid, key, atol=1e-6).outcome == "invalid_answer"
    refusal = Answer(
        decision="refuse",
        method="gate_validation",
        reason="partial_course",
        metrics={},
        estimand="test",
        unit="m",
    )
    assert grade(trial, refusal.model_dump_json().encode(), refusal, atol=0).passed
    assert not grade(
        trial,
        refusal.model_copy(update={"reason": "generic"}).model_dump_json().encode(),
        refusal,
        atol=0,
    ).passed
