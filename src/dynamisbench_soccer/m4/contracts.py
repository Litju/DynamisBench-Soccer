"""Public task/episode contracts and deterministic grader for the M4 prototype."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, ValidationError

Status = Literal["DEVELOPMENT_ONLY_NOT_QUALIFIED"]
STAMP: Status = "DEVELOPMENT_ONLY_NOT_QUALIFIED"


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)

    status: Status = STAMP


class Task(Contract):
    version: Literal["m4.task.v1"] = "m4.task.v1"
    task_id: str = Field(min_length=1)
    evidence: Literal["analytical", "synthetic"]
    prompt: str = Field(min_length=1)
    inputs: dict[str, JsonValue]


class Episode(Contract):
    version: Literal["m4.episode.v1"] = "m4.episode.v1"
    episode_id: str = Field(min_length=1)
    seed: int
    task: Task


class Answer(Contract):
    version: Literal["m4.answer.v1"] = "m4.answer.v1"
    decision: Literal["compute", "refuse"]
    method: str
    reason: str
    metrics: dict[str, float]


class Score(Contract):
    version: Literal["m4.score.v1"] = "m4.score.v1"
    episode_id: str
    numerical: bool = False
    methodological: bool = False
    passed: bool = False
    outcome: Literal["graded", "invalid_answer", "execution_error", "timeout", "output_limit"]


def grade(episode: Episode, output: bytes, key: Answer, *, atol: float) -> Score:
    """Require the full metric set and the matching method, decision and reason."""
    if not math.isfinite(atol) or atol < 0:
        raise ValueError("grading tolerance must be finite and nonnegative")
    try:
        answer = Answer.model_validate_json(output)
    except ValidationError:
        return Score(episode_id=episode.episode_id, outcome="invalid_answer")
    numerical = answer.metrics.keys() == key.metrics.keys() and all(
        math.isfinite(answer.metrics[name]) and abs(answer.metrics[name] - value) <= atol
        for name, value in key.metrics.items()
    )
    methodological = (answer.decision, answer.method, answer.reason) == (
        key.decision,
        key.method,
        key.reason,
    )
    return Score(
        episode_id=episode.episode_id,
        numerical=numerical,
        methodological=methodological,
        passed=numerical and methodological,
        outcome="graded",
    )
