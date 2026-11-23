"""Structured grading controls for formal task checkpoints."""

from __future__ import annotations

import json
import math
from typing import Annotated, Literal

from pydantic import Field, ValidationError, model_validator

from dynamisbench_soccer.m4.contracts import _unique_object
from dynamisbench_soccer.m4.task_registry import Name, Nonnegative, Schema, Text

Reason = Literal["eligible", "not_eligible", "insufficient_evidence", "unsupported_claim"]


class Quantity(Schema):
    value: float
    unit: Text
    dimension: Text
    estimand: Text
    uncertainty: Nonnegative | None


class Operand(Schema):
    artifact: Name
    field: Name
    value: float


class Submission(Schema):
    version: Literal["m4.formal-answer.v1"]
    decision: Literal["compute", "refuse"]
    method: Name
    reason: Reason
    explanation: Text
    selected_trials: tuple[Name, ...]
    citations: tuple[Text, ...]
    claims: tuple[Name, ...]
    quantity: Quantity | None
    operands: tuple[Operand, ...]
    corrects: Name | None = None

    @model_validator(mode="after")
    def decision_matches(self) -> Submission:
        if (self.decision == "compute") != (self.reason == "eligible"):
            raise ValueError("decision conflicts with eligibility")
        if self.decision == "refuse" and (self.quantity is not None or self.operands):
            raise ValueError("a refusal cannot assert a computed result")
        if self.decision == "compute" and (self.quantity is None or not self.operands):
            raise ValueError("a computation requires a quantity and evidence trail")
        return self


class ExpectedMethod(Schema):
    decision: Literal["compute", "refuse"]
    reasons: tuple[Reason, ...] = Field(min_length=1)
    selected_trials: tuple[Name, ...]
    required_citations: tuple[Text, ...]
    allowed_citations: tuple[Text, ...]
    required_claims: tuple[Name, ...]
    allowed_claims: tuple[Name, ...]
    quantity: Quantity | None
    operands: tuple[Operand, ...]
    corrects: Name | None = None


class Outcome(Schema):
    version: Literal["m4.grade.v1"] = "m4.grade.v1"
    outcome: Literal["PASS", "FAIL", "EXCLUDED_FROM_SCORED_BENCHMARK"]
    findings: tuple[str, ...] = ()


def parse_submission(raw: bytes) -> Submission:
    if len(raw) > 16 * 1024:
        raise ValueError("submission exceeds the size limit")
    json.loads(raw, object_pairs_hook=_unique_object)
    return Submission.model_validate_json(raw)


def _compare(
    submission: Submission,
    expected: ExpectedMethod,
    absolute_tolerance: Annotated[float, Field(ge=0)],
) -> tuple[str, ...]:
    findings = []
    if submission.decision != expected.decision or submission.reason not in expected.reasons:
        findings.append("eligibility_or_refusal")
    if submission.selected_trials != expected.selected_trials:
        findings.append("trial_selection")
    if (
        not set(expected.required_citations)
        <= set(submission.citations)
        <= set(expected.allowed_citations)
    ):
        findings.append("source_citations")
    if not set(expected.required_claims) <= set(submission.claims) <= set(expected.allowed_claims):
        findings.append("interpretation_boundary")
    if submission.corrects != expected.corrects:
        findings.append("correction_lineage")
    if submission.operands != expected.operands:
        findings.append("derivation")
    actual, target = submission.quantity, expected.quantity
    if actual is None or target is None:
        if actual is not target:
            findings.append("quantity_absent_or_unexpected")
        return tuple(findings)
    if (actual.unit, actual.dimension) != (target.unit, target.dimension):
        findings.append("unit_or_dimension")
    if actual.estimand != target.estimand:
        findings.append("estimand")
    if not math.isclose(actual.value, target.value, abs_tol=absolute_tolerance, rel_tol=0):
        findings.append("value")
    if actual.uncertainty != target.uncertainty and (
        actual.uncertainty is None
        or target.uncertainty is None
        or not math.isclose(
            actual.uncertainty, target.uncertainty, abs_tol=absolute_tolerance, rel_tol=0
        )
    ):
        findings.append("uncertainty")
    return tuple(findings)


def check_control(raw: bytes, expected: ExpectedMethod, tolerance: float) -> tuple[str, ...]:
    """Return diagnostic error classes for a known-answer control, without a score."""
    if tolerance < 0 or not math.isfinite(tolerance):
        raise ValueError("invalid control tolerance")
    try:
        submission = parse_submission(raw)
    except (ValidationError, ValueError, RecursionError):
        return ("invalid_submission",)
    return _compare(submission, expected, tolerance)
