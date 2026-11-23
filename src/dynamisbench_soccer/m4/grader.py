"""Structured grading controls for formal task checkpoints."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import Field, ValidationError, model_validator

from dynamisbench_soccer.engines.gnss_lps_acquisition import (
    AcquisitionDeclaration,
    AcquisitionRefusal,
)
from dynamisbench_soccer.engines.lps_speed import horizontal_speed, position_error_radius
from dynamisbench_soccer.m4.admission import evaluate
from dynamisbench_soccer.m4.contracts import _unique_object
from dynamisbench_soccer.m4.custody import load_private
from dynamisbench_soccer.m4.task_registry import Name, Nonnegative, RegistryTask, Schema, Text
from dynamisbench_soccer.signals.schema import BundleMeta, SignalBundle

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

    @model_validator(mode="after")
    def consistent_reference(self) -> ExpectedMethod:
        compute = self.decision == "compute"
        if any((reason == "eligible") != compute for reason in self.reasons):
            raise ValueError("reference decision conflicts with eligibility")
        if compute != (self.quantity is not None and bool(self.operands)):
            raise ValueError("reference computation requires quantity and derivation")
        if not compute and (self.quantity is not None or self.operands):
            raise ValueError("reference refusal contains a computed result")
        if not set(self.required_citations) <= set(self.allowed_citations):
            raise ValueError("required citations must be allowed")
        if not set(self.required_claims) <= set(self.allowed_claims):
            raise ValueError("required claims must be allowed")
        return self


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
    absolute_tolerance: float,
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


def _replay(expected: ExpectedMethod, payload: bytes, tolerance: float) -> bool:
    if expected.decision == "refuse":
        return expected.quantity is None and not expected.operands
    if expected.quantity is None or len(expected.operands) != 2:
        return False
    inputs = json.loads(payload)["inputs"]
    try:
        finish, start = [
            json.loads(inputs[item.artifact])[item.field] for item in expected.operands
        ]
    except (KeyError, TypeError, ValueError):
        return False
    if not all(type(value) in (float, int) and math.isfinite(value) for value in (finish, start)):
        return False
    return (
        (finish, start) == tuple(item.value for item in expected.operands)
        and finish >= start
        and math.isclose(finish - start, expected.quantity.value, abs_tol=tolerance, rel_tol=0)
    )


LPS_SPEC = {
    "operation": "native_lps_horizontal_speed",
    "derivative": "three_point",
    "position_filter": "none",
}
LPS_IMPLEMENTATIONS = {
    f"src/dynamisbench_soccer/engines/lps_speed.py::{name}"
    for name in ("horizontal_speed", "position_error_radius")
}


def _replay_lps(
    expected: ExpectedMethod, payload: bytes, artifacts: tuple[str, ...], tolerance: float
) -> bool:
    if len(artifacts) != 1:
        return False
    artifact = artifacts[0]
    data = json.loads(json.loads(payload)["inputs"][artifact])
    row = data["row"]
    if type(row) is not int:
        return False
    channels = {
        name: np.asarray(values, dtype=np.float64) for name, values in data["channels"].items()
    }
    meta = BundleMeta.model_validate(data["meta"])
    if set(channels) != {channel.name for channel in meta.channels}:
        return False
    bundle = SignalBundle(meta, {channel.name: channels[channel.name] for channel in meta.channels})
    declaration = AcquisitionDeclaration.model_validate(data["declaration"])
    try:
        speed = horizontal_speed(bundle, declaration, row)
    except AcquisitionRefusal:
        return (
            expected.decision == "refuse"
            and expected.reasons == ("not_eligible",)
            and not expected.selected_trials
        )
    target = expected.quantity
    if target is None or target.uncertainty is None or expected.selected_trials != (artifact,):
        return False
    time = channels["time"] - channels["time"][0]
    before, center, after = map(float, time[row - 1 : row + 2])
    radius = position_error_radius(before, center, after, data["horizontal_error_m"])
    operands = tuple(
        Operand(artifact=artifact, field=f"{name}.{index}", value=float(channels[name][index]))
        for name in ("time", "position_x", "position_y")
        for index in range(row - 1, row + 2)
    )
    return (
        expected.operands == operands
        and math.isclose(speed, target.value, abs_tol=tolerance, rel_tol=0)
        and math.isclose(radius, target.uncertainty, abs_tol=tolerance, rel_tol=0)
    )


def score_checkpoint(
    root: Path,
    task: RegistryTask,
    catalog_bytes: bytes,
    trusted_catalog_sha256: str,
    custody_root: Path,
    checkpoint_id: str,
    inputs: dict[str, bytes],
    raw: bytes,
) -> Outcome:
    """Recompute admission and load the sealed reference before scoring any submission."""
    admission = evaluate(root, task, catalog_bytes, trusted_catalog_sha256)
    review = admission.reviewed_slice
    if admission.status != "ENGINEERING_ADMITTED" or review is None:
        return Outcome(outcome="EXCLUDED_FROM_SCORED_BENCHMARK", findings=admission.reasons)
    lps = task.engine == "native-lps-speed.v1"
    if (
        task.grader != "formal.v1"
        or task.engine not in {"native-lps-speed.v1", "elapsed-subtraction.v1"}
        or (
            lps
            and (
                review.source != "gnss_lps.lps"
                or review.kind != "native_signal"
                or not set(review.implementations) >= LPS_IMPLEMENTATIONS
            )
        )
    ):
        return Outcome(
            outcome="EXCLUDED_FROM_SCORED_BENCHMARK", findings=("unsupported_runtime_binding",)
        )
    try:
        answer = load_private(custody_root, root, review.answer, admission.task_sha256)
        public = task.candidate_payload(checkpoint_id, inputs, seed=0)
        checkpoint = next(item for item in task.checkpoints if item.id == checkpoint_id)
        keys = answer.checkpoints[checkpoint_id]
        methods = {method.id: method for method in task.methods}
        if keys.keys() != methods.keys() or set(answer.checkpoints) != {
            item.id for item in task.checkpoints
        }:
            raise ValueError("private answer does not cover the declared episode")
        references = {
            name: ExpectedMethod.model_validate_json(json.dumps(value))
            for name, value in keys.items()
        }
        for name, method in methods.items():
            expected = references[name]
            target = expected.quantity
            if (
                expected.corrects != checkpoint.corrects
                or method.specification
                != (LPS_SPEC if lps else {"operation": "finish_minus_start"})
                or (
                    target is not None
                    and (target.unit, target.dimension, target.estimand)
                    != (method.unit, method.dimension, method.estimand)
                )
                or not (
                    _replay_lps(expected, public, checkpoint.artifacts, method.absolute_tolerance)
                    if lps
                    else _replay(expected, public, method.absolute_tolerance)
                )
            ):
                raise ValueError("unsupported or inconsistent replay")
    except (OSError, ValueError, KeyError, TypeError, OverflowError, RecursionError):
        return Outcome(
            outcome="EXCLUDED_FROM_SCORED_BENCHMARK", findings=("invalid_private_reference",)
        )
    try:
        candidate = parse_submission(raw)
    except (ValidationError, ValueError, RecursionError):
        return Outcome(outcome="FAIL", findings=("invalid_submission",))
    if candidate.method not in references:
        return Outcome(outcome="FAIL", findings=("unsupported_method",))
    findings = _compare(
        candidate, references[candidate.method], methods[candidate.method].absolute_tolerance
    )
    return Outcome(outcome="FAIL" if findings else "PASS", findings=findings)
