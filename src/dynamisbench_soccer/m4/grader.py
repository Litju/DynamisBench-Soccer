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
from dynamisbench_soccer.m4 import measurement
from dynamisbench_soccer.m4.admission import ReviewedSlice, evaluate
from dynamisbench_soccer.m4.contracts import _unique_object
from dynamisbench_soccer.m4.custody import load_private
from dynamisbench_soccer.m4.task_registry import Name, Nonnegative, RegistryTask, Schema, Text
from dynamisbench_soccer.signals.schema import BundleMeta, SignalBundle

Reason = measurement.Reason


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


class ScientificResponse(Schema):
    decision: Literal["compute", "refuse"]
    method: Name
    reason: Reason
    selected_trials: tuple[Name, ...]
    citations: tuple[Text, ...]
    claims: tuple[Name, ...]
    quantity: Quantity | None
    operands: tuple[Operand, ...]
    corrects: Name | None = None

    @model_validator(mode="after")
    def decision_matches(self) -> ScientificResponse:
        if (self.decision == "compute") != (self.reason == "eligible"):
            raise ValueError("decision conflicts with eligibility")
        if self.decision == "refuse" and (self.quantity is not None or self.operands):
            raise ValueError("a refusal cannot assert a computed result")
        if self.decision == "compute" and (self.quantity is None or not self.operands):
            raise ValueError("a computation requires a quantity and evidence trail")
        return self


class Submission(ScientificResponse):
    version: Literal["m4.formal-answer.v1"]
    explanation: Text


class MeasurementSubmission(ScientificResponse):
    version: Literal["m5.measurement-answer.v1"]


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


def _check_submission_json(raw: bytes) -> None:
    if len(raw) > 16 * 1024:
        raise ValueError("submission exceeds the size limit")
    json.loads(raw, object_pairs_hook=_unique_object)


def parse_submission(raw: bytes) -> Submission:
    _check_submission_json(raw)
    return Submission.model_validate_json(raw)


def _compare(
    submission: ScientificResponse,
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


def _measurement_binding(task: RegistryTask, review: ReviewedSlice) -> bool:
    method = task.methods[0]
    procedure = measurement.PROCEDURES.get(method.id)
    return (
        task.grader == measurement.GRADER
        and task.task.answer_schema == measurement.ANSWER_VERSION
        and len(task.methods) == 1
        and procedure is not None
        and review.source == procedure.source
        and review.kind == procedure.kind
        and set(review.private_inputs_sha256) == {"expected", "oracle", "provenance"}
        and method.specification == {"operation": method.id}
        and (method.estimand, method.dimension, method.unit)
        == (procedure.estimand, procedure.dimension, procedure.unit)
        and method.source_claims == procedure.claims
        and all(
            f"src/dynamisbench_soccer/engines/{dependency}" in review.implementations
            for dependency in procedure.dependencies
        )
    )


def _replay_measurement(
    expected: ExpectedMethod, task: RegistryTask, payload: bytes, artifacts: tuple[str, ...]
) -> bool:
    if len(artifacts) != 1:
        return False
    artifact = artifacts[0]
    method = task.methods[0]
    reading = measurement.replay(method.id, json.loads(json.loads(payload)["inputs"][artifact]))
    computed = reading.reason == "eligible"
    if computed and (reading.value is None or reading.radius is None):
        return False
    quantity = (
        Quantity(
            value=reading.value,
            unit=method.unit,
            dimension=method.dimension,
            estimand=method.estimand,
            uncertainty=reading.radius,
        )
        if reading.value is not None
        else None
    )
    evidence = MeasurementSubmission(
        version="m5.measurement-answer.v1",
        decision="compute" if computed else "refuse",
        method=method.id,
        reason=reading.reason,
        selected_trials=(artifact,) if computed else (),
        claims=reading.claims,
        citations=method.source_claims,
        quantity=quantity,
        operands=tuple(
            Operand(artifact=artifact, field=field, value=value)
            for field, value in reading.operands
        ),
        corrects=expected.corrects,
    )
    return (
        expected.reasons == (reading.reason,)
        and set(expected.allowed_claims) == set(expected.required_claims) == set(reading.claims)
        and expected.allowed_citations == expected.required_citations == method.source_claims
        and not _compare(evidence, expected, method.absolute_tolerance)
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
    is_measurement = task.engine == measurement.ENGINE
    legacy = (
        task.grader == "formal.v1"
        and task.engine in {"native-lps-speed.v1", "elapsed-subtraction.v1"}
        and (
            not lps
            or (
                review.source == "gnss_lps.lps"
                and review.kind == "native_signal"
                and set(review.implementations) >= LPS_IMPLEMENTATIONS
            )
        )
    )
    if not (legacy or (is_measurement and _measurement_binding(task, review))):
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
            if expected.corrects != checkpoint.corrects or (
                target is not None
                and (target.unit, target.dimension, target.estimand)
                != (method.unit, method.dimension, method.estimand)
            ):
                raise ValueError("inconsistent private target")
            if is_measurement:
                replay_matches = _replay_measurement(expected, task, public, checkpoint.artifacts)
            elif lps:
                replay_matches = method.specification == LPS_SPEC and _replay_lps(
                    expected, public, checkpoint.artifacts, method.absolute_tolerance
                )
            else:
                replay_matches = method.specification == {
                    "operation": "finish_minus_start"
                } and _replay(expected, public, method.absolute_tolerance)
            if not replay_matches:
                raise ValueError("unsupported or inconsistent replay")
    except (OSError, ValueError, KeyError, TypeError, OverflowError, RecursionError):
        return Outcome(
            outcome="EXCLUDED_FROM_SCORED_BENCHMARK", findings=("invalid_private_reference",)
        )
    try:
        candidate: ScientificResponse
        if is_measurement:
            _check_submission_json(raw)
            candidate = MeasurementSubmission.model_validate_json(raw)
        else:
            candidate = parse_submission(raw)
    except (ValidationError, ValueError, RecursionError):
        return Outcome(outcome="FAIL", findings=("invalid_submission",))
    if candidate.method not in references:
        return Outcome(outcome="FAIL", findings=("unsupported_method",))
    findings = _compare(
        candidate, references[candidate.method], methods[candidate.method].absolute_tolerance
    )
    return Outcome(outcome="FAIL" if findings else "PASS", findings=findings)
