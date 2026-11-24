"""Eligibility and computations for the fixed synthetic repeat-session reference."""

from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError
from scipy.stats import norm

from dynamisbench_soccer.statistics.change import (
    BaselinePolicy,
    IndividualChange,
    individual_change,
)
from dynamisbench_soccer.statistics.reliability import PairedReliability, paired_reliability

Identity = Annotated[str, Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_.-]*$")]
Speed = Annotated[float, Field(ge=0)]


class _Input(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid", allow_inf_nan=False)


class _CohortRow(_Input):
    subject: Identity | None = None
    baseline: Speed
    retest: Speed


class _Session(_Input):
    session: Identity | None = None
    trials_m_s: list[Speed]


class _FixedPolicy(_Input):
    window: Literal["fixed"]
    window_sessions: Literal[3]
    min_sessions: Literal[3]
    min_trials: Literal[2]
    missing: Literal["raise"]


class _Study(_Input):
    measurement: Literal["same-method horizontal tag speed session summaries in m/s"]
    population: str
    individual: Identity | None = None
    cohort: list[_CohortRow]
    baseline: list[_Session]
    current: _Session
    policy: _FixedPolicy
    confidence: Annotated[float, Field(gt=0, lt=1)]
    swc_multiplier: Annotated[float, Field(gt=0)]
    declared_single_measure_icc: Annotated[float, Field(ge=0, le=1)]


Refusal = Literal[
    "unsupported_trial_design",
    "missing_identity",
    "duplicate_cohort_identity",
    "duplicate_session_identity",
    "baseline_design_incomplete",
    "cohort_error_unavailable",
]


@dataclass(frozen=True)
class LongitudinalChange:
    reliability: PairedReliability
    change: IndividualChange
    half_width: float
    reaches_swc: bool


def fixed_session_change(raw: dict[str, object]) -> LongitudinalChange | Refusal:
    """Evaluate the declared synthetic design, conditional on its plug-in trial TE."""
    try:
        data = _Study.model_validate(raw)
    except ValidationError:
        return "unsupported_trial_design"
    if data.confidence != 0.95 or data.population != (
        "independently prescribed synthetic runs with independent homogeneous "
        "trial errors and stable cohort underlying speed"
    ):
        return "unsupported_trial_design"
    cohort_ids = {row.subject for row in data.cohort}
    sessions = [row.session for row in (*data.baseline, data.current)]
    if data.individual is None or None in cohort_ids or None in sessions:
        return "missing_identity"
    if len(cohort_ids) != len(data.cohort) or data.individual in cohort_ids:
        return "duplicate_cohort_identity"
    if len(set(sessions)) != len(sessions):
        return "duplicate_session_identity"
    if len(cohort_ids) < 3:
        return "cohort_error_unavailable"
    policy = BaselinePolicy(
        window="fixed", window_sessions=3, min_sessions=3, min_trials=2, missing="raise"
    )
    history = [session.trials_m_s for session in data.baseline[: policy.window_sessions]]
    current = data.current.trials_m_s
    if len(history) < policy.min_sessions or any(
        len(trials) < policy.min_trials for trials in (*history, current)
    ):
        return "baseline_design_incomplete"
    reliability = paired_reliability(
        [(row.baseline, row.retest) for row in data.cohort],
        single_measure_icc=data.declared_single_measure_icc,
        missing="raise",
        swc_multiplier=data.swc_multiplier,
    )
    if not reliability.typical_error:
        return "cohort_error_unavailable"
    result = individual_change(
        history,
        current,
        policy=policy,
        trial_typical_error=reliability.typical_error,
        confidence=data.confidence,
    )
    half_width = float(norm.isf((1 - data.confidence) / 2)) * result.change_se
    return LongitudinalChange(
        reliability, result, half_width, abs(result.change) >= reliability.swc
    )
