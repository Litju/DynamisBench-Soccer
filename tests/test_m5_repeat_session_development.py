"""Exercise conditional same-method reliability and individual change as diagnostics."""

import json
from pathlib import Path

import pytest

from dynamisbench_soccer.m4.grader import MeasurementSubmission
from dynamisbench_soccer.m4.key_store import digest
from dynamisbench_soccer.m4.task_registry import load_registry
from dynamisbench_soccer.m4.trials import TrialContext, run_episode
from dynamisbench_soccer.statistics.change import BaselinePolicy, individual_change
from dynamisbench_soccer.statistics.reliability import paired_reliability

ROOT = Path(__file__).resolve().parents[1]


def test_repeat_sessions_reconstruct_and_refuse_without_scored_admission(tmp_path):
    catalog = (ROOT / "benchmark/reviewed-slices.json").read_bytes()
    context = TrialContext(ROOT, catalog, digest(catalog), tmp_path)
    task = load_registry(ROOT / "benchmark/tasks")["lps-repeat-session-change", 1]
    assert task.engine == "m5-measurement.v1"
    source = (ROOT / "scripts/m5_controls/repeat_session_development.py").read_bytes()
    report = run_episode(context, task, "independent-summary-diagnostic", source, seed=704)
    replay = run_episode(context, task, "independent-summary-diagnostic", source, seed=704)
    assert report.admission.status == "DEVELOPMENT_ONLY_NOT_QUALIFIED"
    assert report.terminal_reason is replay.terminal_reason is None
    assert len(report.attempts) == len(replay.attempts) == 4

    answers = []
    for attempt, repeated in zip(report.attempts, replay.attempts, strict=True):
        assert attempt.grade.outcome == "EXCLUDED_FROM_SCORED_BENCHMARK"
        assert attempt.execution.outcome == "graded" and attempt.execution.workspace_removed
        assert attempt.output_sha256 == repeated.output_sha256
        assert attempt.candidate_payload_sha256 == repeated.candidate_payload_sha256
        answers.append(MeasurementSubmission.model_validate_json(attempt.execution.stdout))

    for index, scope in enumerate(("likely_real", "within_noise")):
        fixture = json.loads((ROOT / task.artifacts[index].path).read_bytes())
        answer = answers[index]
        rel = paired_reliability(
            [(row["baseline"], row["retest"]) for row in fixture["cohort"]],
            single_measure_icc=fixture["declared_single_measure_icc"],
            missing="raise",
            swc_multiplier=fixture["swc_multiplier"],
        )
        change = individual_change(
            [row["trials_m_s"] for row in fixture["baseline"]],
            fixture["current"]["trials_m_s"],
            policy=BaselinePolicy(**fixture["policy"]),
            trial_typical_error=rel.typical_error,
            confidence=fixture["confidence"],
        )
        assert change.baseline_trial_counts == (2, 3, 2)
        assert change.classification == scope
        assert answer.decision == "compute" and answer.quantity is not None
        assert answer.quantity.value == pytest.approx(change.change, abs=1e-10, rel=0)
        assert answer.quantity.uncertainty == pytest.approx(
            change.interval[1] - change.change, abs=1e-10, rel=0
        )
        assert [x.value for x in answer.operands] == pytest.approx(
            [
                rel.typical_error,
                rel.swc,
                change.baseline_mean,
                change.current_mean,
                change.change_se,
                *change.interval,
            ],
            abs=1e-10,
            rel=0,
        )
        assert set(answer.claims) == {
            "conditional_trial_model",
            "te_estimation_uncertainty_excluded",
            scope,
            "point_reaches_swc" if index == 0 else "point_below_swc",
        }

    assert [(a.reason, a.claims) for a in answers[2:]] == [
        ("not_eligible", ("duplicate_cohort_identity",)),
        ("insufficient_evidence", ("baseline_design_incomplete",)),
    ]
