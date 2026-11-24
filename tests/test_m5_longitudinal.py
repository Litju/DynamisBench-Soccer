"""Independent rational-contrast and usage controls for the analytical study."""

import json
import math
from fractions import Fraction
from pathlib import Path
from statistics import NormalDist

import pytest

from dynamisbench_soccer.m4.measurement import replay
from dynamisbench_soccer.m4.task_registry import load_registry
from dynamisbench_soccer.statistics.longitudinal import fixed_session_change

ROOT = Path(__file__).resolve().parents[1]


def _load(index=1):
    path = ROOT / "benchmark/fixtures/m5-repeat-session" / f"repeat-session-{index:02d}.json"
    return json.loads(path.read_bytes())


def _mean(seq):
    return sum(seq, Fraction()) / len(seq)


def _variance(seq):
    # Pairwise squared differences remove the mean algebraically.
    return sum((v - w) ** 2 for i, v in enumerate(seq) for w in seq[:i]) / (
        len(seq) * (len(seq) - 1)
    )


def _contrast_oracle(raw):
    group = [(Fraction(str(r["baseline"])), Fraction(str(r["retest"]))) for r in raw["cohort"]]
    old = [[Fraction(str(x)) for x in s["trials_m_s"]] for s in raw["baseline"][:3]]
    new = [Fraction(str(x)) for x in raw["current"]["trials_m_s"]]
    coefficients = [
        *(Fraction(-1, len(old) * len(s)) for s in old for _ in s),
        *(Fraction(1, len(new)) for _ in new),
    ]
    contrast = sum(
        coef * v for coef, v in zip(coefficients, [v for s in old for v in s] + new, strict=True)
    )
    te2 = _variance([b - a for a, b in group]) / 2
    swc2 = Fraction(str(raw["swc_multiplier"])) ** 2 * _variance([a for a, _ in group])
    se = math.sqrt(te2 * sum(x**2 for x in coefficients))
    half = NormalDist().inv_cdf(0.975) * se
    return (
        float(contrast),
        math.sqrt(te2),
        math.sqrt(swc2),
        float(_mean([_mean(s) for s in old])),
        float(_mean(new)),
        se,
        half,
    )


def _identity_oracle(subjects, individual, sessions):
    if any(not name for name in (*subjects, individual, *sessions)):
        return "missing_identity"
    if individual in subjects or any(subjects.count(name) > 1 for name in subjects):
        return "duplicate_cohort_identity"
    if any(sessions.count(name) > 1 for name in sessions):
        return "duplicate_session_identity"
    return None


@pytest.mark.parametrize(("index", "classification"), [(1, "likely_real"), (2, "within_noise")])
def test_longitudinal_matches_independent_contrast(index, classification):
    raw = _load(index)
    expected, te, swc, old, new, se, half = _contrast_oracle(raw)
    result = fixed_session_change(raw)
    assert not isinstance(result, str)
    change = result.change
    assert change.baseline_indices == (0, 1, 2)
    assert change.baseline_trial_counts == (2, 3, 2)
    assert change.classification == classification
    assert result.reaches_swc is (abs(expected) >= swc)
    assert [
        change.change,
        result.reliability.typical_error,
        result.reliability.swc,
        change.baseline_mean,
        change.current_mean,
        change.change_se,
        result.half_width,
        *change.interval,
    ] == pytest.approx(
        [expected, te, swc, old, new, se, half, expected - half, expected + half], abs=1e-10, rel=0
    )


def test_fixed_window_uses_equal_session_weights_and_preserves_direction():
    raw = _load()
    raw["baseline"][1]["trials_m_s"] = [7.07, 7.1, 7.13]
    raw["baseline"].append({"session": "extra", "trials_m_s": [90, 91]})
    expected, _, _, old, *_ = _contrast_oracle(raw)
    result = fixed_session_change(raw)
    assert not isinstance(result, str)
    assert result.change.change < 0
    assert result.change.change == pytest.approx(expected, abs=1e-10, rel=0)
    pooled = [x for session in raw["baseline"][:3] for x in session["trials_m_s"]]
    assert abs(old - sum(pooled) / len(pooled)) > 0.01
    assert result.change.baseline_mean == pytest.approx(old, abs=1e-10, rel=0)


def test_identity_and_baseline_design_controls():
    raw = _load(3)
    assert (
        fixed_session_change(raw)
        == _identity_oracle(
            [r["subject"] for r in raw["cohort"]],
            raw["individual"],
            [r["session"] for r in (*raw["baseline"], raw["current"])],
        )
        == "duplicate_cohort_identity"
    )
    raw = _load()
    del raw["cohort"][0]["subject"]
    assert fixed_session_change(raw) == "missing_identity"
    raw = _load()
    raw["current"]["session"] = raw["baseline"][0]["session"]
    assert fixed_session_change(raw) == "duplicate_session_identity"
    raw = _load(4)
    assert fixed_session_change(raw) == "baseline_design_incomplete"
    raw["baseline"].append({"session": "extra", "trials_m_s": [6, 7]})
    assert fixed_session_change(raw) == "baseline_design_incomplete"
    raw = _load()
    raw["population"] = "shared correlated sessions"
    assert fixed_session_change(raw) == "unsupported_trial_design"
    raw = _load()
    raw["cohort"] = raw["cohort"][:2]
    assert fixed_session_change(raw) == "cohort_error_unavailable"


def test_declared_sources_and_conditional_replay_are_bound():
    task = load_registry(ROOT / "benchmark/tasks")["lps-repeat-session-change", 1]
    for index, item in enumerate(task.checkpoints, 1):
        inputs = {a.name: (ROOT / a.path).read_bytes() for a in task.artifacts}
        projection = task.candidate_payload(item.id, inputs, seed=704)
        payload = json.loads(projection)
        assert set(payload["inputs"]) == {a.name for a in task.artifacts[:index]}
        reference = replay("fixed-session-change", _load(index))
        assert reference == replay("fixed-session-change", _load(index))
        if reference.reason == "eligible":
            assert "te_estimation_uncertainty_excluded" in reference.claims
            assert "conditional_trial_model" in reference.claims
            assert "empirical_reliability" not in reference.claims
