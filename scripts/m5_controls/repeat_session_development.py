"""Standard-library diagnostic for synthetic same-method retest and session change."""

import json
import math
import statistics
import sys


def diagnose(artifact, data):
    cohort, baseline, current = data["cohort"], data["baseline"], data["current"]
    answer = {
        "version": "m5.measurement-answer.v1",
        "method": "fixed-session-change",
        "citations": ["ST-001", "ST-005", "ST-006"],
        "selected_trials": [],
        "operands": [],
        "quantity": None,
    }
    identities = {row["subject"] for row in cohort}
    if len(identities) != len(cohort) or data["individual"] in identities:
        return dict(
            answer, decision="refuse", reason="not_eligible", claims=["duplicate_cohort_identity"]
        )
    policy = data["policy"]
    baseline = baseline[: policy["window_sessions"]]
    sessions = [*baseline, current]
    trials = [row["trials_m_s"] for row in sessions]
    if (
        len(identities) < 3
        or len({row["session"] for row in sessions}) != len(sessions)
        or len(baseline) < policy["min_sessions"]
        or min(map(len, trials)) < policy["min_trials"]
    ):
        return dict(
            answer,
            decision="refuse",
            reason="insufficient_evidence",
            claims=["baseline_design_incomplete"],
        )

    error = statistics.stdev(row["retest"] - row["baseline"] for row in cohort) / math.sqrt(2)
    swc = data["swc_multiplier"] * statistics.stdev(row["baseline"] for row in cohort)
    *history, now = trials
    prior_mean = statistics.mean(map(statistics.mean, history))
    now_mean = statistics.mean(now)
    change = now_mean - prior_mean
    n = len(history)
    contributions = [1 / len(now), *(1 / (len(session) * n**2) for session in history)]
    critical = statistics.NormalDist().inv_cdf((1 + data["confidence"]) / 2)
    se = error * math.sqrt(sum(contributions))
    width = critical * se
    low, high = change - width, change + width
    operands = [
        ("typical_error", error),
        ("swc", swc),
        ("baseline_mean", prior_mean),
        ("current_mean", now_mean),
        ("change_se", se),
        ("interval_lower", low),
        ("interval_upper", high),
    ]
    return dict(
        answer,
        decision="compute",
        reason="eligible",
        claims=[
            "conditional_trial_model",
            "te_estimation_uncertainty_excluded",
            "likely_real" if low > 0 or high < 0 else "within_noise",
            "point_reaches_swc" if abs(change) >= swc else "point_below_swc",
        ],
        selected_trials=[artifact],
        operands=[
            {"artifact": artifact, "field": field, "value": value} for field, value in operands
        ],
        quantity={
            "value": change,
            "uncertainty": width,
            "unit": "m/s",
            "dimension": "speed",
            "estimand": "horizontal_tag_speed_session_mean_change",
        },
    )


if __name__ == "__main__":
    payload = json.load(sys.stdin)
    (name,) = payload["checkpoint"]["artifacts"]
    print(json.dumps(diagnose(name, json.loads(payload["inputs"][name])), allow_nan=False))
