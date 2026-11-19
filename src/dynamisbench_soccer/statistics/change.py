"""Individual change against prespecified historical sessions."""

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from scipy.stats import norm

from dynamisbench_soccer.statistics.reliability import Missing, _complete


@dataclass(frozen=True)
class BaselinePolicy:
    window: Literal["fixed", "rolling"]
    window_sessions: int
    min_sessions: int
    min_trials: int
    missing: Missing

    def __post_init__(self) -> None:
        if self.window not in ("fixed", "rolling") or self.missing not in ("raise", "omit"):
            raise ValueError("unknown baseline window or missing-data policy")
        sizes = (self.window_sessions, self.min_sessions, self.min_trials)
        if any(type(size) is not int for size in sizes):
            raise ValueError("baseline and trial counts must be integers")
        if not 2 <= self.min_sessions <= self.window_sessions or self.min_trials < 1:
            raise ValueError("require at least two baseline sessions and one trial per session")


@dataclass(frozen=True)
class IndividualChange:
    policy: BaselinePolicy
    baseline_mean: float
    baseline_sd: float
    current_mean: float
    change: float
    z_score: float | None
    change_se: float
    confidence: float
    interval: tuple[float, float]
    probability_in_observed_direction: float
    classification: Literal["likely_real", "within_noise"]
    baseline_indices: tuple[int, ...]
    baseline_trial_counts: tuple[int, ...]
    current_trial_count: int
    trial_typical_error: float


def individual_change(
    history: Sequence[ArrayLike],
    current: ArrayLike,
    *,
    policy: BaselinePolicy,
    trial_typical_error: float,
    confidence: float = 0.95,
) -> IndividualChange:
    """Compare a current session with equally weighted prior session means."""
    if not isfinite(trial_typical_error) or trial_typical_error <= 0:
        raise ValueError("trial_typical_error must be finite and positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    start = max(len(history) - policy.window_sessions, 0) if policy.window == "rolling" else 0
    baseline = []
    for index, session in enumerate(history[start : start + policy.window_sessions], start):
        trials = _complete(session, ndim=1, missing=policy.missing)
        if len(trials) >= policy.min_trials:
            baseline.append((index, len(trials), float(trials.mean())))
        elif policy.missing == "raise":
            raise ValueError("baseline session has too few usable trials")
    if len(baseline) < policy.min_sessions:
        raise ValueError("too few eligible baseline sessions")

    trials = _complete(current, ndim=1, missing=policy.missing)
    if len(trials) < policy.min_trials:
        raise ValueError("current session has too few usable trials")
    indices, counts, means = zip(*baseline, strict=True)
    baseline_mean = float(np.mean(means))
    baseline_sd = float(np.std(means, ddof=1))
    current_mean = float(trials.mean())
    delta = current_mean - baseline_mean
    variance_factor = 1 / len(trials) + sum(1 / count for count in counts) / len(counts) ** 2
    se = trial_typical_error * sqrt(variance_factor)
    if not isfinite(se) or se == 0:
        raise ValueError("change standard error exceeds the supported numeric range")
    half_width = float(norm.isf((1 - confidence) / 2)) * se
    lower, upper = delta - half_width, delta + half_width
    z_score = delta / baseline_sd if baseline_sd > 0 else None
    if not all(isfinite(v) for v in (baseline_sd, lower, upper, z_score or 0)):
        raise ValueError("change estimates exceed the supported numeric range")
    return IndividualChange(
        policy,
        baseline_mean,
        baseline_sd,
        current_mean,
        delta,
        z_score,
        se,
        confidence,
        (lower, upper),
        float(norm.cdf(abs(delta) / se)),
        "likely_real" if lower > 0 or upper < 0 else "within_noise",
        indices,
        counts,
        len(trials),
        trial_typical_error,
    )
