from math import sqrt
from statistics import NormalDist, mean, stdev

import pytest

from dynamisbench_soccer.reference.motion_models import IsometricRamp
from dynamisbench_soccer.statistics.change import BaselinePolicy, individual_change
from dynamisbench_soccer.statistics.reliability import bland_altman, paired_reliability

# First Wright and mini Wright readings, Bland and Altman (1986), Table 1.
WRIGHT = [494, 395, 516, 434, 476, 557, 413, 442, 650, 433, 417, 656, 267, 478, 178, 423, 427]
MINI = [512, 430, 520, 428, 500, 600, 364, 380, 658, 445, 432, 626, 260, 477, 259, 350, 451]


def test_published_bland_altman_and_signed_difference():
    pairs = list(zip(MINI, WRIGHT, strict=True))
    report = bland_altman(pairs, missing="raise")
    assert report.bias == pytest.approx(-2.1, abs=0.05)
    assert report.sd_difference == pytest.approx(38.8, abs=0.05)
    differences = [wright - mini for mini, wright in pairs]
    z = NormalDist().inv_cdf(0.975)
    expected = (
        mean(differences) - z * stdev(differences),
        mean(differences) + z * stdev(differences),
    )
    assert report.limits == pytest.approx(expected, abs=1e-12)
    reversed_report = bland_altman(list(zip(WRIGHT, MINI, strict=True)), missing="raise")
    assert reversed_report.bias == -report.bias
    assert reversed_report.limits == (-report.limits[1], -report.limits[0])


def test_paired_metrics_from_hand_calculable_variances():
    # Differences are ±1 and ±3, so sample variance is 20/3. Baseline variance is 500/3.
    result = paired_reliability(
        [[10, 11], [20, 19], [30, 33], [40, 37], [999, None]],
        single_measure_icc=0.96,
        swc_multiplier=0.3,
        missing="omit",
    )
    assert (result.n_pairs, result.n_omitted) == (4, 1)
    assert result.typical_error == pytest.approx(sqrt(10 / 3))
    assert result.cv_percent == pytest.approx(4 * sqrt(10 / 3))
    assert result.sem == pytest.approx(sqrt(20 / 3))
    assert result.swc == pytest.approx(sqrt(15))
    assert result.mdc90 == pytest.approx(sqrt(40 / 3) * NormalDist().inv_cdf(0.95))
    assert result.mdc95 == pytest.approx(sqrt(40 / 3) * NormalDist().inv_cdf(0.975))
    assert result.snr == pytest.approx(sqrt(4.5))


def test_degenerate_ratios_and_missing_pairs():
    pairs = [[-1, 1], [1, 3], [None, 9]]
    result = paired_reliability(pairs, single_measure_icc=0.75, missing="omit")
    assert result.typical_error == 0
    assert result.snr is None
    assert result.cv_percent is None
    assert result.swc == pytest.approx(0.2 * sqrt(2))
    assert bland_altman(pairs, missing="omit").limits == (2, 2)
    with pytest.raises(ValueError, match="missing"):
        bland_altman(pairs, missing="raise")
    with pytest.raises(ValueError, match="infinite"):
        bland_altman([[1, 2], [float("inf"), 4]], missing="omit")


@pytest.mark.parametrize(
    ("window", "indices", "baseline", "z_score"),
    [("fixed", (0, 1, 2), 6, 3), ("rolling", (1, 2, 3), 9, 2)],
)
def test_change_on_res669_analytical_force(window, indices, baseline, z_score):
    # At 1 second every prescribed ramp is in its hold phase, so force equals peak.
    def session(*peaks):
        return [IsometricRamp(peak_n=peak).force_n(1.0) for peak in peaks]

    history = [session(2, 4), session(5, 7), session(8, 10), session(11, 13)]
    result = individual_change(
        history,
        session(14, 16),
        trial_typical_error=2,
        policy=BaselinePolicy(window, 3, 3, 2, "raise"),
    )
    delta = 15 - baseline
    se = sqrt(8 / 3)
    oracle = NormalDist(delta, se)
    assert result.baseline_indices == indices
    assert result.baseline_mean == baseline
    assert result.baseline_sd == 3
    assert result.change == delta
    assert result.z_score == z_score
    assert result.change_se == pytest.approx(se)
    assert result.interval == pytest.approx((oracle.inv_cdf(0.025), oracle.inv_cdf(0.975)))
    assert result.probability_in_observed_direction == pytest.approx(1 - oracle.cdf(0))
    assert result.classification == "likely_real"


def test_unequal_trial_counts_and_noise_classification():
    history = [[100], [None, 2, 4], [None], [2, 3, 3, 4]]
    result = individual_change(
        history,
        [2, 3, 4],
        trial_typical_error=2,
        policy=BaselinePolicy("rolling", 3, 2, 2, "omit"),
        confidence=0.9,
    )
    assert result.baseline_indices == (1, 3)
    assert result.baseline_trial_counts == (2, 4)
    assert result.current_trial_count == 3
    assert result.baseline_mean == 3
    assert result.z_score is None
    assert result.change_se == pytest.approx(5 / sqrt(12))
    bound = NormalDist(0, 5 / sqrt(12)).inv_cdf(0.95)
    assert result.interval == pytest.approx((-bound, bound))
    assert result.probability_in_observed_direction == 0.5
    assert result.classification == "within_noise"


def test_decrease_and_insufficient_sessions():
    policy = BaselinePolicy("rolling", 2, 2, 2, "omit")
    history = [[10, 10], [10, 10], [11, 11]]
    result = individual_change(history, [5, 5], policy=policy, trial_typical_error=1)
    assert result.change == -5.5
    assert result.interval[1] < 0
    assert result.classification == "likely_real"
    with pytest.raises(ValueError, match="baseline sessions"):
        individual_change([*history, [None]], [5, 5], policy=policy, trial_typical_error=1)
    with pytest.raises(ValueError, match="current session"):
        individual_change(history, [5, None], policy=policy, trial_typical_error=1)
