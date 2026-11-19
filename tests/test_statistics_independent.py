"""Independent numerical verification of reliability statistics (RES-674).

The reference below is a textbook sums-of-squares ANOVA written without the production
code's scaling or effective-degrees-of-freedom path, plus simulations with known
variance components. Neither shares code with ``statistics.reliability``.
"""

import numpy as np
import pytest

from dynamisbench_soccer.statistics.reliability import bland_altman, icc, paired_reliability


def _anova_point_estimates(x: np.ndarray) -> dict[tuple[str, str], float]:
    n, k = x.shape
    grand = x.mean()
    ss_total = ((x - grand) ** 2).sum()
    ss_rows = k * ((x.mean(axis=1) - grand) ** 2).sum()
    ss_cols = n * ((x.mean(axis=0) - grand) ** 2).sum()
    ss_err = ss_total - ss_rows - ss_cols
    ss_within = ss_total - ss_rows
    msr = ss_rows / (n - 1)
    msc = ss_cols / (k - 1)
    mse = ss_err / ((n - 1) * (k - 1))
    msw = ss_within / (n * (k - 1))
    return {
        ("one_way_random", "single"): (msr - msw) / (msr + (k - 1) * msw),
        ("one_way_random", "average"): (msr - msw) / msr,
        ("two_way_random", "single"): (msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n),
        ("two_way_random", "average"): (msr - mse) / (msr + (msc - mse) / n),
        ("two_way_mixed", "single"): (msr - mse) / (msr + (k - 1) * mse),
        ("two_way_mixed", "average"): (msr - mse) / msr,
    }


@pytest.mark.parametrize("seed", range(8))
def test_icc_matches_independent_sums_of_squares(seed):
    rng = np.random.default_rng(seed)
    n, k = int(rng.integers(5, 40)), int(rng.integers(2, 6))
    data = rng.normal(0, 3, (n, 1)) + rng.normal(0, 1, (1, k)) + rng.normal(0, 1.5, (n, k))
    data = data * 1000 + 5e4  # offset and scale must not change the answer
    for (model, kind), expected in _anova_point_estimates(data).items():
        got = icc(data, model=model, type=kind, missing="raise")
        assert got.estimate == pytest.approx(expected, abs=1e-9)
        assert got.interval[0] <= got.estimate <= got.interval[1]


def test_icc_recovers_known_variance_components():
    rng = np.random.default_rng(20261119)
    n, k, sd_b, sd_e = 20000, 3, 2.0, 1.0
    data = rng.normal(0, sd_b, (n, 1)) + rng.normal(0, sd_e, (n, k))
    truth = sd_b**2 / (sd_b**2 + sd_e**2)
    got = icc(data, model="one_way_random", type="single", missing="raise")
    assert got.estimate == pytest.approx(truth, abs=0.01)
    assert got.interval[0] < truth < got.interval[1]


def test_confidence_interval_coverage_is_near_nominal():
    rng = np.random.default_rng(7)
    truth, hits, trials = 4 / 5, 0, 400
    for _ in range(trials):
        data = rng.normal(0, 2, (15, 1)) + rng.normal(0, 1, (15, 3))
        low, high = icc(data, model="one_way_random", type="single", missing="raise").interval
        hits += low <= truth <= high
    assert 0.91 <= hits / trials <= 0.99


def test_typical_error_and_bland_altman_recover_simulated_truth():
    rng = np.random.default_rng(3)
    subject = rng.normal(50, 8, 50000)
    sd_e, bias = 1.5, 0.4
    pairs = np.column_stack(
        (
            subject + rng.normal(0, sd_e, subject.size),
            subject + bias + rng.normal(0, sd_e, subject.size),
        )
    )
    rel = paired_reliability(pairs, single_measure_icc=0.97, missing="raise")
    assert rel.typical_error == pytest.approx(sd_e, rel=0.02)
    ba = bland_altman(pairs, missing="raise")
    assert ba.bias == pytest.approx(bias, abs=0.03)
    assert ba.sd_difference == pytest.approx(np.sqrt(2) * sd_e, rel=0.02)
    assert ba.limits[1] - ba.limits[0] == pytest.approx(2 * 1.959964 * ba.sd_difference, rel=1e-6)
    assert rel.mdc95 == pytest.approx(1.959964 * np.sqrt(2) * rel.sem, rel=1e-6)
