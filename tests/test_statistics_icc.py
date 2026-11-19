import numpy as np
import pytest

from dynamisbench_soccer.statistics.reliability import icc

# Shrout-Fleiss Table 2, also distributed as Stata r19 judges.dta.
JUDGES = np.array(
    [[9, 2, 5, 8], [6, 1, 3, 2], [8, 4, 6, 8], [7, 1, 2, 6], [10, 5, 6, 9], [6, 2, 4, 7]]
)


@pytest.mark.parametrize(
    ("model", "type", "expected"),
    [
        ("one_way_random", "single", (0.1657418, -0.1329323, 0.7225601)),
        ("one_way_random", "average", (0.4427971, -0.8844422, 0.9124154)),
        ("two_way_random", "single", (0.2897638, 0.0187865, 0.7610844)),
        ("two_way_random", "average", (0.6200505, 0.0711368, 0.9272320)),
        ("two_way_mixed", "single", (0.7148407, 0.3424648, 0.9458583)),
        ("two_way_mixed", "average", (0.9093155, 0.6756747, 0.9858917)),
    ],
)
def test_stata_manual_examples(model, type, expected):
    result = icc(JUDGES, model=model, type=type, missing="raise")
    assert (result.estimate, *result.interval) == pytest.approx(expected, abs=5.1e-8)
    assert (result.model, result.type, result.n_subjects, result.n_ratings) == (model, type, 6, 4)
    assert result.definition == (
        "consistency" if model == "two_way_mixed" else "absolute_agreement"
    )


def test_listwise_omission_and_no_imputation():
    ratings = np.vstack((JUDGES, [100, 100, np.nan, 100]))
    result = icc(ratings, model="two_way_random", type="single", missing="omit")
    assert (result.n_subjects, result.n_omitted) == (6, 1)
    assert result.estimate == pytest.approx(0.2897638, abs=5.1e-8)
    with pytest.raises(ValueError, match="missing"):
        icc(ratings, model="two_way_random", type="single", missing="raise")


@pytest.mark.parametrize("model", ["one_way_random", "two_way_random", "two_way_mixed"])
def test_identical_ratings_and_constant_design(model):
    perfect = icc([[1, 1], [2, 2], [3, 3]], model=model, type="single", missing="raise")
    assert (perfect.estimate, *perfect.interval) == (1, 1, 1)
    with pytest.raises(ValueError, match="constant"):
        icc([[1, 1], [1, 1]], model=model, type="single", missing="raise")


def test_negative_icc_and_nondefault_ci():
    result = icc(
        [[1, 3], [3, 1], [2, 2]],
        model="one_way_random",
        type="single",
        missing="raise",
        confidence=0.9,
    )
    assert result.estimate == -1
    assert result.interval == (-1, -1)
