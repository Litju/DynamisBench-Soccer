"""Balanced ANOVA intraclass correlations."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import f

Missing = Literal["raise", "omit"]
ICCModel = Literal["one_way_random", "two_way_random", "two_way_mixed"]
ICCType = Literal["single", "average"]


@dataclass(frozen=True)
class ICC:
    model: ICCModel
    type: ICCType
    definition: Literal["absolute_agreement", "consistency"]
    estimate: float
    confidence: float
    interval: tuple[float, float]
    n_subjects: int
    n_ratings: int
    n_omitted: int
    missing: Missing


def _complete(data: ArrayLike, *, ndim: int, missing: Missing) -> NDArray[np.float64]:
    values = np.asarray(data, dtype=np.float64)
    if missing not in ("raise", "omit"):
        raise ValueError("missing must be 'raise' or 'omit'")
    if values.ndim != ndim or np.isinf(values).any():
        raise ValueError(f"expected {ndim} dimensions with no infinite values")
    absent = np.isnan(values)
    if missing == "raise" and absent.any():
        raise ValueError("missing measurements")
    keep = ~absent.any(axis=1) if ndim == 2 else ~absent
    return values[keep]


def icc(
    ratings: ArrayLike,
    *,
    model: ICCModel,
    type: ICCType,
    missing: Missing,
    confidence: float = 0.95,
) -> ICC:
    """Estimate Shrout-Fleiss ICC(1/2/3, 1/k) from subjects x ratings."""
    if model not in ("one_way_random", "two_way_random", "two_way_mixed"):
        raise ValueError("unknown ICC model")
    if type not in ("single", "average") or not 0 < confidence < 1:
        raise ValueError("invalid ICC type or confidence level")
    raw = np.asarray(ratings)
    values = _complete(raw, ndim=2, missing=missing)
    n, k = values.shape
    if n < 2 or k < 2:
        raise ValueError("ICC requires at least two complete subjects and two ratings")

    # Scale before forming squared deviations to keep mean squares in range.
    scale = float(np.max(np.abs(values)))
    if scale == 0 or np.all(values == values[0, 0]):
        raise ValueError("ICC is undefined for constant ratings")
    values = values / scale
    subject = values.mean(axis=1)
    within = values - subject[:, None]
    df_subject = n - 1
    between = float(k * np.var(subject, ddof=1))
    df_error = df_subject * (k - 1)
    column = within.mean(axis=0)
    error = float(np.sum(np.square(within - column)) / df_error)
    if model == "one_way_random":
        df_error = n * (k - 1)
        error = float(np.sum(np.square(within)) / df_error)

    count = k if type == "single" else 1
    definition: Literal["absolute_agreement", "consistency"] = (
        "consistency" if model == "two_way_mixed" else "absolute_agreement"
    )
    noise = (count - 1) * error
    if model == "two_way_random":
        rater = float(n * np.sum(np.square(column)) / (k - 1))
        noise += count * (rater - error) / n

        # McGraw-Wong effective degrees of freedom, with common factors cancelled.
        rater_weight = (between - error) * rater
        error_weight = (rater + df_subject * between) * error
        weight = abs(rater_weight) + abs(error_weight)
        if weight:
            a, b = rater_weight / weight, error_weight / weight
            df_error = (a + b) ** 2 / (a**2 / (k - 1) + b**2 / df_error)
        elif error:
            raise ValueError("absolute agreement CI is undefined for these mean squares")

    tail = (1 - confidence) / 2
    lower_f = float(f.isf(tail, df_subject, df_error))
    upper_f = float(f.isf(tail, df_error, df_subject))
    denominators = (between + noise, between + lower_f * noise, upper_f * between + noise)
    if not all(np.isfinite(d) and d > 0 for d in denominators):
        raise ValueError("ICC estimate or CI is undefined for these mean squares")
    point, low, high = denominators
    return ICC(
        model,
        type,
        definition,
        (between - error) / point,
        confidence,
        ((between - lower_f * error) / low, (upper_f * between - error) / high),
        n,
        k,
        len(raw) - n,
        missing,
    )
