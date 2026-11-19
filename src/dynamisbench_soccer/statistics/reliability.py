"""Balanced ANOVA intraclass correlations."""

from dataclasses import dataclass
from math import isfinite, sqrt
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.stats import f, norm

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


@dataclass(frozen=True)
class PairedReliability:
    typical_error: float
    cv_percent: float | None
    sem: float
    swc: float
    mdc90: float
    mdc95: float
    snr: float | None
    n_pairs: int
    n_omitted: int
    missing: Missing
    single_measure_icc: float
    swc_multiplier: float


@dataclass(frozen=True)
class Agreement:
    bias: float
    sd_difference: float
    limits: tuple[float, float]
    coverage: float
    n_pairs: int
    n_omitted: int
    missing: Missing


def _pairs(pairs: ArrayLike, missing: Missing) -> tuple[NDArray[np.float64], int]:
    raw = np.asarray(pairs)
    values = _complete(raw, ndim=2, missing=missing)
    if values.shape[1] != 2 or len(values) < 2:
        raise ValueError("expected at least two complete rows of paired measurements")
    return values, len(raw) - len(values)


def paired_reliability(
    pairs: ArrayLike,
    *,
    single_measure_icc: float,
    missing: Missing,
    swc_multiplier: float = 0.2,
) -> PairedReliability:
    """Use rows of (baseline, retest) and a declared single-reading ICC for SEM."""
    if not isfinite(single_measure_icc) or single_measure_icc > 1:
        raise ValueError("single_measure_icc must be finite and at most 1")
    if not isfinite(swc_multiplier) or swc_multiplier <= 0:
        raise ValueError("swc_multiplier must be finite and positive")
    values, omitted = _pairs(pairs, missing)
    typical_error = float(np.std(values[:, 1] - values[:, 0], ddof=1)) / sqrt(2)
    baseline_sd = float(np.std(values[:, 0], ddof=1))
    sem = baseline_sd * sqrt(1 - single_measure_icc)
    swc = swc_multiplier * baseline_sd
    cv = 100 * (typical_error / float(values.mean())) if np.all(values > 0) else None
    snr = swc / typical_error if typical_error else None
    mdc90 = float(norm.isf(0.05)) * sqrt(2) * sem
    mdc95 = float(norm.isf(0.025)) * sqrt(2) * sem
    if not all(isfinite(v) for v in (typical_error, sem, swc, mdc90, mdc95, cv or 0, snr or 0)):
        raise ValueError("paired reliability exceeds the supported numeric range")
    return PairedReliability(
        typical_error,
        cv,
        sem,
        swc,
        mdc90,
        mdc95,
        snr,
        len(values),
        omitted,
        missing,
        single_measure_icc,
        swc_multiplier,
    )


def bland_altman(pairs: ArrayLike, *, missing: Missing, coverage: float = 0.95) -> Agreement:
    """Calculate bias and normal-theory limits for the second reading minus the first."""
    if not 0 < coverage < 1:
        raise ValueError("coverage must be between 0 and 1")
    values, omitted = _pairs(pairs, missing)
    differences = values[:, 1] - values[:, 0]
    bias = float(np.mean(differences))
    sd = float(np.std(differences, ddof=1))
    half_width = float(norm.isf((1 - coverage) / 2)) * sd
    limits = (bias - half_width, bias + half_width)
    if not all(isfinite(v) for v in limits):
        raise ValueError("agreement limits exceed the supported numeric range")
    return Agreement(bias, sd, limits, coverage, len(values), omitted, missing)
