"""Paired agreement statistics and the evidence classification that bounds their meaning.

Statistics describe one dataset. The classification says what that dataset can support:
analytical model agreement never establishes empirical interchangeability, a definition
mismatch makes the numerical difference diagnostic only, and missing or unsupported
evidence is reported as such. Nothing here decides 'interchangeable' from synthetic data.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from scipy import stats

Origin = Literal["none", "analytical", "empirical"]
Classification = Literal[
    "analytical_model_agreement",
    "empirical_device_agreement",
    "definition_mismatch",
    "unsupported_comparison",
    "missing_evidence",
]
MIN_EMPIRICAL_PAIRS = 20


@dataclass(frozen=True)
class PairedAgreement:
    n: int
    bias: float
    sd_difference: float
    typical_error: float
    limits_of_agreement: tuple[float, float]
    proportional_bias_slope: float | None
    proportional_bias_slope_se: float | None
    proportional_bias_p: float | None
    expanded_uncertainty_of_difference: float | None
    within_tolerance: bool | None
    tolerance: float | None
    ratio_bias_percent: float | None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _finite(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def paired_agreement(
    a: ArrayLike,
    b: ArrayLike,
    *,
    tolerance: float | None = None,
    uncertainty_a: float | None = None,
    uncertainty_b: float | None = None,
    coverage_k: float = 2.0,
) -> PairedAgreement:
    """Bias, limits of agreement, proportional bias and tolerance for b - a.

    ``tolerance`` is a declared absolute tolerance that the bias and both limits of agreement
    must respect. ``uncertainty_*`` are declared standard uncertainties; they enter only the
    expanded uncertainty of the difference, never the bias estimate.
    """
    x, y = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if x.shape != y.shape or x.ndim != 1:
        raise ValueError("paired one-dimensional samples of equal length are required")
    if len(x) < 3:
        raise ValueError("at least three pairs are required")
    if not (np.isfinite(x).all() and np.isfinite(y).all()):
        raise ValueError("pairs must be finite; no imputation is performed")
    d = y - x
    mean = (x + y) / 2
    bias = float(d.mean())
    sd = float(d.std(ddof=1))
    half = float(stats.norm.isf(0.025)) * sd
    slope = se = p = None
    if float(np.ptp(mean)) > 0:
        fit = stats.linregress(mean, d)
        slope, se, p = (_finite(v) for v in (fit.slope, fit.stderr, fit.pvalue))
    u = None
    if uncertainty_a is not None and uncertainty_b is not None:
        if uncertainty_a < 0 or uncertainty_b < 0:
            raise ValueError("standard uncertainties must be nonnegative")
        u = coverage_k * math.hypot(uncertainty_a, uncertainty_b)
    within = None
    if tolerance is not None:
        if tolerance <= 0:
            raise ValueError("tolerance must be positive")
        within = bool(max(abs(bias - half), abs(bias + half), abs(bias)) <= tolerance)
    return PairedAgreement(
        n=len(d),
        bias=bias,
        sd_difference=sd,
        typical_error=sd / math.sqrt(2),
        limits_of_agreement=(bias - half, bias + half),
        proportional_bias_slope=slope,
        proportional_bias_slope_se=se,
        proportional_bias_p=p,
        expanded_uncertainty_of_difference=u,
        within_tolerance=within,
        tolerance=tolerance,
        ratio_bias_percent=float(100 * bias / x.mean()) if x.mean() != 0 else None,
    )


@dataclass(frozen=True)
class Verdict:
    classification: Classification
    interchangeable: Literal["unknown", "no"]
    reason: str


def classify(
    *,
    origin: Origin,
    definitions_match: bool,
    n_pairs: int = 0,
    independent_subjects: bool = False,
    data_is_measured: bool = False,
    unsupported_reason: str | None = None,
) -> Verdict:
    """Decide what a comparison can support. Interchangeability is never inferred here."""
    if unsupported_reason is not None:
        return Verdict("unsupported_comparison", "unknown", unsupported_reason)
    if origin == "none":
        return Verdict("missing_evidence", "unknown", "no paired data of any kind")
    if not definitions_match:
        return Verdict(
            "definition_mismatch",
            "no",
            "the estimands differ by definition; the numerical difference is diagnostic only",
        )
    if origin == "analytical":
        return Verdict(
            "analytical_model_agreement",
            "unknown",
            "agreement of idealised observation models; says nothing about real devices",
        )
    if not data_is_measured:
        return Verdict(
            "missing_evidence", "unknown", "empirical origin claimed without measured data"
        )
    if n_pairs < MIN_EMPIRICAL_PAIRS or not independent_subjects:
        return Verdict(
            "missing_evidence",
            "unknown",
            f"fewer than {MIN_EMPIRICAL_PAIRS} independent paired observations",
        )
    return Verdict(
        "empirical_device_agreement",
        "unknown",
        "empirical statistics available; interchangeability still needs tolerance and review",
    )


def is_measured(meta: object) -> bool:
    """True only for empirical, non-vendor-reported bundles (never analytical or synthetic)."""
    category = getattr(meta, "data_category", None)
    value = getattr(category, "value", category)
    return value == "empirical" and not getattr(meta, "vendor_reported", False)
