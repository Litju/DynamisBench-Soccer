"""Symmetric five-sample smoothing for offline reference engines."""

from typing import Literal

import numpy as np

from dynamisbench_soccer.signals.schema import FloatArray

FilterMethod = Literal["none", "boxcar5", "binomial5"]


def smooth_samples(
    signal: FloatArray, method: FilterMethod, *, extend_slope: bool = True
) -> FloatArray:
    if method == "none":
        return signal.copy()
    if method == "boxcar5":
        weights = np.full(5, 0.2)
    elif method == "binomial5":
        weights = np.array([1, 4, 6, 4, 1], dtype=np.float64) / 16
    else:
        raise ValueError("unsupported named filter")
    if not extend_slope:
        return np.convolve(np.pad(signal, 2, mode="edge"), weights, mode="valid")
    # Linear extension avoids introducing artificial reversals at the boundaries.
    left_slope = signal[1] - signal[0]
    right_slope = signal[-1] - signal[-2]
    left = signal[0] - left_slope * np.array([2.0, 1.0])
    right = signal[-1] + right_slope * np.array([1.0, 2.0])
    extended = np.concatenate((left, signal, right))
    return np.convolve(extended, weights, mode="valid")
