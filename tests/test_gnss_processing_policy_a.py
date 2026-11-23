"""GNSS processing variants versus independent local polynomial derivatives."""

import itertools

import numpy as np
import pytest
from test_gnss_lps_acquisition import track_case

from dynamisbench_soccer.engines.gnss_lps import ProcessingOptions, process
from dynamisbench_soccer.engines.gnss_lps_acquisition import AcquisitionRefusal, EnuOrigin

WEIGHTS = {"none": [1], "boxcar5": [1 / 5] * 5, "binomial5": [1 / 16, 1 / 4, 3 / 8, 1 / 4, 1 / 16]}


def _track():
    case, options = track_case("gnss", hz=18, drift=425, latency=0.062)
    bundle = case.observation
    t = case.truth["time"]
    x = t + 0.2 * t**2 + 0.025 * t**4
    doppler = 4 - 0.4 * t + 0.15 * t**2
    bundle["latitude"][:] = 0
    bundle["longitude"][:] = np.rad2deg(np.arcsin(x / 6378137))
    bundle["height"][:] = 0
    bundle["doppler_speed"][:] = doppler
    origin = EnuOrigin(latitude_deg=0, longitude_deg=0, ellipsoidal_height_m=0)
    return bundle, options.model_copy(update={"enu_origin": origin}), t, x, doppler


def _filter(values, name, *, linear):
    weights = WEIGHTS[name]
    half = len(weights) // 2
    n = len(values)
    left, right = (values[1] - values[0], values[-1] - values[-2]) if linear else (0, 0)

    def at(i):
        if i < 0:
            return values[0] + i * left
        if i >= n:
            return values[-1] + (i - n + 1) * right
        return values[i]

    return np.array(
        [sum(w * at(i + offset - half) for offset, w in enumerate(weights)) for i in range(n)]
    )


def _polynomial_slope(values, time, method):
    slopes = []
    for i, center in enumerate(time):
        if method == "five_point" and 2 <= i < len(time) - 2:
            segment = slice(i - 2, i + 3)
        else:
            start = min(max(i - 1, 0), len(time) - 3)
            segment = slice(start, start + 3)
        local_time = time[segment] - center
        powers = np.vander(local_time, N=len(local_time), increasing=True)
        slopes.append(np.linalg.solve(powers, values[segment])[1])
    return np.array(slopes)


def _expected(t, x, doppler, position_filter, doppler_filter, method):
    position = _filter(x, position_filter, linear=True)
    speed = _polynomial_slope(position, t, method)
    acceleration = _polynomial_slope(speed, t, method)
    reported = _filter(doppler, doppler_filter, linear=False)
    change = _polynomial_slope(reported, t, method)
    return position, speed, acceleration, reported, change


@pytest.mark.parametrize("method", ["three_point", "five_point"])
@pytest.mark.parametrize(
    ("position_filter", "doppler_filter"), tuple(itertools.product(WEIGHTS, repeat=2))
)
def test_gnss_variants_and_peaks_match_independent_polynomials(
    method,
    position_filter,
    doppler_filter,
):
    bundle, declaration, t, x, doppler = _track()
    options = ProcessingOptions(
        derivative=method,
        position_filter=position_filter,
        doppler_filter=doppler_filter,
        thresholds=(),
    )
    result = process(bundle, declaration, options)
    k = result.kinematics
    expected = _expected(t, x, doppler, position_filter, doppler_filter, method)
    for observed, known, atol in zip(
        (
            k.position_m[:, 0],
            k.horizontal_positional_speed_m_s,
            k.horizontal_positional_acceleration_m_s2,
            k.doppler_speed_m_s,
            k.doppler_speed_change_m_s2,
        ),
        expected,
        (1e-8, 1e-7, 1e-6, 1e-12, 1e-10),
        strict=True,
    ):
        np.testing.assert_allclose(
            observed,
            np.abs(known) if observed is k.horizontal_positional_acceleration_m_s2 else known,
            rtol=0,
            atol=atol,
        )

    stop = len(t) - (len(t) % 2 == 0)
    sensitivity = result.sensitivity
    assert sensitivity.comparison_duration_s == pytest.approx(t[stop - 1], rel=0, abs=1e-12)
    assert sensitivity.every_second_sample is not None
    for peak, stride, pf, df in (
        (sensitivity.native, 1, position_filter, doppler_filter),
        (
            sensitivity.alternate_filter,
            1,
            "binomial5" if position_filter == "none" else "none",
            "binomial5" if doppler_filter == "none" else "none",
        ),
        (sensitivity.every_second_sample, 2, position_filter, doppler_filter),
    ):
        segment = slice(0, stop, stride)
        _, sp, acc, ds, dc = _expected(t[segment], x[segment], doppler[segment], pf, df, method)
        assert peak.reference_rate_hz == pytest.approx(18 / (1.000425 * stride), rel=0, abs=1e-9)
        assert peak.horizontal_positional_speed_m_s == pytest.approx(max(sp), rel=0, abs=1e-7)
        assert peak.horizontal_positional_acceleration_m_s2 == pytest.approx(
            max(abs(acc)), rel=0, abs=1e-6
        )
        assert peak.doppler_speed_m_s == pytest.approx(max(ds), rel=0, abs=1e-12)
        assert peak.absolute_doppler_speed_change_m_s2 == pytest.approx(
            max(abs(dc)), rel=0, abs=1e-10
        )


def test_gnss_jitter_and_missing_doppler_processing_refuse():
    bundle, declaration, *_ = _track()
    bundle["time"][1:-1:2] += 0.001
    three_point = process(bundle, declaration, ProcessingOptions(derivative="three_point"))
    assert three_point.acquisition.clock.jitter_fraction > 0
    with pytest.raises(AcquisitionRefusal, match="uniform time"):
        process(bundle, declaration, ProcessingOptions(derivative="five_point"))
    clean, declaration = track_case("gnss", doppler=False)
    with pytest.raises(AcquisitionRefusal, match="requires native Doppler"):
        process(clean.observation, declaration, ProcessingOptions(doppler_filter="boxcar5"))
