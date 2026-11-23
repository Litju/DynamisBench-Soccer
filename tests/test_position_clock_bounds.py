"""Independent affine-clock uncertainty controls for the real GNSS and LPS pipeline."""

import numpy as np
import pytest
from test_gnss_lps_acquisition import track_case

from dynamisbench_soccer.engines.gnss_lps import ProcessingOptions, process
from dynamisbench_soccer.engines.gnss_lps_acquisition import AcquisitionRefusal, EnuOrigin
from dynamisbench_soccer.engines.position_clock_bounds import clock_kinematic_bounds


def _quadratic_track(kind, hz):
    case, declaration = track_case(kind, hz=hz, drift=320, latency=0.025)
    observation, t = case.observation, case.truth["time"]
    east = t + 0.25 * t**2
    if kind == "gnss":
        observation["latitude"][:] = 0
        observation["longitude"][:] = np.rad2deg(np.arcsin(east / 6378137))
        observation["height"][:] = 0
        observation["doppler_speed"][:] = 5 - 0.25 * t
        declaration = declaration.model_copy(
            update={
                "enu_origin": EnuOrigin(latitude_deg=0, longitude_deg=0, ellipsoidal_height_m=0)
            }
        )
    else:
        observation["position_x"][:] = east
        observation["position_y"][:] = 0
    return observation, declaration, t


def _closed_rates(t, scale):
    return (1 + 0.5 * t) / scale, 0.5 / scale**2, -0.25 / scale


@pytest.mark.parametrize(("kind", "hz"), [("gnss", 18), ("lps", 25)])
@pytest.mark.parametrize("filter_name", ["none", "boxcar5", "binomial5"])
@pytest.mark.parametrize("method", ["three_point", "five_point"])
def test_clock_bounds_with_native_processing_variants(kind, hz, filter_name, method):
    observation, declaration, t = _quadratic_track(kind, hz)
    result = process(
        observation,
        declaration,
        ProcessingOptions(
            derivative=method,
            position_filter=filter_name,
            doppler_filter=filter_name if kind == "gnss" else "none",
            thresholds=(),
        ),
    )
    fraction = 0.00025
    bounds = clock_kinematic_bounds(result, fraction)
    support = slice(7, -7)
    fast = _closed_rates(t[support], 1 - fraction)
    slow = _closed_rates(t[support], 1 + fraction)
    speed = bounds.horizontal_positional_speed_m_s
    acceleration = bounds.horizontal_positional_acceleration_m_s2
    for actual, expected in (
        (speed.low, slow[0]),
        (speed.high, fast[0]),
        (acceleration.low, slow[1]),
        (acceleration.high, fast[1]),
    ):
        np.testing.assert_allclose(actual[support], expected, atol=1e-7, rtol=0)
    # The fixed latency uncertainty and synthetic affine alignment phase bound
    # shift each absolute timestamp. Their common offset cancels in rate estimates.
    offset = 0.001 + 1e-12
    np.testing.assert_allclose(
        bounds.measurement_time_s.low, t * (1 - fraction) - offset, atol=1e-12, rtol=0
    )
    np.testing.assert_allclose(
        bounds.measurement_time_s.high, t * (1 + fraction) + offset, atol=1e-12, rtol=0
    )
    change = bounds.doppler_speed_change_m_s2
    if kind == "gnss":
        assert change is not None
        np.testing.assert_allclose(change.low[support], fast[2], atol=1e-10, rtol=0)
        np.testing.assert_allclose(change.high[support], slow[2], atol=1e-10, rtol=0)
    else:
        assert change is None


@pytest.mark.parametrize("kind", ["gnss", "lps"])
def test_clock_bounds_refuse_unqualified_scale_and_support_jittered_three_point(kind):
    observation, declaration, _ = _quadratic_track(kind, 25)
    observation["time"][1:-1:2] += 0.001
    result = process(observation, declaration, ProcessingOptions(thresholds=()))
    nominal = clock_kinematic_bounds(result, 0)
    np.testing.assert_array_equal(
        nominal.horizontal_positional_speed_m_s.low,
        result.kinematics.horizontal_positional_speed_m_s,
    )
    with pytest.raises(AcquisitionRefusal, match="drift limit"):
        clock_kinematic_bounds(result, 0.1)
    with pytest.raises(AcquisitionRefusal, match="finite"):
        clock_kinematic_bounds(result, float("nan"))
