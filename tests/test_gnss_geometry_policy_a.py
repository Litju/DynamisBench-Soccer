"""Independent WGS-84 ENU controls for the GNSS acquisition path."""

import math

import numpy as np
import pytest
from test_gnss_lps_acquisition import track_case

from dynamisbench_soccer.engines.gnss_lps_acquisition import AcquisitionRefusal, EnuOrigin, admit


@pytest.mark.parametrize("latitude", [-61, 0, 73])
@pytest.mark.parametrize("longitude", [-179.999, 34, 179.999])
def test_ellipsoid_normal_becomes_local_up(latitude, longitude):
    case, declaration = track_case("gnss", hz=18)
    bundle = case.observation
    heights = 650 + 0.5 * case.truth["time"]
    bundle["latitude"][:] = latitude
    bundle["longitude"][:] = longitude
    bundle["height"][:] = heights
    origin = EnuOrigin(latitude_deg=latitude, longitude_deg=longitude, ellipsoidal_height_m=650)
    result = admit(bundle, declaration.model_copy(update={"enu_origin": origin}))
    expected = np.column_stack((np.zeros_like(heights), np.zeros_like(heights), heights - 650))
    np.testing.assert_allclose(result.position_m, expected, atol=5e-9, rtol=0)
    assert result.frame_id == "local_enu"


def _equator_circle(angle):
    radius = 6378137.0
    return np.column_stack(
        (radius * np.sin(angle), np.zeros_like(angle), -2 * radius * np.sin(angle / 2) ** 2)
    )


def test_equator_direction_and_invalid_geometry():
    case, declaration = track_case("gnss", hz=18, latency=0, drift=0)
    bundle = case.observation
    distance = case.truth["time"] * 2
    earth_radius = 6378137.0
    delta = distance / earth_radius
    bundle["latitude"][:] = 0
    bundle["longitude"][:] = np.rad2deg(delta)
    bundle["height"][:] = 0
    origin = EnuOrigin(latitude_deg=0, longitude_deg=0, ellipsoidal_height_m=0)
    declaration = declaration.model_copy(update={"enu_origin": origin})
    expected = _equator_circle(delta)
    np.testing.assert_allclose(admit(bundle, declaration).position_m, expected, atol=5e-9, rtol=0)
    with pytest.raises(AcquisitionRefusal, match="explicit ENU origin"):
        admit(bundle, declaration.model_copy(update={"enu_origin": None}))
    bundle["longitude"][:] += math.degrees(0.01)
    with pytest.raises(AcquisitionRefusal, match="10 km"):
        admit(bundle, declaration)


def test_meridian_displacement_and_missing_fix_frame():
    case, declaration = track_case("gnss", hz=18, latency=0, drift=0)
    bundle = case.observation
    angle = 1e-7 * case.truth["time"]
    bundle["latitude"][:] = np.rad2deg(angle)
    bundle["longitude"][:] = 0
    bundle["height"][:] = 0
    declaration = declaration.model_copy(
        update={"enu_origin": EnuOrigin(latitude_deg=0, longitude_deg=0, ellipsoidal_height_m=0)}
    )
    # At the equatorial origin, local north is ECEF Z. This independent meridian
    # formula gives a positive north displacement and zero east.
    eccentricity_squared = 0.0066943799901413165
    z = (
        6378137
        * (1 - eccentricity_squared)
        * np.sin(angle)
        / np.sqrt(1 - eccentricity_squared * np.sin(angle) ** 2)
    )
    position = admit(bundle, declaration).position_m
    np.testing.assert_allclose(position[:, 1], z, atol=5e-9, rtol=0)
    np.testing.assert_allclose(position[:, 0], 0, atol=5e-9, rtol=0)
    assert position[-1, 1] > 0
    for name in ("latitude", "longitude", "height"):
        bundle.meta.channel(name).frame = None
    with pytest.raises(AcquisitionRefusal, match="share a geodetic frame"):
        admit(bundle, declaration)
