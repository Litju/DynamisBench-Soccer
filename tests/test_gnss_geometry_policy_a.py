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
    expected = np.column_stack(
        (
            earth_radius * np.sin(delta),
            np.zeros_like(delta),
            -2 * earth_radius * np.sin(delta / 2) ** 2,
        )
    )
    np.testing.assert_allclose(admit(bundle, declaration).position_m, expected, atol=5e-9, rtol=0)
    with pytest.raises(AcquisitionRefusal, match="explicit ENU origin"):
        admit(bundle, declaration.model_copy(update={"enu_origin": None}))
    bundle["longitude"][:] += math.degrees(0.01)
    with pytest.raises(AcquisitionRefusal, match="10 km"):
        admit(bundle, declaration)
