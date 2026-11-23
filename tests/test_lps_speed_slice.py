"""Independent controls for the fixed-grid native LPS task slice."""

from pathlib import Path

import pytest
from test_gnss_lps_acquisition import track_case

from dynamisbench_soccer.engines.gnss_lps_acquisition import AcquisitionRefusal
from dynamisbench_soccer.engines.lps_speed import horizontal_speed, position_error_radius
from dynamisbench_soccer.reference.lps_speed import exact_grid_extreme_speed, linear_tag_speed
from dynamisbench_soccer.registry.models import load

ROOT = Path(__file__).resolve().parents[1]


def _track():
    case, declaration = track_case("lps", hz=25, latency=0, drift=0)
    bundle = case.observation
    assert bundle.meta.acquisition is not None
    bundle.meta.acquisition.alignments[0].uncertainty_s = 0
    time = bundle["time"] - bundle["time"][0]
    for axis, velocity in zip("xyz", (-3, 4, 9), strict=True):
        bundle[f"position_{axis}"][:] = velocity * time
    return bundle, declaration.model_copy(
        update={"timestamp_basis": "measurement", "latency_uncertainty_s": 0}
    )


def test_horizontal_tag_speed_excludes_vertical_motion_and_is_sourced() -> None:
    bundle, declaration = _track()
    assert horizontal_speed(bundle, declaration, 4) == pytest.approx(
        linear_tag_speed(-3, 4), rel=0, abs=1e-12
    )
    bundle["position_z"][:] = 0
    assert horizontal_speed(bundle, declaration, 4) == pytest.approx(5, rel=0, abs=1e-12)
    claims = load(ROOT / "registry").claims
    definition = next(claim for claim in claims if claim.id == "DEV-005")
    assert "positional differentiation" in definition.statement


def test_lps_speed_refuses_unverified_clock_and_signal() -> None:
    bundle, declaration = _track()
    with pytest.raises(AcquisitionRefusal):
        horizontal_speed(bundle, declaration.model_copy(update={"latency_uncertainty_s": 0.001}), 4)
    bundle["signal_valid"][4] = 0
    with pytest.raises(AcquisitionRefusal):
        horizontal_speed(bundle, declaration, 4)


def test_position_only_radius_contains_and_attains_analytical_extreme() -> None:
    bundle, declaration = _track()
    row, bound = 4, 0.012
    center = horizontal_speed(bundle, declaration, row)
    t = bundle["time"] - bundle["time"][0]
    radius = position_error_radius(*t[row - 1 : row + 2], bound)
    assert radius == pytest.approx(25 * bound, rel=0, abs=1e-12)
    for axis, component in (("x", -0.6), ("y", 0.8)):
        bundle[f"position_{axis}"][row - 1] -= component * bound
        bundle[f"position_{axis}"][row + 1] += component * bound
    extreme = horizontal_speed(bundle, declaration, row)
    assert extreme == pytest.approx(exact_grid_extreme_speed(-3, 4, 25, bound), rel=0, abs=1e-12)
    assert extreme - center == pytest.approx(radius, rel=0, abs=1e-12)
