"""RES-703 agreement controls use independently prescribed rational motion."""

import math

import pytest
from test_gnss_lps_acquisition import track_case

from dynamisbench_soccer.agreement.lps import compare_lps_radar
from dynamisbench_soccer.engines.gnss_lps import ProcessingOptions
from dynamisbench_soccer.engines.gnss_lps_acquisition import AcquisitionRefusal


def _motion(hz):
    case, declaration = track_case("lps", hz=hz, latency=0.03, drift=120)
    bundle = case.observation
    for name, velocity in (("position_x", 3), ("position_y", 4), ("position_z", 2)):
        bundle[name][:] = velocity * case.truth["time"]
    return bundle, declaration


@pytest.mark.parametrize("hz", [18, 25])
@pytest.mark.parametrize("derivative", ["three_point", "five_point"])
@pytest.mark.parametrize("position_filter", ["none", "boxcar5", "binomial5"])
def test_matched_lps_radar_control(hz, derivative, position_filter):
    bundle, declaration = _motion(hz)
    result = compare_lps_radar(
        bundle,
        declaration,
        (3, 4),
        0,
        correct_projection=True,
        options=ProcessingOptions(
            derivative=derivative, position_filter=position_filter, thresholds=()
        ),
    )
    assert result.classification == "analytical_model_agreement"
    assert result.interchangeable == "unknown"
    for radar, lps in result.pairs:
        assert radar == pytest.approx(5, rel=0, abs=1e-10)
        assert lps == pytest.approx(5, rel=0, abs=1e-10)


def test_radial_component_is_a_mismatch_and_faults_refuse():
    bundle, declaration = _motion(25)
    opts = ProcessingOptions(thresholds=())
    mismatch = compare_lps_radar(
        bundle,
        declaration,
        (3, 4),
        0,
        correct_projection=False,
        options=opts,
    )
    assert mismatch.classification == "definition_mismatch"
    assert mismatch.interchangeable == "no"
    assert mismatch.statistics["bias"] == pytest.approx(2, rel=0, abs=1e-10)
    with pytest.raises(ValueError, match="does not resolve"):
        compare_lps_radar(
            bundle,
            declaration,
            (3, 4),
            math.atan2(4, 3) + math.pi / 2,
            correct_projection=True,
            options=opts,
        )
    bundle["signal_valid"][20] = 0
    with pytest.raises(AcquisitionRefusal, match="signal integrity"):
        compare_lps_radar(
            bundle,
            declaration,
            (3, 4),
            0,
            correct_projection=True,
            options=opts,
        )
