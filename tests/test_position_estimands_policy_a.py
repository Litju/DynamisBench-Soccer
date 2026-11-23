"""Independent source-to-definition controls for GNSS/LPS native and handoff estimands."""

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
from test_gnss_lps_acquisition import track_case

from dynamisbench_soccer.assessment.handoff import HandoffRefusal, load_speed_report
from dynamisbench_soccer.assessment.sprint import StartSpec, milestones
from dynamisbench_soccer.engines.gnss_lps import ProcessingOptions, SpeedThreshold, process
from dynamisbench_soccer.engines.gnss_lps_acquisition import EnuOrigin
from dynamisbench_soccer.engines.gnss_lps_cli import _json_value
from dynamisbench_soccer.reference.coordinates import ecef_geodetic
from dynamisbench_soccer.registry.models import load

ROOT = Path(__file__).resolve().parents[1]


def _quadratic(kind):
    case, declaration = track_case(kind, hz=25, drift=320, latency=0.02)
    bundle, t = case.observation, case.truth["time"]
    arc = 5 * t + 0.5 * t**2
    x, y, z = 0.6 * arc, -0.8 * arc, 2 + t + 0.3 * t**2
    if kind == "gnss":
        fixes = np.array([ecef_geodetic(row) for row in np.column_stack((6378137 + z, x, y))])
        for index, name in enumerate(("latitude", "longitude", "height")):
            bundle[name][:] = fixes[:, index]
        bundle["doppler_speed"][:] = 2 + 0.5 * t
        origin = EnuOrigin(latitude_deg=0, longitude_deg=0, ellipsoidal_height_m=0)
        declaration = declaration.model_copy(update={"enu_origin": origin})
    else:
        bundle["position_x"][:] = x
        bundle["position_y"][:] = y
        bundle["position_z"][:] = z
    return bundle, declaration, t, arc


@pytest.mark.parametrize("kind", ["gnss", "lps"])
@pytest.mark.parametrize("method", ["three_point", "five_point"])
def test_native_metric_definitions_and_assessment_handoff(kind, method, tmp_path):
    bundle, declaration, t, arc = _quadratic(kind)
    threshold = 5.337
    result = process(
        bundle,
        declaration,
        ProcessingOptions(
            derivative=method,
            thresholds=(
                SpeedThreshold(threshold_m_s=threshold, dwell_s=0.4),
                SpeedThreshold(threshold_m_s=2.753, dwell_s=0.2),
                SpeedThreshold(threshold_m_s=threshold, dwell_s=100),
            ),
        ),
    )
    k = result.kinematics
    np.testing.assert_allclose(k.horizontal_positional_speed_m_s, 5 + t, rtol=0, atol=1e-7)
    np.testing.assert_allclose(k.horizontal_positional_acceleration_m_s2, 1, rtol=0, atol=1e-5)
    np.testing.assert_allclose(k.accumulated_horizontal_distance_m, arc, rtol=0, atol=1e-7)
    first = next(i for i, stamp in enumerate(t) if stamp + 5 >= threshold)
    positional = result.thresholds[0]
    assert positional.estimand == "horizontal_positional_speed"
    assert positional.runs == 1
    assert positional.horizontal_path_distance_m == pytest.approx(arc[-1] - arc[first], abs=1e-7)
    assert positional.duration_s == pytest.approx(t[-1] - t[first], abs=1e-12)
    assert positional.integrated_doppler_distance_m is None
    if kind == "gnss":
        np.testing.assert_allclose(k.doppler_speed_m_s, 2 + 0.5 * t, rtol=0, atol=1e-12)
        np.testing.assert_allclose(k.doppler_speed_change_m_s2, 0.5, rtol=0, atol=1e-10)
        assert result.thresholds[1].estimand == "doppler_speed" and result.thresholds[1].runs == 0
        doppler = result.thresholds[3]
        start = next(stamp for stamp in t if 2 + 0.5 * stamp >= 2.753)
        integral = 2 * (t[-1] - start) + 0.25 * (t[-1] ** 2 - start**2)
        assert doppler.runs == 1 and doppler.horizontal_path_distance_m is None
        assert doppler.integrated_doppler_distance_m == pytest.approx(integral, abs=1e-12)
        assert result.thresholds[4].runs == 0
    else:
        assert k.doppler_speed_m_s is None and len(result.thresholds) == 3
        assert result.thresholds[-1].runs == 0

    path = tmp_path / "native.json"
    report = asdict(result)
    path.write_text(json.dumps(report, default=_json_value))
    declared_source = f"{kind}_position"
    handoff = load_speed_report(path, declared_source=declared_source)
    assert handoff.source == declared_source and "position" in handoff.estimand.lower()
    distance = 5.0
    reached = milestones(handoff.time_s, handoff.speed_m_s, [distance, 1000], StartSpec())
    assert reached["5_m"] == pytest.approx(np.sqrt(25 + 2 * distance) - 5, abs=1e-7)
    assert reached["1000_m"] is None
    if kind == "gnss":
        doppler = load_speed_report(path, declared_source="gnss_doppler")
        assert doppler.estimand != handoff.estimand
        reached = milestones(doppler.time_s, doppler.speed_m_s, [distance], StartSpec())
        assert reached["5_m"] == pytest.approx(2 * (np.sqrt(4 + distance) - 2), abs=1e-7)
        report = json.loads(path.read_bytes())
        report["acquisition"]["declaration"]["doppler_basis"] = "three_dimensional"
        path.write_text(json.dumps(report, default=_json_value))
        with pytest.raises(HandoffRefusal, match="horizontal"):
            load_speed_report(path, declared_source="gnss_doppler")
    else:
        with pytest.raises(HandoffRefusal):
            load_speed_report(path, declared_source="gnss_doppler")
    claim = next(c for c in load(ROOT / "registry").claims if c.id == "DEV-005")
    assert "linke2018" in claim.sources and "Doppler" in claim.statement
