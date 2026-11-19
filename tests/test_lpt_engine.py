import json

import numpy as np
import pytest
from test_lpt_acquisition import BAR, COUNTS, DUAL, LINE, LPT, MOTION, _dual, _modify

from dynamisbench_soccer.cli import main
from dynamisbench_soccer.engines.lpt import MethodSpec, PowerSpec, SensitivitySpec, process
from dynamisbench_soccer.engines.lpt_acquisition import GeometrySpec
from dynamisbench_soccer.reference.exporters import export_case
from dynamisbench_soccer.signals import io

METHODS = ("finite_difference", "butterworth_then_difference", "cubic_spline")
POWER = PowerSpec(system_mass_kg=40, rigid_vertical_translation=True)


def _sine_encoder(hz, attachment="belt"):
    reference = export_case("lpt_encoder.encoder", MOTION, test_id="cmj", hz=2 * hz).observation
    meta = reference.meta.model_dump()
    meta["sampling"].update(nominal_hz=hz, measured_hz=None, irregular=True)
    geometry = meta["acquisition"]["sources"][0]["geometry"]
    geometry.update(
        attachment_point=attachment,
        cable_angle_rad=0.45,
        encoder_counts_per_m=20_000_000,
        unknown=[],
    )
    nominal = np.arange(meta["n_rows"]) / hz
    time = nominal + np.sin(2 * np.pi * 9 * nominal) * 0.06 / hz
    w = 2 * np.pi * 0.75
    z = 0.35 * time + 0.08 * np.sin(w * time)
    counts = np.rint(z * np.cos(0.45) * 20_000_000)
    return (
        _modify(reference, meta=meta, data=dict(time=time + 12, counts=counts)),
        z,
        0.35 + 0.08 * w * np.cos(w * time),
        -0.08 * w**2 * np.sin(w * time),
    )


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("hz", [100, 250])
def test_methods_against_independent_analytic_irregular_quantized_motion(method, hz):
    bundle, z, v, a = _sine_encoder(hz)
    report = process(
        bundle,
        (COUNTS,),
        GeometrySpec(attachment_point="belt"),
        MethodSpec(name=method, summary_trim_s=0.3),
        POWER,
    )
    assert report["estimand"] == "attachment_point_vertical_displacement"
    assert report["analysis_clock"] == "device"
    motion = report["motion"]
    assert abs(motion["elapsed_s"][-1] - 2) < 1e-12
    np.testing.assert_array_equal(report["cables"][0]["native_time_s"], bundle["time"])
    keep = (np.asarray(motion["elapsed_s"]) >= 0.3) & (np.asarray(motion["elapsed_s"]) <= 1.7)
    np.testing.assert_allclose(np.asarray(motion["displacement_m"])[keep], z[keep], atol=2e-4)
    np.testing.assert_allclose(np.asarray(motion["velocity_m_s"])[keep], v[keep], atol=1.5e-3)
    np.testing.assert_allclose(np.asarray(motion["acceleration_m_s2"])[keep], a[keep], atol=0.025)
    np.testing.assert_allclose(
        np.asarray(motion["vertical_mass_model_power_w"])[keep],
        (40 * (9.80665 + a) * v)[keep],
        atol=1.0,
    )


@pytest.mark.parametrize("method", METHODS)
def test_dual_methods_recover_independent_bar_trajectory(method):
    report = process(_dual(), DUAL, BAR, MethodSpec(name=method, summary_trim_s=0.25))
    assert report["estimand"] == "bar_vertical_displacement"
    assert report["analysis_clock"] == "left"
    assert len(report["dual_clock_alignments"]) == 1
    motion = report["motion"]
    t = np.asarray(motion["elapsed_s"])
    keep = (t >= 0.25) & (t <= 0.75)
    np.testing.assert_allclose(
        np.asarray(motion["velocity_m_s"])[keep], 0.4 + 2 * t[keep], atol=5e-4
    )
    np.testing.assert_allclose(np.asarray(motion["acceleration_m_s2"])[keep], 2, atol=0.012)
    assert "vertical_mass_model_power_w" not in motion


def test_sensitivity_perturbs_angles_cutoff_and_timing_on_same_rows():
    bundle, *_ = _sine_encoder(100)
    report = process(
        bundle,
        (COUNTS,),
        GeometrySpec(attachment_point="belt"),
        MethodSpec(name="butterworth_then_difference", summary_trim_s=0.3),
        POWER,
        SensitivitySpec(),
    )
    variants = report["sensitivity"]
    for key in ("cable_angle_upper", "cutoff_lower", "timebase_scale_upper"):
        assert variants[key]["differences"]["velocity_m_s"]["rmse"] > 0
        assert variants[key]["differences"]["vertical_mass_model_power_w"]["rmse"] > 0
    assert variants["cable_angle_upper"]["differences"]["velocity_m_s"]["peak_delta"] > 0
    assert variants["timebase_scale_upper"]["differences"]["velocity_m_s"]["peak_delta"] < 0
    assert variants["finite_difference"]["method"]["name"] == "finite_difference"


def test_missing_geometry_withholds_power_and_coincident_com_stays_an_attachment():
    case = export_case("lpt_encoder.lpt", MOTION, test_id="cmj")
    cable = process(case.observation, (LPT,), GeometrySpec(attachment_point="COM"), power=POWER)
    assert cable["estimand"] == "cable_displacement"
    assert "vertical_mass_model_power_w" not in cable["motion"]
    assert any("power withheld" in msg for msg in cable["limitations"])
    result = process(case.observation, (LPT,), LINE)
    assert result["estimand"] == "attachment_point_vertical_displacement"
    np.testing.assert_allclose(result["motion"]["acceleration_m_s2"][5:], 2, atol=1e-9)


def test_cli_processes_verified_bundle_and_keeps_reference_separate(tmp_path, capsys):
    case = export_case("lpt_encoder.lpt", MOTION, test_id="cmj")
    stem, reference = tmp_path / "observation", tmp_path / "reference"
    io.write(case.observation, stem)
    io.write(case.truth, reference)
    config = tmp_path / "config.json"
    config.write_text(json.dumps({"cables": [LPT.model_dump()], "geometry": LINE.model_dump()}))
    assert main(["lpt-encoder", "process", str(stem), "--config", str(config)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["data_sha256"] == case.observation.meta.data_sha256
    assert report["summary"]["velocity_m_s"]["max"] == pytest.approx(2.4)
    assert main(["lpt-encoder", "process", str(reference), "--config", str(config)]) == 2
    assert "excluding analytical" in capsys.readouterr().err
