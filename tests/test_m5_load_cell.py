"""Independent controls for the fixed native load-cell procedure."""

import json
from pathlib import Path

import numpy as np
import pytest

from dynamisbench_soccer.engines.load_cell import MethodSpec, process
from dynamisbench_soccer.engines.load_cell_acquisition import CellSpec
from dynamisbench_soccer.m4.measurement import CELL_OPERATION, replay
from dynamisbench_soccer.signals.schema import BundleMeta, SignalBundle

ROOT = Path(__file__).resolve().parents[1]


def _fixture(index):
    path = ROOT / f"benchmark/fixtures/m5-force-development/cell-case-{index:02d}.json"
    return json.loads(path.read_bytes())


def _native_seconds(row, t0=0):
    return t0 + row / 1000


def _positive_rectangle_radius(net, lever):
    return net * 0.01 + lever * 2 + 0.02


def _analytical_ramp(row, baseline, rise):
    return baseline + 4 * min(max(row - 120, 0), rise // 4)


@pytest.mark.parametrize(
    ("index", "baseline", "rise", "lever"), [(1, 12, 800, 0.35), (2, 10, 600, 0.42)]
)
def test_native_clock_force_and_processing(index, baseline, rise, lever):
    fixture = _fixture(index)
    meta = BundleMeta.model_validate(fixture["meta"])
    array = np.asarray(fixture["samples"], dtype=np.float64)
    result = process(
        SignalBundle(meta, {"voltage": array}),
        CellSpec.model_validate({"calibration": fixture["setup"].get("analysis_calibration")}),
        MethodSpec(
            onset="absolute_N", onset_absolute_n=3, baseline_window_s=(0, 0.101), windows_ms=(100,)
        ),
    )
    exact_force = [_analytical_ramp(row, baseline, rise) for row in range(meta.n_rows)]
    np.testing.assert_allclose(
        result.time_s, [_native_seconds(row) for row in range(meta.n_rows)], rtol=0, atol=1e-15
    )
    assert result.method_spec["clock_identity"] == "bench-counter:paired-counter-epoch"
    assert result.uncertainty["sync_uncertainty_s"] == 0
    for stream in (
        result.calibrated_force_n,
        result.peak_force_signal_n,
        result.rfd_force_signal_n,
    ):
        np.testing.assert_allclose(stream, exact_force, rtol=0, atol=1e-8)

    reading = replay(CELL_OPERATION, fixture)
    assert reading.reason == "eligible"
    assert reading.value == pytest.approx(rise * lever, rel=0, abs=1e-8)
    assert reading.radius == pytest.approx(_positive_rectangle_radius(rise, lever), rel=0, abs=1e-8)
    assert dict(reading.operands) == pytest.approx(
        {
            "baseline_n": baseline,
            "gross_peak_n": baseline + rise,
            "net_peak_n": rise,
            "onset_s": 0.120,
            "rfd_100ms_n_s": 4_000,
            "impulse_100ms_n_s": (0.100 * 400) / 2,
            "lever_arm_m": lever,
        },
        rel=0,
        abs=1e-8,
    )


def _signed_trace():
    # A negative tare, an earlier negative dip, an exact threshold and a stepped post-onset pulse.
    data = _fixture(1)
    data["setup"]["analysis_calibration"].update(gain=8, offset=-12)
    volts = data["samples"]
    volts[:] = [0.5] * 110 + [0.875] + [1.0] * 99 + [1.5] * (len(volts) - 210)
    volts[105] = 0.25
    data["meta"]["sampling"]["t0_s"] = 0.2
    return data


def test_signed_baseline_strict_crossing_and_asymmetric_impulse():
    data = _signed_trace()
    reading = replay(CELL_OPERATION, data)
    assert reading.reason == "eligible"
    assert dict(reading.operands) == pytest.approx(
        {
            "baseline_n": -8,
            "gross_peak_n": 0,
            "net_peak_n": 8,
            "onset_s": 0.310,
            "rfd_100ms_n_s": (8 - 3) / 0.1,
            "impulse_100ms_n_s": (0.5 * 3 + 99 * 4 + 0.5 * 8) / 1000,
            "lever_arm_m": 0.35,
        },
        rel=0,
        abs=1e-8,
    )
    assert reading.value == pytest.approx(2.8, rel=0, abs=1e-8)
    assert reading.radius == pytest.approx(0.8, rel=0, abs=1e-8)


FAULTS = (
    (3, "not_eligible", "missing_calibration"),
    (4, "not_eligible", "calibration_unit_mismatch"),
    (5, "not_eligible", "adc_saturation"),
    (6, "not_eligible", "protocol_geometry_ineligible"),
    (7, "not_eligible", "wrong_limb"),
    (8, "insufficient_evidence", "moment_arm_unestablished"),
)


@pytest.mark.parametrize(("index", "reason", "finding"), FAULTS)
def test_native_task_refusal(index, reason, finding):
    reading = replay(CELL_OPERATION, _fixture(index))
    assert (reading.reason, reading.claims) == (reason, (finding,))
    assert reading.value is reading.radius is None and not reading.operands


def test_clock_rail_window_and_attachment_faults():
    data = _signed_trace()
    data["meta"]["sampling"]["measured_hz"] = 999
    with pytest.raises(ValueError, match="exact native 1000 Hz"):
        replay(CELL_OPERATION, data)
    data["meta"]["sampling"]["measured_hz"] = 1000
    data["samples"][200] = 0
    assert replay(CELL_OPERATION, data).claims == ("adc_saturation",)
    data["samples"][200] = 1
    geometry = data["meta"]["acquisition"]["sources"][0]["geometry"]
    attachment = geometry["attachment_point"]
    geometry["attachment_point"] = "Strap at the ankle, arm measured along the shank"
    assert replay(CELL_OPERATION, data).claims == ("protocol_geometry_ineligible",)
    geometry["attachment_point"] = attachment
    data["meta"]["n_rows"] = 210
    data["samples"] = data["samples"][:210]
    assert replay(CELL_OPERATION, data).claims == ("no_resolved_window",)
    data["samples"] = [0.5] * 210
    assert replay(CELL_OPERATION, data).claims == ("no_resolved_window",)
