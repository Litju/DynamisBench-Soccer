"""Load-cell engine metrics versus the independent RES-669 IsometricRamp closed form.

Expected values are derived here from the ramp's piecewise-linear definition:
F(t) = b for t <= t0; b + (P - b)(t - t0)/R during the rise; P during the hold. Hence for
0 < w <= R: RFD(0-w) = (P - b)/R and net impulse(0-w) = (P - b) w^2 / (2R). The engine
never imports the reference substrate.
"""

import ast
import json
import subprocess
import sys
from pathlib import Path
from typing import get_args

import numpy as np
import pytest

from dynamisbench_soccer.cli import main
from dynamisbench_soccer.engines import force_platform, load_cell, load_cell_acquisition
from dynamisbench_soccer.engines.load_cell import (
    LIMITATIONS,
    ONSET_METHODS,
    FilterSpec,
    MethodSpec,
    process,
    sensitivity,
)
from dynamisbench_soccer.engines.load_cell_acquisition import (
    KGF_TO_N,
    CellSpec,
    RawRecord,
    acquire,
)
from dynamisbench_soccer.qc.spec import Status
from dynamisbench_soccer.reference.exporters import export_case
from dynamisbench_soccer.reference.faults import ContinuousFault, ContinuousKind, inject
from dynamisbench_soccer.reference.motion_models import IsometricRamp
from dynamisbench_soccer.reference.observations import load_cell_voltage
from dynamisbench_soccer.signals import io
from dynamisbench_soccer.signals.schema import Calibration, Geometry

SENSITIVITY_V_PER_N = 0.0005
OFFSET_V = 0.05
DEFINITION = "perpendicular distance from joint centre to the cell attachment"


def _record(ramp: IsometricRamp, hz: float, **kwargs: object) -> RawRecord:
    time = np.arange(round(ramp.duration_s * hz) + 1) / hz
    volts = np.array(
        [
            load_cell_voltage(ramp, float(t), volts_per_n=SENSITIVITY_V_PER_N, offset_v=OFFSET_V)
            for t in time
        ]
    )
    calibration = Calibration(
        date="2026-01-01",
        method="declared sensitivity record (synthetic)",
        channel="voltage",
        input_unit="N",
        output_unit="V",
        gain=SENSITIVITY_V_PER_N,
        offset=OFFSET_V,
    )
    return RawRecord(volts, "V", hz, time_s=time, calibration=(calibration,), **kwargs)  # type: ignore[arg-type]


def _metric(result: load_cell.Processed, key: str) -> float:
    value = result.metrics[key]
    assert value is not None, result.refusals.get(key)
    return value


RAMPS = [
    IsometricRamp(baseline_n=0, peak_n=2500, onset_s=0.5, rise_s=0.4, hold_s=0.7, release_s=0.3),
    IsometricRamp(baseline_n=15, peak_n=3200, onset_s=0.6, rise_s=0.25, hold_s=1.0, release_s=0.2),
]


@pytest.mark.parametrize("hz", [1000, 2000])
@pytest.mark.parametrize("ramp", RAMPS)
@pytest.mark.parametrize("onset", ["5sd", "reverse_scan"])
def test_analytical_recovery_of_peak_onset_rfd_and_impulse(
    ramp: IsometricRamp, hz: int, onset: str
) -> None:
    result = process(_record(ramp, hz), method=MethodSpec(onset=onset))  # type: ignore[arg-type]
    b, p, t0, rise = ramp.baseline_n, ramp.peak_n, ramp.onset_s, ramp.rise_s
    assert result.onset.status == "detected"
    assert abs(_metric(result, "onset_time_s") - t0) <= 1 / hz + 1e-9
    assert _metric(result, "baseline_n") == pytest.approx(b, abs=1e-8)
    assert _metric(result, "peak_force_n") == pytest.approx(p, rel=1e-10)
    assert _metric(result, "net_peak_force_n") == pytest.approx(p - b, rel=1e-10)
    assert _metric(result, "time_to_peak_s") == pytest.approx(rise, abs=1 / hz + 1e-9)
    for w_ms in (50, 100, 200):
        w = w_ms / 1000
        assert _metric(result, f"rfd_0_{w_ms}ms_n_s") == pytest.approx((p - b) / rise, rel=1e-6)
        assert _metric(result, f"net_impulse_0_{w_ms}ms_n_s") == pytest.approx(
            (p - b) * w**2 / (2 * rise), rel=1e-6
        )
    gross = (
        b * t0 + (b + p) * rise / 2 + p * ramp.hold_s + (b + p) * ramp.release_s / 2
    )  # whole-record closed form
    assert _metric(result, "gross_impulse_record_n_s") == pytest.approx(gross, rel=1e-9)
    assert result.verdict in {"accept", "accept_with_flags"}
    assert {r.rule_id for r in result.qc if r.status is Status.FAIL} <= {"LC-ZERO"}


def test_noisy_ramp_with_named_filter_variants_recovers_within_stated_tolerance() -> None:
    ramp = RAMPS[0]
    clean = _record(ramp, 1000)
    rng = np.random.default_rng(696)
    noisy = RawRecord(
        clean.values + rng.normal(0, 0.5 * SENSITIVITY_V_PER_N, len(clean.values)),  # 0.5 N SD
        "V",
        1000,
        time_s=clean.time_s,
        calibration=clean.calibration,
    )
    method = MethodSpec(
        peak_filter=FilterSpec(kind="butterworth_lowpass", cutoff_hz=20),
        rfd_filter=FilterSpec(kind="butterworth_lowpass", cutoff_hz=100),
    )
    result = process(noisy, method=method)
    slope = (ramp.peak_n - ramp.baseline_n) / ramp.rise_s
    # Stated tolerance: onset within 5 ms, peak within 0.5 %, RFD 0-200 ms within 3 %.
    assert abs(_metric(result, "onset_time_s") - ramp.onset_s) <= 0.005
    assert _metric(result, "peak_force_n") == pytest.approx(ramp.peak_n, rel=0.005)
    assert _metric(result, "rfd_0_200ms_n_s") == pytest.approx(slope, rel=0.03)
    assert result.method_spec["peak_filter"] != result.method_spec["rfd_filter"]
    assert not np.array_equal(result.peak_force_signal_n, result.rfd_force_signal_n)


def test_onset_variant_names_match_force_platform_engine() -> None:
    fp = set(get_args(force_platform.MethodSpec.model_fields["onset"].annotation))
    lc = set(get_args(MethodSpec.model_fields["onset"].annotation))
    assert lc == fp == {"5sd", "reverse_scan", "first_derivative", "absolute_N"}
    assert set(ONSET_METHODS) == fp


def test_onset_variants_on_ideal_ramp_have_known_method_offsets() -> None:
    ramp, hz = RAMPS[0], 1000
    slope = (ramp.peak_n - ramp.baseline_n) / ramp.rise_s
    record = _record(ramp, hz)
    found = {name: process(record, method=MethodSpec(onset=name)).onset for name in ONSET_METHODS}  # type: ignore[arg-type]
    assert found["5sd"].time_s == pytest.approx(ramp.onset_s, abs=1e-9)
    assert found["reverse_scan"].time_s == pytest.approx(ramp.onset_s, abs=1e-9)
    # Fixed 20 N threshold: first sample above 20 N, minus one sample of backdating.
    lag = (np.floor(20 / slope * hz) + 1 - 1) / hz
    assert found["absolute_N"].time_s == pytest.approx(ramp.onset_s + lag, abs=1e-9)
    # Zero-lag derivative smoothing spreads the kink both ways: detected early, bounded.
    assert ramp.onset_s - 0.03 <= found["first_derivative"].time_s <= ramp.onset_s
    assert found["first_derivative"].threshold_unit == "N/s"


def test_isometric_baseline_does_not_use_body_weight() -> None:
    source = Path(load_cell.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module
    }
    assert "dynamisbench_soccer.engines.force_platform" not in imported
    assert "dynamisbench_soccer.engines.force_platform_acquisition" not in imported
    assert "body_weight" not in source and "bw" not in source.split()


def test_torque_requires_declared_lever_arm_and_definition_and_scales_with_it() -> None:
    ramp = RAMPS[1]
    geometry = Geometry(rig_description="rigid chair", lever_arm_m=0.4)
    force_only = process(_record(ramp, 1000, geometry=geometry))
    assert not force_only.torque.available
    assert "moment-arm definition" in (force_only.torque.reason or "")
    assert force_only.metrics["peak_torque_n_m"] is None
    assert "torque refused" in force_only.refusals["rtd_0_50ms_n_m_s"]
    missing = process(_record(ramp, 1000), CellSpec(moment_arm_definition=DEFINITION))
    assert missing.torque.reason == "torque refused: no declared lever arm"
    unknown = Geometry(rig_description="rigid chair", unknown=["lever_arm_m"], lever_arm_m=0.4)
    assert not process(_record(ramp, 1000, geometry=unknown)).torque.available
    cell = CellSpec(moment_arm_definition=DEFINITION)
    right = process(_record(ramp, 1000, geometry=geometry), cell)
    wrong = process(
        _record(ramp, 1000, geometry=geometry),
        CellSpec(moment_arm_definition=DEFINITION, lever_arm_m=0.5),
    )
    net = ramp.peak_n - ramp.baseline_n
    assert _metric(right, "net_peak_torque_n_m") == pytest.approx(net * 0.4, rel=1e-10)
    assert _metric(right, "rtd_0_100ms_n_m_s") == pytest.approx(net / ramp.rise_s * 0.4, rel=1e-6)
    for key in ("peak_torque_n_m", "rtd_0_50ms_n_m_s", "net_angular_impulse_0_200ms_n_m_s"):
        assert _metric(wrong, key) / _metric(right, key) == pytest.approx(0.5 / 0.4, rel=1e-12)
    for key in ("peak_force_n", "rfd_0_50ms_n_s"):
        assert _metric(wrong, key) == _metric(right, key)  # force estimand is unaffected


def test_gravity_correction_is_an_explicit_named_variant() -> None:
    ramp = RAMPS[0]
    geometry = Geometry(rig_description="rigid chair", lever_arm_m=0.4)
    cell = CellSpec(moment_arm_definition=DEFINITION, gravity_torque_n_m=12.5)
    default = process(_record(ramp, 1000, geometry=geometry), cell)
    assert default.method_spec["gravity_correction"] == "none"
    corrected = process(
        _record(ramp, 1000, geometry=geometry),
        cell,
        MethodSpec(gravity_correction="declared_gravity_torque"),
    )
    assert _metric(corrected, "peak_torque_n_m") - _metric(default, "peak_torque_n_m") == (
        pytest.approx(12.5)
    )
    assert _metric(corrected, "rtd_0_50ms_n_m_s") == _metric(default, "rtd_0_50ms_n_m_s")
    assert _metric(corrected, "peak_force_n") == _metric(default, "peak_force_n")
    with pytest.raises(ValueError, match="gravity_torque_n_m"):
        process(
            _record(ramp, 1000, geometry=geometry),
            CellSpec(moment_arm_definition=DEFINITION),
            MethodSpec(gravity_correction="declared_gravity_torque"),
        )


def test_exported_bundle_end_to_end_with_declared_definition() -> None:
    case = export_case("load_cell_isometric.load_cell", hz=1000)
    result = process(case.observation, CellSpec(moment_arm_definition=DEFINITION))
    ramp = IsometricRamp()
    assert result.verdict == "accept"
    assert _metric(result, "peak_torque_n_m") == pytest.approx(ramp.peak_n * 0.4)
    report = json.loads(result.report_json())
    assert report["engine"] == "load-cell.v1"
    assert report["estimands"]["torque"] is not None
    assert report["method_spec"]["geometry"]["rig_compliance_m_per_n"] == 0
    assert "not modelled" in report["uncertainty"]["rig_compliance"]
    assert any("not interchangeable with force-plate IMTP" in x for x in report["limitations"])
    assert any("Rig compliance" in x for x in report["limitations"])
    assert report["method_spec"]["input_digest"]


def test_non_uniform_timestamps_refuse_rfd_and_impulse_and_filtering() -> None:
    case = export_case("load_cell_isometric.load_cell", hz=1000)
    dropped = inject(case.observation, ContinuousFault(ContinuousKind.DROP, ("voltage",), stride=3))
    result = process(dropped)
    for w in (50, 100, 200):
        assert result.metrics[f"rfd_0_{w}ms_n_s"] is None
        assert "non-uniform" in result.refusals[f"rfd_0_{w}ms_n_s"]
        assert result.metrics[f"net_impulse_0_{w}ms_n_s"] is None
    assert result.metrics["peak_force_n"] == pytest.approx(IsometricRamp().peak_n)
    assert next(r for r in result.qc if r.rule_id == "LC-TIMING").status is Status.FAIL
    with pytest.raises(ValueError, match="uniform sampling"):
        process(dropped, method=MethodSpec(peak_filter=FilterSpec(kind="butterworth_lowpass")))


def test_rate_below_rfd_minimum_refuses_windows() -> None:
    result = process(_record(RAMPS[0], 500))
    assert result.metrics["rfd_0_50ms_n_s"] is None
    assert "below RFD minimum" in result.refusals["rfd_0_50ms_n_s"]
    assert result.metrics["peak_force_n"] == pytest.approx(RAMPS[0].peak_n)


def test_unzeroed_saturated_and_mislabelled_inputs_are_flagged_end_to_end() -> None:
    case = export_case("load_cell_isometric.load_cell", hz=1000)
    unzeroed = process(
        inject(case.observation, ContinuousFault(ContinuousKind.OFFSET, ("voltage",), 0.1))
    )
    zero = next(r for r in unzeroed.qc if r.rule_id == "LC-ZERO")
    assert zero.status is Status.FAIL and zero.values["baseline_n"] == pytest.approx(100)
    assert _metric(unzeroed, "net_peak_force_n") == pytest.approx(2500)  # net metrics robust
    clipped = process(
        inject(case.observation, ContinuousFault(ContinuousKind.CLIPPING, ("voltage",), 2.0)),
        CellSpec(raw_range=(-2.0, 2.0)),
    )
    assert clipped.verdict == "reject"
    assert next(r for r in clipped.qc if r.rule_id == "LC-RAILS").status is Status.FAIL
    ramp = IsometricRamp()
    time = np.arange(0, ramp.duration_s, 0.001)
    kgf_values = np.array([ramp.force_n(float(t)) for t in time]) / KGF_TO_N
    mislabel = process(RawRecord(kgf_values, "N", 1000), CellSpec(expected_peak_n=(1000, 6000)))
    assert mislabel.verdict == "reject"
    units = next(r for r in mislabel.qc if r.rule_id == "LC-UNITS")
    assert "possible kgf data declared as N" in units.detail
    honest = process(RawRecord(kgf_values, "kgf", 1000), CellSpec(expected_peak_n=(1000, 6000)))
    assert _metric(honest, "peak_force_n") == pytest.approx(ramp.peak_n, rel=1e-12)


def test_creep_and_drift_flags_on_declared_windows() -> None:
    time = np.arange(0, 4, 0.001)
    force = np.where(time < 0.5, 0.0, np.minimum((time - 0.5) * 10000, 2000.0))
    hold = time >= 0.7
    force[hold] -= 100 * (time[hold] - 0.7)  # 5 % decline over the 1 s near-peak hold
    force[time >= 3.0] = 30.0  # does not return to zero
    record = RawRecord(force, "N", 1000)
    result = process(record, method=MethodSpec(end_baseline_window_s=(3.2, 4.0)))
    creep = next(r for r in result.qc if r.rule_id == "LC-CREEP")
    drift = next(r for r in result.qc if r.rule_id == "LC-DRIFT")
    assert creep.status is Status.FAIL and "heuristic" in creep.detail
    assert drift.status is Status.FAIL and drift.values["drift_n"] == pytest.approx(30)


def test_absent_effort_reports_no_onset_and_refuses_windows() -> None:
    result = process(RawRecord(np.zeros(2000), "N", 1000))
    assert result.onset.index is None
    assert result.metrics["rfd_0_50ms_n_s"] is None
    assert next(r for r in result.qc if r.rule_id == "LC-ONSET").status is Status.FAIL


def test_sensitivity_reports_gain_filter_and_onset_variants() -> None:
    record = _record(RAMPS[0], 1000)
    report = sensitivity(record, method=MethodSpec(sensitivity_cutoffs_hz=(20.0, 50.0, 600.0)))
    variants = report["variants"]
    assert isinstance(variants, dict)
    for label, sign in (("gain_-0.01", -1), ("gain_+0.01", 1)):
        for key in ("net_peak_force_n", "rfd_0_100ms_n_s", "net_impulse_0_200ms_n_s"):
            assert variants[label][key]["relative"] == pytest.approx(sign * 0.01, abs=1e-9)
        assert variants[label]["onset_time_s"]["delta"] == 0
    assert {"butterworth_20hz", "butterworth_50hz"} <= set(variants)
    assert "butterworth_600hz" in report["skipped"]  # at/above Nyquist
    assert {f"onset_{n}" for n in ONSET_METHODS if n != "5sd"} <= set(variants)
    assert variants["onset_absolute_N"]["onset_time_s"]["delta"] > 0
    assert "rig compliance" in report["not_modelled"]
    json.dumps(report, allow_nan=False)


def test_limitations_are_recorded_in_outputs() -> None:
    result = process(_record(RAMPS[0], 1000))
    assert set(LIMITATIONS) <= set(result.limitations)
    assert result.limitations[-1].startswith("torque refused")
    report = result.report()
    assert report["qualification"].startswith("engineering calculations only")
    assert any("off-axis" in x for x in report["limitations"])  # type: ignore[union-attr]
    off_axis = next(r for r in result.qc if r.rule_id == "LC-OFFAXIS")
    assert off_axis.status is Status.NOT_APPLICABLE


def test_engine_never_imports_reference_substrate() -> None:
    for module in (load_cell, load_cell_acquisition):
        tree = ast.parse(Path(module.__file__ or "").read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith("dynamisbench_soccer.reference")
            if isinstance(node, ast.Import):
                assert not any(
                    a.name.startswith("dynamisbench_soccer.reference") for a in node.names
                )
    code = (
        "import sys\n"
        "import dynamisbench_soccer.engines.load_cell\n"
        "import dynamisbench_soccer.engines.load_cell_cli\n"
        "leaked = sorted(m for m in sys.modules if m.startswith('dynamisbench_soccer.reference'))\n"
        "assert not leaked, leaked\n"
        "print('clean')\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "clean"


def test_cli_processes_a_bundle_file_with_sensitivity(tmp_path: Path) -> None:
    case = export_case("load_cell_isometric.load_cell", hz=1000)
    stem = tmp_path / "isometric"
    io.write(case.observation, stem)
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps({"cell": {"moment_arm_definition": DEFINITION, "capacity_n": 5000}})
    )
    out = tmp_path / "report.json"
    args = ["load-cell", "process", str(stem), "--config", str(config), "--out", str(out)]
    code = main([*args, "--sensitivity"])
    assert code == 0
    report = json.loads(out.read_text())
    assert report["metrics"]["peak_force_n"] == pytest.approx(2500)
    assert report["metrics"]["peak_torque_n_m"] == pytest.approx(1000)
    assert report["sensitivity"]["variants"]["gain_+0.01"]
    overwrite = ["load-cell", "process", str(stem), "--config", str(config), "--out", str(config)]
    assert main(overwrite) == 2
    assert main(["load-cell", "process", str(stem), "--out", str(stem) + ".json"]) == 2
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"cell": {"capacity_n": -1}}))
    assert main(["load-cell", "process", str(stem), "--config", str(bad), "--out", str(out)]) == 2
    assert json.loads(out.read_text())["verdict"] == "reject"


def test_acquired_records_can_be_reprocessed_without_redeclaring_the_cell() -> None:
    acquired = acquire(_record(RAMPS[0], 1000))
    assert process(acquired).metrics == process(_record(RAMPS[0], 1000)).metrics
    with pytest.raises(ValueError, match="already carries"):
        process(acquired, CellSpec())
