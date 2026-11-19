"""Load-cell acquisition, calibration and integrity flags versus independent RES-669 truth.

Expected values come from the IsometricRamp closed form and load_cell_voltage, which the
production engine never imports.
"""

import math
from pathlib import Path

import numpy as np
import pytest

from dynamisbench_soccer.engines import load_cell_acquisition as lca
from dynamisbench_soccer.engines.load_cell_acquisition import (
    KGF_TO_N,
    CalibrationPoints,
    CellSpec,
    RawRecord,
    acquire,
    check_anti_aliasing,
    check_calibration_fit,
    check_drift,
    check_off_axis,
    check_saturation,
    check_temperature,
    check_timing,
    check_units,
    check_zero,
    fit_points,
    fit_record,
    from_bundle,
    input_digest,
    window_mask,
)
from dynamisbench_soccer.qc.spec import Status
from dynamisbench_soccer.reference.exporters import export_case
from dynamisbench_soccer.reference.faults import ContinuousFault, ContinuousKind, inject
from dynamisbench_soccer.reference.motion_models import IsometricRamp
from dynamisbench_soccer.reference.observations import load_cell_voltage
from dynamisbench_soccer.signals.schema import Calibration, DeviceClass, SignalBundle

GAIN_V_PER_N = 0.001  # RES-669 prescribed ideal sensitivity (exporter spec)
OFFSET_V = 0.2


def _truth(ramp: IsometricRamp, time: np.ndarray) -> np.ndarray:
    return np.array([ramp.force_n(float(t)) for t in time])


def _volts(ramp: IsometricRamp, time: np.ndarray, gain: float = GAIN_V_PER_N) -> np.ndarray:
    return np.array(
        [load_cell_voltage(ramp, float(t), volts_per_n=gain, offset_v=OFFSET_V) for t in time]
    )


@pytest.mark.parametrize("hz", [500, 1000, 2000])
def test_exported_voltage_recovers_closed_form_force(hz: int) -> None:
    case = export_case("load_cell_isometric.load_cell", hz=hz)
    before = input_digest(from_bundle(case.observation))
    acquired = acquire(case.observation)
    ramp = IsometricRamp()
    np.testing.assert_allclose(
        acquired.force_n, _truth(ramp, acquired.timing.time_s), rtol=0, atol=1e-9
    )
    assert acquired.calibration.source == "record"
    assert acquired.calibration.gain_n_per_unit == pytest.approx(1 / GAIN_V_PER_N, rel=1e-12)
    assert acquired.timing.uniform
    assert acquired.timing.measured_hz == pytest.approx(hz, rel=1e-9)
    assert acquired.record.device_class is DeviceClass.LOAD_CELL
    assert input_digest(from_bundle(case.observation)) == before


def test_isometric_dynamometer_newton_channel_passes_through_without_reapplying_records() -> None:
    case = export_case("load_cell_isometric.isometric_rig", hz=1000)
    acquired = acquire(case.observation)
    assert acquired.calibration.source == "passthrough"
    assert acquired.record.calibration  # upstream record retained as provenance only
    np.testing.assert_allclose(
        acquired.force_n, _truth(IsometricRamp(), acquired.timing.time_s), atol=1e-9
    )
    with pytest.raises(ValueError, match="apply twice"):
        acquire(
            case.observation,
            CellSpec(
                calibration=Calibration(
                    date="2000-01-01",
                    method="x",
                    channel="force",
                    input_unit="V",
                    output_unit="N",
                    gain=1000,
                    offset=0,
                )
            ),
        )


def test_two_point_calibration_recovers_sensitivity_exactly() -> None:
    ramp = IsometricRamp(peak_n=3000)
    loads = (0.0, 2000.0)
    readings = tuple(GAIN_V_PER_N * f + OFFSET_V for f in loads)
    fit = fit_points(
        CalibrationPoints(
            date="2026-01-01",
            method="dead-weight two-point",
            raw_unit="V",
            load_unit="N",
            applied_loads=loads,
            readings=readings,
        )
    )
    assert fit.gain_n_per_unit == pytest.approx(1 / GAIN_V_PER_N, rel=1e-12)
    assert fit.offset_n == pytest.approx(-OFFSET_V / GAIN_V_PER_N, rel=1e-12)
    assert fit.n_points == 2 and max(abs(r) for r in fit.residuals_n) < 1e-9
    assert check_calibration_fit(fit, 0.005).status is Status.NOT_APPLICABLE
    time = np.arange(0, ramp.duration_s, 0.001)
    record = RawRecord(_volts(ramp, time), "V", 1000)
    acquired = acquire(
        record,
        CellSpec(
            calibration_points=CalibrationPoints(
                date="2026-01-01",
                method="dead-weight two-point",
                raw_unit="V",
                load_unit="N",
                applied_loads=loads,
                readings=readings,
            )
        ),
    )
    np.testing.assert_allclose(acquired.force_n, _truth(ramp, time), atol=1e-8)


def test_multi_point_least_squares_reports_residuals_and_fit_quality() -> None:
    loads = np.array([0.0, 500.0, 1000.0, 1500.0, 2000.0, 2500.0])
    # Declared readings with a small quadratic non-linearity (2 mV at full scale).
    readings = GAIN_V_PER_N * loads + OFFSET_V + 0.002 * (loads / 2500) ** 2
    points = CalibrationPoints(
        date="2026-01-01",
        method="six-point dead-weight",
        raw_unit="V",
        load_unit="N",
        applied_loads=tuple(loads),
        readings=tuple(readings),
    )
    fit = fit_points(points)
    # Independent closed-form OLS of load on reading.
    x, y = readings, loads
    slope = np.sum((x - x.mean()) * (y - y.mean())) / np.sum((x - x.mean()) ** 2)
    intercept = y.mean() - slope * x.mean()
    residuals = y - (slope * x + intercept)
    assert fit.gain_n_per_unit == pytest.approx(slope, rel=1e-10)
    assert fit.offset_n == pytest.approx(intercept, rel=1e-9)
    np.testing.assert_allclose(fit.residuals_n, residuals, atol=1e-8)
    assert fit.rms_residual_n == pytest.approx(math.sqrt(np.mean(residuals**2)), rel=1e-8)
    assert fit.max_abs_residual_n == pytest.approx(np.max(np.abs(residuals)), rel=1e-8)
    assert 0.9999 < fit.r_squared < 1
    assert fit.gain_standard_error is not None and fit.gain_standard_error > 0
    assert fit.record is not None and fit.record.uncertainty is not None
    assert abs(sum(fit.residuals_n)) < 1e-8
    assert check_calibration_fit(fit, 0.01).status is Status.PASS
    assert check_calibration_fit(fit, 1e-6).status is Status.FAIL
    assert fit.summary()["n_points"] == 6


def test_kgf_points_and_inputs_use_exact_factor() -> None:
    assert KGF_TO_N == 9.80665
    fit = fit_points(
        CalibrationPoints(
            date="2026-01-01",
            method="kgf masses",
            raw_unit="mV",
            load_unit="kgf",
            applied_loads=(0.0, 100.0),
            readings=(0.0, 2.0),
        )
    )
    assert fit.gain_n_per_unit == pytest.approx(100 * 9.80665 / 2, rel=1e-14)
    values = np.array([0.0, 10.0, 250.0])
    acquired = acquire(RawRecord(values, "kgf", 1000))
    assert acquired.calibration.source == "kgf_exact"
    np.testing.assert_array_equal(acquired.force_n, values * 9.80665)


def test_calibration_record_forms_and_unit_checks() -> None:
    sensitivity = Calibration(
        date="2000-01-01",
        method="mV sensitivity",
        channel="voltage",
        input_unit="N",
        output_unit="mV",
        gain=1.0,
        offset=200.0,
    )
    readout = Calibration(
        date="2000-01-01",
        method="mV readout",
        channel="voltage",
        input_unit="mV",
        output_unit="N",
        gain=1.0,
        offset=-200.0,
    )
    a, b = fit_record(sensitivity, "mV"), fit_record(readout, "mV")
    assert (a.gain_n_per_unit, a.offset_n) == pytest.approx((b.gain_n_per_unit, b.offset_n))
    with pytest.raises(ValueError, match="does not match raw unit"):
        fit_record(sensitivity, "V")
    kgf_readout = readout.model_copy(update={"output_unit": "kgf"})
    assert fit_record(kgf_readout, "mV").gain_n_per_unit == pytest.approx(KGF_TO_N)
    with pytest.raises(ValueError, match="gain and offset"):
        fit_record(readout.model_copy(update={"offset": None}), "mV")
    with pytest.raises(ValueError, match="do not map"):
        fit_record(readout.model_copy(update={"input_unit": "N"}), "mV")


def test_unknown_units_and_missing_calibration_are_refused() -> None:
    with pytest.raises(ValueError, match="unknown raw unit 'lbf'"):
        acquire(RawRecord(np.zeros(10), "lbf", 1000))
    with pytest.raises(ValueError, match="none is invented"):
        acquire(RawRecord(np.zeros(10), "V", 1000))
    with pytest.raises(ValueError, match="integers"):
        acquire(
            RawRecord(np.array([0.0, 1.5, 2.0]), "count", 1000),
            CellSpec(calibration=_count_record()),
        )
    with pytest.raises(ValueError):  # schema refuses unrecognised calibration units
        Calibration(date="2000-01-01", method="x", channel="v", input_unit="lbf", output_unit="V")


def _count_record() -> Calibration:
    return Calibration(
        date="2026-01-01",
        method="ADC readout",
        channel="counts",
        input_unit="count",
        output_unit="N",
        gain=0.1,
        offset=-3276.8,
    )


def test_non_finite_samples_are_rejected_on_both_paths() -> None:
    values = np.ones(100)
    values[40] = np.nan
    with pytest.raises(ValueError, match="1 non-finite raw samples rejected"):
        acquire(RawRecord(values, "N", 1000))
    values[40] = np.inf
    with pytest.raises(ValueError, match="non-finite"):
        acquire(RawRecord(values, "N", 1000))
    case = export_case("load_cell_isometric.load_cell", hz=1000)
    case.observation.data["voltage"][10] = np.nan  # mutated after construction
    with pytest.raises(ValueError, match="non-finite"):
        acquire(case.observation)
    with pytest.raises(ValueError, match="non-finite timestamps"):
        acquire(RawRecord(np.ones(3), "N", 1000, time_s=np.array([0, np.nan, 0.002])))


def test_non_uniform_timestamps_are_detected_and_flagged() -> None:
    case = export_case("load_cell_isometric.load_cell", hz=1000)
    dropped = inject(case.observation, ContinuousFault(ContinuousKind.DROP, ("voltage",), stride=3))
    acquired = acquire(dropped)
    assert not acquired.timing.uniform
    timing, _ = check_timing(acquired.timing, 1000, 0.01)
    assert timing.status is Status.FAIL and "refused" in timing.detail
    with pytest.raises(ValueError, match="strictly increasing"):
        acquire(RawRecord(np.ones(3), "N", 1000, time_s=np.array([0.0, 0.0, 0.001])))


def test_rate_mismatch_and_low_rate_are_flagged() -> None:
    record = RawRecord(np.zeros(200), "N", 1000, time_s=np.arange(200) / 950.0)
    _, rate = check_timing(acquire(record).timing, 1000, 0.01)
    assert rate.status is Status.FAIL
    _, low = check_timing(acquire(RawRecord(np.zeros(200), "N", 500)).timing, 1000, 0.01)
    assert low.status is Status.FAIL and "declared metadata only" in low.detail


def test_unzeroed_cell_is_flagged_by_pre_tension_check() -> None:
    case = export_case("load_cell_isometric.load_cell", hz=1000)
    # 0.15 V on a 1 mV/N cell: a 150 N unremoved zero offset.
    unzeroed = inject(case.observation, ContinuousFault(ContinuousKind.OFFSET, ("voltage",), 0.15))
    for bundle, expected in ((case.observation, Status.PASS), (unzeroed, Status.FAIL)):
        acquired = acquire(bundle)
        mask = window_mask(acquired.timing.time_s, (0.0, 0.3))
        baseline = float(np.mean(acquired.force_n[mask]))
        result = check_zero(baseline, 20, bundle.meta.zeroing is not None)
        assert result.status is expected
    assert baseline == pytest.approx(150, abs=1e-9)
    assert "unzeroed cell or deliberate pre-tension" in result.detail


def test_zero_drift_between_declared_windows() -> None:
    time = np.arange(0, 3, 0.001)
    force = 5.0 * time  # 5 N/s zero drift
    assert check_drift(time, force, (0, 0.5), None, 2).status is Status.NOT_APPLICABLE
    drift = check_drift(time, force, (0, 0.5), (2.5, 3.0), 2)
    assert drift.status is Status.FAIL
    assert drift.values["drift_n"] == pytest.approx(12.5, abs=1e-6)


def test_saturation_and_clipping_are_flagged_against_declared_range_and_capacity() -> None:
    case = export_case("load_cell_isometric.load_cell", hz=1000)
    clipped = inject(case.observation, ContinuousFault(ContinuousKind.CLIPPING, ("voltage",), 2.0))
    cell = CellSpec(raw_range=(-2.0, 2.0), capacity_n=5000)
    rails, capacity = check_saturation(acquire(clipped, cell))
    assert rails.status is Status.FAIL and rails.values["samples_at_rail"] > 100
    assert capacity.status is Status.PASS
    clean_rails, _ = check_saturation(acquire(case.observation, CellSpec(raw_range=(-10, 10))))
    assert clean_rails.status is Status.PASS
    _, overload = check_saturation(acquire(case.observation, CellSpec(capacity_n=2000)))
    assert overload.status is Status.FAIL
    undeclared_rails, undeclared_capacity = check_saturation(acquire(clipped))
    assert undeclared_rails.status is Status.NOT_APPLICABLE
    assert "undetectable" in undeclared_rails.detail
    assert undeclared_capacity.status is Status.NOT_APPLICABLE
    counts = np.array([100.0, 30000.0, 65535.0, 65535.0, 200.0])
    cell = CellSpec(calibration=_count_record(), raw_range=(0, 65535))
    adc_rails, _ = check_saturation(acquire(RawRecord(counts, "count", 1000), cell))
    assert adc_rails.status is Status.FAIL and adc_rails.values["samples_at_rail"] == 2


def test_unit_mislabel_is_caught_against_declared_plausibility_and_capacity() -> None:
    ramp = IsometricRamp()  # 2500 N peak
    time = np.arange(0, ramp.duration_s, 0.001)
    newton = _truth(ramp, time)
    cell = CellSpec(expected_peak_n=(1000, 6000), capacity_n=10000)
    honest = acquire(RawRecord(newton, "N", 1000), cell)
    assert check_units(honest, float(honest.force_n.max())).status is Status.PASS
    # kgf data declared as N: ~255 "N".
    kgf_as_n = acquire(RawRecord(newton / KGF_TO_N, "N", 1000), cell)
    flag = check_units(kgf_as_n, float(kgf_as_n.force_n.max()))
    assert flag.status is Status.FAIL and "possible kgf data declared as N" in flag.detail
    # N data declared as kgf: ~24.5 kN, also an overload of the declared 10 kN cell.
    n_as_kgf = acquire(RawRecord(newton, "kgf", 1000), cell)
    flag = check_units(n_as_kgf, float(n_as_kgf.force_n.max()))
    assert flag.status is Status.FAIL and "possible N data declared as kgf" in flag.detail
    assert check_saturation(n_as_kgf)[1].status is Status.FAIL
    # Documented limitation: without declarations the mislabel is not detectable.
    blind = acquire(RawRecord(newton / KGF_TO_N, "N", 1000))
    limitation = check_units(blind, float(blind.force_n.max()))
    assert limitation.status is Status.NOT_APPLICABLE
    assert "undetectable" in limitation.detail


def test_temperature_off_axis_and_anti_aliasing_are_declared_or_stated() -> None:
    assert check_temperature(CellSpec(), 0.005).status is Status.NOT_APPLICABLE
    warm = CellSpec(temperature_c=(20, 30), span_temperature_coefficient_per_c=0.0002)
    hot = check_temperature(warm, 0.001)
    assert hot.status is Status.FAIL and hot.values["span_change_fraction"] == pytest.approx(0.002)
    with pytest.raises(ValueError, match="both"):
        CellSpec(temperature_c=(20, 30))
    off = check_off_axis()
    assert off.status is Status.NOT_APPLICABLE and "cannot detect off-axis" in off.detail
    timing = acquire(RawRecord(np.zeros(100), "N", 1000)).timing
    assert check_anti_aliasing(CellSpec(), timing).status is Status.NOT_APPLICABLE
    assert "ASSUMED" in check_anti_aliasing(CellSpec(), timing).detail
    assert check_anti_aliasing(CellSpec(anti_aliasing_hz=200), timing).status is Status.PASS
    assert check_anti_aliasing(CellSpec(anti_aliasing_hz=600), timing).status is Status.FAIL


def test_wrong_device_and_ambiguous_inputs_are_refused() -> None:
    plate = export_case("force_platform.single", hz=1000).observation
    with pytest.raises(ValueError, match="device class"):
        acquire(plate)
    case = export_case("load_cell_isometric.load_cell", hz=1000)
    with pytest.raises(ValueError, match="exactly one"):
        acquire(case.observation, CellSpec(channel="missing"))
    with pytest.raises(ValueError, match="one calibration source"):
        CellSpec(
            calibration=_count_record(),
            calibration_points=CalibrationPoints(
                date="2026-01-01",
                method="x",
                raw_unit="V",
                load_unit="N",
                applied_loads=(0, 1),
                readings=(0, 1),
            ),
        )
    assert isinstance(case.observation, SignalBundle)


def test_acquisition_module_never_imports_reference_substrate() -> None:
    source = Path(lca.__file__).read_text(encoding="utf-8")
    assert "dynamisbench_soccer.reference" not in source
    assert "import reference" not in source
