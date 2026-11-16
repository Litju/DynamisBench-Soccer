"""Production outputs versus separate RES-669 truth; production imports no oracle."""

from itertools import product

import numpy as np
import pytest

from dynamisbench_soccer.engines.force_platform import MethodSpec, cumulative, filter_force, process
from dynamisbench_soccer.engines.force_platform_acquisition import PlateSpec, input_digest
from dynamisbench_soccer.reference.exporters import export_case
from dynamisbench_soccer.reference.faults import (
    ContinuousFault,
    ContinuousKind,
    ProtocolFault,
    ProtocolKind,
    inject,
)
from dynamisbench_soccer.reference.kinematics import G, cmj, dj, hops, sj
from dynamisbench_soccer.signals.schema import BundleMeta, Side, SignalBundle


@pytest.mark.parametrize("rate", [600, 1000, 2000])
@pytest.mark.parametrize("family", ["force_platform.single", "force_platform.dual"])
@pytest.mark.parametrize("factory", [cmj, sj, dj, hops])
def test_reconstructs_all_vertical_fixtures_single_dual_and_rates(
    rate: int, family: str, factory
) -> None:  # type: ignore[no-untyped-def]
    motion = factory()
    pair = export_case(family, motion, hz=rate)
    external = export_case(family, cmj(), hz=rate).observation if factory in (dj, hops) else None
    method = MethodSpec(
        integration_start="record_start" if external is not None else "quiet_start",
        initial_velocity_m_s=motion.initial_velocity_m_s if external is not None else None,
    )
    before = input_digest(pair.observation)
    result = process(pair.observation, method=method, weighing=external)
    truth = np.array(
        [
            [motion.state(float(t)).velocity_m_s, motion.state(float(t)).displacement_m]
            for t in result.time_s
        ]
    )
    assert result.mass_kg == pytest.approx(75, abs=1e-10)
    assert np.max(np.abs(result.velocity_m_s - truth[:, 0])) < 0.06
    assert np.max(np.abs(result.displacement_m - truth[:, 1])) < 0.07
    intervals = motion.flight_intervals()
    assert len(result.flights) == len(intervals)
    for actual, (off, on) in zip(result.flights, intervals, strict=True):
        assert abs(actual.takeoff_s - off) <= 1 / rate + 1e-10
        assert abs(actual.landing_s - on) <= 1 / rate + 1e-10
        assert abs(actual.takeoff_velocity_m_s - motion.state(off).velocity_m_s) < 0.06
        assert abs(actual.height_flight_time_m - G * (on - off) ** 2 / 8) < 0.003
    np.testing.assert_allclose(
        result.grf_power_w - result.net_power_w,
        result.body_weight_n * result.velocity_m_s,
        atol=1e-10,
    )
    assert input_digest(pair.observation) == before
    assert result.method_spec["input_digests"][0] == before
    assert "diagnostic_only" in result.report_json()
    assert result.cop_m is None


@pytest.mark.parametrize("rate", [600, 1000, 2000])
@pytest.mark.parametrize("family", ["force_platform.single", "force_platform.dual"])
@pytest.mark.parametrize(
    "filtering,integration,onset",
    list(
        product(
            ["raw", "butterworth", "chebyshev"],
            ["trapezoid", "simpson"],
            ["5sd", "reverse_scan", "first_derivative", "absolute_N"],
        )
    ),
)
@pytest.mark.parametrize("factory", [cmj, sj, dj, hops])
def test_named_alternatives_against_independent_cmj_truth(
    rate: int, family: str, filtering: str, integration: str, onset: str, factory
) -> None:  # type: ignore[no-untyped-def]
    motion = factory()
    external = factory in (dj, hops)
    result = process(
        export_case(family, motion, hz=rate).observation,
        method=MethodSpec(
            filter=filtering,
            integration=integration,
            onset=onset,
            integration_start="record_start" if external else "quiet_start",
            initial_velocity_m_s=motion.initial_velocity_m_s if external else None,
        ),
        weighing=export_case(family, cmj(), hz=rate).observation if external else None,
    )
    takeoff = result.flights[0]
    assert abs(takeoff.takeoff_velocity_m_s - motion.takeoff_velocity_m_s()) < (
        0.15 if external else 0.075
    )
    if not external:
        assert result.onset.time_s is not None
        assert abs(result.onset.time_s - 1) < 0.15
    exact = np.array([motion.state(float(t)).displacement_m for t in result.time_s])
    assert np.max(np.abs(result.displacement_m - exact)) < 0.1
    assert result.method_spec["filter"] == filtering
    assert result.method_spec["integration"] == integration
    assert result.method_spec["onset"] == onset


@pytest.mark.parametrize("rate", [600, 1000, 2000])
@pytest.mark.parametrize("family", ["force_platform.single", "force_platform.dual"])
@pytest.mark.parametrize("contact", ["fixed_N", "system_weight_drop", "k_sd_unloaded"])
@pytest.mark.parametrize("start", ["quiet_start", "onset", "record_start"])
def test_contact_and_integration_start_alternatives(
    rate: int, family: str, contact: str, start: str
) -> None:
    motion = cmj()
    result = process(
        export_case(family, motion, hz=rate).observation,
        method=MethodSpec(contact=contact, unloaded_window_s=(1.9, 2), integration_start=start),
    )
    assert abs(result.flights[0].takeoff_velocity_m_s - motion.takeoff_velocity_m_s()) < 0.03
    assert abs(result.flights[0].takeoff_s - motion.takeoff_s()) <= 1 / rate + 1e-10


def test_smooth_quadrature_has_independent_polynomial_reference_and_convergence() -> None:
    errors = []
    for n in (21, 41, 81):
        time = np.linspace(0, 2, n)
        trap = cumulative(time**2, time, "trapezoid")
        simpson = cumulative(time**2, time, "simpson")
        np.testing.assert_allclose(simpson, time**3 / 3, atol=1e-12)
        errors.append(abs(trap[-1] - 8 / 3))
    assert errors[0] / errors[1] == pytest.approx(4, rel=1e-9)
    assert errors[1] / errors[2] == pytest.approx(4, rel=1e-9)


@pytest.mark.parametrize("filtering", ["raw", "butterworth", "chebyshev"])
def test_filters_preserve_calibrated_dc_and_do_not_mutate_input(filtering: str) -> None:
    time = np.arange(3000, dtype=np.float64) / 1000
    force = np.full(3000, 735.49875)
    before = force.copy()
    filtered = filter_force(force, time, MethodSpec(filter=filtering))
    np.testing.assert_allclose(filtered, force, atol=1e-9)
    np.testing.assert_array_equal(force, before)


def test_production_qc_rejects_observable_faults_and_uses_assignment_and_isolated_loads() -> None:
    b = export_case("force_platform.dual", hz=1000).observation
    noisy = inject(
        b, ContinuousFault(ContinuousKind.NOISE, ("fz_left", "fz_right"), magnitude=0.1, seed=7)
    )
    clean = process(noisy, assignment_reference={1: Side.LEFT, 2: Side.RIGHT})
    assert clean.verdict != "reject"
    swapped = inject(noisy, ProtocolFault(ProtocolKind.SIDE_SWAP))
    wrong_sides = process(noisy, assignment_reference={1: Side.RIGHT, 2: Side.LEFT})
    assert wrong_sides.verdict == "reject"
    assert any(r.rule_id == "FP-ASSIGN" and r.status.value == "fail" for r in wrong_sides.qc)
    original = process(noisy)
    np.testing.assert_allclose(process(swapped).force_n, original.force_n, atol=1e-12)
    movement = inject(noisy, ProtocolFault(ProtocolKind.STEP_OFF, index=10))
    with pytest.raises(ValueError, match="stable quiet stance"):
        process(movement)
    trial_data = {k: v.copy() for k, v in b.data.items()}
    trial_data["fz_left"][:] = 500
    trial_data["fz_right"][:] = 20 * np.sin(np.arange(b.meta.n_rows) * 0.1)
    isolated = SignalBundle(b.meta, trial_data)
    result = process(noisy, isolated_loads={1: isolated})
    assert any(r.rule_id == "FP-CROSSTALK-1" and r.status.value == "fail" for r in result.qc)
    assert result.method_spec["cross_talk_inputs"]["1"] == input_digest(isolated)


def test_conditional_input_uncertainty_is_computed_without_inventing_hardware_budget() -> None:
    b = export_case("force_platform.single", hz=1000).observation
    base = process(b, method=MethodSpec(calibration_force_bound_n=1, body_weight_bound_n=0.5))
    assert not base.uncertainty["complete_measurement_uncertainty"]
    assert base.uncertainty["velocity_input_error_envelope_m_s"] > 0
    perturb = inject(b, ContinuousFault(ContinuousKind.OFFSET, ("fz",), magnitude=0.5))
    changed = process(perturb)
    assert (
        np.max(np.abs(changed.velocity_m_s - base.velocity_m_s))
        <= base.uncertainty["velocity_input_error_envelope_m_s"]
    )
    assert process(b).uncertainty["velocity_input_error_envelope_m_s"] is None


def test_external_weighing_and_initial_velocity_are_required_for_drop_jump() -> None:
    b = export_case("force_platform.single", dj(), hz=1000).observation
    with pytest.raises(ValueError):
        process(b)
    with pytest.raises(ValueError, match="explicit initial velocity"):
        process(
            b,
            weighing=export_case("force_platform.single", hz=1000).observation,
            method=MethodSpec(integration_start="record_start"),
        )


def test_no_onset_is_absent_not_index_zero() -> None:
    original = export_case("force_platform.single", hz=1000).observation
    data = {n: v.copy() for n, v in original.data.items()}
    data["fz"][:] = 75 * G
    static = SignalBundle(original.meta, data)
    result = process(static)
    assert result.onset.index is None
    assert result.onset.time_s is None
    assert not result.flights
    with pytest.raises(ValueError, match="unidentified onset"):
        process(static, method=MethodSpec(integration_start="onset"))


def test_invalid_processing_parameters_and_data_reject() -> None:
    b = export_case("force_platform.single", hz=1000).observation
    with pytest.raises(ValueError):
        MethodSpec(quiet_window_s=(0, 0.5))
    with pytest.raises(ValueError, match="Nyquist"):
        process(b, method=MethodSpec(filter="butterworth", cutoff_hz=600))
    with pytest.raises(ValueError, match="unloaded window"):
        process(b, method=MethodSpec(contact="k_sd_unloaded"))
    with pytest.raises(ValueError, match="overlap"):
        process(b, plates=(PlateSpec(plate=1, zero_window_s=(0, 0.5)),))
    with pytest.raises(ValueError):
        process(export_case("force_platform.single", hz=100).observation)
    broken = {n: v.copy() for n, v in b.data.items()}
    broken["time"][5] = broken["time"][4]
    with pytest.raises(ValueError):
        process(SignalBundle(b.meta, broken))


def test_vendor_values_cannot_change_computed_metrics() -> None:
    from dynamisbench_soccer.signals.schema import VendorValue

    b = export_case("force_platform.single", hz=1000).observation
    before = process(b)
    raw = b.meta.model_dump()
    raw["vendor_reported"] = {"jump_height": VendorValue(value=999, unit="m").model_dump()}
    after = process(SignalBundle(BundleMeta.model_validate(raw), b.data))
    assert after.metrics == before.metrics
    assert after.flights == before.flights


def test_filtered_variants_do_not_claim_raw_input_uncertainty_envelopes() -> None:
    motion = dj()
    b = export_case("force_platform.single", motion, hz=600).observation
    weighing = export_case("force_platform.single", cmj(), hz=600).observation
    result = process(
        b,
        method=MethodSpec(
            filter="butterworth",
            cutoff_hz=0.1,
            integration_start="record_start",
            initial_velocity_m_s=motion.initial_velocity_m_s,
            calibration_force_bound_n=1,
            body_weight_bound_n=0,
        ),
        weighing=weighing,
    )
    assert result.uncertainty["velocity_input_error_envelope_m_s"] is None
    assert "filter-operator" in result.uncertainty["input_envelope_unavailable"]


def test_resolved_qc_rules_are_validated_and_bound_to_result() -> None:
    from dynamisbench_soccer.engines.force_platform import _default_rules

    b = export_case("force_platform.single", hz=600).observation
    default = process(b)
    custom = _default_rules()
    custom.version = "changed-review-rules"
    custom.by_id("QC-S03").params["min_run"] = 100000.0
    adjusted = process(b, rules=custom)
    assert default.verdict != adjusted.verdict
    assert default.method_spec["qc_rules_sha256"] != adjusted.method_spec["qc_rules_sha256"]
    assert adjusted.method_spec["qc_rules"]["version"] == "changed-review-rules"
    custom.by_id("QC-S03").params = {}
    with pytest.raises(ValueError, match="QC parameter names"):
        process(b, rules=custom)
    unknown = _default_rules()
    unknown.rule[0].id = "QC-Z99"
    with pytest.raises(ValueError, match="invalid QC rules"):
        process(b, rules=unknown)


def test_separate_weighing_cannot_use_another_subjects_bodyweight() -> None:
    motion = dj()
    b = export_case("force_platform.single", motion, hz=600).observation
    w = export_case("force_platform.single", cmj(), hz=600).observation
    raw = w.meta.model_dump()
    raw["subject"]["pseudonym"] = "OTHER-SUBJECT"
    wrong = SignalBundle(BundleMeta.model_validate(raw), w.data)
    with pytest.raises(ValueError, match="weighing record must match"):
        process(
            b,
            weighing=wrong,
            method=MethodSpec(
                integration_start="record_start", initial_velocity_m_s=motion.initial_velocity_m_s
            ),
        )
