"""Jump assessment versus independent closed forms from the RES-669 phase definitions.

CMJ: quiet 1 s, a = -2 for 0.25 s, +2.5 for 0.2 s, +9 for 0.3 s, take-off at v = 2.7 m/s.
Hence time to take-off = 0.75 s, minimum velocity = -0.5 m/s, unweighting depth = 2/g,
propulsive net impulse = m * 2.7, flight time = 2 v / g and h = v^2 / 2g.
"""

import ast
import json
import math
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from dynamisbench_soccer.assessment import jumps
from dynamisbench_soccer.assessment.jumps import (
    LIMITATIONS,
    DropEvidence,
    JumpSignal,
    JumpSpec,
    assess,
)
from dynamisbench_soccer.cli import main
from dynamisbench_soccer.engines.force_platform import MethodSpec, process
from dynamisbench_soccer.engines.force_platform_acquisition import PlateSpec
from dynamisbench_soccer.reference.exporters import export_case
from dynamisbench_soccer.reference.faults import ContinuousFault, ContinuousKind, inject
from dynamisbench_soccer.reference.kinematics import cmj, dj, hops, sj
from dynamisbench_soccer.signals import io
from dynamisbench_soccer.signals.schema import SignalBundle

G = 9.80665
HZ = 1000
MASS = 75.0
DROP = DropEvidence(
    touchdown_velocity_m_s=-math.sqrt(2 * G * 0.30),
    source="independent synthetic initial condition",
)
PLATE = (PlateSpec(plate=1, assume_field_axes=True),)


def _signal(trajectory, weigh: bool = False, **method: object) -> JumpSignal:
    observation = export_case("force_platform.single", trajectory, hz=HZ).observation
    weighing = export_case("force_platform.single", sj(), hz=HZ).observation if weigh else None
    result = process(
        observation,
        plates=PLATE,
        weighing=weighing,
        method=MethodSpec(**method),  # type: ignore[arg-type]
    )
    return JumpSignal.from_engine(result)


def _dj_signal() -> JumpSignal:
    observation = export_case("force_platform.single", dj(), hz=HZ).observation
    # A sub-newton contact-only ripple avoids the flat-maximum clipping heuristic.
    force = observation["fz"]
    data = {**observation.data, "fz": force + (force > 0) * np.sin(np.arange(len(force))) * 0.1}
    weighing = export_case("force_platform.single", sj(), hz=HZ).observation
    return JumpSignal.from_engine(
        process(
            SignalBundle(observation.meta, data),
            plates=PLATE,
            weighing=weighing,
            method=MethodSpec(
                integration_start="record_start", initial_velocity_m_s=DROP.touchdown_velocity_m_s
            ),
        )
    )


def test_cmj_events_phases_and_estimands_match_closed_forms() -> None:
    result = assess(_signal(cmj()), JumpSpec(protocol="cmj"))
    assert result.eligible and result.refusals == {}
    assert result.events["takeoff_s"] == pytest.approx(1.75, abs=2 / HZ)
    assert result.events["landing_s"] == pytest.approx(1.75 + 2 * 2.7 / G, abs=2 / HZ)
    assert result.phases["unweighting_s"] == pytest.approx(0.25, abs=3 / HZ)
    assert result.phases["braking_s"] == pytest.approx(0.20, abs=3 / HZ)
    assert result.phases["propulsion_s"] == pytest.approx(0.30, abs=3 / HZ)
    assert result.phases["time_to_takeoff_s"] == pytest.approx(0.75, abs=3 / HZ)
    e = result.estimands
    assert e["flight_time_s"] == pytest.approx(2 * 2.7 / G, abs=2 / HZ)
    assert e["takeoff_velocity_m_s"] == pytest.approx(2.7, abs=0.01)
    assert e["height_impulse_momentum_m"] == pytest.approx(2.7**2 / (2 * G), abs=0.003)
    assert e["height_flight_time_m"] == pytest.approx(2.7**2 / (2 * G), abs=0.003)
    assert e["unweighting_depth_fraction_of_bw"] == pytest.approx(2 / G, abs=0.002)
    assert e["peak_negative_velocity_m_s"] == pytest.approx(-0.5, abs=0.01)
    assert e["propulsive_net_impulse_n_s"] == pytest.approx(MASS * 2.7, rel=0.01)
    assert e["peak_force_n"] == pytest.approx(MASS * (G + 9.0), rel=1e-3)
    assert e["rsi_modified_m_s"] == pytest.approx(e["height_impulse_momentum_m"] / 0.75, rel=0.01)


def test_heights_are_separate_named_estimands_with_difference_reported() -> None:
    e = assess(_signal(cmj()), JumpSpec(protocol="cmj")).estimands
    assert "height_flight_time_m" in e and "height_impulse_momentum_m" in e
    assert e["height_difference_flight_minus_impulse_m"] == pytest.approx(
        e["height_flight_time_m"] - e["height_impulse_momentum_m"]
    )
    assert any("never merged" in text for text in LIMITATIONS)
    assert any("interchangeab" in text for text in LIMITATIONS)


def test_squat_jump_values_and_countermovement_eligibility_both_ways() -> None:
    squat = assess(_signal(sj()), JumpSpec(protocol="sj"))
    assert squat.eligible
    assert squat.estimands["unweighting_depth_fraction_of_bw"] == pytest.approx(0.0, abs=1e-3)
    assert squat.estimands["height_impulse_momentum_m"] == pytest.approx(
        2.4**2 / (2 * G), abs=0.003
    )
    assert squat.estimands["flight_time_s"] == pytest.approx(2 * 2.4 / G, abs=2 / HZ)
    wrong = assess(_signal(sj()), JumpSpec(protocol="cmj"))
    assert not wrong.eligible
    assert any(
        c["check"] == "countermovement_present" and not c["passed"] for c in wrong.eligibility
    )
    counter = assess(_signal(cmj()), JumpSpec(protocol="sj"))
    assert not counter.eligible
    assert any(c["check"] == "no_countermovement" and not c["passed"] for c in counter.eligibility)


def test_drop_jump_contact_time_height_and_rsi_variants() -> None:
    signal = _dj_signal()
    flight_basis = assess(signal, JumpSpec(protocol="dj", drop_height_m=0.30), drop_evidence=DROP)
    rebound = 2.6
    assert flight_basis.eligible
    e = flight_basis.estimands
    assert e["ground_contact_time_s"] == pytest.approx(0.18, abs=2 / HZ)
    assert e["flight_time_s"] == pytest.approx(2 * rebound / G, abs=2 / HZ)
    assert e["rsi_m_s"] == pytest.approx(rebound**2 / (2 * G) / 0.18, rel=0.02)
    momentum = assess(
        signal,
        JumpSpec(protocol="dj", drop_height_m=0.30, rsi_height_basis="impulse_momentum"),
        drop_evidence=DROP,
    )
    assert momentum.estimands["rsi_m_s"] == pytest.approx(
        momentum.estimands["height_impulse_momentum_m"]
        / momentum.estimands["ground_contact_time_s"]
    )
    assert momentum.estimands["rsi_m_s"] != flight_basis.estimands["rsi_m_s"]  # distinct numbers


def test_drop_height_contradiction_and_missing_declaration() -> None:
    signal = _dj_signal()
    wrong = assess(signal, JumpSpec(protocol="dj", drop_height_m=0.60), drop_evidence=DROP)
    assert not wrong.eligible
    assert any(
        c["check"] == "impact_speed_matches_drop_height" and not c["passed"]
        for c in wrong.eligibility
    )
    undeclared = assess(signal, JumpSpec(protocol="dj"))
    assert not undeclared.eligible and not undeclared.estimands
    assert "no declared drop height" in undeclared.refusals["drop_height"]


def test_repeated_hops_per_hop_and_aggregate_values() -> None:
    result = assess(
        _signal(hops(5), True, integration_start="record_start", initial_velocity_m_s=0.0),
        JumpSpec(protocol="repeated_hops"),
    )
    assert result.eligible
    v = 2.0
    assert result.estimands["hop_count"] == 5
    assert result.estimands["mean_contact_time_s"] == pytest.approx(0.2, abs=2 / HZ)
    assert result.estimands["mean_flight_time_s"] == pytest.approx(2 * v / G, abs=2 / HZ)
    assert result.estimands["mean_height_flight_time_m"] == pytest.approx(v**2 / (2 * G), abs=0.003)
    assert result.estimands["mean_rsi_m_s"] == pytest.approx(v**2 / (2 * G) / 0.2, rel=0.03)
    assert all(c["contact_time_s"] == pytest.approx(0.2, abs=2 / HZ) for c in result.contacts)
    too_few = assess(
        _signal(hops(2), True, integration_start="record_start", initial_velocity_m_s=0.0),
        JumpSpec(protocol="repeated_hops", min_hops=3),
    )
    assert not too_few.eligible


def test_bilateral_quantities_need_declared_sides_and_match_split() -> None:
    base = _signal(cmj())
    split = replace(base, plate_force_n={1: 0.4 * base.force_n, 2: 0.6 * base.force_n})
    unlabelled = assess(split, JumpSpec(protocol="cmj"))
    assert unlabelled.bilateral["available"] is False
    labelled = assess(split, JumpSpec(protocol="cmj", plate_sides={1: "left", 2: "right"}))
    assert labelled.bilateral["available"] is True
    assert labelled.bilateral["impulse_asymmetry_weaker_over_stronger_percent"] == pytest.approx(
        100 * 0.4 / 0.6, rel=1e-6
    )
    swapped = assess(split, JumpSpec(protocol="cmj", plate_sides={1: "right", 2: "left"}))
    assert swapped.bilateral["impulse_n_s"]["left"] == pytest.approx(  # type: ignore[index]
        labelled.bilateral["impulse_n_s"]["right"]  # type: ignore[index]
    )


def test_single_leg_eligibility_requires_unloaded_contralateral_plate() -> None:
    base = _signal(cmj())
    sides = {1: "left", 2: "right"}
    one_leg = replace(base, plate_force_n={1: base.force_n, 2: 0.01 * base.force_n})
    assert assess(one_leg, JumpSpec(protocol="slcmj", plate_sides=sides)).eligible  # type: ignore[arg-type]
    two_leg = replace(base, plate_force_n={1: 0.5 * base.force_n, 2: 0.5 * base.force_n})
    both = assess(two_leg, JumpSpec(protocol="slcmj", plate_sides=sides))  # type: ignore[arg-type]
    assert not both.eligible
    assert any(c["check"] == "contralateral_unloaded" and not c["passed"] for c in both.eligibility)
    assert not assess(base, JumpSpec(protocol="slcmj")).eligible  # one plate, no sides


def test_no_flight_is_refused_not_estimated() -> None:
    time = np.arange(0, 3, 1 / HZ)
    force = MASS * G * (1 + 0.05 * np.sin(2 * math.pi * time / 1.5) * (time > 1))
    signal = JumpSignal(time, force, np.zeros_like(time), np.zeros_like(time), MASS * G, MASS)
    result = assess(signal, JumpSpec(protocol="cmj"))
    assert not result.eligible
    assert "no complete flight" in result.refusals["jump"]
    assert "height_flight_time_m" not in result.estimands


def test_non_uniform_sampling_makes_assessment_ineligible() -> None:
    base = _signal(cmj())
    keep = np.arange(len(base.time_s)) % 9 != 4
    jittered = JumpSignal(
        base.time_s[keep],
        base.force_n[keep],
        base.velocity_m_s[keep],
        base.displacement_m[keep],
        base.body_weight_n,
        base.mass_kg,
    )
    result = assess(jittered, JumpSpec(protocol="cmj"))
    assert not result.eligible
    assert any(c["check"] == "uniform_sampling" and not c["passed"] for c in result.eligibility)


def test_device_rejection_is_carried_into_the_assessment() -> None:
    observation = export_case("force_platform.single", cmj(), hz=HZ).observation
    clipped = inject(observation, ContinuousFault(ContinuousKind.CLIPPING, ("fz",), 1000.0))
    device = process(clipped, plates=PLATE)
    result = assess(JumpSignal.from_engine(device), JumpSpec(protocol="cmj"))
    assert result.device_verdict == "reject"
    assert result.report()["diagnostic_only"] is True
    assert result.estimands["peak_force_n"] == pytest.approx(1000.0, abs=1e-6)


def test_threshold_and_bodyweight_sensitivity_are_reported() -> None:
    result = assess(_signal(cmj()), JumpSpec(protocol="cmj", body_weight_error_fraction=0.02))
    u = result.uncertainty
    assert u["flight_time_resolution_s"] == pytest.approx(1 / HZ)
    assert set(u["flight_time_by_contact_threshold_s"]) == {"10_n", "20_n", "50_n"}  # type: ignore[arg-type]
    low, high = u["height_impulse_momentum_bw_sensitivity_m"]  # type: ignore[misc]
    v = result.estimands["takeoff_velocity_m_s"]
    assert low == pytest.approx((0.98 * v) ** 2 / (2 * G))
    assert high == pytest.approx((1.02 * v) ** 2 / (2 * G))


def test_assessment_never_imports_reference_substrate_or_claims_equivalence() -> None:
    tree = ast.parse(Path(jumps.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("dynamisbench_soccer.reference")


def test_cli_end_to_end_and_exit_codes(tmp_path: Path) -> None:
    stem = tmp_path / "cmj"
    io.write(export_case("force_platform.single", cmj(), hz=HZ).observation, stem)
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "jump": {"protocol": "cmj"},
                "plates": [{"plate": 1, "assume_field_axes": True}],
            }
        )
    )
    out = tmp_path / "report.json"
    args = ["assess", "jump", str(stem), "--config", str(config)]
    code = main([*args, "--out", str(out)])
    report = json.loads(out.read_text())
    assert report["protocol"] == "cmj" and report["estimands"]["flight_time_s"] > 0.5
    assert report["device"]["engine"] == "force-platform.v1"
    assert code in {0, 1}
    assert main([*args, "--out", str(config)]) == 2
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"jump": {"protocol": "unknown"}}))
    assert main(["assess", "jump", str(stem), "--config", str(bad), "--out", str(out)]) == 2


def _contact_result(latency: float = 0.0):
    from dynamisbench_soccer.engines.contact import ContactPolicy, DurationRange, contact
    from dynamisbench_soccer.engines.event_time import ClockTiming, EdgeTiming, Timing
    from dynamisbench_soccer.reference.clocks import DeviceClock

    clock = ClockTiming(resolution_s=0.001, sync_latency_s=0, sync_latency_error_s=0)
    edge = EdgeTiming(latency_s=latency, latency_error_s=0.002)
    policy = ContactPolicy.model_validate(
        {
            "timing": Timing(clock_model="wired", clocks={"device": clock}, authority="RES-669"),
            "system": "contact_mat",
            "initial_event": "contact_snapshot",
            "switch_on": edge,
            "switch_off": edge,
            "debounce_s": 0.0,
            "expected_flights": 1,
            "flight": DurationRange(min_s=0.15, max_s=0.9),
            "ground_contact": DurationRange(min_s=0.04, max_s=0.5),
            "ballistic_equal_com_height": True,
        }
    )
    case = export_case(
        "timing_contact.contact", cmj(), clock=DeviceClock(0.0, 0.0), latency_s=latency
    )
    return contact(case.observation, policy)


def test_force_contact_and_imu_flight_estimates_stay_separate_labelled_sources() -> None:
    from dynamisbench_soccer.engines.imu import ImuSpec
    from dynamisbench_soccer.engines.imu import process as imu_process

    truth = 2 * 2.7 / G
    force = jumps.force_platform_estimate(assess(_signal(cmj()), JumpSpec(protocol="cmj")))
    contact_estimate = jumps.contact_system_estimates(_contact_result())
    imu = jumps.imu_estimates(
        imu_process(
            export_case("imu.inertial", cmj(), hz=500).observation,
            ImuSpec(initial_orientation=((1, 0, 0), (0, 1, 0), (0, 0, 1))),
        )
    )
    assert len(contact_estimate) == 1 and len(imu) == 1
    for estimate in (force, contact_estimate[0], imu[0]):
        assert estimate.flight_time_s == pytest.approx(truth, abs=0.004)
        assert estimate.flight_time_low_s <= estimate.flight_time_s <= estimate.flight_time_high_s
    comparison = jumps.compare_sources([force, contact_estimate[0], imu[0]])
    assert comparison["merged_value"] is None
    assert len(comparison["pairs"]) == 3  # type: ignore[arg-type]
    assert {p["source"] for p in comparison["estimates"]} == {  # type: ignore[attr-defined,union-attr]
        "force_platform:contact_threshold",
        "contact_system:contact_mat",
        "imu:specific_force_magnitude",
    }
    for pair in comparison["pairs"]:  # type: ignore[attr-defined]
        assert "not established" in pair["empirical_agreement"]


def test_latency_bias_shows_up_as_a_difference_not_a_correction() -> None:
    biased = jumps.contact_system_estimates(_contact_result(latency=0.02))[0]
    clean = jumps.contact_system_estimates(_contact_result())[0]
    assert clean.flight_time_s == pytest.approx(2 * 2.7 / G, abs=0.002)
    assert biased.flight_time_low_s <= biased.flight_time_s <= biased.flight_time_high_s
    with pytest.raises(ValueError, match="expected a contact-system result"):
        jumps.contact_system_estimates(object())
    with pytest.raises(ValueError, match="finite, positive and ordered"):
        jumps._flight_estimate("x", "y", 0.5, 0.6, 0.7)


def _lpt_report(trajectory, hz: int = 200) -> dict:
    from test_lpt_acquisition import LINE, LPT

    from dynamisbench_soccer.engines.lpt import MethodSpec as LptMethod
    from dynamisbench_soccer.engines.lpt import process as lpt_process

    case = export_case("lpt_encoder.lpt", trajectory, test_id="cmj", hz=hz)
    return lpt_process(case.observation, (LPT,), LINE, LptMethod(name="cubic_spline"))


def test_lpt_heights_are_attachment_estimands_with_own_definitions() -> None:
    motion = cmj()
    heights = jumps.lpt_estimates(_lpt_report(motion))
    apex_t = motion.takeoff_s() + motion.takeoff_velocity_m_s() / G
    apex = motion.state(apex_t).displacement_m
    assert heights.height_peak_displacement_above_start_m == pytest.approx(apex, abs=0.003)
    assert heights.peak_velocity_m_s == pytest.approx(motion.takeoff_velocity_m_s(), abs=0.05)
    assert heights.height_peak_velocity_m == pytest.approx(
        motion.takeoff_velocity_m_s() ** 2 / (2 * G), abs=0.01
    )
    assert "not comparable" in heights.definitions["comparability"]
    force = assess(_signal(cmj()), JumpSpec(protocol="cmj")).estimands["height_flight_time_m"]
    assert heights.height_peak_displacement_above_start_m != pytest.approx(force, abs=1e-3)


def test_lpt_cable_only_reports_refuse_jump_heights() -> None:
    with pytest.raises(ValueError, match="cable displacement is not vertical"):
        jumps.lpt_estimates({"estimand": "cable_displacement", "motion": {}})
    with pytest.raises(ValueError, match="processed LPT report"):
        jumps.lpt_estimates({"estimand": None})


def test_filtered_engine_force_never_moves_contact_events() -> None:
    raw = _signal(cmj())
    filtered = _signal(cmj(), filter="butterworth", cutoff_hz=50)
    a = assess(raw, JumpSpec(protocol="cmj"))
    b = assess(filtered, JumpSpec(protocol="cmj"))
    assert filtered.raw_force_n is not None
    assert b.events["takeoff_s"] == a.events["takeoff_s"]
    assert b.estimands["flight_time_s"] == a.estimands["flight_time_s"]
    # The filtered kinetics change the take-off velocity only by the filter's edge smearing.
    assert b.estimands["takeoff_velocity_m_s"] == pytest.approx(
        a.estimands["takeoff_velocity_m_s"], abs=0.05
    )


def test_cmj_and_unjustified_initial_velocity_cannot_be_accepted_as_drop_jumps() -> None:
    spec = JumpSpec(protocol="dj", drop_height_m=0.30)
    cmj_signal = _signal(cmj())
    for signal, evidence in ((cmj_signal, DROP), (_dj_signal(), None)):
        report = assess(signal, spec, drop_evidence=evidence).report()
        assert not report["eligible"] and report["diagnostic_only"]
        assert report["estimands"] == {} and report["contacts"] == []
        assert "rsi_m_s" in report["diagnostics"]["estimands"]
    forged = replace(cmj_signal, velocity_m_s=cmj_signal.velocity_m_s + DROP.touchdown_velocity_m_s)
    assert not assess(forged, spec, drop_evidence=DROP).eligible
    never_leaves = replace(
        cmj_signal, raw_force_n=None, force_n=np.full_like(cmj_signal.force_n, MASS * G)
    )
    assert not assess(never_leaves, spec, drop_evidence=DROP).eligible
