"""RES-671: eligibility and QC context travel with every exported jump estimate."""

import dataclasses

import numpy as np
import pytest
from test_jump_assessment import (
    HZ,
    PLATE,
    _accepted_signal,
    _contact_result,
    _lpt_report,
    _signal,
    cmj,
    export_case,
)

from dynamisbench_soccer.assessment import jumps
from dynamisbench_soccer.assessment.jumps import JumpSpec, assess
from dynamisbench_soccer.engines.force_platform import process
from dynamisbench_soccer.reference.kinematics import hops, sj
from dynamisbench_soccer.signals.schema import SignalBundle


def test_rejected_force_assessment_cannot_become_an_accepted_estimate() -> None:
    rejected = assess(_signal(cmj()), JumpSpec(protocol="cmj"))  # clipping rule rejects the fixture
    assert rejected.device_verdict == "reject" and rejected.report()["diagnostic_only"]
    with pytest.raises(ValueError, match="ineligible or device-rejected"):
        jumps.force_platform_estimate(rejected)
    diagnostic = jumps.force_platform_estimate(rejected, allow_diagnostic=True)
    assert diagnostic.diagnostic_only and diagnostic.qc_verdict == "reject"
    assert diagnostic.protocol_eligible is True  # protocol checks passed; the device did not
    # negative control: the accepted fixture exports a non-diagnostic estimate with context
    accepted = jumps.force_platform_estimate(
        assess(_accepted_signal(cmj()), JumpSpec(protocol="cmj"))
    )
    assert not accepted.diagnostic_only and accepted.qc_verdict != "reject"
    assert accepted.protocol == "cmj" and accepted.basis


def test_diagnostic_estimates_cannot_enter_an_accepted_comparison() -> None:
    accepted = jumps.force_platform_estimate(
        assess(_accepted_signal(cmj()), JumpSpec(protocol="cmj"))
    )
    rejected = jumps.force_platform_estimate(
        assess(_signal(cmj()), JumpSpec(protocol="cmj")), allow_diagnostic=True
    )
    contact = jumps.contact_system_estimates(_contact_result())[0]
    with pytest.raises(ValueError, match="diagnostic-only"):
        jumps.compare_sources([accepted, rejected, contact])
    marked = jumps.compare_sources([accepted, rejected], allow_diagnostic=True)
    assert all(p["diagnostic_only"] for p in marked["pairs"])  # type: ignore[attr-defined]
    clean = jumps.compare_sources([accepted, contact])
    assert not any(p["diagnostic_only"] for p in clean["pairs"])  # type: ignore[attr-defined]


def test_estimates_from_different_protocols_are_never_compared() -> None:
    a = jumps.force_platform_estimate(assess(_accepted_signal(cmj()), JumpSpec(protocol="cmj")))
    b = dataclasses.replace(a, protocol="sj", source="force_platform:other")
    with pytest.raises(ValueError, match="different protocols"):
        jumps.compare_sources([a, b])


def test_ineligible_protocol_is_refused_even_when_the_device_is_accepted() -> None:
    cmj_as_sj = assess(_accepted_signal(cmj()), JumpSpec(protocol="sj"))
    assert not cmj_as_sj.eligible and cmj_as_sj.device_verdict != "reject"
    with pytest.raises(ValueError, match="ineligible"):
        jumps.force_platform_estimate(cmj_as_sj)
    assert jumps.force_platform_estimate(cmj_as_sj, allow_diagnostic=True).diagnostic_only


def test_rejected_imu_contact_and_lpt_sources_are_refused_unless_diagnostic() -> None:
    class Rejected:
        verdict = "reject"
        flights = ()

    with pytest.raises(ValueError, match="rejected by QC"):
        jumps.imu_estimates(Rejected())
    with pytest.raises(ValueError, match="rejected by QC"):
        jumps.contact_system_estimates(Rejected())
    assert jumps.imu_estimates(Rejected(), allow_diagnostic=True) == []
    report = _lpt_report(cmj())
    assert jumps.lpt_estimates(report).qc_verdict == "processed"
    with pytest.raises(ValueError, match="rejected by QC"):
        jumps.lpt_estimates({**report, "verdict": "reject"})
    assert jumps.lpt_estimates(
        {**report, "verdict": "reject"}, allow_diagnostic=True
    ).diagnostic_only


def test_exported_estimate_carries_qc_basis_and_uncertainty_assumptions() -> None:
    est = jumps.contact_system_estimates(_contact_result())[0]
    assert est.basis.startswith("trigger:") and est.uncertainty_assumptions
    assert "qc_verdict" in dataclasses.asdict(est)


def _rj_signal(n: int):  # type: ignore[no-untyped-def]
    observation = export_case("force_platform.single", hops(n), hz=HZ).observation
    force = observation["fz"]
    data = {**observation.data, "fz": force + (force > 0) * np.sin(np.arange(len(force))) * 0.1}
    result = process(
        SignalBundle(observation.meta, data),
        plates=PLATE,
        weighing=export_case("force_platform.single", sj(), hz=HZ).observation,
        method=__import__("dynamisbench_soccer.engines.force_platform", fromlist=["x"]).MethodSpec(
            integration_start="record_start", initial_velocity_m_s=0.0
        ),
    )
    return jumps.JumpSignal.from_engine(result)


def test_ten_five_selection_is_exact_and_names_both_variants() -> None:
    result = assess(_rj_signal(10), JumpSpec(protocol="repeated_hops"))
    assert result.eligible
    summary = jumps.rj_10_5_selection(result)
    assert summary["n_hops"] == 10 and summary["variants_are_not_interchangeable"]
    best = summary["best_5_of_10_mean"]
    assert len(best["selected_hop_indices"]) == 5  # type: ignore[index]
    assert best["rsi_m_s"] == pytest.approx(summary["all_hops_mean_rsi_m_s"], rel=1e-3)  # type: ignore[index]
    assert summary["peak_single_rsi"]["rsi_m_s"] >= best["rsi_m_s"]  # type: ignore[index]


@pytest.mark.parametrize("n", [5, 9])
def test_ten_five_refuses_incomplete_sets_instead_of_padding(n: int) -> None:
    result = assess(_rj_signal(n), JumpSpec(protocol="repeated_hops"))
    with pytest.raises(ValueError, match="exactly 10"):
        jumps.rj_10_5_selection(result)


def test_ten_five_refuses_other_protocols_and_rejected_sources() -> None:
    with pytest.raises(ValueError, match="repeated_hops"):
        jumps.rj_10_5_selection(assess(_accepted_signal(cmj()), JumpSpec(protocol="cmj")))
    rejected = assess(
        _signal(hops(10), True, integration_start="record_start", initial_velocity_m_s=0.0),
        JumpSpec(protocol="repeated_hops"),
    )
    # the unrippled fixture trips the clipping rule: a rejected source never gives a summary
    assert rejected.device_verdict == "reject"
    with pytest.raises(ValueError, match="no accepted 10/5"):
        jumps.rj_10_5_selection(rejected)


def test_a_stripped_diagnostic_flag_cannot_bypass_the_comparison_gate() -> None:
    accepted = jumps.force_platform_estimate(
        assess(_accepted_signal(cmj()), JumpSpec(protocol="cmj"))
    )
    contact = jumps.contact_system_estimates(_contact_result())[0]
    stripped = dataclasses.replace(accepted, qc_verdict="reject", diagnostic_only=False)
    with pytest.raises(ValueError, match="QC-rejected"):
        jumps.compare_sources([stripped, contact])
    ineligible = dataclasses.replace(accepted, protocol_eligible=False)
    with pytest.raises(ValueError, match="protocol-ineligible"):
        jumps.compare_sources([ineligible, contact])
    assert jumps.compare_sources([accepted, contact])["merged_value"] is None


def test_ten_five_selection_picks_the_best_five_and_reports_flight_time_basis() -> None:
    result = assess(
        _rj_signal(10), JumpSpec(protocol="repeated_hops", rsi_height_basis="impulse_momentum")
    )
    rsi = [0.30, 0.55, 0.20, 0.60, 0.25, 0.58, 0.35, 0.57, 0.22, 0.56]
    varied = dataclasses.replace(
        result, contacts=[{**c, "rsi_m_s": r} for c, r in zip(result.contacts, rsi, strict=True)]
    )
    summary = jumps.rj_10_5_selection(varied)
    assert summary["best_5_of_10_mean"]["selected_hop_indices"] == [1, 3, 5, 7, 9]  # type: ignore[index]
    assert summary["best_5_of_10_mean"]["rsi_m_s"] == pytest.approx(
        np.mean([0.55, 0.60, 0.58, 0.57, 0.56])
    )  # type: ignore[index]
    assert summary["peak_single_rsi"]["selected_hop_indices"] == [3]  # type: ignore[index]
    assert summary["rsi_height_basis"] == "flight_time"
