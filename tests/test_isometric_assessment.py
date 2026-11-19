"""Isometric protocol assessment versus the RES-669 piecewise-linear force ramp.

With baseline b, peak P, rise time R: RFD(0-w) = (P - b)/R, net impulse(0-w) = (P - b) w^2/(2R)
for w <= R, and time to peak = R. These are derived here; the assessment layer never
imports the substrate.
"""

import ast
import json
from pathlib import Path

import numpy as np
import pytest

from dynamisbench_soccer.assessment import isometric
from dynamisbench_soccer.assessment.isometric import (
    LIMITATIONS,
    IsoSpec,
    NordicSpec,
    assess,
    assess_nordic,
    run_trials,
)
from dynamisbench_soccer.cli import main
from dynamisbench_soccer.engines.load_cell import MethodSpec
from dynamisbench_soccer.engines.load_cell_acquisition import CellSpec
from dynamisbench_soccer.engines.nordic_hip import NordicSpec as NordicEngineSpec
from dynamisbench_soccer.engines.nordic_hip import SideDeclaration
from dynamisbench_soccer.engines.nordic_hip import process as nordic_process
from dynamisbench_soccer.reference.exporters import export_case
from dynamisbench_soccer.reference.faults import ContinuousFault, ContinuousKind, inject
from dynamisbench_soccer.reference.motion_models import IsometricRamp
from dynamisbench_soccer.signals import io
from dynamisbench_soccer.signals.schema import SignalBundle

RISE = 0.4
CELL = CellSpec(lever_arm_m=0.4, moment_arm_definition="joint centre to cell attachment")


def _bundle(
    peak: float = 2500.0, baseline: float = 0.0, knee: float | None = 130.0
) -> SignalBundle:
    case = export_case(
        "load_cell_isometric.load_cell", IsometricRamp(baseline_n=baseline, peak_n=peak), hz=1000
    ).observation
    geometry = case.meta.geometry.model_copy(update={"knee_angle_deg": knee})
    return SignalBundle(case.meta.model_copy(update={"geometry": geometry}), case.data)


def _methods(**kw: object) -> MethodSpec:
    return MethodSpec(windows_ms=(50, 100, 150, 200, 250), **kw)  # type: ignore[arg-type]


def _run(bundles: list[SignalBundle], spec: IsoSpec, **kw: object):
    method = _methods()
    results = run_trials(bundles, CELL, method)
    return assess(
        results,
        spec,
        geometry=bundles[0].meta.geometry,
        bundles=bundles,
        cell=CELL,
        method=method,
        **kw,  # type: ignore[arg-type]
    )


def test_imtp_quantities_match_ramp_closed_forms() -> None:
    result = _run(
        [_bundle(), _bundle()],
        IsoSpec(protocol="imtp", angle_range_deg=(125, 145), body_mass_kg=80),
    )
    assert result.eligible
    s = result.summary
    assert s["peak_force_n"] == pytest.approx(2500, abs=1e-6)
    assert s["time_to_peak_s"] == pytest.approx(RISE, abs=2e-3)
    assert s["rfd_headline_n_s"] == pytest.approx(2500 / RISE, rel=1e-6)
    assert s["impulse_headline_n_s"] == pytest.approx(2500 * 0.2**2 / (2 * RISE), rel=1e-6)
    assert s["peak_force_n_per_kg"] == pytest.approx(2500 / 80)
    trial = result.trials[0]
    for w in (50, 100, 150, 200, 250):
        assert trial["rfd_windows_n_s"][f"rfd_0_{w}ms_n_s"] == pytest.approx(2500 / RISE, rel=1e-6)
        assert trial["net_impulse_windows_n_s"][f"net_impulse_0_{w}ms_n_s"] == pytest.approx(
            2500 * (w / 1000) ** 2 / (2 * RISE), rel=1e-6
        )
    assert trial["peak_torque_n_m"] == pytest.approx(1000)


def test_baseline_offset_is_removed_from_net_quantities() -> None:
    result = _run([_bundle(peak=2600, baseline=100)], IsoSpec(protocol="imtp", min_trials=1))
    t = result.trials[0]
    assert t["peak_force_n"] == pytest.approx(2600)
    assert t["net_peak_force_n"] == pytest.approx(2500)
    assert t["rfd_headline_n_s"] == pytest.approx(2500 / RISE, rel=1e-6)


def test_repetition_policies_and_consistency() -> None:
    trials = [_bundle(2500), _bundle(2400), _bundle(2450)]
    expected = {"best_peak": 2500, "mean_peak": 2450, "first_valid": 2500, "all_valid": 2450}
    for policy, value in expected.items():
        result = _run(trials, IsoSpec(protocol="imtp", policy=policy))  # type: ignore[arg-type]
        assert result.summary["peak_force_n"] == pytest.approx(value, abs=1e-6), policy
    spread = _run(trials, IsoSpec(protocol="imtp")).consistency
    assert spread["peak_spread_fraction_of_best"] == pytest.approx(100 / 2500)
    assert spread["consistent"] is True
    loose = _run([_bundle(2500), _bundle(2200)], IsoSpec(protocol="imtp"))
    assert loose.consistency["consistent"] is False and not loose.eligible
    only_second_best = _run(
        [_bundle(2400), _bundle(2500)], IsoSpec(protocol="imtp", policy="first_valid")
    )
    assert only_second_best.summary["peak_force_n"] == pytest.approx(2400)


def test_protocol_geometry_eligibility_is_declared_not_assumed() -> None:
    inside = _run(
        [_bundle(knee=130)], IsoSpec(protocol="imtp", min_trials=1, angle_range_deg=(125, 145))
    )
    outside = _run(
        [_bundle(knee=100)], IsoSpec(protocol="imtp", min_trials=1, angle_range_deg=(125, 145))
    )
    unknown = _run([_bundle(knee=None)], IsoSpec(protocol="imtp", min_trials=1))
    assert inside.eligible
    assert not outside.eligible
    assert any(
        c["check"] == "knee_angle_deg_in_protocol_range" and not c["passed"]
        for c in outside.eligibility
    )
    assert not unknown.eligible
    assert any(
        c["check"] == "knee_angle_deg_declared" and not c["passed"] for c in unknown.eligibility
    )
    hip = _run([_bundle(knee=None)], IsoSpec(protocol="hip_adduction", min_trials=1))
    assert hip.eligible  # the export declares hip flexion 45 deg
    assert hip.protocol == "hip_adduction"


def test_onset_variants_are_reported_and_headline_uses_only_the_declared_one() -> None:
    result = _run([_bundle()], IsoSpec(protocol="imtp", min_trials=1))
    assert set(result.onset_variants) == set(isometric.ONSET_METHODS)
    for name, variant in result.onset_variants.items():
        # The 20 N absolute threshold is crossed 20 / 6250 s after the true start (0.5 s).
        assert 0.45 <= variant["onset_s"] <= 0.5 + 20 / 6250 + 1e-3, name  # type: ignore[operator]
        assert variant["rfd_headline_n_s"] == pytest.approx(2500 / RISE, rel=0.2), name
    assert result.trials[0]["onset_method"] == "5sd"


def test_trials_that_peak_at_the_record_edge_or_start_loaded_are_invalid() -> None:
    full = _bundle()
    cut = 950
    truncated = SignalBundle(
        full.meta.model_copy(update={"n_rows": cut}), {k: v[:cut] for k, v in full.data.items()}
    )
    edge = _run([truncated], IsoSpec(protocol="imtp", min_trials=1, require_peak_before_end_s=0.1))
    assert edge.trials[0]["peak_at_record_end"] is True and not edge.eligible
    assert "no valid trial" in edge.refusals["summary"]
    loaded = _run(
        [_bundle(peak=2600, baseline=300)],
        IsoSpec(protocol="imtp", min_trials=1, max_baseline_n=100),
    )
    assert not loaded.eligible and loaded.trials[0]["valid"] is False


def test_missing_body_mass_refuses_relative_force() -> None:
    result = _run([_bundle()], IsoSpec(protocol="imtp", min_trials=1))
    assert "relative_force" in result.refusals and "peak_force_n_per_kg" not in result.summary


def test_calibration_gain_sensitivity_scales_linearly() -> None:
    result = _run(
        [_bundle()], IsoSpec(protocol="imtp", min_trials=1, calibration_gain_errors=(-0.02, 0.03))
    )
    sens = result.uncertainty["calibration_gain_sensitivity"]
    assert sens["-0.020"]["peak_force_n"] == pytest.approx(2500 * 0.98)  # type: ignore[index]
    assert sens["+0.030"]["rfd_headline_n_s"] == pytest.approx(2500 / RISE * 1.03)  # type: ignore[index]
    assert "rig_compliance" in result.uncertainty


def test_corrupted_trials_do_not_become_valid_evidence() -> None:
    case = export_case("load_cell_isometric.load_cell", hz=1000).observation
    geometry = case.meta.geometry.model_copy(update={"knee_angle_deg": 130.0})
    base = SignalBundle(case.meta.model_copy(update={"geometry": geometry}), case.data)
    clipped = inject(base, ContinuousFault(ContinuousKind.CLIPPING, ("voltage",), 2.0))
    result = assess(
        [run_trials([clipped], CellSpec(raw_range=(-2.0, 2.0)), _methods())[0]],
        IsoSpec(protocol="imtp", min_trials=1),
        geometry=geometry,
    )
    assert result.trials[0]["verdict"] == "reject" and result.trials[0]["valid"] is False
    assert not result.eligible and "no valid trial" in result.refusals["summary"]


def _nordic(peak_scale: float, basis: str = "raw_calibrated"):
    from dynamisbench_soccer.reference.motion_models import IsometricRamp as _  # noqa: F401

    case = export_case("nordic_hip.nordic", hz=1000).observation
    data = {k: (v * peak_scale if k.startswith("force") else v) for k, v in case.data.items()}
    spec = NordicEngineSpec(
        protocol="nordic_eccentric",
        force_basis=basis,  # type: ignore[arg-type]
        sides=(
            SideDeclaration(channel="force_left", side="left"),
            SideDeclaration(channel="force_right", side="right"),
        ),
        lever_arm_m=0.4,
        moment_arm_definition="knee joint centre to strap",
    )
    return nordic_process(SignalBundle(case.meta, data), spec)


def test_nordic_protocol_policy_and_eligibility() -> None:
    results = [_nordic(1.0), _nordic(0.9)]
    best = assess_nordic(results, NordicSpec(policy="best_peak", min_trials=2))
    assert best["eligible"] is True
    assert best["summary"]["peak_force_n_left"] == pytest.approx(160, abs=0.1)  # type: ignore[index]
    assert best["summary"]["peak_force_n_right"] == pytest.approx(240, abs=0.1)  # type: ignore[index]
    mean = assess_nordic(results, NordicSpec(policy="mean_peak", min_trials=2))
    assert mean["summary"]["peak_force_n_left"] == pytest.approx(152, abs=0.1)  # type: ignore[index]
    inconsistent = assess_nordic(results, NordicSpec(min_trials=2, consistency_limit_fraction=0.05))
    assert inconsistent["eligible"] is False
    needs_break = assess_nordic(results, NordicSpec(require_break_point=True))
    assert needs_break["eligible"] is False  # no measured angle: no break point
    with pytest.raises(ValueError, match="cannot be combined"):
        assess_nordic([_nordic(1.0), _nordic(1.0, "vendor_corrected")], NordicSpec())
    with pytest.raises(ValueError, match="at least one"):
        assess_nordic([], NordicSpec())


def test_limitations_state_no_muscle_force_or_interchangeability() -> None:
    text = " ".join(LIMITATIONS)
    assert "not muscle force" in text and "interchangeability" in text
    tree = ast.parse(Path(isometric.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or "").startswith("dynamisbench_soccer.reference")


def test_cli_end_to_end(tmp_path: Path) -> None:
    stems = []
    for i, peak in enumerate((2500.0, 2450.0)):
        stem = tmp_path / f"trial{i}"
        io.write(_bundle(peak), stem)
        stems.append(str(stem))
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "isometric": {"protocol": "imtp", "angle_range_deg": [125, 145]},
                "method": {"windows_ms": [50, 100, 200]},
                "cell": json.loads(CELL.model_dump_json()),
            }
        )
    )
    out = tmp_path / "report.json"
    assert main(["assess", "isometric", *stems, "--config", str(config), "--out", str(out)]) == 0
    report = json.loads(out.read_text())
    assert report["protocol"] == "imtp" and report["summary"]["peak_force_n"] == pytest.approx(2500)
    assert (
        main(["assess", "isometric", *stems, "--config", str(config), "--out", stems[0] + ".json"])
        == 2
    )  # the sidecar of an input is protected
    assert main(["assess", "isometric", *stems, "--config", str(config), "--out", str(config)]) == 2
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"isometric": {"protocol": "nope"}}))
    assert main(["assess", "isometric", *stems, "--config", str(bad), "--out", str(out)]) == 2


def test_numpy_float_summary_is_json_safe() -> None:
    result = _run([_bundle()], IsoSpec(protocol="imtp", min_trials=1))
    json.loads(result.report_json())
    assert isinstance(np.float64(1.0), float)
