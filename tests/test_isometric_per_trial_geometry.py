"""RES-672: geometry and onset sensitivity are bound to each individual trial."""

import pytest
from test_isometric_assessment import CELL, _bundle, _methods

from dynamisbench_soccer.assessment.isometric import IsoSpec, assess, run_trials

RANGE = (125.0, 145.0)


def _run(bundles, **spec):
    method = _methods()
    results = run_trials(bundles, CELL, method)
    return assess(
        results,
        IsoSpec(protocol="imtp", angle_range_deg=RANGE, min_trials=1, **spec),
        bundles=bundles,
        cell=CELL,
        method=method,
    )


def test_first_valid_second_invalid_geometry_excludes_the_second_trial() -> None:
    r = _run([_bundle(2400, knee=130), _bundle(2600, knee=90)])
    assert [t["valid"] for t in r.trials] == [True, False]
    assert r.trials[1]["diagnostic_only"] and r.trials[1]["geometry_exclusions"]
    assert r.summary["peak_force_n"] == pytest.approx(2400)  # best peak belongs to invalid trial
    assert r.trials[1]["peak_force_n"] == pytest.approx(2600)  # preserved as diagnostic


def test_first_invalid_second_valid_uses_only_the_valid_trial() -> None:
    r = _run([_bundle(2600, knee=90), _bundle(2400, knee=130)])
    assert [t["valid"] for t in r.trials] == [False, True]
    assert r.summary["peak_force_n"] == pytest.approx(2400)
    assert r.eligible


def test_missing_geometry_in_a_later_trial_is_refused_for_that_trial() -> None:
    r = _run([_bundle(2400, knee=130), _bundle(2600, knee=None)])
    assert r.trials[1]["valid"] is False
    assert any("knee_angle_deg_declared" in x for x in r.trials[1]["geometry_exclusions"])
    assert r.summary["peak_force_n"] == pytest.approx(2400)


def test_all_trials_invalid_geometry_is_ineligible_with_negative_control() -> None:
    bad = _run([_bundle(knee=90), _bundle(knee=95)])
    assert not bad.eligible and "summary" in bad.refusals
    good = _run([_bundle(knee=130), _bundle(knee=131)])
    assert good.eligible


def test_pooling_inconsistent_in_range_angles_is_refused() -> None:
    wide = _run([_bundle(2400, knee=126), _bundle(2450, knee=144)], policy="all_valid")
    assert not wide.eligible
    assert any(
        c["check"] == "knee_angle_deg_consistent_across_pooled_trials" and not c["passed"]
        for c in wide.eligibility
    )
    tight = _run([_bundle(2400, knee=130), _bundle(2450, knee=130.5)], policy="all_valid")
    assert tight.eligible


@pytest.mark.parametrize("policy", ["first_valid", "best_peak", "all_valid", "mean_peak"])
def test_policies_select_only_among_eligible_trials(policy: str) -> None:
    r = _run(
        [_bundle(2900, knee=60), _bundle(2400, knee=130), _bundle(2500, knee=131)], policy=policy
    )
    assert r.trials[0]["valid"] is False
    expected = {"first_valid": 2400, "best_peak": 2500, "all_valid": 2450, "mean_peak": 2450}
    assert r.summary["peak_force_n"] == pytest.approx(expected[policy], abs=1e-6)


def test_onset_sensitivity_is_bound_to_the_selected_trial() -> None:
    # Trial 0 is invalid (geometry); trial 2 has the best valid peak. Sensitivity must use trial 2.
    r = _run([_bundle(2000, knee=60), _bundle(2400, knee=130), _bundle(2600, knee=130)])
    assert r.onset_variants
    for name, v in r.onset_variants.items():
        assert v["source_trial_indices"] == [2], name
    first_valid = _run(
        [_bundle(2000, knee=60), _bundle(2400, knee=130), _bundle(2600, knee=130)],
        policy="first_valid",
    )
    assert all(v["source_trial_indices"] == [1] for v in first_valid.onset_variants.values())
    varied = _run([_bundle(2400, knee=130), _bundle(2600, knee=130)], policy="all_valid")
    assert all(v["source_trial_indices"] == [0, 1] for v in varied.onset_variants.values())


def test_sensitivity_values_follow_the_selected_trial_not_trial_zero() -> None:
    r = _run([_bundle(1000, knee=130), _bundle(3000, knee=130)])  # best = trial 1
    # RFD scales with peak: the sensitivity must reflect 3000 N, not trial zero's 1000 N.
    peak_rfd = r.trials[1]["rfd_headline_n_s"]
    assert any(
        v["rfd_headline_n_s"] == pytest.approx(peak_rfd, rel=0.25)
        for v in r.onset_variants.values()
        if v.get("rfd_headline_n_s") is not None
    )
    assert all(
        v["rfd_headline_n_s"] > 1.5 * r.trials[0]["rfd_headline_n_s"]
        for v in r.onset_variants.values()
        if v.get("rfd_headline_n_s") is not None
    )


def test_lever_arm_conflict_between_trial_geometry_and_cell_excludes_trial() -> None:
    b = _bundle(2400, knee=130)
    geo = b.meta.geometry.model_copy(update={"lever_arm_m": 0.55})
    conflicted = type(b)(b.meta.model_copy(update={"geometry": geo}), b.data)
    r = _run([_bundle(2400, knee=130), conflicted])
    assert r.trials[1]["valid"] is False
    assert any("lever_arm_conflict" in x for x in r.trials[1]["geometry_exclusions"])


def test_mismatched_bundle_count_is_refused() -> None:
    method = _methods()
    results = run_trials([_bundle(), _bundle()], CELL, method)
    with pytest.raises(ValueError, match="one-to-one"):
        assess(results, IsoSpec(protocol="imtp"), bundles=[_bundle()], cell=CELL, method=method)


def test_cli_binds_geometry_per_trial_and_reports_the_diagnostic_trial(tmp_path) -> None:
    import json

    from test_isometric_assessment import CELL as CELL_

    from dynamisbench_soccer.cli import main
    from dynamisbench_soccer.signals import io

    stems = []
    for i, (peak, knee) in enumerate(((2400.0, 130.0), (2900.0, 70.0))):
        stem = tmp_path / f"trial{i}"
        io.write(_bundle(peak, knee=knee), stem)
        stems.append(str(stem))
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "isometric": {"protocol": "imtp", "angle_range_deg": [125, 145], "min_trials": 1},
                "method": {"windows_ms": [50, 100, 200]},
                "cell": json.loads(CELL_.model_dump_json()),
            }
        )
    )
    out = tmp_path / "report.json"
    assert main(["assess", "isometric", *stems, "--config", str(config), "--out", str(out)]) == 0
    report = json.loads(out.read_text())
    assert report["summary"]["peak_force_n"] == pytest.approx(2400)
    assert report["trials"][1]["diagnostic_only"] is True
    assert all(v["source_trial_indices"] == [0] for v in report["onset_variants"].values())
