"""End-to-end qualification verification (RES-675).

Runs the executable matrix for all eight device families, recomputes the per-subfamily states
and the M3 gate from the ledger (and proves they are computed, not declared), checks that every
existing CLI command family still answers, and holds the frozen engine-v0.1 interface.
"""

import ast
import dataclasses
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import e2e_cases
import pytest

from dynamisbench_soccer.cli import main
from dynamisbench_soccer.qualification import interface
from dynamisbench_soccer.qualification import ledger as ledger_model
from dynamisbench_soccer.qualification.ledger import Evidence, Rating
from dynamisbench_soccer.qualification.state import computed_states, m3_gate, subfamily_state

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ledger_model.load(ROOT / "qualification/ledger.toml")
FAMILIES = {
    "force_platform", "lpt_encoder", "load_cell_isometric", "nordic_hip",
    "timing_contact", "radar_laser", "gnss_lps", "imu",
}  # fmt: skip


@pytest.mark.parametrize("case", e2e_cases.CASES, ids=lambda c: f"{c.subfamily}:{c.name}")
def test_every_matrix_case_passes(case) -> None:
    outcome = case.run()
    assert outcome.passed, (outcome.observed, outcome.expected, outcome.tolerance, outcome.detail)


def test_matrix_covers_all_eight_families_and_fifteen_subfamilies() -> None:
    assert {c.family for c in e2e_cases.CASES} == FAMILIES
    assert {c.subfamily for c in e2e_cases.CASES} == set(ledger_model.SUBFAMILIES)
    assert len({(c.subfamily, c.name) for c in e2e_cases.CASES}) == len(e2e_cases.CASES)
    for sub in ledger_model.SUBFAMILIES:
        cases = [c for c in e2e_cases.CASES if c.subfamily == sub]
        assert len(cases) >= 2, sub


def test_every_family_exercises_corruption_refusal_and_a_processing_axis() -> None:
    for family in FAMILIES:
        axes = [c.axes for c in e2e_cases.CASES if c.family == family]
        seen = {a: {x[a] for x in axes if a in x} for a in e2e_cases.AXES}
        assert seen["corruption"] or seen["refusal"], family
        assert seen["refusal"] or seen["eligibility"], family
        assert seen["rate_hz"] or seen["event_method"] or seen["diff_integ"], family


def test_native_rates_filters_events_and_differentiation_are_swept_where_relevant() -> None:
    coverage = {f: {a: set() for a in e2e_cases.AXES} for f in FAMILIES}
    for c in e2e_cases.CASES:
        for axis, value in c.axes.items():
            coverage[c.family][axis].add(value)
    assert len(coverage["force_platform"]["rate_hz"]) >= 3
    assert len(coverage["force_platform"]["event_method"]) >= 4
    assert (
        len(coverage["force_platform"]["diff_integ"]) >= 2
        and len(coverage["force_platform"]["filter"]) >= 1
    )
    assert len(coverage["lpt_encoder"]["diff_integ"]) >= 3
    assert len(coverage["load_cell_isometric"]["event_method"]) >= 4
    assert (
        len(coverage["radar_laser"]["rate_hz"]) >= 3
        and len(coverage["radar_laser"]["diff_integ"]) >= 2
    )
    assert len(coverage["gnss_lps"]["diff_integ"]) >= 2
    assert len(coverage["imu"]["rate_hz"]) >= 3
    assert len(coverage["nordic_hip"]["rate_hz"]) >= 3


def test_states_are_computed_from_evidence_not_declared() -> None:
    states = {s.subfamily: s for s in computed_states(LEDGER)}
    assert set(states) == set(ledger_model.SUBFAMILIES)
    gate = m3_gate(LEDGER, ROOT)
    assert gate.passed is False and gate.structural_problems == ()
    assert gate.engineering_verified == 0
    assert all(s.state in {"SPECIFIED", "IN_PROGRESS", "NOT_STARTED"} for s in states.values())
    target = next(s for s in LEDGER.subfamilies if s.id == "force_platform.single")
    verified = {d: Rating.VERIFIED for d in ledger_model.DIMENSIONS}
    j = Evidence(
        target.id,
        "J",
        "analytical_reference",
        "repo:README.md",
        "RES-669",
        "x",
        independence_from=target.engine_id,
        implementation="independent closed form",
    )

    def state_for(**changes) -> str:
        sub = dataclasses.replace(target, **changes)
        model = dataclasses.replace(LEDGER, evidence=(*LEDGER.evidence, j))
        return subfamily_state(model, sub).state

    assert state_for() == "SPECIFIED"
    assert state_for(ratings=verified, blockers=("open",)) == "SPECIFIED"
    assert state_for(ratings=verified, blockers=()) == "ENGINEERING_VERIFIED"
    no_n = {**verified, "N": Rating.ABSENT}
    assert (
        state_for(ratings=no_n, blockers=()) == "ENGINEERING_VERIFIED"
    )  # N is not an M3 dimension
    one_missing = {**verified, "L": Rating.PARTIAL}
    assert state_for(ratings=one_missing, blockers=()) == "SPECIFIED"
    absent = dict.fromkeys(ledger_model.DIMENSIONS, Rating.ABSENT)
    assert state_for(ratings=absent, blockers=()) == "NOT_STARTED"
    no_reference = dataclasses.replace(
        LEDGER,
        evidence=tuple(
            e for e in LEDGER.evidence if not (e.subfamily == target.id and e.dimension == "J")
        ),
    )
    unreferenced = dataclasses.replace(target, ratings=verified, blockers=())
    assert (
        subfamily_state(no_reference, unreferenced).state == "SPECIFIED"
    )  # no independent reference


def test_gate_passes_only_when_every_subfamily_is_engineering_verified() -> None:
    verified = {d: Rating.VERIFIED for d in ledger_model.DIMENSIONS}
    extra = tuple(
        Evidence(
            s.id,
            "J",
            "analytical_reference",
            "repo:README.md",
            "RES-669",
            "x",
            independence_from=s.engine_id,
            implementation="independent closed form",
        )
        for s in LEDGER.subfamilies
    )
    clean = tuple(dataclasses.replace(s, ratings=verified, blockers=()) for s in LEDGER.subfamilies)
    model = dataclasses.replace(LEDGER, subfamilies=clean, evidence=(*LEDGER.evidence, *extra))
    assert m3_gate(model).engineering_verified == 15
    one_blocked = (
        dataclasses.replace(clean[0], blockers=("needs measured calibration",)),
        *clean[1:],
    )
    assert m3_gate(dataclasses.replace(model, subfamilies=one_blocked)).engineering_verified == 14
    assert m3_gate(dataclasses.replace(model, subfamilies=one_blocked)).passed is False


def test_every_unverified_subfamily_names_its_owner_blocker_and_required_experiment() -> None:
    for s in computed_states(LEDGER):
        if s.state == "ENGINEERING_VERIFIED":
            continue
        assert s.blockers, s.subfamily
        assert set(s.required_experiments) == set(s.unverified_m3)
        assert all(s.owners[d].startswith("RES-") for d in s.unverified_m3)


def test_cli_states_reports_the_computed_gate(capsys: pytest.CaptureFixture[str]) -> None:
    code = main(["qualification", "states"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 1 and payload["m3_gate"]["passed"] is False
    assert len(payload["subfamilies"]) == 15


def test_frozen_interface_matches_the_live_interface() -> None:
    frozen = json.loads((ROOT / "docs/verification/engine-v0.1-interface.json").read_text())
    assert frozen == json.loads(interface.snapshot_json())
    assert frozen["interface_version"] == "engine-v0.1"
    assert main(["qualification", "interface"]) == 0


def test_every_existing_command_family_still_answers_help() -> None:
    frozen = json.loads((ROOT / "docs/verification/engine-v0.1-interface.json").read_text())
    required = {
        "adapters", "clean", "data", "env-check", "force-platform", "gnss-lps", "imu", "load-cell",
        "lpt-encoder", "nordic-hip", "ontology", "qc", "qualification", "radar-laser", "registry",
        "signals", "timing", "assess", "agreement",
    }  # fmt: skip
    tops = {path.split()[0] for path in frozen["cli"] if path != "dbs"}
    assert required <= tops, required - tops
    for path in sorted(frozen["cli"]):
        if path == "dbs":
            continue
        with pytest.raises(SystemExit) as exit_info:
            main([*path.split(), "--help"])
        assert exit_info.value.code == 0, path


def test_engines_never_import_assessment_agreement_or_reference() -> None:
    forbidden = (
        "dynamisbench_soccer.assessment",
        "dynamisbench_soccer.agreement",
        "dynamisbench_soccer.reference",
    )
    for path in (ROOT / "src/dynamisbench_soccer/engines").glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom):
                assert not (node.module or "").startswith(forbidden), path.name


def test_committed_verification_report_is_reproduced() -> None:
    spec = importlib.util.spec_from_file_location(
        "build_verification", ROOT / "scripts/build_verification.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_verification"] = module
    spec.loader.exec_module(module)
    payload, text = module.build()
    committed = json.loads((ROOT / "docs/verification/engine-v0.1.json").read_text())
    assert committed == json.loads(json.dumps(payload))
    assert (ROOT / "docs/verification/engine-v0.1.md").read_text() == text
    assert payload["case_totals"]["failed"] == []
    headings = (
        "M3 gate (computed)",
        "Blockers and the experiment",
        "Dimension ratings",
        "assessment capability",
        "numerical verification matrix",
        "Cross-device comparisons",
        "Frozen interface",
        "Reproduce",
    )
    for heading in headings:
        assert heading in text


def test_build_script_check_mode_passes_from_a_clean_process() -> None:
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts/build_verification.py"), "--check"],
        capture_output=True, text=True, cwd=ROOT, check=False,
    )  # fmt: skip
    assert result.returncode == 0, result.stderr
