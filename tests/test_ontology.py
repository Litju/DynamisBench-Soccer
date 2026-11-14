from pathlib import Path

import pytest

from dynamisbench_soccer.cli import main
from dynamisbench_soccer.ontology.boundaries import screen, screenable
from dynamisbench_soccer.ontology.models import (
    COMPETENCIES,
    BatteryTest,
    Status,
    load,
    validate,
)
from dynamisbench_soccer.registry.models import V01_TESTS
from dynamisbench_soccer.registry.models import load as load_registry


@pytest.fixture
def bundle(repo_root: Path):  # type: ignore[no-untyped-def]
    return load(repo_root / "ontology")


def test_ontology_is_consistent_with_registry(bundle, repo_root: Path) -> None:  # type: ignore[no-untyped-def]
    assert validate(bundle, load_registry(repo_root / "registry")) == []


def test_tests_equal_registry_ids(bundle) -> None:  # type: ignore[no-untyped-def]
    assert tuple(t.id for t in bundle.ontology.test) == V01_TESTS


def test_registry_gap_has_no_core_cells(bundle, repo_root: Path) -> None:  # type: ignore[no-untyped-def]
    gaps = {g.subject for g in load_registry(repo_root / "registry").gaps}
    for test in gaps:
        row = bundle.coverage.row_for(test)
        assert all(row.cell(c) is not Status.CORE for c in COMPETENCIES)


def test_validation_catches_core_cell_on_gap(bundle, repo_root: Path) -> None:  # type: ignore[no-untyped-def]
    row = bundle.coverage.row_for("iso_ham")
    row.C3 = Status.CORE
    problems = validate(bundle, load_registry(repo_root / "registry"))
    assert any("iso_ham is a registry gap" in p for p in problems)


def test_estimand_naming_rules() -> None:
    with pytest.raises(ValueError):
        BatteryTest(id="x", name="x", device_class="d", estimands=["Jump Height"])
    with pytest.raises(ValueError):
        BatteryTest(id="x", name="x", device_class="d", estimands=["a.b", "a.b"])


def test_two_jump_height_estimands_are_distinct(bundle) -> None:  # type: ignore[no-untyped-def]
    cmj = next(t for t in bundle.ontology.test if t.id == "cmj")
    assert {"jump_height.takeoff_velocity", "jump_height.flight_time"} <= set(cmj.estimands)


def test_epistemic_classes_present(bundle) -> None:  # type: ignore[no-untyped-def]
    ids = {c.id for c in bundle.ontology.epistemic_class}
    assert ids == {"measured", "derived", "inference", "unsupported_explanation"}


def test_boundary_fixtures_classified(bundle) -> None:  # type: ignore[no-untyped-def]
    rules = bundle.boundaries
    for rule in screenable(rules):
        for text in rule.violating:
            assert any(h.rule_id == rule.id for h in screen(text, rules)), (rule.id, text)
    # no compliant example of ANY rule may trigger ANY reference pattern
    for rule in rules.rule:
        for text in rule.compliant:
            assert screen(text, rules) == [], (rule.id, text)


def test_clearance_rule_is_mandatory_fail(bundle) -> None:  # type: ignore[no-untyped-def]
    rule = next(r for r in bundle.boundaries.rule if r.id == "DB-01")
    assert rule.severity == "fail"


def test_cli_screen_exit_codes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    bad = tmp_path / "bad.txt"
    bad.write_text("Great numbers. The player is cleared to return to play.")
    good = tmp_path / "good.txt"
    good.write_text("NHE force is 94% of baseline; the medical team decides on return to play.")
    assert main(["ontology", "screen", "--file", str(bad)]) == 1
    assert main(["ontology", "screen", "--file", str(good)]) == 0
    assert "DB-01" in capsys.readouterr().out
