from pathlib import Path

import pytest
from pydantic import ValidationError

from dynamisbench_soccer.qc import spec
from dynamisbench_soccer.qc.checks import EVALUATORS
from dynamisbench_soccer.qc.faults import FIXTURES
from dynamisbench_soccer.registry.models import load as load_registry
from dynamisbench_soccer.signals import examples


@pytest.fixture(scope="module")
def rules(repo_root_module: Path) -> spec.RuleSet:
    return spec.load(repo_root_module / "qc" / "rules.toml")


@pytest.fixture(scope="module")
def repo_root_module() -> Path:
    return Path(__file__).resolve().parents[1]


def test_rules_valid_against_evaluators_and_registry(
    rules: spec.RuleSet, repo_root_module: Path
) -> None:
    assert spec.validate(rules, load_registry(repo_root_module / "registry")) == []


def test_every_rule_has_positive_and_negative_fixture(rules: spec.RuleSet) -> None:
    assert set(FIXTURES) == {r.id for r in rules.rule} == set(EVALUATORS)


@pytest.mark.parametrize("rule_id", sorted(FIXTURES))
def test_negative_fixture_passes(rule_id: str, rules: spec.RuleSet) -> None:
    rule, fx = rules.by_id(rule_id), FIXTURES[rule_id]
    ref = fx.reference() if fx.reference else None
    bundle = fx.negative()
    assert bundle.meta.test_id in rule.applies_to
    outcome = EVALUATORS[rule_id](bundle, rule.params, ref)
    assert outcome.status is spec.Status.PASS, outcome.detail


@pytest.mark.parametrize("rule_id", sorted(FIXTURES))
def test_positive_fixture_fails(rule_id: str, rules: spec.RuleSet) -> None:
    rule, fx = rules.by_id(rule_id), FIXTURES[rule_id]
    ref = fx.reference() if fx.reference else None
    bundle = fx.positive()
    assert bundle.meta.test_id in rule.applies_to
    outcome = EVALUATORS[rule_id](bundle, rule.params, ref)
    assert outcome.status is spec.Status.FAIL
    assert outcome.detail


@pytest.mark.parametrize(
    "name", ["cmj_dual_plate", "nhe", "hip_squeeze", "sprint_gates", "sprint_radar"]
)
def test_clean_examples_are_accepted(name: str, rules: spec.RuleSet) -> None:
    results = spec.run(examples.ALL[name](), rules)
    assert results, "at least one rule applies"
    failed = [(r.rule_id, r.detail) for r in results if r.status is spec.Status.FAIL]
    assert failed == []
    assert spec.verdict(results) == "accept"


def test_verdict_levels(rules: spec.RuleSet) -> None:
    from dynamisbench_soccer.qc import faults

    assert spec.verdict(spec.run(faults.double_contact(), rules)) == "reject"
    flagged = spec.run(faults.mains(faults.cmj()), rules)
    assert spec.verdict(flagged) == "accept_with_flags"
    # an unzeroed plate (+25 N/plate) is flagged by QC-S01 AND hides the flight phase
    # from the 30 N take-off threshold, so QC-T03 rejects the trial: faults cascade.
    cascade = spec.run(faults.offset(faults.cmj(), 25.0), rules)
    failed = {r.rule_id for r in cascade if r.status is spec.Status.FAIL}
    assert failed == {"QC-S01", "QC-T03"}
    assert spec.verdict(cascade) == "reject"


def test_parameter_overrides(rules: spec.RuleSet) -> None:
    from dynamisbench_soccer.qc import faults

    bundle = faults.offset(faults.cmj(), 25.0)
    results = spec.run(bundle, rules, overrides={"QC-S01": {"max_offset_n": 100.0}})
    s01 = next(r for r in results if r.rule_id == "QC-S01")
    assert s01.status is spec.Status.PASS


def test_evidence_rules_require_claims() -> None:
    with pytest.raises(ValidationError, match="must cite"):
        spec.Rule(
            id="QC-X01",
            name="x",
            level="signal",
            applies_to=["cmj"],
            severity="flag",
            basis="evidence",
            claims=[],
            rationale="a rationale that is long enough",
        )
