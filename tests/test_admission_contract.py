"""Admission contract: evidence is required, the qualified scope never changes (RES-702)."""

import hashlib
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from dynamisbench_soccer.cli import main
from dynamisbench_soccer.qualification import admission
from dynamisbench_soccer.qualification import ledger as ledger_model
from dynamisbench_soccer.registry.models import load as load_registry

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/admission/candidate.toml"
REGISTRY = load_registry(ROOT / "registry")
CLAIMS = {c.id for c in REGISTRY.claims}
GAPS = {g.subject for g in REGISTRY.gaps}


def _base() -> dict:
    import tomllib

    return tomllib.loads(FIXTURE.read_text())


def _evaluate(**changes) -> admission.AdmissionResult:
    raw = {**_base(), **changes}
    return admission.evaluate(
        admission.Proposal.model_validate(raw),
        registry_claim_ids=CLAIMS,
        registry_gap_subjects=GAPS,
    )


def test_complete_proposal_is_admission_ready_but_confers_nothing() -> None:
    result = admission.evaluate(
        admission.load(FIXTURE), registry_claim_ids=CLAIMS, registry_gap_subjects=GAPS
    )
    assert result.decision == "ADMISSION_READY" and result.problems == ()
    assert result.initial_state == "NOT_STARTED" and result.scope_changed is False
    assert "confers none" in result.report()["qualification"]  # type: ignore[operator]


def test_admission_cannot_alter_the_qualified_scope(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(ROOT)
    before = hashlib.sha256((ROOT / "qualification/ledger.toml").read_bytes()).hexdigest()
    families, subfamilies = set(ledger_model.FAMILIES), set(ledger_model.SUBFAMILIES)
    assert main(["qualification", "admit", str(FIXTURE)]) == 0
    assert hashlib.sha256((ROOT / "qualification/ledger.toml").read_bytes()).hexdigest() == before
    assert set(ledger_model.FAMILIES) == families and len(families) == 8
    assert set(ledger_model.SUBFAMILIES) == subfamilies and len(subfamilies) == 15
    ledger = ledger_model.load(ROOT / "qualification/ledger.toml")
    assert ledger_model.validate(ledger, ROOT) == []
    smuggled = ledger_model.Ledger(
        ledger.version,
        (
            *ledger.subfamilies,
            ledger_model.Subfamily(
                "fixture_candidate.sensor",
                "fixture_candidate",
                "continuous_signal",
                "RES-702",
                "fixture_engine",
                (),
                dict.fromkeys(ledger_model.DIMENSIONS, ledger_model.Rating.VERIFIED),
                (),
                dict.fromkeys(ledger_model.DIMENSIONS, "RES-702"),
            ),
        ),
        ledger.evidence,
        ledger.comparisons,
        ledger.reviewers,
    )
    problems = ledger_model.validate(smuggled)
    assert any("eight required device families" in p for p in problems)
    assert any("all 15 required" in p for p in problems)
    assert any("rated without evidence" in p for p in problems)  # no rating bypasses evidence


@pytest.mark.parametrize(
    ("changes", "expected"),
    [
        (
            {"family": "force_platform", "subfamilies": ["force_platform.new"]},
            "existing qualified family",
        ),
        ({"family": "Bad-Name", "subfamilies": ["Bad-Name.x"]}, "snake_case"),
        ({"paradigm": "telepathy"}, "unknown acquisition paradigm"),
        ({"methods": ["orientation_estimation"]}, "not valid for the declared paradigm"),
        ({"paradigm": "discrete_event", "methods": ["debounce"], "engine_id": "x_engine"}, None),
        ({"registry_claims": [], "registry_gaps": []}, "registry-backed metrology evidence"),
        ({"registry_claims": ["NOPE-999"]}, "unknown registry claims"),
        ({"registry_gaps": ["not_a_subject"]}, "unknown registry gaps"),
        ({"references": []}, "independent reference plan"),
        ({"non_equivalences": []}, "non-equivalences"),
        ({"data_rights_statement": "ok"}, "data-rights statement"),
        ({"requested_ratings": {"A": "verified"}}, "cannot request ratings"),
    ],
)
def test_each_missing_or_invalid_requirement_is_rejected(
    changes: dict, expected: str | None
) -> None:
    result = _evaluate(**changes)
    if expected is None:
        assert result.decision == "ADMISSION_READY"  # a valid discrete-event method set is fine
        return
    assert result.decision == "REJECTED"
    assert any(expected in p for p in result.problems), result.problems


def test_discrete_event_families_cannot_use_continuous_filters() -> None:
    result = _evaluate(paradigm="discrete_event", methods=["butterworth"])
    assert result.decision == "REJECTED"
    assert any(
        "continuous" in p or "not valid for the declared paradigm" in p for p in result.problems
    )


def test_colliding_subfamily_and_owner_problems() -> None:
    clash = _evaluate(subfamilies=["fixture_candidate.sensor"], family="fixture_candidate")
    assert clash.decision == "ADMISSION_READY"
    raw = _base()
    raw["owners"] = {k: v for k, v in raw["owners"].items() if k != "G"}
    result = admission.evaluate(
        admission.Proposal.model_validate(raw),
        registry_claim_ids=CLAIMS,
        registry_gap_subjects=GAPS,
    )
    assert any("every dimension A-O" in p for p in result.problems)
    raw["owners"] = {**_base()["owners"], "G": "not-an-issue"}
    bad = admission.evaluate(
        admission.Proposal.model_validate(raw),
        registry_claim_ids=CLAIMS,
        registry_gap_subjects=GAPS,
    )
    assert any("RES-<n>" in p for p in bad.problems)


def test_references_must_be_independent_and_never_vendor() -> None:
    circular = {
        "dimension": "J",
        "kind": "analytical_reference",
        "implementation": "fixture_engine",
        "independent_of": "fixture_engine",
    }
    vendor = {
        "dimension": "J",
        "kind": "independent_test",
        "implementation": "vendor reported numbers",
        "independent_of": "fixture_engine",
    }
    wrong_dimension = {
        "dimension": "K",
        "kind": "analytical_reference",
        "implementation": "closed form",
        "independent_of": "fixture_engine",
    }
    assert any("computed by the engine" in p for p in _evaluate(references=[circular]).problems)
    assert any("vendor output" in p for p in _evaluate(references=[vendor]).problems)
    only_k = _evaluate(references=[wrong_dimension])
    assert any("dimension J" in p for p in only_k.problems)


def test_non_equivalence_must_name_a_qualified_subfamily() -> None:
    bad = {
        "against": "nonexistent.thing",
        "quantity": "force",
        "study_plan": "a sufficiently long study plan description",
    }
    result = _evaluate(non_equivalences=[bad])
    assert any("unknown subfamily" in p for p in result.problems)


def test_scope_impact_and_subfamily_prefix_are_enforced_by_the_schema() -> None:
    with pytest.raises(ValidationError):
        admission.Proposal.model_validate({**_base(), "scope_impact": "expand"})
    with pytest.raises(ValidationError, match="must start with"):
        admission.Proposal.model_validate({**_base(), "subfamilies": ["other.sensor"]})
    with pytest.raises(ValidationError):
        admission.Proposal.model_validate({**_base(), "unexpected": 1})


def test_cli_exit_codes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(ROOT)
    assert main(["qualification", "admit", str(FIXTURE)]) == 0
    assert json.loads(capsys.readouterr().out)["decision"] == "ADMISSION_READY"
    rejected = tmp_path / "bad.toml"
    rejected.write_text(
        FIXTURE.read_text()
        .replace('registry_claims = ["DEV-006"]', "registry_claims = []")
        .replace('registry_gaps = ["iso_ham"]', "registry_gaps = []")
    )
    assert main(["qualification", "admit", str(rejected)]) == 1
    assert json.loads(capsys.readouterr().out)["decision"] == "REJECTED"
    broken = tmp_path / "broken.toml"
    broken.write_text("family = ")
    assert main(["qualification", "admit", str(broken)]) == 2
