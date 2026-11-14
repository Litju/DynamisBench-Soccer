"""RES-692: positive and adversarial stage/evidence controls."""

from dataclasses import replace
from pathlib import Path

import pytest

from dynamisbench_soccer.cli import main
from dynamisbench_soccer.qualification.ledger import (
    DIMENSIONS,
    SUBFAMILIES,
    Evidence,
    Ledger,
    Rating,
    Reviewer,
    Stage,
    gate,
    load,
    markdown,
    validate,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def ledger() -> Ledger:
    return load(ROOT / "qualification/ledger.toml")


def verified(ledger: Ledger) -> Ledger:
    subs = tuple(
        replace(s, ratings=dict.fromkeys(DIMENSIONS, Rating.VERIFIED), blockers=())
        for s in ledger.subfamilies
    )
    evidence = tuple(
        Evidence(
            s.id,
            d,
            "analytical_reference",
            "repo:tests/test_qualification.py",
            "RES-692",
            "Independent closed-form verification",
            s.engine_id,
            "independent-producer",
        )
        for s in subs
        for d in DIMENSIONS
    )
    comparisons = tuple(replace(c, verdict="not_interchangeable") for c in ledger.comparisons)
    return replace(ledger, subfamilies=subs, evidence=evidence, comparisons=comparisons)


def test_current_ledger_and_generated_report(ledger: Ledger) -> None:
    assert not validate(ledger, ROOT)
    assert {s.id for s in ledger.subfamilies} == SUBFAMILIES
    assert len(ledger.subfamilies) == 15
    assert (ROOT / "docs/qualification-ledger.md").read_text() == markdown(ledger)
    for stage in Stage:
        assert not gate(ledger, stage).passed
        assert gate(ledger, stage).passed_families == 0


@pytest.mark.parametrize("dropped", sorted(SUBFAMILIES))
def test_every_subfamily_required(ledger: Ledger, dropped: str) -> None:
    obj = replace(ledger, subfamilies=tuple(s for s in ledger.subfamilies if s.id != dropped))
    assert validate(obj)
    assert not gate(obj, Stage.M3).passed


def test_stages_and_specified_are_separate(ledger: Ledger) -> None:
    obj = verified(ledger)
    assert not validate(obj, ROOT)
    assert gate(obj, Stage.M3).passed
    assert gate(obj, Stage.M5).passed
    assert not gate(obj, Stage.M8).passed
    subs = tuple(replace(s, ratings={**s.ratings, "N": Rating.ABSENT}) for s in obj.subfamilies)
    obj = replace(obj, subfamilies=subs)
    assert gate(obj, Stage.M3).passed
    assert not gate(obj, Stage.M5).passed
    subs = tuple(replace(s, ratings={**s.ratings, "A": Rating.SPECIFIED}) for s in subs)
    assert not gate(replace(obj, subfamilies=subs), Stage.M3).passed


def test_m8_requires_two_distinct_reviewers_per_family(ledger: Ledger) -> None:
    obj = verified(ledger)
    reviewers = tuple(
        Reviewer(f, who, True, "accepted", "repo:tests/test_qualification.py")
        for f in {s.family for s in obj.subfamilies}
        for who in ("A", "B")
    )
    assert gate(replace(obj, reviewers=reviewers), Stage.M8).passed
    assert not gate(replace(obj, reviewers=reviewers[:-1]), Stage.M8).passed
    assert not gate(
        replace(obj, reviewers=tuple(replace(r, identity="A") for r in reviewers)), Stage.M8
    ).passed


@pytest.mark.parametrize("kind", ["adapter_import", "vendor_reproduction"])
def test_import_and_vendor_evidence_cannot_qualify(ledger: Ledger, kind: str) -> None:
    obj = verified(ledger)
    obj = replace(obj, evidence=tuple(replace(e, kind=kind) for e in obj.evidence))
    assert any("adapter/vendor-only" in e for e in validate(obj))
    assert not gate(obj, Stage.M3).passed


def test_blockers_prevent_gate_and_report_promotion(ledger: Ledger) -> None:
    obj = verified(ledger)
    obj = replace(
        obj, subfamilies=(replace(obj.subfamilies[0], blockers=("open",)), *obj.subfamilies[1:])
    )
    assert not gate(obj, Stage.M3).passed
    assert "| force_platform | single, dual | IN_PROGRESS" in markdown(obj)


@pytest.mark.parametrize(
    "field,value",
    [
        ("independence_from", "another-engine"),
        ("implementation", "fp_single"),
        ("implementation", ""),
    ],
)
def test_circular_or_wrong_reference_rejected(ledger: Ledger, field: str, value: str) -> None:
    obj = verified(ledger)
    evidence = tuple(
        replace(e, **{field: value})
        if e.subfamily == "force_platform.single" and e.dimension == "J"
        else e
        for e in obj.evidence
    )
    assert any("circular" in e for e in validate(replace(obj, evidence=evidence)))


@pytest.mark.parametrize("method", ["butterworth", "moving_average", "filter", "magic"])
def test_discrete_processing_restricted(ledger: Ledger, method: str) -> None:
    subs = tuple(
        replace(s, methods=(method,)) if s.id == "timing_contact.gates" else s
        for s in ledger.subfamilies
    )
    assert validate(replace(ledger, subfamilies=subs))


def test_unresolved_agreement_rejected(ledger: Ledger) -> None:
    obj = replace(verified(ledger), comparisons=ledger.comparisons)
    assert any("unresolved non-equivalence" in e for e in validate(obj))
    comparisons = tuple(
        replace(
            c,
            verdict="conditionally_interchangeable",
            evidence_level="analytical",
            refs=("study",),
            reviewer_status="accepted",
        )
        for c in obj.comparisons
    )
    assert any("analytical-only" in e for e in validate(replace(obj, comparisons=comparisons)))


@pytest.mark.parametrize(
    "ref,kind",
    [
        ("repo:../outside", "schema"),
        ("repo:docs/missing.md", "schema"),
        ("FAKE-999", "registry_claim"),
        ("unknown-gap", "registry_gap"),
    ],
)
def test_dangling_evidence_rejected(ledger: Ledger, ref: str, kind: str) -> None:
    evidence = (replace(ledger.evidence[0], ref=ref, kind=kind), *ledger.evidence[1:])
    assert validate(replace(ledger, evidence=evidence), ROOT)


def test_not_applicable_needs_review(ledger: Ledger) -> None:
    obj = verified(ledger)
    subs = tuple(
        replace(s, ratings={**s.ratings, "B": Rating.NOT_APPLICABLE}) for s in obj.subfamilies
    )
    assert any("N/A without" in e for e in validate(replace(obj, subfamilies=subs)))


def test_cli_validation_and_blocked_gate() -> None:
    assert main(["qualification", "validate"]) == 0
    assert main(["qualification", "ledger", "--check"]) == 0
    assert main(["qualification", "gate"]) == 1


def test_gate_resolves_evidence_not_only_structure(ledger: Ledger) -> None:
    obj = verified(ledger)
    obj = replace(obj, evidence=tuple(replace(e, ref="repo:missing") for e in obj.evidence))
    assert not gate(obj, Stage.M3, ROOT).passed


def test_study_and_review_references_resolve(ledger: Ledger) -> None:
    obj = verified(ledger)
    assert validate(
        replace(
            obj,
            evidence=(
                replace(obj.evidence[0], kind="empirical_study", ref="fabricated"),
                *obj.evidence[1:],
            ),
        ),
        ROOT,
    )
    comparisons = (
        replace(
            obj.comparisons[0],
            verdict="conditionally_interchangeable",
            evidence_level="empirical",
            reviewer_status="accepted",
            refs=("fabricated",),
        ),
        *obj.comparisons[1:],
    )
    assert validate(replace(obj, comparisons=comparisons), ROOT)
    assert validate(replace(obj, reviewers=(Reviewer("imu", "A", True, "accepted", "fake"),)), ROOT)
