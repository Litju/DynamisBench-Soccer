"""RES-675/RES-692: both M3 gate entry points give one answer on every ledger (Policy A).

`ledger.gate(Stage.M3)` and `state.m3_gate` (behind `dbs qualification gate|states`) both call
the canonical evaluator in `qualification/gate.py`. These tests feed identical valid, invalid,
adversarial and incomplete ledgers to both and require identical decisions and reasons.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest
from receipt_fixtures import IMPLEMENTATION, TEST, fixture_receipts, receipt

from dynamisbench_soccer.cli import main
from dynamisbench_soccer.qualification import ledger as L
from dynamisbench_soccer.qualification.ledger import Evidence, Rating, Stage
from dynamisbench_soccer.qualification.state import computed_states, m3_gate

ROOT = Path(__file__).resolve().parents[1]
LEDGER = L.load(ROOT / "qualification/ledger.toml")
VERIFIED = {d: Rating.VERIFIED for d in L.M3_DIMENSIONS}


def _complete(kind: str = "independent_test") -> L.Ledger:
    extra = tuple(
        Evidence(
            s.id, d, kind, "repo:README.md", "RES-675", "agreement control",
            independence_from=s.engine_id, implementation="independent control",
        )
        for s in LEDGER.subfamilies
        for d in VERIFIED
    )  # fmt: skip
    subs = tuple(
        dataclasses.replace(s, ratings={**s.ratings, **VERIFIED, "N": Rating.ABSENT}, blockers=())
        for s in LEDGER.subfamilies
    )
    return dataclasses.replace(
        LEDGER,
        subfamilies=subs,
        evidence=(*LEDGER.evidence, *extra),
        comparisons=tuple(
            dataclasses.replace(c, verdict="not_interchangeable") for c in LEDGER.comparisons
        ),
    )


def _cases() -> dict[str, tuple[L.Ledger, tuple, bool]]:
    """name -> (ledger, receipts, expected pass)."""
    base = _complete()
    good = fixture_receipts(base)
    first = base.subfamilies[0]
    out: dict[str, tuple[L.Ledger, tuple, bool]] = {"valid_complete": (base, good, True)}
    out["committed_ledger"] = (LEDGER, (), False)
    out["no_receipts_at_all"] = (base, (), False)  # verified ratings without receipts
    out["blocker"] = (
        dataclasses.replace(
            base,
            subfamilies=(dataclasses.replace(first, blockers=("open",)), *base.subfamilies[1:]),
        ),
        good,
        False,
    )
    out["partial_L"] = (
        dataclasses.replace(
            base,
            subfamilies=(
                dataclasses.replace(first, ratings={**first.ratings, "L": Rating.PARTIAL}),
                *base.subfamilies[1:],
            ),
        ),
        good,
        False,
    )
    out["missing_one_receipt"] = (
        base,
        tuple(r for r in good if r.id != f"FX-{first.id}-F"),
        False,
    )
    out["O_without_registry_basis"] = (
        dataclasses.replace(
            base,
            evidence=tuple(
                e
                for e in base.evidence
                if not (
                    e.subfamily == first.id
                    and e.dimension == "O"
                    and e.kind in {"registry_claim", "registry_gap"}
                )
            ),
        ),
        good,
        False,
    )
    out["no_J_evidence"] = (
        dataclasses.replace(base, evidence=tuple(e for e in base.evidence if e.dimension != "J")),
        good,
        False,
    )
    out["unexecuted_comparison"] = (
        dataclasses.replace(
            base,
            comparisons=tuple(
                dataclasses.replace(c, evidence_level="none") for c in base.comparisons
            ),
        ),
        good,
        False,
    )
    out["stale_digest"] = (
        base,
        (
            dataclasses.replace(good[0], digests={**good[0].digests, IMPLEMENTATION: "0" * 64}),
            *good[1:],
        ),
        False,
    )
    out["reference_is_implementation"] = (
        base,
        (receipt((first.id,), "A", rid=good[0].id, reference=(IMPLEMENTATION,)), *good[1:]),
        False,
    )
    out["reference_calls_engine"] = (
        base,
        (receipt((first.id,), "A", rid=good[0].id, reference=(TEST,)), *good[1:]),
        False,
    )
    out["empty_independence"] = (
        base,
        (receipt((first.id,), "A", rid=good[0].id, independence="same"), *good[1:]),
        False,
    )
    out["zero_tolerance"] = (
        base,
        (receipt((first.id,), "A", rid=good[0].id, tolerance=0.0), *good[1:]),
        False,
    )
    return out


CASES = _cases()


@pytest.mark.parametrize("name", sorted(CASES))
def test_both_gates_decide_identically_for_the_same_ledger(name: str) -> None:
    model, receipts, expected = CASES[name]
    ledger_gate = L.gate(model, Stage.M3, ROOT, receipts=receipts)
    state_gate = m3_gate(model, ROOT, receipts=receipts)
    assert ledger_gate.passed == state_gate.passed == expected, name
    blocked_families = {f for f in ledger_gate.missing if f != "__ledger__"}
    assert blocked_families == {
        s.family
        for s in computed_states(model, ROOT, receipts=receipts)
        if s.state not in {"ENGINEERING_VERIFIED", "SCIENTIFICALLY_QUALIFIED"}
    } or ("__ledger__" in ledger_gate.missing and not expected)
    if expected:
        assert state_gate.engineering_verified == 15 and not state_gate.structural_problems


def test_committed_ledger_cli_commands_agree(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(ROOT)
    gate_code = main(["qualification", "gate", "--stage", "M3"])
    gate_out = capsys.readouterr().out
    states_code = main(["qualification", "states"])
    payload = json.loads(capsys.readouterr().out)
    assert gate_code == states_code == (0 if payload["m3_gate"]["passed"] else 1)
    assert "computational-reference readiness" in gate_out
    assert payload["policy"].startswith("A:")


def test_analytical_evidence_kinds_alone_no_longer_pass_the_gate() -> None:
    """Replaces the Policy-neutral pin: file + evidence-kind strings are not receipts."""
    for kind in ("analytical_reference", "independent_test", "empirical_study"):
        model = _complete(kind)
        assert not L.gate(model, Stage.M3, ROOT, receipts=()).passed
        assert not m3_gate(model, ROOT, receipts=()).passed


def test_synthetic_receipts_cannot_stand_in_for_physical_validity() -> None:
    model = _complete()
    states = computed_states(model, ROOT, receipts=fixture_receipts(model))
    assert all(s.state == "ENGINEERING_VERIFIED" for s in states)
    for s in states:
        # computational verification leaves every empirical requirement open
        assert not s.empirical_records and not s.empirical_dimensions_covered
        assert set(s.empirical_requirements) == set(L.M3_DIMENSIONS)
        assert s.empirical_interchangeability in {"unknown", "not_interchangeable"}
        assert set(s.analytical_only_dimensions) == set(L.M3_DIMENSIONS)
