"""RES-675/RES-692: the two M3 gate implementations must never disagree (policy-neutral)."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest

from dynamisbench_soccer.qualification import ledger as L
from dynamisbench_soccer.qualification.ledger import Evidence, Rating, Stage
from dynamisbench_soccer.qualification.state import m3_gate

ROOT = Path(__file__).resolve().parents[1]
LEDGER = L.load(ROOT / "qualification/ledger.toml")
VERIFIED = {d: Rating.VERIFIED for d in L.M3_DIMENSIONS}


def _complete(kind: str = "independent_test") -> L.Ledger:
    extra = tuple(
        Evidence(
            s.id,
            d,
            kind,
            "repo:README.md",
            "RES-675",
            "agreement control",
            independence_from=s.engine_id,
            implementation="independent control",
        )
        for s in LEDGER.subfamilies
        for d in VERIFIED
    )
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


def _mutations() -> dict[str, L.Ledger]:
    base = _complete()
    first = base.subfamilies[0]
    out = {"complete": base, "committed": LEDGER}
    out["blocker"] = dataclasses.replace(
        base, subfamilies=(dataclasses.replace(first, blockers=("open",)), *base.subfamilies[1:])
    )
    out["partial_L"] = dataclasses.replace(
        base,
        subfamilies=(
            dataclasses.replace(first, ratings={**first.ratings, "L": Rating.PARTIAL}),
            *base.subfamilies[1:],
        ),
    )
    out["no_J_evidence"] = dataclasses.replace(
        base, evidence=tuple(e for e in base.evidence if e.dimension != "J")
    )
    out["unresolved_comparison"] = dataclasses.replace(
        base, comparisons=tuple(dataclasses.replace(c, verdict="unknown") for c in base.comparisons)
    )
    out["vendor_only"] = dataclasses.replace(
        base,
        evidence=tuple(
            dataclasses.replace(e, kind="vendor_reproduction") if e.dimension == "B" else e
            for e in base.evidence
        ),
    )
    return out


@pytest.mark.parametrize("name", sorted(_mutations()))
def test_ledger_gate_and_computed_state_gate_agree(name: str) -> None:
    model = _mutations()[name]
    ledger_gate = L.gate(model, Stage.M3, ROOT)
    state_gate = m3_gate(model, ROOT)
    assert ledger_gate.passed == state_gate.passed, name
    if name == "complete":
        assert ledger_gate.passed and state_gate.engineering_verified == 15
    else:
        assert not ledger_gate.passed


def test_committed_ledger_is_blocked_by_both_gates_with_no_state_promoted() -> None:
    assert not L.gate(LEDGER, Stage.M3, ROOT).passed
    gate = m3_gate(LEDGER, ROOT)
    assert not gate.passed and gate.engineering_verified == 0
    assert set(gate.blocking) == {s.id for s in LEDGER.subfamilies}


def test_analytical_evidence_alone_can_satisfy_the_gate_so_the_verified_flag_is_policy_a() -> None:
    """Documented ambiguity: required-experiment text is empirical (policy B) but the gate
    accepts analytical_reference evidence for every dimension (policy A). See
    docs/verification/m3-gate-policy-decision.md. Not changed without owner authorization."""
    analytical = _complete("analytical_reference")
    assert L.gate(analytical, Stage.M3, ROOT).passed
    assert m3_gate(analytical, ROOT).passed


def test_states_expose_which_verified_dimensions_rest_on_analytical_evidence_only() -> None:
    from dynamisbench_soccer.qualification.state import computed_states

    states = computed_states(_complete("analytical_reference"))
    assert all(s.state == "ENGINEERING_VERIFIED" for s in states)
    assert all(s.analytical_only_dimensions for s in states)
    empirical = computed_states(_complete("empirical_study"))
    assert all(not s.analytical_only_dimensions for s in empirical)
