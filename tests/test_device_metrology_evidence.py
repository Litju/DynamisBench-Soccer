"""RES-694: evidence identity and complete subfamily gap linkage."""

import json
from pathlib import Path

from dynamisbench_soccer.qualification.ledger import load as load_ledger
from dynamisbench_soccer.qualification.state import computed_states
from dynamisbench_soccer.registry.models import load, validate

ROOT = Path(__file__).resolve().parents[1]


def test_primary_metadata_receipts_match_registry() -> None:
    registry = load(ROOT / "registry")
    receipts = json.loads((ROOT / "registry/verification/RES-694-crossref.json").read_text())
    assert len(receipts) == 8
    assert not validate(registry)
    for sid, receipt in receipts.items():
        source = registry.sources[sid]
        assert source.doi.lower() == receipt["doi"].lower()
        assert source.title == receipt["title"]
        assert source.first_author == receipt["first_author"]
        assert source.year == receipt["year"]
        assert source.verification.method.value == "crossref"


def test_every_subfamily_has_claim_and_explicit_gap_for_metrology() -> None:
    ledger = load_ledger(ROOT / "qualification/ledger.toml")
    registry = load(ROOT / "registry")
    gaps = {g.subject for g in registry.gaps}
    claims = {c.id for c in registry.claims}
    states = {s.subfamily: s for s in computed_states(ledger, ROOT)}
    assert "iso_ham" in gaps
    for sub in ledger.subfamilies:
        for dimension in "IJLO":
            refs = [
                e for e in ledger.evidence if e.subfamily == sub.id and e.dimension == dimension
            ]
            assert any(e.kind == "registry_claim" and e.ref in claims for e in refs)
            assert any(
                e.kind == "registry_gap" and e.ref == f"metrology.{sub.id}" and e.ref in gaps
                for e in refs
            )
            # Policy A: the metrology gap is an *empirical* requirement and stays open. A verified
            # rating here is computational only and must never rest on empirical records.
            if sub.ratings[dimension].value == "verified":
                assert dimension in states[sub.id].analytical_only_dimensions
                assert dimension not in states[sub.id].empirical_dimensions_covered
            assert dimension in states[sub.id].empirical_requirements
