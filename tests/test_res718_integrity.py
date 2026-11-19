"""RES-718 boundaries: rejected trials stay rejected and the pilot is never verification."""

import json
from pathlib import Path

from dynamisbench_soccer.qualification import ledger as ledger_model

ROOT = Path(__file__).resolve().parents[1]
ACCEPTED_KINDS = {"independent_test", "analytical_reference", "empirical_study", "external_review"}


def test_recorded_trial_rejections_and_rules_are_preserved() -> None:
    text = (ROOT / "docs/res718/README.md").read_text(encoding="utf-8")
    assert "Both received a trial-level QC rejection" in text
    for rule in ("FP-RAILS", "QC-T04", "QC-T05"):
        assert rule in text
    assert "No trial-level result was admitted" in text
    assert "engineering and scientific qualification\nremain unchanged" in text


def test_pilot_is_not_counted_as_verification_evidence_in_the_ledger() -> None:
    model = ledger_model.load(ROOT / "qualification" / "ledger.toml")
    for evidence in model.evidence:
        assert not (evidence.issue == "RES-718" and evidence.kind in ACCEPTED_KINDS)
        assert "res718" not in evidence.ref.lower()


def test_manifest_keeps_synthetic_category_and_cross_validation_only_use() -> None:
    manifest = json.loads((ROOT / "data/manifests/biosyngen-res718-force.json").read_text())
    assert manifest["data_category"] == "synthetic"
    assert manifest["usage"] == "cross-validation-only"


def test_comparison_script_gates_protected_reference_on_both_run_receipts() -> None:
    source = (ROOT / "scripts/compare_biosyngen_force.py").read_text(encoding="utf-8")
    assert "receipt" in source.lower()
    # The comparator must never rewrite the processor's verdict or relax the QC rules.
    for forbidden in ("verdict =", "disable_qc", "rules=None", "qc.load"):
        assert forbidden not in source
