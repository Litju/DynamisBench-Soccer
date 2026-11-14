"""Checks on the committed evidence registry (RES-663 acceptance)."""

from pathlib import Path

from dynamisbench_soccer.registry.cli import MIN_CLAIMS
from dynamisbench_soccer.registry.models import V01_TESTS, load, validate
from dynamisbench_soccer.registry.report import contested_markdown

REQUIRED_ANCHOR_DOIS = {
    "10.1123/ijspp.2025-0055",  # elite youth battery reliability / sensitivity
    "10.1007/s40279-026-02404-9",  # RTP criteria after lower-limb muscle injury
    "10.1136/bjsports-2016-097206",  # hamstring RTP Delphi
    "10.1136/bjsports-2019-101969",  # IOC surveillance consensus (PMC7146946)
}


def test_registry_is_valid(repo_root: Path) -> None:
    reg = load(repo_root / "registry")
    assert validate(reg, min_claims=MIN_CLAIMS) == []
    assert len(reg.claims) >= 60


def test_every_v01_test_covered_or_declared_gap(repo_root: Path) -> None:
    reg = load(repo_root / "registry")
    gaps = {g.subject for g in reg.gaps}
    for test in V01_TESTS:
        assert reg.claims_for(test) or test in gaps, test


def test_required_anchors_are_cited(repo_root: Path) -> None:
    reg = load(repo_root / "registry")
    cited = {reg.sources[s].doi for c in reg.claims for s in c.sources}
    assert cited >= REQUIRED_ANCHOR_DOIS


def test_methods_anchors_for_onset_integration_filtering_and_estimands(repo_root: Path) -> None:
    reg = load(repo_root / "registry")
    metrics = {m for c in reg.claims for m in c.metrics}
    assert {"movement_onset", "velocity", "jump_height.takeoff_velocity",
            "jump_height.flight_time", "sampling_rate"} <= metrics  # fmt: skip


def test_contested_doc_is_in_sync(repo_root: Path) -> None:
    reg = load(repo_root / "registry")
    doc = (repo_root / "docs" / "contested-methods.md").read_text(encoding="utf-8")
    assert doc == contested_markdown(reg)
