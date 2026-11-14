"""Consistency between dataset manifests and docs/data-audit.md (RES-666)."""

import re
from pathlib import Path

from dynamisbench_soccer.data.manifest import DatasetManifest, Usage


def _manifests(repo_root: Path) -> list[DatasetManifest]:
    return [
        DatasetManifest.load(p) for p in sorted((repo_root / "data" / "manifests").glob("*.json"))
    ]


def test_every_manifest_has_an_audit_section(repo_root: Path) -> None:
    doc = (repo_root / "docs" / "data-audit.md").read_text(encoding="utf-8")
    headings = set(re.findall(r"^### ([a-z0-9._-]+)$", doc, flags=re.MULTILINE))
    manifests = _manifests(repo_root)
    assert manifests
    for m in manifests:
        assert m.audit_ref == f"docs/data-audit.md#{m.dataset_id}"
        assert m.dataset_id in headings, m.dataset_id
    assert headings == {m.dataset_id for m in manifests}


def test_usage_decisions_are_justified(repo_root: Path) -> None:
    for m in _manifests(repo_root):
        assert m.handling_notes, m.dataset_id
        if m.usage is Usage.EXCLUDED:
            assert m.handling_notes.startswith("EXCLUDED:")
            assert m.files == []
        else:
            assert m.files and all(f.url for f in m.files)
            assert m.privacy_notes


def test_no_empirical_task_input_in_v01(repo_root: Path) -> None:
    # Audit outcome: v0.1 tasks use analytical/synthetic signals only.
    assert not [m.dataset_id for m in _manifests(repo_root) if m.usage is Usage.TASK_INPUT]


def test_pickle_risk_is_recorded(repo_root: Path) -> None:
    white = next(m for m in _manifests(repo_root) if m.dataset_id == "white2022-cmj-vgrf-acc")
    assert "allow_pickle=True" in white.handling_notes
    assert "NEVER unpickle" in white.handling_notes


def test_accepted_datasets_are_open_licensed(repo_root: Path) -> None:
    for m in _manifests(repo_root):
        if m.usage is not Usage.EXCLUDED:
            assert m.license == "CC-BY-4.0"
            assert m.source.doi
