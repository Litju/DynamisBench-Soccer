"""RES-675: every committed receipt is bound, by content, to an independent SUPPORTED audit."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "scripts/bind_audit.py", *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_every_committed_receipt_is_bound_to_a_supported_audit_entry() -> None:
    result = _run("--check")
    assert result.returncode == 0, result.stderr + result.stdout


def test_the_audit_record_states_its_limitations_and_never_promotes_unsupported_cells() -> None:
    audit = json.loads((ROOT / "qualification/receipt_audit.json").read_text())
    assert audit["limitations"] and "model-based" in " ".join(audit["limitations"]).lower()
    from dynamisbench_soccer.qualification import receipts

    committed = {r.id for r in receipts.load(ROOT / "qualification/receipts.toml")}
    supported = {i for i, e in audit["receipts"].items() if e["verdict"] == "SUPPORTED"}
    assert committed == supported  # exactly the SUPPORTED receipts, nothing else


def test_editing_a_receipt_after_audit_breaks_the_binding(tmp_path: Path) -> None:
    import dataclasses
    import importlib.util

    spec = importlib.util.spec_from_file_location("bind_audit", ROOT / "scripts/bind_audit.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from dynamisbench_soccer.qualification import receipts

    first = receipts.load(ROOT / "qualification/receipts.toml")[0]
    edited = dataclasses.replace(first, tolerance=first.tolerance * 10 + 1)
    assert module.content_digest(first) != module.content_digest(edited)
    redigested = dataclasses.replace(first, digests={k: "0" * 64 for k in first.digests})
    assert module.content_digest(first) != module.content_digest(redigested)
