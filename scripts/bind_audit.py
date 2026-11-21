"""Bind each SUPPORTED entry of qualification/receipt_audit.json to its receipt's exact content.

    uv run python scripts/bind_audit.py          # record the bindings (a deliberate re-review act)
    uv run python scripts/bind_audit.py --check  # fail if a receipt differs from what was audited

The digest covers every receipt field, including the code digests, so editing a receipt, or any
code it cites, requires re-issuing it and re-recording the audit: it can no longer stay SUPPORTED
silently. Recording a binding is only legitimate after an independent re-review of the receipt.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "qualification/receipt_audit.json"
sys.path.insert(0, str(ROOT / "src"))

from dynamisbench_soccer.qualification import receipts  # noqa: E402


def content_digest(receipt: receipts.Receipt) -> str:
    payload = json.dumps(receipt.__dict__, sort_keys=True, default=list)
    return hashlib.sha256(payload.encode()).hexdigest()


def main(check: bool) -> int:
    audit = json.loads(AUDIT.read_text())
    loaded = receipts.load(ROOT / "qualification/receipts.toml")
    problems = []
    for r in loaded:
        entry = audit["receipts"].get(r.id)
        if entry is None or entry["verdict"] != "SUPPORTED":
            problems.append(f"{r.id}: no SUPPORTED audit entry")
            continue
        digest = content_digest(r)
        if check:
            if entry.get("bound_digest") != digest:
                problems.append(f"{r.id}: receipt changed since it was audited")
        else:
            entry["bound_digest"] = digest
    if check:
        for p in problems:
            print(f"FAIL {p}", file=sys.stderr)
        print(f"{len(loaded)} receipts: audit binding {'OK' if not problems else 'BROKEN'}")
        return 1 if problems else 0
    AUDIT.write_text(json.dumps(audit, indent=1, sort_keys=True) + "\n")
    print(f"bound {len(loaded)} receipts to the audit record")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main("--check" in sys.argv))
