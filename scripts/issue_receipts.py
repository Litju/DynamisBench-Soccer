"""Fill the SHA-256 digests of every symbol cited by qualification/receipts.toml.

    uv run python scripts/issue_receipts.py          # write digests
    uv run python scripts/issue_receipts.py --check  # fail if any digest is missing or stale

Digests only bind a receipt to the exact code it cites. They confer no independence: whether
the cited reference is truly independent and the cited test truly exercises the claim is the
independent reviewer's judgement, recorded in the pull request.
"""

from __future__ import annotations

import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from dynamisbench_soccer.qualification import receipts  # noqa: E402


def issue(path: Path, check: bool) -> int:
    text = path.read_text(encoding="utf-8")
    loaded = receipts.load(path)
    raw = tomllib.loads(text).get("receipt", [])
    blocks = re.split(r"(?m)^(?=\[\[receipt\]\])", text)
    prefix, bodies = blocks[0], [b.split("[receipt.digests]")[0] for b in blocks[1:]]
    stale = 0
    out = [prefix]
    for body, r, row in zip(bodies, loaded, raw, strict=True):
        digests = {}
        for ref in (*r.implementation, *r.reference, *r.tests):
            d = receipts.resolve_digest(ROOT, ref)
            if d is None:
                print(f"FAIL {r.id}: {ref} does not resolve", file=sys.stderr)
                return 1
            digests[ref] = d
        if digests != row.get("digests", {}):
            stale += 1
        lines = "".join(f'"{k}" = "{v}"\n' for k, v in sorted(digests.items()))
        out.append(body.rstrip("\n") + "\n\n[receipt.digests]\n" + lines + "\n")
    if check:
        if stale:
            print(f"FAIL {stale} receipt(s) have missing or stale digests", file=sys.stderr)
            return 1
        print(f"{len(loaded)} receipts: digests current")
        return 0
    path.write_text("".join(out).rstrip("\n") + "\n", encoding="utf-8")
    print(f"wrote digests for {len(loaded)} receipts ({stale} changed)")
    return 0


if __name__ == "__main__":
    raise SystemExit(
        max(
            issue(path, "--check" in sys.argv)
            for path in (
                ROOT / f"qualification/{table}.toml" for table in receipts.RECEIPT_INVENTORIES
            )
        )
    )
