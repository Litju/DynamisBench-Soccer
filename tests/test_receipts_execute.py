"""RES-692/RES-675: every test cited by a committed receipt actually runs and passes.

Receipt verification is structural. This closes the loop: the cited tests are executed in a
subprocess, so a receipt can never rest on a failing, erroring or silently skipped test.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from dynamisbench_soccer.qualification import receipts

ROOT = Path(__file__).resolve().parents[1]


def test_every_cited_receipt_test_runs_and_passes() -> None:
    cited = sorted(
        {t for r in receipts.load(ROOT / "qualification/receipts.toml") for t in r.tests}
    )
    if not cited:
        return
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", *cited],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-1000:]
    assert " skipped" not in result.stdout and " xfailed" not in result.stdout
    assert " passed" in result.stdout, result.stdout[-500:]


def test_committed_receipt_digests_are_current_and_the_check_tool_reads_digested_files() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/issue_receipts.py", "--check"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr + result.stdout
