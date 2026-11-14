"""Guard against committing data into Git (ADR-0001 rules 5-6).

Checks every file tracked by Git: it fails on data-file extensions outside
``tests/fixtures/`` and on any file larger than the size threshold.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MAX_BYTES = 512 * 1024
FIXTURE_MAX_BYTES = 256 * 1024
DATA_EXTENSIONS = {
    ".parquet", ".feather", ".arrow", ".h5", ".hdf5", ".c3d", ".mat",
    ".zip", ".gz", ".tgz", ".xz", ".7z", ".npy", ".npz", ".pkl",
}  # fmt: skip
#: Plain-text exports are only allowed as small test fixtures.
TEXT_DATA_EXTENSIONS = {".csv", ".tsv", ".txt"}
ALLOWED_LARGE = {"uv.lock", "LICENSE"}


@dataclass(frozen=True)
class Violation:
    path: str
    reason: str


def tracked_files(repo: Path) -> list[str]:
    out = subprocess.run(
        ["git", "ls-files", "-z"], cwd=repo, check=True, capture_output=True
    ).stdout
    return [p for p in out.decode().split("\0") if p]


def check(repo: Path, files: list[str] | None = None) -> list[Violation]:
    violations = []
    for rel in files if files is not None else tracked_files(repo):
        path = repo / rel
        if not path.is_file():
            continue
        size = path.stat().st_size
        suffix = path.suffix.lower()
        in_fixtures = rel.startswith("tests/fixtures/")
        if suffix in DATA_EXTENSIONS:
            violations.append(Violation(rel, f"data file type {suffix} must not be committed"))
        elif suffix in TEXT_DATA_EXTENSIONS and not in_fixtures and not rel.startswith("docs/"):
            violations.append(Violation(rel, f"{suffix} data only allowed under tests/fixtures/"))
        elif in_fixtures and size > FIXTURE_MAX_BYTES:
            violations.append(Violation(rel, f"fixture {size} B > {FIXTURE_MAX_BYTES} B"))
        elif size > DEFAULT_MAX_BYTES and rel not in ALLOWED_LARGE:
            violations.append(Violation(rel, f"{size} B > {DEFAULT_MAX_BYTES} B threshold"))
    return violations
