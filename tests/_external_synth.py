"""SYNTHETIC source-shaped fixtures for RES-720 tests. Never empirical data.

They imitate only the *format* of ``shorts`` ``laser-gun-data.csv`` (header, 1 kHz
grid, millimetre distance quantisation) using an analytical mono-exponential sprint,
so numerical diagnostics have a known closed-form answer.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from dynamisbench_soccer.external.models import SourceLock

REPO = "example/synthetic"
REF = "0" * 39 + "1"
LICENSE_TEXT = (
    "MIT License\n\nCopyright (c) 2024 Synthetic Holder\n\n"
    "Permission is hereby granted, free of charge, to any person obtaining a copy\n"
)
MSS, TAU, TD = 8.0, 1.2, 0.3


def sprint(
    n: int = 3000, t0: float = -0.012, mss: float = MSS, tau: float = TAU, td: float = TD
) -> dict[str, np.ndarray]:
    t = np.round(t0 + np.arange(n) / 1000.0, 3)
    s = np.clip(t - td, 0, None)
    v = mss * (1 - np.exp(-s / tau))
    d = mss * (s - tau * (1 - np.exp(-s / tau)))
    raw = v + 0.05 * np.sin(2 * np.pi * 4 * t)
    return {"time": t, "distance": d, "velocity": v, "raw_velocity": raw}


def csv_bytes(
    cols: dict[str, np.ndarray], header: str = "time,distance,velocity,raw_velocity"
) -> bytes:
    lines = [header]
    for t, d, v, r in zip(
        cols["time"], cols["distance"], cols["velocity"], cols["raw_velocity"], strict=True
    ):
        lines.append(f"{t:.3f},{d:.3f},{v:.8f},{r:.7f}")
    return ("\n".join(lines) + "\n").encode("ascii")


def _entry(path: str, role: str, data: bytes) -> dict[str, object]:
    return {
        "path": path,
        "role": role,
        "sha256": hashlib.sha256(data).hexdigest(),
        "size_bytes": len(data),
        "git_blob_sha1": hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest(),
        "url": f"https://raw.githubusercontent.com/{REPO}/{REF}/{path}",
    }


EVIDENCE_PATH = "docs/acquisition-log.txt"
EVIDENCE_TEXT = "Synthetic acquisition log.\nSprint recorded on 2024-01-06 at the test track.\n"
EVIDENCE_EXCERPT = "Sprint recorded on 2024-01-06"
EVIDENCE_DATE = "2024-01-06"


def date_claim(**overrides: object) -> dict[str, object]:
    claim: dict[str, object] = {
        "date": EVIDENCE_DATE,
        "evidence_path": EVIDENCE_PATH,
        "excerpt": EVIDENCE_EXCERPT,
        "reviewed_by": "synthetic reviewer",
        "reviewed_on": "2026-11-17",
    }
    claim.update(overrides)
    return claim


def upstream(
    root: Path,
    csv: bytes | None = None,
    source_id: str = "synthetic-laser",
    *,
    evidence: str | None = None,
    claim: dict[str, object] | None = None,
) -> SourceLock:
    """Write a fake upstream checkout under ``root`` and return its source lock.

    ``evidence`` adds a pinned ``acquisition_evidence`` file; ``claim`` adds the lock's
    ``acquisition_date`` claim (both SYNTHETIC).
    """
    files = {
        "data-raw/laser-gun-data.csv": ("data", csv if csv is not None else csv_bytes(sprint())),
        "LICENSE.md": ("license", LICENSE_TEXT.encode()),
    }
    if evidence is not None:
        files[EVIDENCE_PATH] = ("acquisition_evidence", evidence.encode())
    for rel, (_, data) in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(data)
    payload: dict[str, object] = {
        "source_id": source_id,
        "repository": REPO,
        "pinned_ref": REF,
        "files": [_entry(rel, role, data) for rel, (role, data) in files.items()],
    }
    if claim is not None:
        payload["acquisition_date"] = claim
    return SourceLock.model_validate(payload)


def dated_upstream(root: Path, csv: bytes | None = None) -> SourceLock:
    return upstream(root, csv, evidence=EVIDENCE_TEXT, claim=date_claim())


def rights_payload(source_id: str = "synthetic-laser", **overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "source_id": source_id,
        "pinned_ref": REF,
        "assessed_on": "2026-11-17",
        "assessed_by": "test",
        "license_id": "MIT",
        "license_files": ["LICENSE.md"],
        "license_grant_phrase": "Permission is hereby granted, free of charge",
        "copyright_holder": "Synthetic Holder",
        "license_scope": "synthetic test scope statement",
        "dataset_specific_terms": "none",
        "measurement_data_rights": "pending",
        "participant_consent": "not_documented",
        "minors_possible": False,
        "personal_identifiers_in_source": "none",
        "local_inspection_permitted": True,
        "local_inspection_basis": "synthetic local inspection basis",
        "canonical_admission": "pending",
        "redistribution_allowed": False,
        "blockers": ["synthetic blocker"],
        "evidence_urls": ["https://example.org/evidence"],
    }
    base.update(overrides)
    return base


def verified_payload(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "measurement_data_rights": "verified",
        "participant_consent": "documented",
        "canonical_admission": "verified",
        "blockers": [],
    }
    values.update(overrides)
    return rights_payload(**values)


def dump(obj: object, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
