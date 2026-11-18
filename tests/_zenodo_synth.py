"""SYNTHETIC Zenodo record, archive and records for RES-721 tests. Never empirical data."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from dynamisbench_soccer.external.zenodo_models import (
    ZenodoLock,
    ZenodoRightsRecord,
    metadata_digest,
)

RECORD_ID = 999001
KEY = "synthetic-uwb.zip"
BASE = "synthetic-uwb/"
HEADER = "SessionID,CIR1,CIR2,CIR3,CIR4,PSA1,PSA2,PSA3,PSA4,d1,d2,d3,d4,UWB_x,UWB_y,OPT_x,OPT_y"
ANCHORS = np.array([[0.0, 0.0], [5.0, 0.0], [5.0, 3.0], [0.0, 3.0]])
ANCHOR_H = 1.2


def uwb_csv(
    n_per_session: int = 300,
    sessions: int = 3,
    *,
    lag: int = 0,
    scale: float = 1.0,
    rot_deg: float = 0.0,
    shift: tuple[float, float] = (0.0, 0.0),
    noise: float = 0.0,
    sentinel: bool = False,
    lever: tuple[float, float] = (0.0, 0.0),
) -> bytes:
    """A planar path; UWB = similarity(OPT) shifted by ``lag`` rows (+ optional noise)."""
    rng = np.random.default_rng(7)
    lines = [HEADER]
    th = np.radians(rot_deg)
    rot = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    for s in range(1, sessions + 1):
        k = np.arange(n_per_session + abs(lag))
        path = np.c_[2.5 + 1.8 * np.cos(k / 40 + s), 1.5 + 1.0 * np.sin(k / 25 + s)]
        # positive lag: the device trails the reference (device row i+lag ~ reference row i)
        src = path[:n_per_session] if lag >= 0 else path[-lag : -lag + n_per_session]
        opt = path[lag : lag + n_per_session] if lag >= 0 else path[:n_per_session]
        uwb = scale * (src @ rot.T) + np.array(shift) + noise * rng.standard_normal(src.shape)
        if lever != (0.0, 0.0):  # body-frame tag/marker offset: (along, cross) the motion
            v = np.gradient(src, axis=0)
            tan = v / np.linalg.norm(v, axis=1)[:, None]
            nrm = np.c_[-tan[:, 1], tan[:, 0]]
            uwb = uwb + lever[0] * tan + lever[1] * nrm
        d = np.sqrt(((opt[:, None, :] - ANCHORS[None]) ** 2).sum(-1) + ANCHOR_H**2)
        for i in range(n_per_session):
            dd = d[i].copy()
            if sentinel and s == 1 and i == 5:
                dd[0] = -49.13
            lines.append(
                f"{s},2000,1700,1650,2000,125,117,117,125,"
                + ",".join(f"{x:.2f}" for x in dd)
                + ","
                + ",".join(repr(float(v)) for v in (*uwb[i], *opt[i]))
            )
    return ("\n".join(lines) + "\n").encode()


def build_archive(
    path: Path,
    members: dict[str, bytes] | None = None,
    *,
    extra: list[tuple[zipfile.ZipInfo, bytes]] | None = None,
) -> dict[str, bytes]:
    files = members or {
        BASE + "UWB_Walk_dataset.csv": uwb_csv(),
        BASE + "UWB_Trolley_dataset.csv": uwb_csv(sessions=2),
        BASE + "README_data.md": b"# Synthetic README\n",
        "__MACOSX/" + BASE + "._README_data.md": b"\x00\x05\x16\x07",
    }
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in files.items():
            z.writestr(name, data)
        for info, data in extra or []:
            z.writestr(info, data)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buf.getvalue())
    return files


def record(archive: Path, *, license_id: str = "cc-by-4.0") -> dict[str, Any]:
    data = archive.read_bytes()
    return {
        "id": RECORD_ID,
        "doi": f"10.5281/zenodo.{RECORD_ID}",
        "stats": {"views": 1},
        "metadata": {
            "title": "Synthetic UWB record",
            "publication_date": "2020-01-01",
            "license": {"id": license_id},
            "access_right": "open",
            "creators": [{"name": "Doe, Jane", "affiliation": "Synthetic Lab"}],
        },
        "files": [
            {
                "key": KEY,
                "size": len(data),
                "checksum": "md5:" + hashlib.md5(data, usedforsecurity=False).hexdigest(),
            }
        ],
    }


def lock(archive: Path, meta: dict[str, Any], files: dict[str, bytes]) -> ZenodoLock:
    data = archive.read_bytes()
    members = [
        {
            "name": n,
            "role": "documentation" if n.endswith(".md") else "data",
            "sha256": hashlib.sha256(b).hexdigest(),
            "size_bytes": len(b),
        }
        for n, b in files.items()
        if not n.startswith("__MACOSX/")
    ]
    excluded = [
        {"name": n, "reason": "macOS AppleDouble resource fork"}
        for n in files
        if n.startswith("__MACOSX/")
    ]
    return ZenodoLock.model_validate(
        {
            "source_id": "synthetic-uwb",
            "record_id": RECORD_ID,
            "doi": f"10.5281/zenodo.{RECORD_ID}",
            "license_id": "cc-by-4.0",
            "metadata_digest": metadata_digest(meta),
            "archives": [
                {
                    "key": KEY,
                    "url": f"https://zenodo.org/api/records/{RECORD_ID}/files/{KEY}/content",
                    "md5": hashlib.md5(data, usedforsecurity=False).hexdigest(),
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "size_bytes": len(data),
                    "members": members,
                    "excluded": excluded,
                }
            ],
        }
    )


def rights(**overrides: Any) -> ZenodoRightsRecord:
    base: dict[str, Any] = {
        "source_id": "synthetic-uwb",
        "pinned_ref": f"zenodo-{RECORD_ID}",
        "assessed_on": "2026-11-17",
        "assessed_by": "test",
        "license_id": "CC-BY-4.0",
        "license_grant_phrase": "cc-by-4.0",
        "copyright_holder": "Synthetic Lab",
        "license_scope": "synthetic record-level license scope",
        "dataset_specific_terms": "none",
        "measurement_data_rights": "verified",
        "participant_consent": "not_applicable",
        "minors_possible": False,
        "personal_identifiers_in_source": "none",
        "local_inspection_permitted": True,
        "local_inspection_basis": "synthetic local inspection basis",
        "canonical_admission": "verified",
        "redistribution_allowed": False,
        "blockers": [],
        "evidence_urls": ["https://example.org/record"],
    }
    base.update(overrides)
    return ZenodoRightsRecord.model_validate(base)


def relicensed(meta: dict[str, Any], license_id: str) -> dict[str, Any]:
    out = copy.deepcopy(meta)
    out["metadata"]["license"]["id"] = license_id
    return out


def dump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = obj.model_dump(mode="json") if hasattr(obj, "model_dump") else obj
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
