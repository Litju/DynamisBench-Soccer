"""Empirical device-validity evidence (RES-692 / RES-675, Policy A).

Computational-reference readiness (M3) says the *software* is correct under declared
conditions. Empirical validity says a *physical instrument* measures a stated estimand to a
stated uncertainty against a criterion, under stated conditions (JCGM 200 VIM: measurement
uncertainty, calibration, metrological traceability). The two are never merged. Only
measured data can establish the second: synthetic, analytical, simulated or vendor-reported
records are rejected here by construction.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path

M3_DIMENSIONS = ("A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "O")
DATA_CATEGORIES = frozenset({"measured", "synthetic", "analytical", "simulated", "vendor_reported"})
ACCEPTED_CATEGORY = "measured"
CRITERION_KINDS = frozenset(
    {
        "reference_standard",  # traceable calibration standard (known mass, length, time base)
        "criterion_instrument",  # a higher-order independent instrument, itself traceable
        "adjudicated_events",  # independently annotated events
        "paired_instrument",  # a second device in a paired agreement design
    }
)
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
MIN_TEXT = 20


@dataclass(frozen=True)
class EmpiricalRecord:
    id: str
    subfamily: str
    dimension: str
    instrument: str  # make/model/serial or configuration identifying the physical device
    estimand: str
    protocol: str
    conditions: str
    criterion_kind: str
    criterion: str
    uncertainty: str  # expanded uncertainty or LoA with coverage and method
    data_category: str
    data_sha256: str
    rights_ref: str  # documented rights/provenance clearance
    n_independent_units: int
    issue: str


def load(path: Path) -> tuple[EmpiricalRecord, ...]:
    if not path.is_file():
        return ()
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    return tuple(
        EmpiricalRecord(
            id=r["id"],
            subfamily=r["subfamily"],
            dimension=r["dimension"],
            instrument=r["instrument"],
            estimand=r["estimand"],
            protocol=r["protocol"],
            conditions=r["conditions"],
            criterion_kind=r["criterion_kind"],
            criterion=r["criterion"],
            uncertainty=r["uncertainty"],
            data_category=r["data_category"],
            data_sha256=r["data_sha256"],
            rights_ref=r["rights_ref"],
            n_independent_units=int(r["n_independent_units"]),
            issue=r["issue"],
        )
        for r in raw.get("record", [])
    )


def _clearance_ok(root: Path, ref: str) -> bool:
    """The clearance must be a real file under the repository's docs/ or data/ trees."""
    target = (root / ref).resolve()
    allowed = [(root / name).resolve() for name in ("docs", "data")]
    return target.is_file() and any(target.is_relative_to(a) for a in allowed)


MIN_PAIRED_UNITS = 20  # the approved paired-agreement floor for dimension L


def verify(record: EmpiricalRecord, subfamily_ids: set[str], root: Path | None = None) -> list[str]:
    rid = record.id
    errors: list[str] = []
    if record.dimension not in M3_DIMENSIONS:
        errors.append(f"{rid}: dimension {record.dimension!r} is not an M3 dimension")
    if record.dimension == "L":
        if record.criterion_kind != "paired_instrument":
            errors.append(f"{rid}: L needs a paired_instrument criterion")
        if record.n_independent_units < MIN_PAIRED_UNITS:
            errors.append(f"{rid}: L needs at least {MIN_PAIRED_UNITS} independent units")
    if root is not None and not _clearance_ok(root, record.rights_ref):
        errors.append(
            f"{rid}: rights/provenance clearance {record.rights_ref!r} must be a file under "
            "docs/ or data/ inside the repository"
        )
    if record.subfamily not in subfamily_ids:
        errors.append(f"{rid}: unknown subfamily {record.subfamily}")
    if record.data_category not in DATA_CATEGORIES:
        errors.append(f"{rid}: unknown data category {record.data_category!r}")
    elif record.data_category != ACCEPTED_CATEGORY:
        errors.append(
            f"{rid}: {record.data_category} data cannot establish physical-device validity"
        )
    if record.criterion_kind not in CRITERION_KINDS:
        errors.append(f"{rid}: unknown criterion kind {record.criterion_kind!r}")
    for name in ("instrument", "estimand", "protocol", "conditions", "criterion", "uncertainty"):
        if len(getattr(record, name).strip()) < MIN_TEXT:
            errors.append(f"{rid}: {name} is not substantive")
    if not SHA256.fullmatch(record.data_sha256):
        errors.append(f"{rid}: data_sha256 must be a sha256 digest of the measured dataset")
    if not record.rights_ref.strip():
        errors.append(f"{rid}: rights/provenance clearance reference required")
    if record.n_independent_units < 1:
        errors.append(f"{rid}: at least one independent unit required")
    if not record.issue.startswith("RES-"):
        errors.append(f"{rid}: invalid owning issue")
    return errors


def verify_all(
    records: tuple[EmpiricalRecord, ...], subfamily_ids: set[str], root: Path | None = None
) -> dict[str, list[str]]:
    seen: set[str] = set()
    out: dict[str, list[str]] = {}
    for r in records:
        errs = verify(r, subfamily_ids, root)
        if r.id in seen:
            errs.append(f"{r.id}: duplicate id")
        seen.add(r.id)
        out[r.id] = errs
    return out
