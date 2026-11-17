"""Rights verification: check license evidence bytes and enforce the admission decision.

A tool cannot grant legal permission. This module only (1) proves that the license
evidence named by a :class:`RightsRecord` is present, pinned, hash-verified and says
what the record claims, and (2) refuses every action the record does not permit.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import secrets
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from dynamisbench_soccer.external.models import (
    FileRole,
    RightsRecord,
    RightsStatus,
    SourceLock,
)
from dynamisbench_soccer.external.zenodo_models import (
    ZenodoLock,
    ZenodoRightsRecord,
    metadata_digest,
)

AnyLock = SourceLock | ZenodoLock


class RightsBlocked(PermissionError):
    """Raised when a source's rights record does not permit the requested action."""

    def __init__(self, action: str, blockers: list[str]) -> None:
        self.action = action
        self.blockers = list(blockers)
        super().__init__(f"{action} blocked: " + "; ".join(blockers))


#: Per-process key for sealing checks. Only :func:`verify` holds it; a seal over every
#: field means a hand-built, edited (``dataclasses.replace``) or re-targeted check carries
#: no authority. (Deliberate in-process tampering with module privates is out of scope.)
_KEY: bytes = secrets.token_bytes(32)


@dataclass(frozen=True)
class RightsCheck:
    """Result of :func:`verify`, bound to one source and commit. Only sealed checks count."""

    license_evidence_verified: bool
    problems: list[str] = field(default_factory=list)
    local_inspection_permitted: bool = False
    canonical_admission: RightsStatus = RightsStatus.PENDING
    redistribution_allowed: bool = False
    blockers: list[str] = field(default_factory=list)
    source_id: str = ""
    pinned_ref: str = ""
    lock_digest: str = ""
    record_digest: str = ""
    seal: str = field(default="", repr=False, compare=False)

    def _payload(self) -> bytes:
        body = {k: v for k, v in asdict(self).items() if k != "seal"}
        return json.dumps(body, sort_keys=True, default=str).encode()


def _seal(check: RightsCheck) -> str:
    return hmac.new(_KEY, check._payload(), hashlib.sha256).hexdigest()


def digest(model: BaseModel) -> str:
    """Content digest of a lock or record; a check is bound to the exact content it saw."""
    return hashlib.sha256(model.model_dump_json().encode()).hexdigest()


def issued(check: RightsCheck) -> bool:
    if not isinstance(check.seal, str) or not check.seal:
        return False
    return hmac.compare_digest(check.seal, _seal(check))


def _authentic(action: str, check: RightsCheck, lock: AnyLock) -> None:
    if not issued(check):
        raise RightsBlocked(action, ["rights check was not issued by rights.verify"])
    if (check.source_id, check.pinned_ref) != (lock.source_id, lock.pinned_ref):
        raise RightsBlocked(action, ["rights check was issued for a different source or commit"])
    if check.lock_digest != digest(lock):
        raise RightsBlocked(action, ["rights check was issued for different source-lock content"])


def verify(record: RightsRecord, lock: SourceLock, files: dict[str, Path]) -> RightsCheck:
    """Check the record against the lock and the already hash-verified evidence files."""
    problems: list[str] = []
    if record.source_id != lock.source_id:
        problems.append(f"rights source_id {record.source_id!r} != lock {lock.source_id!r}")
    if record.pinned_ref != lock.pinned_ref:
        problems.append("rights record assessed a different pinned commit than the lock")
    license_paths = {f.path for f in lock.by_role(FileRole.LICENSE)}
    texts: list[str] = []
    for rel in record.license_files:
        if rel not in license_paths:
            problems.append(f"{rel}: not a license file of the source lock")
        elif rel not in files:
            problems.append(f"{rel}: license evidence not acquired")
        else:
            data = files[rel].read_bytes()
            entry = lock.file(rel)
            if len(data) != entry.size_bytes or hashlib.sha256(data).hexdigest() != entry.sha256:
                problems.append(f"{rel}: license evidence bytes differ from the source lock")
            else:
                texts.append(data.decode("utf-8"))
    joined = "\n".join(texts)
    if texts and record.license_grant_phrase not in joined:
        problems.append("license grant phrase not found in the pinned license text")
    if texts and record.copyright_holder not in joined:
        problems.append("copyright holder not named in the pinned license text")
    return _issue(record, lock, problems)


def verify_zenodo(
    record: ZenodoRightsRecord, lock: ZenodoLock, metadata: Mapping[str, Any]
) -> RightsCheck:
    """Check a Zenodo rights record against the lock and freshly fetched record metadata.

    The license evidence is ``metadata.license.id`` on the live record. The lock pins a
    digest of the rights-relevant metadata, so a relicense, a replaced file or changed
    creators after assessment fails closed.
    """
    problems: list[str] = []
    if record.source_id != lock.source_id:
        problems.append(f"rights source_id {record.source_id!r} != lock {lock.source_id!r}")
    if record.pinned_ref != lock.pinned_ref:
        problems.append("rights record assessed a different Zenodo record than the lock")
    try:
        current = metadata_digest(metadata)
        license_now = str(metadata["metadata"]["license"]["id"])
        affiliations = {str(c.get("affiliation")) for c in metadata["metadata"]["creators"]} | {
            str(c.get("name")) for c in metadata["metadata"]["creators"]
        }
        record_id = int(metadata["id"])
    except (KeyError, TypeError, ValueError) as exc:
        problems.append(f"record metadata is malformed: {exc!r}")
    else:
        if record_id != lock.record_id:
            problems.append(f"metadata is for record {record_id}, not {lock.record_id}")
        if current != lock.metadata_digest:
            problems.append(
                "record metadata differs from the pinned digest (relicensed or changed)"
            )
        if license_now != lock.license_id or license_now != record.license_grant_phrase:
            problems.append(f"record license {license_now!r} is not the assessed license")
        if record.copyright_holder not in affiliations:
            problems.append("copyright holder is not a creator or creator affiliation")
    return _issue(record, lock, problems)


def _issue(record: RightsRecord, lock: AnyLock, problems: list[str]) -> RightsCheck:
    verified = not problems
    blockers = list(record.blockers)
    if not verified:
        blockers = [*problems, *blockers]
    check = RightsCheck(
        license_evidence_verified=verified,
        problems=problems,
        local_inspection_permitted=verified and record.local_inspection_permitted,
        canonical_admission=record.canonical_admission if verified else RightsStatus.PENDING,
        redistribution_allowed=verified and record.redistribution_allowed,
        blockers=blockers,
        source_id=lock.source_id,
        pinned_ref=lock.pinned_ref,
        lock_digest=digest(lock),
        record_digest=digest(record),
    )
    return replace(check, seal=_seal(check))


def require_local_inspection(check: RightsCheck, lock: AnyLock) -> None:
    _authentic("local quarantine inspection", check, lock)
    if not check.local_inspection_permitted:
        raise RightsBlocked("local quarantine inspection", check.blockers or check.problems)


def require_canonical_admission(check: RightsCheck, lock: AnyLock) -> None:
    _authentic("canonical admission", check, lock)
    if check.canonical_admission is not RightsStatus.VERIFIED:
        raise RightsBlocked("canonical admission", check.blockers)
