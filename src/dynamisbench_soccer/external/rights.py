"""Rights verification: check license evidence bytes and enforce the admission decision.

A tool cannot grant legal permission. This module only (1) proves that the license
evidence named by a :class:`RightsRecord` is present, pinned, hash-verified and says
what the record claims, and (2) refuses every action the record does not permit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from dynamisbench_soccer.external.models import (
    FileRole,
    RightsRecord,
    RightsStatus,
    SourceLock,
)


class RightsBlocked(PermissionError):
    """Raised when a source's rights record does not permit the requested action."""

    def __init__(self, action: str, blockers: list[str]) -> None:
        self.action = action
        self.blockers = list(blockers)
        super().__init__(f"{action} blocked: " + "; ".join(blockers))


#: Marks checks produced by :func:`verify`; a hand-built RightsCheck carries no authority.
_ISSUED: object = object()


@dataclass(frozen=True)
class RightsCheck:
    """Result of :func:`verify`. Only checks issued by ``verify`` authorise anything."""

    license_evidence_verified: bool
    problems: list[str] = field(default_factory=list)
    local_inspection_permitted: bool = False
    canonical_admission: RightsStatus = RightsStatus.PENDING
    redistribution_allowed: bool = False
    blockers: list[str] = field(default_factory=list)
    issued_by: object = field(default=None, repr=False, compare=False)


def issued(check: RightsCheck) -> bool:
    return check.issued_by is _ISSUED


def _authentic(action: str, check: RightsCheck) -> None:
    if not issued(check):
        raise RightsBlocked(action, ["rights check was not issued by rights.verify"])


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
            texts.append(files[rel].read_text(encoding="utf-8"))
    joined = "\n".join(texts)
    if texts and record.license_grant_phrase not in joined:
        problems.append("license grant phrase not found in the pinned license text")
    if texts and record.copyright_holder not in joined:
        problems.append("copyright holder not named in the pinned license text")
    verified = not problems
    blockers = list(record.blockers)
    if not verified:
        blockers = [*problems, *blockers]
    return RightsCheck(
        license_evidence_verified=verified,
        problems=problems,
        local_inspection_permitted=verified and record.local_inspection_permitted,
        canonical_admission=record.canonical_admission if verified else RightsStatus.PENDING,
        redistribution_allowed=verified and record.redistribution_allowed,
        blockers=blockers,
        issued_by=_ISSUED,
    )


def require_local_inspection(check: RightsCheck) -> None:
    _authentic("local quarantine inspection", check)
    if not check.local_inspection_permitted:
        raise RightsBlocked("local quarantine inspection", check.blockers or check.problems)


def require_canonical_admission(check: RightsCheck) -> None:
    _authentic("canonical admission", check)
    if check.canonical_admission is not RightsStatus.VERIFIED:
        raise RightsBlocked("canonical admission", check.blockers)
