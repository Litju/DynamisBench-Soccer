"""Evidence registry schema.

* ``registry/sources.toml`` - bibliographic sources, each with how it was verified.
* ``registry/claims/*.toml`` - atomic scientific claims, each linked to sources.

A claim is the smallest statement a task, grader or reference implementation may rely
on. "No claim without a source" is enforced by :func:`validate`.
"""

from __future__ import annotations

import tomllib
from collections import Counter
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

DOI_PATTERN = r"^10\.\d{4,9}/\S+$"
CLAIM_ID_PATTERN = r"^[A-Z]{2,4}-\d{3}$"

#: Canonical v0.1 test identifiers. The ontology (RES-664) must use the same ids.
V01_TESTS: tuple[str, ...] = (
    "cmj", "sj", "slcmj", "dj", "rj_10_5", "imtp", "nhe", "hip_add_abd", "iso_ham",
    "sprint_gates", "sprint_fv", "cod_505", "ift_30_15", "rsa",
)  # fmt: skip
#: Cross-cutting topics that are not a single test.
TOPICS: tuple[str, ...] = (
    "force_plate_processing", "asymmetry", "reliability_stats", "soccer_context",
    "injury_surveillance", "return_to_play", "maturation",
)  # fmt: skip


class SourceType(StrEnum):
    CONSENSUS = "consensus"
    SYSTEMATIC_REVIEW = "systematic-review"
    PRIMARY = "primary"
    METHODS = "methods"
    BOOK = "book"
    PREPRINT = "preprint"


class VerificationMethod(StrEnum):
    CROSSREF = "crossref"  # DOI resolved on Crossref and title/first author matched
    URL = "url"  # landing page recorded, metadata not machine-verified
    ISBN = "isbn"


class Verification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    method: VerificationMethod
    checked: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")


class Source(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z][a-z0-9_]*\d{4}[a-z]?$")
    type: SourceType
    first_author: str
    year: int = Field(ge=1950, le=2100)
    title: str
    venue: str
    doi: str | None = Field(default=None, pattern=DOI_PATTERN)
    url: str | None = None
    isbn: str | None = None
    verification: Verification

    @model_validator(mode="after")
    def _identifier(self) -> Source:
        if not (self.doi or self.url or self.isbn):
            raise ValueError(f"source {self.id} needs a doi, url or isbn")
        if self.verification.method is VerificationMethod.CROSSREF and not self.doi:
            raise ValueError(f"source {self.id}: crossref verification requires a doi")
        return self


class ClaimKind(StrEnum):
    DEFINITION = "definition"  # how a quantity is defined or computed
    METHOD = "method"  # a processing/protocol choice and its consequences
    RELIABILITY = "reliability"
    SENSITIVITY = "sensitivity"
    VALIDITY = "validity"
    CONTEXT = "context"  # soccer-specific empirical finding
    CRITERION = "criterion"  # RTP / decision criterion (never a clearance rule)
    TAXONOMY = "taxonomy"


class EvidenceGrade(StrEnum):
    A = "A"  # consensus statement or systematic review / meta-analysis
    B = "B"  # peer-reviewed primary or methods study
    C = "C"  # preprint, conference abstract, URL-only record, or indirect population


class Population(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sport: str = "any"
    level: str = "any"
    sex: Literal["male", "female", "mixed", "any"] = "any"
    age_group: str = "any"


class Statistic(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str  # e.g. "ICC", "CV%", "SNR"
    value: str  # reported text; ranges kept verbatim, e.g. "0.70-0.98"
    model: str | None = None  # e.g. ICC form


class Claim(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=CLAIM_ID_PATTERN)
    subjects: list[str] = Field(min_length=1, description="test ids and/or topics")
    metrics: list[str] = Field(default_factory=list)
    kind: ClaimKind
    statement: str = Field(min_length=20)
    population: Population = Field(default_factory=Population)
    statistic: Statistic | None = None
    sources: list[str] = Field(min_length=1)
    evidence_grade: EvidenceGrade
    contested: bool = False
    variants: list[str] = Field(default_factory=list)
    reviewer_status: Literal["unreviewed", "reviewed", "disputed"] = "unreviewed"

    @model_validator(mode="after")
    def _contested_has_variants(self) -> Claim:
        if self.contested and len(self.variants) < 2:
            raise ValueError(f"{self.id}: contested claims must list >= 2 variants")
        unknown = set(self.subjects) - set(V01_TESTS) - set(TOPICS)
        if unknown:
            raise ValueError(f"{self.id}: unknown subjects {sorted(unknown)}")
        return self


class Gap(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subject: str
    reason: str
    opened: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")


class Registry(BaseModel):
    sources: dict[str, Source]
    claims: list[Claim]
    gaps: list[Gap] = Field(default_factory=list)

    def claims_for(self, subject: str) -> list[Claim]:
        return [c for c in self.claims if subject in c.subjects]


class RegistryError(ValueError):
    pass


def load(root: Path) -> Registry:
    """Load ``root/sources.toml`` and every ``root/claims/*.toml``."""
    raw_sources = tomllib.loads((root / "sources.toml").read_text(encoding="utf-8"))
    sources = [Source.model_validate(s) for s in raw_sources.get("source", [])]
    dupes = [k for k, n in Counter(s.id for s in sources).items() if n > 1]
    if dupes:
        raise RegistryError(f"duplicate source ids: {dupes}")
    claims: list[Claim] = []
    for path in sorted((root / "claims").glob("*.toml")):
        data = tomllib.loads(path.read_text(encoding="utf-8"))
        claims += [Claim.model_validate(c) for c in data.get("claim", [])]
    gaps_path = root / "gaps.toml"
    gaps = []
    if gaps_path.is_file():
        raw_gaps = tomllib.loads(gaps_path.read_text(encoding="utf-8"))
        gaps = [Gap.model_validate(g) for g in raw_gaps.get("gap", [])]
    return Registry(sources={s.id: s for s in sources}, claims=claims, gaps=gaps)


def validate(reg: Registry, min_claims: int = 0) -> list[str]:
    """Cross-reference checks. Returns human-readable problems (empty means valid)."""
    problems = []
    dupes = [k for k, n in Counter(c.id for c in reg.claims).items() if n > 1]
    if dupes:
        problems.append(f"duplicate claim ids: {dupes}")
    for claim in reg.claims:
        for sid in claim.sources:
            if sid not in reg.sources:
                problems.append(f"{claim.id}: unknown source {sid!r}")
        grades = {reg.sources[s].type for s in claim.sources if s in reg.sources}
        if claim.evidence_grade is EvidenceGrade.A and not grades & {
            SourceType.CONSENSUS,
            SourceType.SYSTEMATIC_REVIEW,
        }:
            problems.append(f"{claim.id}: grade A requires a consensus or systematic review")
        if (
            any(
                s in reg.sources and reg.sources[s].verification.method is VerificationMethod.URL
                for s in claim.sources
            )
            and claim.evidence_grade is not EvidenceGrade.C
            and len(claim.sources) == 1
        ):
            problems.append(f"{claim.id}: claims resting only on a URL-only source must be grade C")
    used = {s for c in reg.claims for s in c.sources}
    for sid in sorted(set(reg.sources) - used):
        problems.append(f"source {sid} is not cited by any claim")
    gap_subjects = {g.subject for g in reg.gaps}
    for test in V01_TESTS:
        n = len(reg.claims_for(test))
        if n == 0 and test not in gap_subjects:
            problems.append(f"v0.1 test {test!r} has no claim and is not a declared gap")
        if n > 0 and test in gap_subjects:
            problems.append(f"{test!r} is declared a gap but has claims; close the gap")
    if len(reg.claims) < min_claims:
        problems.append(f"{len(reg.claims)} claims < required {min_claims}")
    return problems


def coverage(reg: Registry) -> dict[str, int]:
    return {subject: len(reg.claims_for(subject)) for subject in (*V01_TESTS, *TOPICS)}
