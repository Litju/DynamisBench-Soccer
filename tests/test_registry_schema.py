from pathlib import Path

import pytest
from pydantic import ValidationError

from dynamisbench_soccer.registry.models import Claim, Source, load, validate

SOURCE = """
[[source]]
id = "smith2020"
type = "primary"
first_author = "Smith"
year = 2020
title = "A study"
venue = "Journal"
doi = "10.1234/abc.1"
verification = { method = "crossref", checked = "2026-11-14" }

[[source]]
id = "web2024"
type = "primary"
first_author = "(not recorded)"
year = 2024
title = "A web record"
venue = "Repository"
url = "https://example.org/record"
verification = { method = "url", checked = "2026-11-14" }
"""


def _claim(**kw: object) -> str:
    fields = {
        "id": '"FP-001"',
        "subjects": '["cmj"]',
        "kind": '"definition"',
        "statement": '"Jump height is computed from take-off velocity squared over 2g."',
        "sources": '["smith2020"]',
        "evidence_grade": '"B"',
    }
    fields.update({k: str(v) for k, v in kw.items()})
    return "[[claim]]\n" + "\n".join(f"{k} = {v}" for k, v in fields.items()) + "\n"


def _write(root: Path, claims: str, sources: str = SOURCE) -> Path:
    (root / "claims").mkdir(parents=True)
    (root / "sources.toml").write_text(sources)
    (root / "claims" / "a.toml").write_text(claims)
    return root


def test_valid_minimal_registry(tmp_path: Path) -> None:
    reg = load(_write(tmp_path, _claim() + _claim(id='"FP-002"', sources='["web2024"]',
                                                    evidence_grade='"C"')))  # fmt: skip
    problems = validate(reg)
    # only coverage problems (other v0.1 tests have no claims) remain
    assert all("v0.1 test" in p for p in problems)


def test_unknown_source_and_uncited_source(tmp_path: Path) -> None:
    reg = load(_write(tmp_path, _claim(sources='["nobody1999"]')))
    problems = validate(reg)
    assert any("unknown source 'nobody1999'" in p for p in problems)
    assert any("smith2020 is not cited" in p for p in problems)


def test_claim_without_source_is_rejected() -> None:
    with pytest.raises(ValidationError):
        Claim.model_validate(
            {
                "id": "FP-001",
                "subjects": ["cmj"],
                "kind": "definition",
                "sources": [],
                "statement": "x" * 30,
                "evidence_grade": "B",
            }
        )


def test_contested_claim_needs_variants(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="variants"):
        load(_write(tmp_path, _claim(contested="true", variants='["only_one"]')))


def test_unknown_subject_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="unknown subjects"):
        load(_write(tmp_path, _claim(subjects='["yoga"]')))


@pytest.mark.parametrize("doi", ["10.1/x", "doi:10.1234/x", "https://doi.org/10.1234/x"])
def test_bad_doi_format(doi: str) -> None:
    with pytest.raises(ValidationError):
        Source.model_validate(
            {
                "id": "a2020",
                "type": "primary",
                "first_author": "A",
                "year": 2020,
                "title": "t",
                "venue": "v",
                "doi": doi,
                "verification": {"method": "crossref", "checked": "2026-11-14"},
            }
        )


def test_crossref_verification_requires_doi() -> None:
    with pytest.raises(ValidationError, match="crossref"):
        Source.model_validate(
            {
                "id": "a2020",
                "type": "primary",
                "first_author": "A",
                "year": 2020,
                "title": "t",
                "venue": "v",
                "url": "https://x.org",
                "verification": {"method": "crossref", "checked": "2026-11-14"},
            }
        )


def test_grade_rules(tmp_path: Path) -> None:
    claims = _claim(evidence_grade='"A"') + _claim(
        id='"FP-002"', sources='["web2024"]', evidence_grade='"B"'
    )
    problems = validate(load(_write(tmp_path, claims)))
    assert any("FP-001: grade A requires" in p for p in problems)
    assert any("FP-002: claims resting only on a URL-only source" in p for p in problems)


def test_duplicate_claim_ids(tmp_path: Path) -> None:
    problems = validate(load(_write(tmp_path, _claim() + _claim())))
    assert any("duplicate claim ids" in p for p in problems)
