"""Negative regressions for the independent review of PR #11 (RES-720 repairs).

Each test reproduces a bypass the reviewer demonstrated against 1cea22d and asserts it
is now refused. All fixtures are SYNTHETIC.
"""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from _external_synth import (
    REF,
    TD,
    csv_bytes,
    date_claim,
    dump,
    rights_payload,
    sprint,
    upstream,
)
from pydantic import ValidationError

from dynamisbench_soccer.cli import main
from dynamisbench_soccer.external import acquire, pipeline, rights, shorts_laser, sourceqc
from dynamisbench_soccer.external import timebase as tbm
from dynamisbench_soccer.external.models import RightsRecord, RightsStatus, dates_in
from dynamisbench_soccer.qc import spec


def _lock_with(tmp_path: Path, evidence: str, **claim: object) -> Any:
    return upstream(tmp_path / "up", evidence=evidence, claim=date_claim(**claim))


# --- finding 1: date evidence can no longer be fooled -------------------------------------


@pytest.mark.parametrize(
    ("excerpt", "date", "why"),
    [
        ("Sprint recorded on 16 January 2024", "2024-01-06", "states 2024-01-16"),
        ("Sprint recorded on 11 January 2024", "2024-01-01", "states 2024-01-11"),
        ("Sprint recorded on January 16, 2024", "2024-01-06", "states 2024-01-16"),
        ("Sprint recorded on 12024-01-06 trial", "2024-01-06", "exactly one date"),
        (
            "Data recorded 2024-01-06. Package released 2024-03-01.",
            "2024-03-01",
            "exactly one date",
        ),
        ("Package released to CRAN on 2024-01-06", "2024-01-06", "recorded/collected"),
        ("Sprint recorded on 06/01/2024 at the track", "2024-01-06", "ambiguous"),
        ("Sprint recorded on 2024.01.06 at the track", "2024-01-06", "ambiguous"),
        # re-review: unrecognised spellings must fail closed, not hide a second date
        ("Data recorded 6th January 2024; released 2024-03-01", "2024-03-01", "other date"),
        ("Data recorded 06-01-2024; released 2024-03-01", "2024-03-01", "other date"),
        ("Data recorded 6 Jan 2024; released 2024-03-01", "2024-03-01", "other date"),
        ("Data recorded 6 january 2024; released 2024-03-01", "2024-03-01", "other date"),
        ("Data recorded January 6 2024; released 2024-03-01", "2024-03-01", "other date"),
        ("Data recorded 20240106; released 2024-03-01", "2024-03-01", "other date"),
        ("Data recorded 2024-01-06T10:00 at the track", "2024-01-06", "exactly one date"),
        ("Data recorded 2024\u201101\u201106 at the track", "2024-01-06", "ASCII"),
        ("License attested by the author on 2024-01-06", "2024-01-06", "recorded/collected"),
        ("Data were unrecorded; package built 2024-01-06", "2024-01-06", "recorded/collected"),
        ("Data were not recorded on 2024-01-06 here", "2024-01-06", "negated"),
    ],
)
def test_date_claim_rejects_substring_multi_and_ambiguous_dates(
    tmp_path: Path, excerpt: str, date: str, why: str
) -> None:
    with pytest.raises(ValidationError, match=why):
        _lock_with(tmp_path, f"Log.\n{excerpt}\n", excerpt=excerpt, date=date)


def test_bare_date_excerpt_and_future_date_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="at least 20"):
        _lock_with(tmp_path, "2024-01-06\n", excerpt="2024-01-06")
    with pytest.raises(ValidationError, match="after the evidence review date"):
        _lock_with(
            tmp_path,
            "Sprint recorded on 2031-05-05 at the track.\n",
            excerpt="Sprint recorded on 2031-05-05",
            date="2031-05-05",
        )


def test_dates_in_is_word_bounded() -> None:
    assert [d.isoformat() for d in dates_in("on 16 January 2024")] == ["2024-01-16"]
    assert dates_in("ref 2024-01-06-b and x2024-01-06") == []
    assert dates_in("31 February 2024") == []  # invalid calendar date never counts


# --- finding 2/4: admission and inspection need an *issued* rights check ------------------


def _forged() -> rights.RightsCheck:
    return rights.RightsCheck(
        license_evidence_verified=True,
        local_inspection_permitted=True,
        canonical_admission=RightsStatus.VERIFIED,
        redistribution_allowed=True,
    )


def test_forged_rights_check_cannot_authorise_conversion(tmp_path: Path) -> None:
    up = tmp_path / "up"
    lock = upstream(
        up, evidence="Sprint recorded on 2024-01-06 at the track.\n", claim=date_claim()
    )
    data = (up / shorts_laser.DATA_PATH).read_bytes()
    import hashlib

    table = shorts_laser.parse(up / shorts_laser.DATA_PATH, hashlib.sha256(data).hexdigest())
    outcome = shorts_laser.convert(
        table,
        lock,
        {f.path: up / f.path for f in lock.files},
        admission=_forged(),
        time_base=tbm.assess(table),
    )
    assert not outcome.converted
    assert [b.split(":")[0] for b in outcome.blockers] == ["canonical admission"]


def test_forged_or_copied_check_cannot_acquire_data(tmp_path: Path) -> None:
    lock = upstream(tmp_path / "up")
    with pytest.raises(rights.RightsBlocked, match=r"not issued by rights\.verify"):
        acquire.acquire_data(lock, _forged(), source_dir=tmp_path / "up", root=tmp_path / "q")
    files = acquire.acquire(lock, source_dir=tmp_path / "up", root=tmp_path / "q")
    real = rights.verify(RightsRecord.model_validate(rights_payload()), lock, files)
    stripped = dataclasses.replace(real, seal="")
    with pytest.raises(rights.RightsBlocked, match="not issued"):
        acquire.acquire_data(lock, stripped, source_dir=tmp_path / "up", root=tmp_path / "q")
    with pytest.raises(rights.RightsBlocked, match="not issued"):
        rights.require_canonical_admission(_forged(), lock)


# --- re-review findings: edited, re-sealed and cross-source checks -----------------------


def _issued(tmp_path: Path, name: str, **overrides: object) -> tuple[Any, Any, Path]:
    up = tmp_path / name
    lock = upstream(up, source_id=name)
    files = acquire.acquire(lock, source_dir=up, root=tmp_path / "q")
    record = RightsRecord.model_validate(rights_payload(source_id=name, **overrides))
    return lock, rights.verify(record, lock, files), up


def test_edited_copy_of_a_real_check_keeps_no_authority(tmp_path: Path) -> None:
    lock, denied, up = _issued(tmp_path, "src-a", local_inspection_permitted=False)
    promoted = dataclasses.replace(denied, local_inspection_permitted=True)  # seal kept
    assert promoted.seal == denied.seal
    with pytest.raises(rights.RightsBlocked, match="not issued"):
        acquire.acquire_data(lock, promoted, source_dir=up, root=tmp_path / "q")
    _, pending, _ = _issued(tmp_path, "src-b")
    admitted = dataclasses.replace(pending, canonical_admission=RightsStatus.VERIFIED)
    with pytest.raises(rights.RightsBlocked, match="not issued"):
        rights.require_canonical_admission(admitted, lock)


def test_reused_seal_on_a_new_check_keeps_no_authority(tmp_path: Path) -> None:
    lock, real, up = _issued(tmp_path, "src-a")
    minted = rights.RightsCheck(
        True,
        local_inspection_permitted=True,
        canonical_admission=RightsStatus.VERIFIED,
        source_id=real.source_id,
        pinned_ref=real.pinned_ref,
        seal=real.seal,
    )
    with pytest.raises(rights.RightsBlocked, match="not issued"):
        acquire.acquire_data(lock, minted, source_dir=up, root=tmp_path / "q")


def test_check_for_one_source_cannot_authorise_another(tmp_path: Path) -> None:
    lock_a, _, up_a = _issued(tmp_path, "src-a", local_inspection_permitted=False)
    _, ok_b, _ = _issued(tmp_path, "src-b")
    assert rights.issued(ok_b) and ok_b.local_inspection_permitted
    with pytest.raises(rights.RightsBlocked, match="different source"):
        acquire.acquire_data(lock_a, ok_b, source_dir=up_a, root=tmp_path / "q")
    with pytest.raises(rights.RightsBlocked, match="different source"):
        acquire.verify_quarantine(lock_a, tmp_path / "q", ok_b)
    assert not (tmp_path / "q" / "src-a" / REF / shorts_laser.DATA_PATH).exists()


def test_cross_source_verified_admission_cannot_convert(tmp_path: Path) -> None:
    up = tmp_path / "dated"
    dated = upstream(
        up,
        source_id="src-a",
        evidence="Sprint recorded on 2024-01-06 here.\n",
        claim=date_claim(excerpt="Sprint recorded on 2024-01-06"),
    )
    _, other_ok, _ = _issued(tmp_path, "src-b", **_verified_overrides())
    assert other_ok.canonical_admission is RightsStatus.VERIFIED
    import hashlib

    data = (up / shorts_laser.DATA_PATH).read_bytes()
    table = shorts_laser.parse(up / shorts_laser.DATA_PATH, hashlib.sha256(data).hexdigest())
    outcome = shorts_laser.convert(
        table,
        dated,
        {f.path: up / f.path for f in dated.files},
        admission=other_ok,
        time_base=tbm.assess(table),
    )
    assert not outcome.converted and outcome.blockers[0].startswith("canonical admission")


def _verified_overrides() -> dict[str, object]:
    return {
        "measurement_data_rights": "verified",
        "participant_consent": "documented",
        "canonical_admission": "verified",
        "blockers": [],
    }


def test_license_evidence_is_rehashed_not_trusted(tmp_path: Path) -> None:
    up = tmp_path / "up"
    lock = upstream(up)
    forged = tmp_path / "forged-license.md"
    forged.write_text("Permission is hereby granted, free of charge\nSynthetic Holder\n")
    check = rights.verify(
        RightsRecord.model_validate(rights_payload()), lock, {"LICENSE.md": forged}
    )
    assert not check.license_evidence_verified
    assert any("differ from the source lock" in p for p in check.problems)


def test_future_review_date_rejected_at_binding(tmp_path: Path) -> None:
    up = tmp_path / "up"
    lock = upstream(
        up,
        evidence="Sprint recorded on 2024-01-06 here.\n",
        claim=date_claim(excerpt="Sprint recorded on 2024-01-06", reviewed_on="2099-01-01"),
    )
    with pytest.raises(shorts_laser.SessionEvidenceError, match="future"):
        shorts_laser.bind_session_date(lock, {f.path: up / f.path for f in lock.files})


def test_low_level_acquire_refuses_data_role_without_check(tmp_path: Path) -> None:
    lock = upstream(tmp_path / "up")
    with pytest.raises(rights.RightsBlocked, match="no rights check"):
        acquire.acquire(
            lock, source_dir=tmp_path / "up", root=tmp_path / "q", roles=acquire.DATA_ROLES
        )
    assert not (tmp_path / "q" / lock.source_id / REF / shorts_laser.DATA_PATH).exists()


# --- finding 3 / 8: quarantine verification never reads data without permission -----------


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root ignores modes")
def test_unreadable_cached_data_is_never_opened(tmp_path: Path) -> None:
    """API-independent: any open/read/hash of the cached CSV would raise PermissionError."""
    lock = upstream(tmp_path / "up")
    files = acquire.acquire(lock, source_dir=tmp_path / "up", root=tmp_path / "q")
    ok = rights.verify(RightsRecord.model_validate(rights_payload()), lock, files)
    acquire.acquire_data(lock, ok, source_dir=tmp_path / "up", root=tmp_path / "q")
    cached = tmp_path / "q" / lock.source_id / REF / shorts_laser.DATA_PATH
    cached.chmod(0o000)
    try:
        assert acquire.verify_quarantine(lock, tmp_path / "q") == []  # evidence only
        denied = rights.verify(
            RightsRecord.model_validate(rights_payload(local_inspection_permitted=False)),
            lock,
            files,
        )
        with pytest.raises(rights.RightsBlocked):
            acquire.verify_quarantine(lock, tmp_path / "q", denied)
    finally:
        cached.chmod(0o444)


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root ignores modes")
def test_pipeline_never_opens_unreadable_cached_data(tmp_path: Path, repo_root: Path) -> None:
    up = tmp_path / "up"
    lock = upstream(up, csv_bytes(sprint(n=600)))
    ext = tmp_path / "external"
    dump(lock.model_dump(mode="json"), ext / lock.source_id / "source-lock.json")
    sha = lock.file(shorts_laser.DATA_PATH).sha256
    cached = tmp_path / "q" / lock.source_id / REF / shorts_laser.DATA_PATH
    cached.parent.mkdir(parents=True)
    cached.write_bytes((up / shorts_laser.DATA_PATH).read_bytes())
    cached.chmod(0o000)
    (up / shorts_laser.DATA_PATH).chmod(0o000)  # the local source copy too
    dump(rights_payload(local_inspection_permitted=False), ext / lock.source_id / "rights.json")
    try:
        report = pipeline.run_shorts_laser(
            tmp_path / "run",
            external_root=ext,
            rules_path=repo_root / "qc/rules.toml",
            source_dir=up,
            quarantine=tmp_path / "q",
            expected_ref=REF,
            expected_sha256=sha,
            source_id=lock.source_id,
        )
    finally:
        cached.chmod(0o444)
        (up / shorts_laser.DATA_PATH).chmod(0o644)
    status = {s["name"]: s["status"] for s in report["stages"]}
    assert status["data_integrity"] == "blocked" and report["outcome"] == "stopped_at_blocked_gate"


# --- finding 5: non-committed records are marked unreviewed --------------------------------


def test_override_root_is_marked_unreviewed(
    tmp_path: Path, repo_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    up = tmp_path / "up"
    lock = upstream(up, csv_bytes(sprint(n=600)))
    ext = tmp_path / "external"
    dump(lock.model_dump(mode="json"), ext / lock.source_id / "source-lock.json")
    dump(rights_payload(local_inspection_permitted=False), ext / lock.source_id / "rights.json")
    report = pipeline.run_shorts_laser(
        tmp_path / "run",
        external_root=ext,
        rules_path=repo_root / "qc/rules.toml",
        source_dir=up,
        quarantine=tmp_path / "q",
        expected_ref=REF,
        expected_sha256=lock.file(shorts_laser.DATA_PATH).sha256,
        source_id=lock.source_id,
    )
    assert report["records"]["committed_default_root"] is False
    assert "UNREVIEWED RECORDS" in (tmp_path / "run" / "report.md").read_text()
    other = tmp_path / "other-root"
    fake = upstream(tmp_path / "up2", source_id="shorts-laser")  # fails the adapter pin offline
    dump(fake.model_dump(mode="json"), other / "shorts-laser" / "source-lock.json")
    dump(rights_payload(source_id="shorts-laser"), other / "shorts-laser" / "rights.json")
    assert (
        main(
            ["external", "run", "shorts-laser", "--root", str(other), "--out", str(tmp_path / "o")]
        )
        == 1
    )
    assert "not the committed, reviewed" in capsys.readouterr().err


def test_committed_report_records_reviewed_root(repo_root: Path) -> None:
    path = repo_root / "docs/external-data/reports/shorts-laser-diagnostic-report.json"
    assert json.loads(path.read_text())["records"]["committed_default_root"] is True


# --- finding 7 / 8: time-base edge cases ---------------------------------------------------


def _table(tmp_path: Path, cols: dict[str, np.ndarray]) -> shorts_laser.SourceTable:
    import hashlib

    data = csv_bytes(cols)
    (tmp_path / "t.csv").write_bytes(data)
    return shorts_laser.parse(tmp_path / "t.csv", hashlib.sha256(data).hexdigest())


def test_short_table_time_base_is_undetermined_not_qualified(
    tmp_path: Path, repo_root: Path
) -> None:
    table = _table(tmp_path, sprint(n=40))
    tb = tbm.assess(table)
    assert tb.qualification == "undetermined" and tb.statement().startswith("TEMPORALLY")
    src05 = next(
        c
        for c in sourceqc.run(table, spec.load(repo_root / "qc/rules.toml"))
        if c.rule_id == "SRC-05"
    )
    assert src05.status is spec.Status.FAIL
    assert json.dumps(dataclasses.asdict(tb), allow_nan=False)  # no NaN reaches a report


@pytest.mark.parametrize(("lead_ms", "expected"), [(5, "qualified"), (6, "unqualified")])
def test_time_base_tolerance_edge(tmp_path: Path, lead_ms: int, expected: str) -> None:
    cols = sprint(n=6000)
    cols["distance"] = sprint(n=6000, td=TD - lead_ms / 1000)["distance"]
    tb = tbm.assess(_table(tmp_path, cols))
    assert tb.best_lag_ms == pytest.approx(lead_ms, abs=1.0)
    assert tb.qualification == expected
