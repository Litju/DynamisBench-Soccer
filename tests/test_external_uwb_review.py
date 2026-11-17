"""Regressions for the independent RES-721 review (SYNTHETIC fixtures only)."""

from __future__ import annotations

import copy
import hashlib
import io as stdio
import json
import os
import urllib.error
import zipfile
from pathlib import Path
from typing import Any

import pytest
from _zenodo_synth import (
    BASE,
    KEY,
    build_archive,
    dump,
    lock,
    record,
    rights,
    uwb_csv,
)
from pydantic import ValidationError

from dynamisbench_soccer.cli import main
from dynamisbench_soccer.external import comparability as cmp
from dynamisbench_soccer.external import planar_profile, uwb_optitrack, uwb_pipeline, zenodo
from dynamisbench_soccer.external import rights as rg
from dynamisbench_soccer.external.zenodo_models import metadata_digest

NOT_ROOT = pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root")


def _check(data: bytes) -> cmp.SessionCheck:
    t = uwb_optitrack.parse("walk", "m", hashlib.sha256(data).hexdigest(), data)
    m = t.session == 1
    return cmp.check_session("walk", 1, t.device[m], t.reference[m], t.ranges[m])


# --- finding 1/2: mounting offsets (lever arm) are caught, never reported as error ----------


def test_cross_track_mounting_offset_blocks_at_zero_lag() -> None:
    c = _check(uwb_csv(lever=(0.0, 0.04)))
    assert c.best_row_lag == 0 and c.temporal_ok  # no lag can explain a cross-track offset
    assert c.cross_track_bias == pytest.approx(0.04, abs=0.005)
    assert not c.motion_ok and not c.comparable
    assert any("mounting offset" in r for r in c.reasons)


def test_along_track_offset_is_reported_as_ambiguous_not_as_delay() -> None:
    c = _check(uwb_csv(lever=(0.06, 0.0)))
    assert not c.comparable and not c.motion_ok
    assert c.along_track_bias == pytest.approx(0.06, abs=0.01)
    assert all("not separable" in r or "mounting offset" in r for r in c.reasons if "motion" in r)


def test_rms_lag_criterion_resolves_injected_lags_exactly() -> None:
    for lag in (-4, -1, 1, 2, 5):
        assert _check(uwb_csv(lag=lag, noise=0.01)).best_row_lag == lag


# --- finding 10: each gate blocks on its own (no gate can be dropped silently) ---------------


def _pipeline(tmp_path: Path, walk: bytes, trolley: bytes) -> dict[str, Any]:
    archive = tmp_path / "up" / KEY
    files = build_archive(
        archive,
        {
            BASE + "UWB_Walk_dataset.csv": walk,
            BASE + "UWB_Trolley_dataset.csv": trolley,
            BASE + "README_data.md": b"# synthetic\n",
        },
    )
    meta = record(archive)
    root = tmp_path / "ext"
    dump(lock(archive, meta, files), root / "synthetic-uwb" / "source-lock.json")
    dump(rights(), root / "synthetic-uwb" / "rights.json")
    return uwb_pipeline.run_uwb_optitrack(
        tmp_path / "run",
        external_root=root,
        manifests=tmp_path / "m",
        metadata=meta,
        source_dir=archive.parent,
        quarantine=tmp_path / "q",
        source_id="synthetic-uwb",
        store_root=tmp_path / "s",
        staging=tmp_path / "st",
    )


@pytest.mark.parametrize(
    ("kw", "gate"),
    [
        ({"lag": 3}, "temporal_ok"),
        ({"scale": 0.97}, "frame_ok"),
        ({"lever": (0.0, 0.04)}, "motion_ok"),
    ],
)
def test_each_gate_alone_blocks_metrics(tmp_path: Path, kw: dict[str, Any], gate: str) -> None:
    r = _pipeline(tmp_path, uwb_csv(**kw, noise=0.005), uwb_csv(sessions=2, **kw, noise=0.005))
    assert "error_metrics" not in r
    failing = {
        g
        for x in r["comparability"]["sessions"]
        for g in ("temporal_ok", "frame_ok", "motion_ok")
        if not x[g]
    }
    assert gate in failing


# --- finding 3/8: impossible ranges and wording ----------------------------------------------


def test_impossibly_short_ranges_make_geometry_inconsistent() -> None:
    data = uwb_csv()
    t = uwb_optitrack.parse("walk", "m", hashlib.sha256(data).hexdigest(), data)
    ranges = t.ranges.copy()
    ranges[:40, 0] = 0.05  # far below the fitted vertical separation (~1.2)
    g = cmp.anchor_geometry(t.reference, ranges)[0]
    assert g["h2"] > 0 and g["impossibly_short_ranges"] >= 40
    assert not g["geometrically_consistent"]


def test_lag_reason_does_not_overclaim_a_delay() -> None:
    c = _check(uwb_csv(lag=2))
    assert any("not established" in r for r in c.reasons)
    assert not any("are not simultaneous" in r for r in c.reasons)


# --- finding 4: offline / malformed input fails closed with a report ------------------------


def test_offline_metadata_fetch_fails_closed_with_report(tmp_path: Path, repo_root: Path) -> None:
    def offline(url: str, timeout: float) -> Any:
        raise urllib.error.URLError("network unreachable")

    r = uwb_pipeline.run_uwb_optitrack(
        tmp_path / "run",
        external_root=repo_root / "data/external",
        opener=offline,
        quarantine=tmp_path / "q",
        store_root=tmp_path / "s",
        staging=tmp_path / "st",
    )
    assert r["outcome"] == "failed" and (tmp_path / "run" / "report.json").is_file()
    assert {s["name"]: s["status"] for s in r["stages"]}["rights_verification"] == "failed"


def test_cli_rejects_unreadable_metadata(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bad = tmp_path / "m.json"
    bad.write_text("{not json")
    assert (
        main(
            [
                "external",
                "run",
                "uwb-optitrack",
                "--metadata",
                str(bad),
                "--out",
                str(tmp_path / "o"),
            ]
        )
        == 1
    )
    assert "--metadata unreadable" in capsys.readouterr().err


# --- finding 5: digest covers free-text terms and every rights-relevant field ---------------


@pytest.mark.parametrize(
    "change",
    [
        lambda m: m["metadata"].update(description="Non-commercial use only."),
        lambda m: m["metadata"].update(notes="Additional terms apply."),
        lambda m: m["metadata"].update(access_right="restricted"),
        lambda m: m["metadata"]["creators"].pop(),
        lambda m: m["metadata"].update(embargo_date="2030-01-01"),
        lambda m: m["metadata"].update(version="v2"),
    ],
)
def test_digest_detects_rights_relevant_changes(tmp_path: Path, change: Any) -> None:
    archive = tmp_path / KEY
    build_archive(archive)
    meta = record(archive)
    meta["metadata"]["creators"].append({"name": "Roe, R", "affiliation": "Synthetic Lab"})
    changed = copy.deepcopy(meta)
    change(changed)
    assert metadata_digest(changed) != metadata_digest(meta)


def test_missing_affiliation_is_not_the_string_none(tmp_path: Path) -> None:
    archive = tmp_path / KEY
    files = build_archive(archive)
    meta = record(archive)
    meta["metadata"]["creators"] = [{"name": "Doe, Jane"}]
    lk = lock(archive, meta, files)
    check = rg.verify_zenodo(rights(copyright_holder="None"), lk, meta)
    assert any("not a creator" in p for p in check.problems)


# --- finding 6: archive audit zip quirks -----------------------------------------------------


def _audit(tmp_path: Path, extra: list[tuple[zipfile.ZipInfo, bytes]]) -> list[str]:
    archive = tmp_path / KEY
    files = build_archive(archive)
    lk = lock(archive, record(archive), files)
    buf = stdio.BytesIO(archive.read_bytes())
    with zipfile.ZipFile(buf, "a") as z:
        for info, data in extra:
            z.writestr(info, data)
    archive.write_bytes(buf.getvalue())
    return zenodo.audit_archive(lk.archives[0], archive)


def test_audit_rejects_duplicate_names(tmp_path: Path) -> None:
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # zipfile warns on duplicate names
        problems = _audit(tmp_path, [(zipfile.ZipInfo(BASE + "UWB_Walk_dataset.csv"), b"x")])
    assert any("duplicate entry name" in p for p in problems)


def test_audit_rejects_directory_entries_with_data_or_unsafe_names(tmp_path: Path) -> None:
    fat_dir = zipfile.ZipInfo(BASE + "hidden/")
    escape = zipfile.ZipInfo("../evil/")
    problems = _audit(tmp_path, [(fat_dir, b"0" * 6000), (escape, b"")])
    assert any("directory entry carries" in p for p in problems)
    assert any("'../evil/': unsafe path" in p for p in problems)


def test_unlisted_member_fails_the_pipeline_end_to_end(tmp_path: Path) -> None:
    archive = tmp_path / "up" / KEY
    files = build_archive(archive)
    meta = record(archive)
    lk = lock(archive, meta, files)
    buf = stdio.BytesIO(archive.read_bytes())
    with zipfile.ZipFile(buf, "a") as z:
        z.writestr(BASE + "payload.csv", b"1,2\n")
    archive.write_bytes(buf.getvalue())  # lock no longer matches: rebuild hashes for this test
    data = archive.read_bytes()
    raw = lk.model_dump(mode="json")
    raw["archives"][0].update(
        sha256=hashlib.sha256(data).hexdigest(),
        size_bytes=len(data),
        md5=hashlib.md5(data, usedforsecurity=False).hexdigest(),
    )
    meta = record(archive)
    raw["metadata_digest"] = metadata_digest(meta)
    root = tmp_path / "ext"
    dump(raw, root / "synthetic-uwb" / "source-lock.json")
    dump(rights(), root / "synthetic-uwb" / "rights.json")
    r = uwb_pipeline.run_uwb_optitrack(
        tmp_path / "run",
        external_root=root,
        manifests=tmp_path / "m",
        metadata=meta,
        source_dir=archive.parent,
        quarantine=tmp_path / "q",
        source_id="synthetic-uwb",
        store_root=tmp_path / "s",
        staging=tmp_path / "st",
    )
    st = {s["name"]: s["status"] for s in r["stages"]}
    assert st["archive_audit"] == "failed" and "safe_parse" not in st


# --- finding 7: committed manifest fields are all checked -----------------------------------


@pytest.mark.parametrize(("field", "value"), [("license", "MIT"), ("redistribution_allowed", True)])
def test_committed_manifest_rights_fields_checked(tmp_path: Path, field: str, value: Any) -> None:
    archive = tmp_path / "up" / KEY
    files = build_archive(
        archive,
        {
            BASE + "UWB_Walk_dataset.csv": uwb_csv(),
            BASE + "UWB_Trolley_dataset.csv": uwb_csv(),
            BASE + "README_data.md": b"# s\n",
        },
    )
    meta = record(archive)
    lk = lock(archive, meta, files)
    root = tmp_path / "ext"
    dump(lk, root / "synthetic-uwb" / "source-lock.json")
    dump(rights(), root / "synthetic-uwb" / "rights.json")
    check = rg.verify_zenodo(rights(), lk, meta)
    m = zenodo.to_manifest(
        lk, rights(), check, title="t", citation="c", privacy_notes="p", handling_notes="h"
    )
    m.model_copy(update={field: value}).save(tmp_path / "m" / "synthetic-uwb.json")
    r = uwb_pipeline.run_uwb_optitrack(
        tmp_path / "run",
        external_root=root,
        manifests=tmp_path / "m",
        metadata=meta,
        source_dir=archive.parent,
        quarantine=tmp_path / "q",
        source_id="synthetic-uwb",
        store_root=tmp_path / "s",
        staging=tmp_path / "st",
    )
    ing = next(s for s in r["stages"] if s["name"] == "ingestion")
    assert ing["status"] == "failed" and field in ing["detail"]


# --- finding 9: profile cannot carry a frame claim, extra system or soccer scenario ----------


def test_profile_rejects_frame_claims_extra_systems_and_soccer_scenarios() -> None:
    data = uwb_csv(n_per_session=60)
    raw = uwb_optitrack.profile(
        uwb_optitrack.parse("walk", "m", hashlib.sha256(data).hexdigest(), data)
    ).model_dump(mode="json")
    bad_frame = json.loads(json.dumps(raw))
    bad_frame["systems"][0]["frame"] = "field_frame_soccer_pitch"
    extra = json.loads(json.dumps(raw))
    extra["systems"].append(
        {**raw["systems"][0], "id": "z_axis", "x_column": "UWB_z", "y_column": "UWB_y"}
    )
    mixed = json.loads(json.dumps(raw))
    mixed["systems"][0]["y_column"] = "OPT_y"
    soccer = json.loads(json.dumps(raw))
    soccer["sessions"][0]["scenario"] = "sprint_fv"
    for bad in (bad_frame, extra, mixed, soccer):
        with pytest.raises(ValidationError):
            planar_profile.PlanarProfile.model_validate(bad)


@NOT_ROOT
def test_permission_based_tests_are_meaningful_here() -> None:
    """The chmod-000 tests elsewhere rely on a non-root runner; assert that precondition."""
    assert os.geteuid() != 0
