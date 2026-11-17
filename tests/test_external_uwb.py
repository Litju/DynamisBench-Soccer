"""RES-721 gates 4-5: strict UWB parse, planar profile, comparability before metrics."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from _zenodo_synth import (
    BASE,
    KEY,
    build_archive,
    dump,
    lock,
    record,
    relicensed,
    rights,
    uwb_csv,
)
from pydantic import ValidationError

from dynamisbench_soccer.cli import main
from dynamisbench_soccer.external import comparability as cmp
from dynamisbench_soccer.external import planar_profile, uwb_optitrack, uwb_pipeline
from dynamisbench_soccer.external.pipeline import dumps


def _table(data: bytes, scenario: str = "walk") -> uwb_optitrack.UwbTable:
    return uwb_optitrack.parse(scenario, "m.csv", hashlib.sha256(data).hexdigest(), data)


# --- strict parsing -------------------------------------------------------------------------


def test_parse_reads_positions_ranges_and_sessions() -> None:
    t = _table(uwb_csv(n_per_session=60, sessions=2))
    assert t.n_rows == 120 and t.device.shape == t.reference.shape == (120, 2)
    assert t.ranges.shape == (120, 4) and set(t.session.tolist()) == {1, 2}


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda b: b.replace(b"OPT_y", b"OPT_z", 1), "header"),
        (lambda b: b"\xef\xbb\xbf" + b, "byte-order"),
        (lambda b: b.replace(b"\n", b"\r\n"), "carriage"),
        (lambda b: b.replace(b"\n1,2000", b"\n1,2000,7", 1), "fields"),
        (lambda b: b.replace(b"\n1,2000", b"\nNaN,2000", 1), "finite number"),
        (lambda b: b.replace(b"\n1,2000", b"\n1.5,2000", 1), "positive integer"),
        (
            lambda b: b + b"1,2000,1700,1650,2000,125,117,117,125,1,1,1,1,0,0,0,0\n",
            "non-decreasing",
        ),
    ],
)
def test_parse_is_strict(mutate: Any, message: str) -> None:
    with pytest.raises(uwb_optitrack.UwbFormatError, match=message):
        _table(mutate(uwb_csv(n_per_session=60, sessions=2)))


def test_readme_counts_include_header() -> None:
    t = uwb_optitrack.parse("walk", "m", "0" * 64, uwb_csv(n_per_session=932, sessions=3))
    rec = uwb_optitrack.readme_reconciliation(t)
    assert rec == {
        "readme_stated": 2797,
        "data_rows": 2796,
        "readme_counts_header_line": True,
        "matches_data_rows": False,
    }


# --- planar profile -------------------------------------------------------------------------


def test_profile_cannot_hold_z_soccer_task_or_clock() -> None:
    p = uwb_optitrack.profile(_table(uwb_csv(n_per_session=60)))
    assert (p.dimensionality, p.z_coordinate, p.soccer_task) == ("2D", "not_recorded", False)
    assert (p.timestamps, p.pairing, p.units, p.usage) == (
        "absent",
        "row_index_within_session",
        "undeclared",
        "cross-validation-only",
    )
    raw = p.model_dump(mode="json")
    for bad in (
        {"z_coordinate": "synthesised_zero"},
        {"soccer_task": True},
        {"test_id": "sprint_fv"},
        {"timestamps": "row_index_as_time"},
        {"units": "metre"},
        {"dimensionality": "3D"},
    ):
        with pytest.raises(ValidationError):
            planar_profile.PlanarProfile.model_validate({**raw, **bad})
    assert not any(f == "z" or f.endswith("_z") for f in planar_profile.PlanarProfile.model_fields)


def test_profile_needs_exactly_one_reference() -> None:
    raw = uwb_optitrack.profile(_table(uwb_csv(n_per_session=60))).model_dump(mode="json")
    raw["systems"][1]["role"] = "device_estimate"
    with pytest.raises(ValidationError, match="depositor_reference"):
        planar_profile.PlanarProfile.model_validate(raw)


def test_committed_schema_doc_is_current(repo_root: Path) -> None:
    doc = repo_root / planar_profile.SCHEMA_DOC
    assert doc.read_text(encoding="utf-8") == planar_profile.json_schema()


# --- comparability --------------------------------------------------------------------------


def _session(data: bytes) -> cmp.SessionCheck:
    t = _table(data)
    m = t.session == 1
    return cmp.check_session("walk", 1, t.device[m], t.reference[m], t.ranges[m])


def test_identical_frames_and_rows_are_comparable() -> None:
    c = _session(uwb_csv(noise=0.01))
    assert c.comparable and c.best_row_lag == 0
    assert c.scale == pytest.approx(1, abs=0.005) and abs(c.rotation_deg) < 0.5


@pytest.mark.parametrize(
    ("kw", "lag", "temporal", "frame", "why"),
    [
        ({"lag": 3}, 3, False, True, "row lag 3"),
        ({"lag": -2}, -2, False, True, "row lag -2"),
        ({"scale": 0.97}, 0, True, False, "scale"),
        ({"rot_deg": 3.0}, 0, True, False, "rotation"),
        ({"shift": (0.0, -0.27)}, 0, True, False, "translation"),
    ],
)
def test_time_and_frame_mismatches_block(
    kw: dict[str, Any], lag: int, temporal: bool, frame: bool, why: str
) -> None:
    c = _session(uwb_csv(**kw))
    assert c.best_row_lag == lag and (c.temporal_ok, c.frame_ok) == (temporal, frame)
    assert not c.comparable and any(why in r for r in c.reasons)


def test_measured_transform_matches_injected_one() -> None:
    c = _session(uwb_csv(scale=0.97, rot_deg=2.0, shift=(0.06, -0.03)))
    assert c.scale == pytest.approx(0.97, abs=1e-6) and c.rotation_deg == pytest.approx(
        2.0, abs=1e-4
    )
    assert c.translation == pytest.approx((0.06, -0.03), abs=1e-6)


def test_checks_never_modify_data() -> None:
    t = _table(uwb_csv(lag=3, scale=0.97))
    before = (t.device.copy(), t.reference.copy(), t.ranges.copy())
    cmp.check_session("walk", 1, t.device, t.reference, t.ranges)
    cmp.anchor_geometry(t.reference, t.ranges)
    assert all(
        np.array_equal(a, b) for a, b in zip(before, (t.device, t.reference, t.ranges), strict=True)
    )


def test_short_session_is_undetermined_not_comparable() -> None:
    c = _session(uwb_csv(n_per_session=20))
    assert not c.comparable and "undetermined" in c.reasons[0]


def test_anchor_geometry_recovers_synthetic_anchors_and_flags_sentinel() -> None:
    from _zenodo_synth import ANCHOR_H, ANCHORS

    t = _table(uwb_csv(sentinel=True))
    rows = cmp.anchor_geometry(t.reference, t.ranges)
    for g, (ax, ay) in zip(rows, ANCHORS, strict=True):
        assert g["geometrically_consistent"] and g["h2"] == pytest.approx(ANCHOR_H**2, abs=0.05)
        assert (g["x"], g["y"]) == pytest.approx((ax, ay), abs=0.05)
    assert rows[0]["rows_used"] == t.n_rows - 1  # the -49.13 sentinel row is excluded


# --- end to end -----------------------------------------------------------------------------


def _setup(tmp_path: Path, walk: bytes, trolley: bytes, **rk: Any) -> dict[str, Any]:
    archive = tmp_path / "up" / KEY
    files = build_archive(
        archive,
        {
            BASE + "UWB_Walk_dataset.csv": walk,
            BASE + "UWB_Trolley_dataset.csv": trolley,
            BASE + "README_data.md": b"# synthetic\n",
            "__MACOSX/" + BASE + "._README_data.md": b"\x00\x05",
        },
    )
    meta = record(archive)
    lk = lock(archive, meta, files)
    root = tmp_path / "ext"
    dump(lk, root / "synthetic-uwb" / "source-lock.json")
    dump(rights(**rk), root / "synthetic-uwb" / "rights.json")
    return {"archive": archive, "meta": meta, "root": root, "lock": lk}


def _run(tmp_path: Path, s: dict[str, Any], **kw: Any) -> dict[str, Any]:
    return uwb_pipeline.run_uwb_optitrack(
        tmp_path / "run",
        external_root=s["root"],
        manifests=tmp_path / "manifests",
        metadata=kw.pop("metadata", s["meta"]),
        source_dir=s["archive"].parent,
        quarantine=tmp_path / "q",
        source_id="synthetic-uwb",
        store_root=tmp_path / "store",
        staging=tmp_path / "stage",
        **kw,
    )


def _status(r: dict[str, Any]) -> dict[str, str]:
    return {s["name"]: s["status"] for s in r["stages"]}


def test_comparable_synthetic_source_reaches_metrics(tmp_path: Path) -> None:
    s = _setup(tmp_path, uwb_csv(noise=0.01), uwb_csv(sessions=2, noise=0.01))
    r = _run(tmp_path, s)
    assert r["outcome"] == "completed" and set(_status(r).values()) == {"passed"}
    m = r["error_metrics"]["walk"]
    assert m["median"] == pytest.approx(0.01 * np.sqrt(np.log(4)), rel=0.2)  # Rayleigh median
    assert "not ground truth" in r["error_metrics_note"]


def test_lagged_or_offset_source_blocks_metrics(tmp_path: Path) -> None:
    s = _setup(tmp_path, uwb_csv(shift=(0.0, -0.27)), uwb_csv(sessions=2, lag=3))
    r = _run(tmp_path, s)
    st = _status(r)
    assert st["comparability"] == st["error_metrics"] == "blocked" and "error_metrics" not in r
    assert r["comparability"]["data_modified"] is False
    assert r["outcome"] == "completed_with_blocked_gates: comparability, error_metrics"
    assert "Not computed" in (tmp_path / "run" / "report.md").read_text()


def test_rights_without_inspection_never_read_archive(tmp_path: Path) -> None:
    s = _setup(
        tmp_path,
        uwb_csv(),
        uwb_csv(sessions=2),
        local_inspection_permitted=False,
        canonical_admission="pending",
        blockers=["synthetic"],
    )
    s["archive"].chmod(0o000)
    try:
        r = _run(tmp_path, s)
    finally:
        s["archive"].chmod(0o644)
    assert _status(r)["data_integrity"] == "blocked" and r["outcome"] == "stopped_at_blocked_gate"


def test_relicensed_record_fails_before_archive(tmp_path: Path) -> None:
    s = _setup(tmp_path, uwb_csv(), uwb_csv(sessions=2))
    s["archive"].chmod(0o000)
    try:
        r = _run(tmp_path, s, metadata=relicensed(s["meta"], "cc-by-nc-4.0"))
    finally:
        s["archive"].chmod(0o644)
    assert _status(r) == {"pinned_source": "passed", "rights_verification": "failed"}


def test_committed_manifest_must_agree_with_lock(tmp_path: Path) -> None:
    s = _setup(tmp_path, uwb_csv(), uwb_csv(sessions=2))
    from dynamisbench_soccer.external import rights as rg
    from dynamisbench_soccer.external import zenodo

    check = rg.verify_zenodo(rights(), s["lock"], s["meta"])
    m = zenodo.to_manifest(
        s["lock"], rights(), check, title="t", citation="c", privacy_notes="p", handling_notes="h"
    )
    m = m.model_copy(update={"files": [m.files[0].model_copy(update={"sha256": "0" * 64})]})
    m.save(tmp_path / "manifests" / "synthetic-uwb.json")
    assert _status(_run(tmp_path, s))["ingestion"] == "failed"


def test_cli_uses_saved_metadata_and_fails_closed_offline(
    tmp_path: Path, repo_root: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    meta = tmp_path / "meta.json"
    meta.write_text(json.dumps({"id": 3452007}))  # malformed: rights cannot be verified
    code = main(
        [
            "external",
            "run",
            "uwb-optitrack",
            "--root",
            str(repo_root / "data/external"),
            "--metadata",
            str(meta),
            "--out",
            str(tmp_path / "o"),
        ]
    )
    out = capsys.readouterr().out
    assert code == 1 and "rights_verification" in out and "malformed" in out
    report = json.loads((tmp_path / "o" / "report.json").read_text())
    assert "data_integrity" not in {s["name"] for s in report["stages"]}


# --- the committed real-source report -------------------------------------------------------


def test_committed_report_blocks_metrics_honestly(repo_root: Path) -> None:
    base = repo_root / "docs/external-data/reports/uwb-optitrack-comparability-report"
    text = base.with_suffix(".json").read_text(encoding="utf-8")
    r = json.loads(text)
    assert dumps(r) == text and r["records"]["committed_default_root"] is True
    assert r["source"]["md5"] == "d5221f12a5f093eacc21db797c0ac13a"
    st = _status(r)
    assert st["rights_verification"] == st["ingestion"] == "passed"
    assert st["comparability"] == st["error_metrics"] == "blocked" and "error_metrics" not in r
    sessions = r["comparability"]["sessions"]
    assert len(sessions) == 10 and not any(x["comparable"] for x in sessions)
    lags = {(x["scenario"], x["session_id"]): x["best_row_lag"] for x in sessions}
    assert [lags[("walk", i)] for i in range(1, 6)] == [1, 1, 0, 0, 0]
    assert [lags[("trolley", i)] for i in range(1, 6)] == [3, 3, 3, 4, 3]
    assert sum(not x["temporal_ok"] for x in sessions) == 7
    assert all(not x["frame_ok"] for x in sessions)
    walk = [x for x in sessions if x["scenario"] == "walk"]
    trolley = [x for x in sessions if x["scenario"] == "trolley"]
    assert all(x["translation"][1] == pytest.approx(-0.27, abs=0.03) for x in walk)
    assert all(x["scale"] == pytest.approx(0.97, abs=0.01) for x in trolley)
    assert r["comparability"]["range_sentinels"] == {"walk": 4, "trolley": 0}
    assert not any(
        g["geometrically_consistent"] for g in r["comparability"]["anchor_geometry"]["trolley"]
    )
    assert r["comparability"]["data_modified"] is False
    assert all(v["readme_counts_header_line"] for v in r["readme_reconciliation"].values())
    assert all(
        p["z_coordinate"] == "not_recorded" and p["soccer_task"] is False for p in r["profiles"]
    )
    assert "doi:10.5281/zenodo.3452007" in r["attribution"]
