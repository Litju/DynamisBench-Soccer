"""End-to-end pipeline for ``uwb-optitrack`` (Zenodo 3452007, RES-721).

pinned_source -> rights_verification (live or saved record metadata) -> data_integrity
(archive, only after rights permit inspection) -> archive_audit -> ingestion (canonical
store if admitted, else quarantine) -> safe_parse (in-memory members) -> profile
(planar validation profile v1) -> comparability -> error_metrics (only if every gating
comparability check passes) -> report.{json,md}
"""

from __future__ import annotations

import contextlib
import platform
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from dynamisbench_soccer import __version__
from dynamisbench_soccer.data import store
from dynamisbench_soccer.data.manifest import DatasetManifest
from dynamisbench_soccer.data.store import IntegrityError
from dynamisbench_soccer.external import acquire, comparability, rights, uwb_optitrack, zenodo
from dynamisbench_soccer.external.acquire import Opener
from dynamisbench_soccer.external.models import RightsStatus
from dynamisbench_soccer.external.pipeline import DEFAULT_ROOT, PipelineStop, Run, dumps
from dynamisbench_soccer.external.zenodo_models import load_zenodo_lock, load_zenodo_rights

PIPELINE_VERSION = "1.0.0"
DEFAULT_THRESHOLDS = comparability.Thresholds()
MANIFESTS = Path("data/manifests")
ATTRIBUTION = (
    "Raza U, Khan A, Kou R, Farnham T, Premalal T, Stanoev A, Thompson W (2019). Dataset: "
    "Indoor Localization with Narrow-band, Ultra-Wideband, and Motion Capture Systems. "
    "Zenodo. doi:10.5281/zenodo.3452007. CC BY 4.0."
)


def run_uwb_optitrack(
    out_dir: Path,
    *,
    external_root: Path = DEFAULT_ROOT,
    manifests: Path = MANIFESTS,
    metadata: dict[str, Any] | None = None,
    source_dir: Path | None = None,
    quarantine: Path | None = None,
    opener: Opener | None = None,
    source_id: str = uwb_optitrack.SOURCE_ID,
    store_root: Path | None = None,
    staging: Path | None = None,
    thresholds: comparability.Thresholds = DEFAULT_THRESHOLDS,
) -> dict[str, Any]:
    run = Run()
    report: dict[str, Any] = {
        "kind": "dbs-external-planar-validation-report",
        "source_id": source_id,
        "pipeline": f"dbs-external-uwb@{PIPELINE_VERSION}",
        "adapter": f"{uwb_optitrack.ADAPTER_ID}@{uwb_optitrack.ADAPTER_VERSION}",
        "environment": {
            "dynamisbench_soccer": __version__,
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "records": {"committed_default_root": external_root.resolve() == DEFAULT_ROOT.resolve()},
        "attribution": ATTRIBUTION,
        "thresholds": asdict(thresholds),
    }
    with contextlib.suppress(PipelineStop):
        _stages(
            run,
            report,
            external_root,
            manifests,
            metadata,
            source_dir,
            quarantine,
            opener,
            source_id,
            store_root,
            staging,
            thresholds,
        )
    report["stages"] = [asdict(s) for s in run.stages]
    statuses = {s.name: s.status for s in run.stages}
    if "failed" in statuses.values():
        report["outcome"] = "failed"
    elif "comparability" in statuses:
        blocked = [n for n, st in statuses.items() if st == "blocked"]
        report["outcome"] = (
            "completed" if not blocked else ("completed_with_blocked_gates: " + ", ".join(blocked))
        )
    else:
        report["outcome"] = "stopped_at_blocked_gate"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(dumps(report), encoding="utf-8")
    (out_dir / "report.md").write_text(render(report), encoding="utf-8")
    return report


def _stages(
    run: Run,
    report: dict[str, Any],
    external_root: Path,
    manifests: Path,
    metadata: dict[str, Any] | None,
    source_dir: Path | None,
    quarantine: Path | None,
    opener: Opener | None,
    source_id: str,
    store_root: Path | None,
    staging: Path | None,
    th: comparability.Thresholds,
) -> None:
    lock = load_zenodo_lock(source_id, external_root)
    record = load_zenodo_rights(source_id, external_root)
    (archive,) = lock.archives
    report["source"] = {
        "record_id": lock.record_id,
        "doi": lock.doi,
        "archive": archive.key,
        "md5": archive.md5,
        "sha256": archive.sha256,
        "size_bytes": archive.size_bytes,
    }
    run.add("pinned_source", "passed", "Zenodo record, archive and every member pinned")

    # rights: public metadata only; no measurement byte is touched before this passes
    try:
        meta = metadata if metadata is not None else zenodo.fetch_metadata(lock, opener=opener)
    except (IntegrityError, ValueError) as exc:
        run.add("rights_verification", "failed", f"record metadata unavailable: {exc}")
    check = rights.verify_zenodo(record, lock, meta)
    report["rights"] = {
        "license_id": record.license_id,
        "record_license": lock.license_id,
        "license_evidence_verified": check.license_evidence_verified,
        "measurement_data_rights": str(record.measurement_data_rights),
        "participant_consent": record.participant_consent,
        "canonical_admission": str(check.canonical_admission),
        "redistribution_allowed": check.redistribution_allowed,
        "problems": list(check.problems),
        "blockers": list(check.blockers),
        "record_digest": check.record_digest,
    }
    if not check.license_evidence_verified:
        run.add("rights_verification", "failed", "; ".join(check.problems))
    if check.canonical_admission is RightsStatus.VERIFIED:
        run.add("rights_verification", "passed", "record license cc-by-4.0 matches pinned digest")
    else:
        run.add(
            "rights_verification",
            "blocked",
            "canonical admission " + str(check.canonical_admission),
            blockers=list(check.blockers),
        )
    if not check.local_inspection_permitted:
        run.add("data_integrity", "blocked", "rights do not permit inspection; archive not read")
        return

    try:
        path = zenodo.acquire_archive(
            lock, archive.key, check, source_dir=source_dir, root=quarantine, opener=opener
        )
    except IntegrityError as exc:
        run.add("data_integrity", "failed", str(exc))
    run.add("data_integrity", "passed", "published MD5, SHA-256 and size verified")
    problems = zenodo.audit_archive(archive, path)
    if problems:
        run.add("archive_audit", "failed", "; ".join(problems))
    run.add(
        "archive_audit",
        "passed",
        f"{len(archive.members)} pinned and {len(archive.excluded)} excluded members; no "
        "unsafe paths, links or bombs",
    )

    if check.canonical_admission is RightsStatus.VERIFIED:
        manifest = zenodo.to_manifest(
            lock, record, check, title="", citation="", privacy_notes="", handling_notes=""
        )
        committed = manifests / f"{lock.source_id}.json"
        if committed.is_file():
            pinned = DatasetManifest.load(committed)
            if [(f.path, f.sha256, f.size_bytes) for f in pinned.files] != [
                (f.path, f.sha256, f.size_bytes) for f in manifest.files
            ] or pinned.usage is not manifest.usage:
                run.add("ingestion", "failed", "committed manifest disagrees with the lock")
            manifest = pinned
        staged = acquire.admit_canonical(
            manifest, path.parent, store_root=store_root, staging=staging
        )
        blob = store.blob_path(archive.sha256, store_root).resolve()
        if (staged / archive.key).resolve() != blob:
            run.add("ingestion", "failed", "staged link does not resolve to the store blob")
        path = blob
        run.add(
            "ingestion",
            "passed",
            "canonical store ingest, verify and read-only stage "
            f"({manifest.key}, {manifest.usage})",
        )
    else:
        run.add("ingestion", "blocked", "canonical store refused by rights gate; quarantine only")

    tables = []
    names = [m.name for m in archive.members]
    try:
        for scenario in uwb_optitrack.SCENARIO_SUFFIX:
            member = uwb_optitrack.member_for(scenario, names)
            data = zenodo.read_member(lock, archive.key, path, member, check)
            tables.append(
                uwb_optitrack.parse(scenario, member, archive.member(member).sha256, data)
            )
    except (IntegrityError, uwb_optitrack.UwbFormatError) as exc:
        run.add("safe_parse", "failed", str(exc))
    run.add("safe_parse", "passed", "in-memory strict parse of both UWB/OptiTrack members")
    report["readme_reconciliation"] = {
        t.scenario: uwb_optitrack.readme_reconciliation(t) for t in tables
    }
    report["profiles"] = [uwb_optitrack.profile(t).model_dump(mode="json") for t in tables]
    run.add("profile", "passed", "dbs-planar-validation@1: 2D, no Z, no soccer task, no clock")

    checks: list[comparability.SessionCheck] = []
    anchors: dict[str, Any] = {}
    for t in tables:
        for sid in np.unique(t.session):
            m = t.session == sid
            checks.append(
                comparability.check_session(
                    t.scenario, int(sid), t.device[m], t.reference[m], t.ranges[m], th
                )
            )
        anchors[t.scenario] = comparability.anchor_geometry(t.reference, t.ranges)
    report["comparability"] = {
        "sessions": [comparability.as_dict(c) for c in checks],
        "anchor_geometry": anchors,
        "range_sentinels": {t.scenario: int(np.sum(t.ranges < 0)) for t in tables},
        "data_modified": False,
        "lag_or_transform_applied": False,
    }
    failing = [c for c in checks if not c.comparable]
    if failing:
        run.add(
            "comparability",
            "blocked",
            f"{len(failing)}/{len(checks)} sessions not comparable "
            f"(temporal {sum(not c.temporal_ok for c in checks)}, "
            f"frame {sum(not c.frame_ok for c in checks)})",
        )
        run.add(
            "error_metrics",
            "blocked",
            "not computed: UWB and OptiTrack are not shown to "
            "share a time base and frame; a difference would not be device error",
        )
        return
    run.add(
        "comparability", "passed", "all sessions: zero row lag and identity frame within tolerance"
    )
    report["error_metrics"] = {
        t.scenario: comparability.planar_disagreement(t.device, t.reference) for t in tables
    }
    report["error_metrics_note"] = (
        "same-row UWB minus OptiTrack in source units (undeclared); OptiTrack is a reference "
        "measurement, not ground truth; no clock evidence beyond depositor row order"
    )
    run.add("error_metrics", "passed", "planar disagreement computed after comparability passed")


def _f(x: Any, d: int = 4) -> str:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return str(x)
    return str(x) if isinstance(x, int) else f"{x:.{d}g}"


def render(r: dict[str, Any]) -> str:
    out: list[str] = []
    a = out.append
    a(f"# External-source comparability report: `{r['source_id']}`")
    a("")
    a("<!-- GENERATED by `dbs external run uwb-optitrack`; regenerate rather than edit. -->")
    a("")
    a(
        f"**Outcome:** `{r['outcome']}` · pipeline `{r['pipeline']}` · adapter `{r['adapter']}` · "
        + ", ".join(f"{k} {v}" for k, v in r["environment"].items())
    )
    a("")
    a("> A 2D instrument-comparability study of **non-soccer** testbed data. OptiTrack is a")
    a("> reference measurement, not ground truth. Nothing here qualifies a device or a task.")
    a("")
    a(f"**Attribution (CC BY 4.0):** {r['attribution']}")
    a("")
    if not r["records"]["committed_default_root"]:
        a("> **WARNING: UNREVIEWED RECORDS.** Records were loaded from a non-committed root.")
        a("")
    a("## Stages")
    a("")
    a("| Stage | Status | Detail |")
    a("| -- | -- | -- |")
    for s in r["stages"]:
        a(f"| {s['name']} | **{s['status']}** | {s['detail']} |")
    a("")
    if "source" in r:
        s = r["source"]
        a(
            f"Record {s['record_id']} (`{s['doi']}`) · `{s['archive']}` · {s['size_bytes']} B · "
            f"MD5 `{s['md5']}` · SHA-256 `{s['sha256']}`"
        )
        a("")
    if "readme_reconciliation" in r:
        a("## README sample counts")
        a("")
        for k, v in r["readme_reconciliation"].items():
            a(
                f"* {k}: README states {v['readme_stated']}, data rows {v['data_rows']} "
                f"(README count includes the header line: {v['readme_counts_header_line']})"
            )
        a("")
    if "comparability" in r:
        c = r["comparability"]
        t = r["thresholds"]
        a("## Comparability (measured, never applied)")
        a("")
        a(
            f"Gates: best row lag ≤ {t['max_row_lag']} (search ±{t['lag_search_rows']}); "
            f"|scale - 1| ≤ {t['max_scale_deviation']}; |rotation| ≤ {t['max_rotation_deg']}°; "
            f"|translation| ≤ {t['max_translation']} (source units). Data modified: "
            f"{c['data_modified']}; lag or transform applied: {c['lag_or_transform_applied']}."
        )
        a("")
        a(
            "| Scenario | Session | n | Best row lag | Median offset @0 / @best | Scale | "
            "Rotation ° | Translation | Similarity residual median | Range sentinels | Comparable |"
        )
        a("| -- | --: | --: | --: | -- | --: | --: | -- | --: | --: | -- |")
        for x in c["sessions"]:
            tx, ty = x["translation"]
            a(
                f"| {x['scenario']} | {x['session_id']} | {x['n_rows']} | {x['best_row_lag']} | "
                f"{_f(x['median_offset_at_zero_lag'])} / {_f(x['median_offset_at_best_lag'])} | "
                f"{_f(x['scale'])} | {_f(x['rotation_deg'], 3)} | ({_f(tx, 3)}, {_f(ty, 3)}) | "
                f"{_f(x['similarity_residual_median'])} | {x['range_sentinels']} | "
                f"**{x['comparable']}** |"
            )
        a("")
        a(
            "Median offsets are **alignment diagnostics** used to find the lag. They are not "
            "accuracy, because the systems are not shown to be comparable."
        )
        a("")
        a("### Anchor-range geometry (flag): fit d² = |p - a|² + h² to OptiTrack positions")
        a("")
        a("| Scenario | Anchor | x | y | h² | median abs(range residual) | Consistent (h² ≥ 0) |")
        a("| -- | --: | --: | --: | --: | --: | -- |")
        for scen, rows in c["anchor_geometry"].items():
            for g in rows:
                a(
                    f"| {scen} | {int(g['anchor'])} | {_f(g['x'], 3)} | {_f(g['y'], 3)} | "
                    f"{_f(g['h2'], 3)} | {_f(g['range_residual_median_abs'], 3)} | "
                    f"{g['geometrically_consistent']} |"
                )
        a("")
    a("## Error metrics")
    a("")
    if "error_metrics" in r:
        a(r["error_metrics_note"] + ".")
        a("")
        for scen, m in r["error_metrics"].items():
            a(f"* {scen}: " + ", ".join(f"{k} {_f(v)}" for k, v in m.items()))
    else:
        a(
            "**Not computed.** Comparability was not established, so a UWB - OptiTrack difference "
            "would mix device error with time, frame, scale and reference-point mismatch."
        )
    return "\n".join(out).rstrip() + "\n"
