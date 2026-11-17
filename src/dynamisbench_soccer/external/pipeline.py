"""End-to-end external-source pipeline for ``shorts-laser`` (RES-720).

pinned source -> evidence integrity (license/docs only) -> rights verification ->
data integrity (measurement bytes, only if rights permit inspection) -> ingestion
(canonical only if rights permit, else quarantine) -> strict parse -> profile ->
schema-v2 conversion (where valid) -> QC -> numerical diagnostics -> report.

Each stage records ``passed``, ``blocked`` (a documented gate refused it) or
``failed``. A blocked gate never falls through to a weaker path. The only fallback
is quarantine inspection, and only when the rights record expressly permits it.
"""

from __future__ import annotations

import contextlib
import json
import platform
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from dynamisbench_soccer import __version__
from dynamisbench_soccer.data import store
from dynamisbench_soccer.data.store import IntegrityError
from dynamisbench_soccer.external import (
    acquire,
    diagnostics,
    rights,
    shorts_laser,
    sourceqc,
    timebase,
)
from dynamisbench_soccer.external.acquire import Opener
from dynamisbench_soccer.external.models import RightsStatus, load_lock, load_rights
from dynamisbench_soccer.qc import spec
from dynamisbench_soccer.signals import io

PIPELINE_VERSION = "1.0.0"
DEFAULT_ROOT = Path("data/external")
StageStatus = Literal["passed", "blocked", "failed"]


@dataclass
class Stage:
    name: str
    status: StageStatus
    detail: str
    evidence: dict[str, Any] = field(default_factory=dict)


class PipelineStop(RuntimeError):
    pass


@dataclass
class Run:
    stages: list[Stage] = field(default_factory=list)

    def add(self, name: str, status: StageStatus, detail: str, **evidence: Any) -> None:
        self.stages.append(Stage(name, status, detail, evidence))
        if status == "failed":
            raise PipelineStop(f"{name}: {detail}")


def run_shorts_laser(
    out_dir: Path,
    *,
    external_root: Path = Path("data/external"),
    rules_path: Path = Path("qc/rules.toml"),
    source_dir: Path | None = None,
    quarantine: Path | None = None,
    opener: Opener | None = None,
    expected_ref: str = shorts_laser.PINNED_REF,
    expected_sha256: str = shorts_laser.PINNED_SHA256,
    source_id: str = shorts_laser.SOURCE_ID,
    store_root: Path | None = None,
    staging: Path | None = None,
) -> dict[str, Any]:
    """Run every stage, write ``report.json`` + ``report.md`` to ``out_dir``, return it."""
    run = Run()
    report: dict[str, Any] = {
        "kind": "dbs-external-diagnostic-report",
        "source_id": source_id,
        "pipeline": f"dbs-external@{PIPELINE_VERSION}",
        "adapter": f"{shorts_laser.ADAPTER_ID}@{shorts_laser.ADAPTER_VERSION}",
        "environment": {
            "dynamisbench_soccer": __version__,
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
    }
    with contextlib.suppress(PipelineStop):
        _stages(
            run,
            report,
            out_dir,
            external_root,
            rules_path,
            source_dir,
            quarantine,
            opener,
            expected_ref,
            expected_sha256,
            source_id,
            store_root,
            staging,
        )
    report["stages"] = [asdict(s) for s in run.stages]
    report["outcome"] = _outcome(run)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(dumps(report), encoding="utf-8")
    from dynamisbench_soccer.external.report import render

    (out_dir / "report.md").write_text(render(report), encoding="utf-8")
    return report


def _outcome(run: Run) -> str:
    statuses = {s.name: s.status for s in run.stages}
    if "failed" in statuses.values():
        return "failed"
    if statuses.get("diagnostics") == "passed":
        blocked = [n for n, st in statuses.items() if st == "blocked"]
        return "completed" if not blocked else "completed_with_blocked_gates: " + ", ".join(blocked)
    return "stopped_at_blocked_gate"


def _stages(
    run: Run,
    report: dict[str, Any],
    out_dir: Path,
    external_root: Path,
    rules_path: Path,
    source_dir: Path | None,
    quarantine: Path | None,
    opener: Opener | None,
    expected_ref: str,
    expected_sha256: str,
    source_id: str,
    store_root: Path | None,
    staging: Path | None,
) -> None:
    # 1. pinned source, bound to the adapter's version
    lock, record = load_lock(source_id, external_root), load_rights(source_id, external_root)
    data_entry = lock.file(shorts_laser.DATA_PATH)
    pin = {
        "repository": lock.repository,
        "pinned_ref": lock.pinned_ref,
        "data_path": data_entry.path,
        "data_sha256": data_entry.sha256,
        "data_size_bytes": data_entry.size_bytes,
        "git_blob_sha1": data_entry.git_blob_sha1,
    }
    report["source"] = pin
    report["records"] = {
        "committed_default_root": external_root.resolve() == DEFAULT_ROOT.resolve(),
        "note": "only records committed under data/external/ are reviewed in Git; any other "
        "root is an unreviewed local override",
    }
    if lock.pinned_ref != expected_ref or data_entry.sha256 != expected_sha256:
        run.add("pinned_source", "failed", "lock does not pin the adapter's source version", **pin)
    run.add("pinned_source", "passed", "commit-pinned lock matches adapter version", **pin)

    # 2. evidence integrity: license, documentation and scripts only, never data bytes
    try:
        files = acquire.acquire(lock, source_dir=source_dir, root=quarantine, opener=opener)
    except IntegrityError as exc:
        run.add("evidence_integrity", "failed", str(exc))
    run.add(
        "evidence_integrity",
        "passed",
        "size, SHA-256 and Git blob SHA-1 verified for every evidence file; "
        "measurement data not yet acquired",
        files={p: lock.file(p).sha256 for p in sorted(files)},
    )

    # 3. rights, decided before any measurement byte is acquired or read
    check = rights.verify(record, lock, files)
    report["rights"] = {
        "license_id": record.license_id,
        "license_evidence_verified": check.license_evidence_verified,
        "measurement_data_rights": str(record.measurement_data_rights),
        "participant_consent": record.participant_consent,
        "minors_possible": record.minors_possible,
        "canonical_admission": str(check.canonical_admission),
        "redistribution_allowed": check.redistribution_allowed,
        "local_inspection_permitted": check.local_inspection_permitted,
        "blockers": list(check.blockers),
        "record_digest": check.record_digest,
    }
    if not check.license_evidence_verified:
        run.add("rights_verification", "failed", "; ".join(check.problems))
    if check.canonical_admission is RightsStatus.VERIFIED:
        run.add("rights_verification", "passed", "license evidence verified; admission permitted")
    else:
        run.add(
            "rights_verification",
            "blocked",
            "license evidence verified; canonical admission " + str(check.canonical_admission),
            blockers=check.blockers,
        )
    if not check.local_inspection_permitted:
        run.add(
            "data_integrity",
            "blocked",
            "rights record does not permit local inspection; measurement-data bytes were "
            "not acquired, opened or hashed",
            data_bytes_acquired=False,
        )
        return

    # 4. data integrity, then ingestion
    try:
        files |= acquire.acquire_data(
            lock, check, source_dir=source_dir, root=quarantine, opener=opener
        )
    except IntegrityError as exc:
        run.add("data_integrity", "failed", str(exc))
    run.add(
        "data_integrity",
        "passed",
        "measurement-data size, SHA-256 and Git blob SHA-1 verified after the rights gate",
        files={data_entry.path: data_entry.sha256},
    )
    if check.canonical_admission is RightsStatus.VERIFIED:
        manifest = acquire.to_manifest(
            lock,
            record,
            check,
            title=f"{source_id} (shorts)",
            citation=f"{lock.repository}@{lock.pinned_ref}",
        )
        root = files[shorts_laser.DATA_PATH].parents[len(Path(shorts_laser.DATA_PATH).parts) - 1]
        staged = acquire.admit_canonical(manifest, root, store_root=store_root, staging=staging)
        blob = store.blob_path(data_entry.sha256, store_root).resolve()
        if (staged / shorts_laser.DATA_PATH).resolve() != blob:
            run.add("ingestion", "failed", "staged link does not resolve to the store blob")
        data_path = blob
        run.add(
            "ingestion",
            "passed",
            "canonical store ingest, verify and read-only stage",
            manifest=manifest.key,
        )
    else:
        data_path = files[shorts_laser.DATA_PATH]
        run.add(
            "ingestion",
            "blocked",
            "canonical store refused by rights gate; read-only quarantine copy inspected "
            "under the license grant only",
            mode="quarantine",
        )

    # 5-6. strict parse and profile
    try:
        table = shorts_laser.parse(data_path, expected_sha256)
    except (shorts_laser.SourceIntegrityError, shorts_laser.SourceFormatError) as exc:
        run.add("safe_parse", "failed", str(exc))
    run.add(
        "safe_parse",
        "passed",
        "hash-bound strict parse; exact header, ASCII, finite values",
        n_rows=table.n_rows,
    )
    report["profile"] = shorts_laser.profile(table)
    report["signal_stages"] = [
        {
            "source_column": c.source_column,
            "channel": c.channel,
            "stage": str(c.stage),
            "quantity": str(c.quantity),
            "meaning": c.meaning,
        }
        for c in shorts_laser.SOURCE_COLUMNS
    ]
    report["package_derived_not_ingested"] = dict(shorts_laser.PACKAGE_DERIVED_NOT_INGESTED)
    run.add("profile", "passed", "Gate-2 shape audit recorded")

    # 7. schema-v2 conversion (time base measured once; reported, never applied)
    time_base = timebase.assess(table)
    outcome = shorts_laser.convert(
        table,
        lock,
        files,
        admission=check,
        time_base=time_base,
    )
    report["conversion"] = {
        "converted": outcome.converted,
        "blockers": outcome.blockers,
        "determinations": [{"field": f, "decision": d} for f, d in outcome.determinations],
    }
    rules = spec.load(rules_path)
    if outcome.bundle is not None:
        stem = out_dir / "bundles" / outcome.bundle.meta.bundle_id
        parquet, _ = io.write(outcome.bundle, stem)
        bundle = io.read(stem)  # verifies Parquet SHA-256, schema and counts
        batches = sum(len(b["time"]) for b in io.iter_batches(stem))
        report["conversion"]["bundle"] = {
            "bundle_id": bundle.meta.bundle_id,
            "data_sha256": bundle.meta.data_sha256,
            "rows_read_back": bundle.meta.n_rows,
            "rows_streamed": batches,
            "parquet_name": parquet.name,
        }
        run.add("schema_v2_conversion", "passed", "strict v2 bundle written and read back")
        results = spec.run(bundle, rules)
        report["bundle_qc"] = {
            "verdict": spec.verdict(results),
            "results": [
                {
                    "rule_id": r.rule_id,
                    "name": r.name,
                    "severity": str(r.severity),
                    "status": str(r.status),
                    "detail": r.detail,
                    "values": dict(r.values),
                }
                for r in results
            ],
        }
    else:
        run.add(
            "schema_v2_conversion",
            "blocked",
            "; ".join(outcome.blockers),
            blockers=outcome.blockers,
        )

    # 8. QC
    checks = sourceqc.run(table, rules, time_base)
    report["source_qc"] = {
        "verdict": sourceqc.verdict(checks),
        "rules_version": rules.version,
        "results": [
            {**asdict(c), "severity": str(c.severity), "status": str(c.status)} for c in checks
        ],
    }
    run.add(
        "qc",
        "passed",
        f"source QC verdict {report['source_qc']['verdict']}"
        + (
            f"; bundle QC verdict {report['bundle_qc']['verdict']}"
            if "bundle_qc" in report
            else "; bundle QC not run (no v2 bundle)"
        ),
    )

    # 9. diagnostics
    report["diagnostics"] = diagnostics.compute(table, time_base)
    run.add("diagnostics", "passed", "pipeline-derived numerical diagnostics computed")


def _clean(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_clean(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def dumps(report: dict[str, Any]) -> str:
    return json.dumps(_clean(report), indent=2, sort_keys=True, allow_nan=False) + "\n"
