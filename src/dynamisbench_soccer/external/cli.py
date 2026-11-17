"""``dbs external ...`` commands."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dynamisbench_soccer.data import paths
from dynamisbench_soccer.data.store import IntegrityError
from dynamisbench_soccer.external import (
    acquire,
    pipeline,
    planar_profile,
    rights,
    shorts_laser,
    uwb_optitrack,
    uwb_pipeline,
    zenodo,
)
from dynamisbench_soccer.external.models import (
    EXTERNAL_ROOT,
    RightsRecord,
    SourceLock,
    load_lock,
    load_rights,
    source_ids,
)
from dynamisbench_soccer.external.zenodo_models import (
    ZenodoLock,
    load_zenodo_lock,
    load_zenodo_rights,
    lock_kind,
)


def _cmd_validate(args: argparse.Namespace) -> int:
    root = Path(args.root)
    failed = 0
    ids = source_ids(root)
    for sid in ids:
        try:
            lock, record = _load(sid, root)
            if record.pinned_ref != lock.pinned_ref:
                raise ValueError("rights record and lock pin different sources")
            print(f"OK   {sid} @ {lock.pinned_ref[:14]} admission={record.canonical_admission}")
        except Exception as exc:  # report every invalid source
            failed += 1
            print(f"FAIL {sid}: {exc}", file=sys.stderr)
    print(f"external: {len(ids)} source(s): {'OK' if not failed else f'{failed} invalid'}")
    return 1 if failed else 0


def _cmd_rights(args: argparse.Namespace) -> int:
    _, record = _load(args.source_id, Path(args.root))
    redistribution = "allowed" if record.redistribution_allowed else "NOT allowed"
    print(
        f"{record.source_id}: license {record.license_id}; "
        f"measurement-data rights {record.measurement_data_rights}; "
        f"canonical admission {record.canonical_admission}; redistribution {redistribution}"
    )
    for b in record.blockers:
        print(f"  BLOCKER {b}")
    return 0


def _load(sid: str, root: Path) -> tuple[SourceLock | ZenodoLock, RightsRecord]:
    """Dispatch on the committed lock kind (GitHub commit or Zenodo record)."""
    if lock_kind(sid, root) == "dbs-zenodo-source-lock":
        return load_zenodo_lock(sid, root), load_zenodo_rights(sid, root)
    return load_lock(sid, root), load_rights(sid, root)


def _acquire_zenodo(args: argparse.Namespace, root: Path) -> int:
    lock, record = load_zenodo_lock(args.source_id, root), load_zenodo_rights(args.source_id, root)
    source_dir = Path(args.source_dir) if args.source_dir else None
    try:
        # Rights evidence (public record metadata) first; archive bytes only afterwards.
        if args.metadata:
            metadata = json.loads(Path(args.metadata).read_text(encoding="utf-8"))
        else:
            metadata = zenodo.fetch_metadata(lock)
        check = rights.verify_zenodo(record, lock, metadata)
        archives = [
            zenodo.acquire_archive(lock, a.key, check, source_dir=source_dir) for a in lock.archives
        ]
        problems = [
            p
            for a, path in zip(lock.archives, archives, strict=True)
            for p in zenodo.audit_archive(a, path)
        ]
    except (IntegrityError, rights.RightsBlocked) as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        return 1
    for p in problems:
        print(f"FAIL archive audit: {p}", file=sys.stderr)
    for a, path in zip(lock.archives, archives, strict=True):
        print(f"verified md5 {a.md5} sha256 {a.sha256} {a.size_bytes:>8} {path}")
    print(f"canonical admission {check.canonical_admission}")
    return 1 if problems else 0


def _cmd_acquire(args: argparse.Namespace) -> int:
    root = Path(args.root)
    if lock_kind(args.source_id, root) == "dbs-zenodo-source-lock":
        return _acquire_zenodo(args, root)
    lock, record = load_lock(args.source_id, root), load_rights(args.source_id, root)
    source_dir = Path(args.source_dir) if args.source_dir else None
    try:
        # Evidence first; measurement bytes only after the rights gate permits it.
        files = acquire.acquire(lock, source_dir=source_dir)
        check = rights.verify(record, lock, files)
        files |= acquire.acquire_data(lock, check, source_dir=source_dir)
    except (IntegrityError, rights.RightsBlocked) as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        return 1
    for entry in lock.files:
        print(f"verified {entry.sha256} {entry.size_bytes:>8} {files[entry.path]}")
    print(f"quarantine only; canonical admission {check.canonical_admission}")
    return 0


def _cmd_run_uwb(args: argparse.Namespace) -> int:
    root = Path(args.root)
    if root.resolve() != EXTERNAL_ROOT.resolve():
        print(
            f"WARNING records root {args.root} is not the committed, reviewed "
            f"{EXTERNAL_ROOT}; results are an unreviewed local override",
            file=sys.stderr,
        )
    metadata = (
        json.loads(Path(args.metadata).read_text(encoding="utf-8")) if args.metadata else None
    )
    out = Path(args.out) if args.out else paths.runs_root() / "external" / args.source_id
    report = uwb_pipeline.run_uwb_optitrack(
        out,
        external_root=root,
        metadata=metadata,
        source_dir=Path(args.source_dir) if args.source_dir else None,
    )
    for stage in report["stages"]:
        print(f"{stage['status']:<8} {stage['name']:<22} {stage['detail']}")
    print(f"outcome: {report['outcome']}")
    for flag, name in ((args.report_md, "report.md"), (args.report_json, "report.json")):
        if flag:
            Path(flag).parent.mkdir(parents=True, exist_ok=True)
            Path(flag).write_text((out / name).read_text(encoding="utf-8"), encoding="utf-8")
            print(f"wrote {flag}")
    return 1 if report["outcome"] == "failed" else 0


def _cmd_planar_schema(args: argparse.Namespace) -> int:
    text = planar_profile.json_schema()
    target = Path(args.out)
    if args.write:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        print(f"wrote {target}")
    elif args.check:
        if not target.is_file() or target.read_text(encoding="utf-8") != text:
            print(f"FAIL {target} out of date; run `dbs external planar-schema --write`")
            return 1
        print(f"{target} up to date")
    else:
        sys.stdout.write(text)
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    if args.source_id == uwb_optitrack.SOURCE_ID:
        return _cmd_run_uwb(args)
    if args.source_id != shorts_laser.SOURCE_ID:
        print(f"FAIL no adapter for {args.source_id!r}", file=sys.stderr)
        return 1
    if Path(args.root).resolve() != EXTERNAL_ROOT.resolve():
        print(
            f"WARNING records root {args.root} is not the committed, reviewed "
            f"{EXTERNAL_ROOT}; results are an unreviewed local override",
            file=sys.stderr,
        )
    out = Path(args.out) if args.out else paths.runs_root() / "external" / args.source_id
    report = pipeline.run_shorts_laser(
        out,
        external_root=Path(args.root),
        rules_path=Path(args.rules),
        source_dir=Path(args.source_dir) if args.source_dir else None,
    )
    for stage in report["stages"]:
        print(f"{stage['status']:<8} {stage['name']:<22} {stage['detail']}")
    print(f"outcome: {report['outcome']}")
    print(f"report: {out / 'report.md'}")
    for flag, name in ((args.report_md, "report.md"), (args.report_json, "report.json")):
        if flag:
            Path(flag).parent.mkdir(parents=True, exist_ok=True)
            Path(flag).write_text((out / name).read_text(encoding="utf-8"), encoding="utf-8")
            print(f"wrote {flag}")
    return 1 if report["outcome"] == "failed" else 0


def register(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    ext = sub.add_parser("external", help="pinned external sources and their rights gate")
    esub = ext.add_subparsers(dest="external_command", required=True)
    v = esub.add_parser("validate", help="validate committed source locks and rights records")
    v.add_argument("--root", default=str(EXTERNAL_ROOT))
    v.set_defaults(handler=_cmd_validate)
    r = esub.add_parser("rights", help="print a source's rights disposition and blockers")
    r.add_argument("source_id")
    r.add_argument("--root", default=str(EXTERNAL_ROOT))
    r.set_defaults(handler=_cmd_rights)
    a = esub.add_parser("acquire", help="fetch pinned bytes into verified quarantine")
    a.add_argument("source_id")
    a.add_argument("--source-dir", default=None, help="local copy laid out like upstream")
    a.add_argument("--metadata", default=None, help="Zenodo only: saved record metadata JSON")
    a.add_argument("--root", default=str(EXTERNAL_ROOT))
    a.set_defaults(handler=_cmd_acquire)
    run = esub.add_parser("run", help="run the full source pipeline and write a report")
    run.add_argument("source_id")
    run.add_argument("--source-dir", default=None, help="local copy laid out like upstream")
    run.add_argument("--out", default=None, help="run directory (default under DBS_CACHE_HOME)")
    run.add_argument("--report-md", default=None, help="also copy the Markdown report here")
    run.add_argument("--report-json", default=None, help="also copy the JSON report here")
    run.add_argument("--rules", default="qc/rules.toml")
    run.add_argument("--metadata", default=None, help="Zenodo only: saved record metadata JSON")
    run.add_argument("--root", default=str(EXTERNAL_ROOT))
    run.set_defaults(handler=_cmd_run)
    ps = esub.add_parser("planar-schema", help="emit the planar validation profile JSON Schema")
    ps.add_argument("--out", default=planar_profile.SCHEMA_DOC)
    mode = ps.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    ps.set_defaults(handler=_cmd_planar_schema)
