"""``dbs data ...``, ``dbs clean`` and ``dbs repo-check`` commands."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dynamisbench_soccer import repocheck
from dynamisbench_soccer.data import clean, store
from dynamisbench_soccer.data.manifest import (
    DataCategory,
    DatasetManifest,
    Source,
    Usage,
    describe_directory,
)


def _load(path: str) -> DatasetManifest:
    return DatasetManifest.load(Path(path))


def _cmd_create(args: argparse.Namespace) -> int:
    manifest = DatasetManifest(
        dataset_id=args.dataset_id,
        version=args.version,
        title=args.title,
        data_category=DataCategory(args.category),
        usage=Usage(args.usage),
        license=args.license,
        redistribution_allowed=args.redistributable,
        source=Source(kind=args.source_kind, location=args.source_location),
        files=describe_directory(Path(args.directory)),
    )
    manifest.save(Path(args.output))
    print(f"wrote {args.output}: {len(manifest.files)} files, {manifest.total_bytes()} bytes")
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    failed = 0
    for name in args.manifests:
        try:
            m = _load(name)
            print(f"OK   {name} ({m.key}, {m.data_category}, {m.usage})")
        except Exception as exc:  # report every invalid manifest, not just the first
            failed += 1
            print(f"FAIL {name}: {exc}", file=sys.stderr)
    return 1 if failed else 0


def _cmd_ingest(args: argparse.Namespace) -> int:
    store.ingest_local(_load(args.manifest), Path(args.source_dir))
    print("ingested and verified")
    return 0


def _cmd_fetch(args: argparse.Namespace) -> int:
    store.fetch(_load(args.manifest))
    print("fetched and verified")
    return 0


def _cmd_verify(args: argparse.Namespace) -> int:
    manifest = _load(args.manifest)
    problems = store.verify(manifest)
    for p in problems:
        print(f"FAIL {p.path}: {p.reason}", file=sys.stderr)
    print(f"{manifest.key}: {'OK' if not problems else f'{len(problems)} problem(s)'}")
    return 1 if problems else 0


def _cmd_stage(args: argparse.Namespace) -> int:
    print(store.stage(_load(args.manifest)))
    return 0


def _cmd_clean(args: argparse.Namespace) -> int:
    targets = [t for t in clean.TARGETS if getattr(args, t)] or list(clean.TARGETS)
    candidates = clean.plan(targets, Path(args.repo_root).resolve(), args.older_than_hours)
    total = sum(c.size_bytes for c in candidates)
    for c in candidates:
        print(f"{'would remove' if args.dry_run else 'remove'} [{c.target}] {c.path}")
    if args.dry_run:
        print(f"dry run: {len(candidates)} item(s), {total} bytes")
        return 0
    freed = clean.apply(candidates)
    print(f"removed {len(candidates)} item(s), {freed} bytes")
    return 0


def _cmd_repo_check(args: argparse.Namespace) -> int:
    violations = repocheck.check(Path(args.repo_root).resolve())
    for v in violations:
        print(f"FAIL {v.path}: {v.reason}", file=sys.stderr)
    print("repo-check OK" if not violations else f"repo-check: {len(violations)} violation(s)")
    return 1 if violations else 0


def register(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    data = sub.add_parser("data", help="external dataset manifests, store and staging")
    dsub = data.add_subparsers(dest="data_command", required=True)

    c = dsub.add_parser("manifest-create", help="hash a directory into a new manifest")
    c.add_argument("directory")
    c.add_argument("--output", required=True)
    c.add_argument("--dataset-id", required=True)
    c.add_argument("--version", required=True)
    c.add_argument("--title", required=True)
    c.add_argument("--category", choices=[x.value for x in DataCategory], required=True)
    c.add_argument("--usage", choices=[x.value for x in Usage], required=True)
    c.add_argument("--license", required=True)
    c.add_argument("--redistributable", action="store_true")
    c.add_argument("--source-kind", choices=["url", "local", "generated"], default="local")
    c.add_argument("--source-location", required=True)
    c.set_defaults(handler=_cmd_create)

    v = dsub.add_parser("validate", help="schema-validate manifest files")
    v.add_argument("manifests", nargs="+")
    v.set_defaults(handler=_cmd_validate)

    for name, handler, help_ in (
        ("fetch", _cmd_fetch, "download declared files into the store and verify"),
        ("verify", _cmd_verify, "re-hash stored files against a manifest"),
        ("stage", _cmd_stage, "verify, then link a dataset into a temporary staging dir"),
    ):
        p = dsub.add_parser(name, help=help_)
        p.add_argument("manifest")
        p.set_defaults(handler=handler)

    i = dsub.add_parser("ingest", help="copy local files into the store and verify")
    i.add_argument("manifest")
    i.add_argument("source_dir")
    i.set_defaults(handler=_cmd_ingest)

    cl = sub.add_parser("clean", help="remove run artifacts, workspaces and caches")
    for target in clean.TARGETS:
        cl.add_argument(f"--{target}", action="store_true", help=f"clean {target}")
    cl.add_argument("--dry-run", action="store_true", help="list without deleting")
    cl.add_argument("--older-than-hours", type=float, default=None)
    cl.add_argument("--repo-root", default=".")
    cl.set_defaults(handler=_cmd_clean)

    rc = sub.add_parser("repo-check", help="fail if data or oversized files are tracked")
    rc.add_argument("--repo-root", default=".")
    rc.set_defaults(handler=_cmd_repo_check)
