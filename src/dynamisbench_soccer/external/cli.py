"""``dbs external ...`` commands."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dynamisbench_soccer.data.store import IntegrityError
from dynamisbench_soccer.external import acquire, rights
from dynamisbench_soccer.external.models import (
    EXTERNAL_ROOT,
    load_lock,
    load_rights,
    source_ids,
)


def _cmd_validate(args: argparse.Namespace) -> int:
    root = Path(args.root)
    failed = 0
    ids = source_ids(root)
    for sid in ids:
        try:
            lock, record = load_lock(sid, root), load_rights(sid, root)
            if record.pinned_ref != lock.pinned_ref:
                raise ValueError("rights record and lock pin different commits")
            print(f"OK   {sid} @ {lock.pinned_ref[:12]} admission={record.canonical_admission}")
        except Exception as exc:  # report every invalid source
            failed += 1
            print(f"FAIL {sid}: {exc}", file=sys.stderr)
    print(f"external: {len(ids)} source(s): {'OK' if not failed else f'{failed} invalid'}")
    return 1 if failed else 0


def _cmd_rights(args: argparse.Namespace) -> int:
    record = load_rights(args.source_id, Path(args.root))
    redistribution = "allowed" if record.redistribution_allowed else "NOT allowed"
    print(
        f"{record.source_id}: license {record.license_id}; "
        f"measurement-data rights {record.measurement_data_rights}; "
        f"canonical admission {record.canonical_admission}; redistribution {redistribution}"
    )
    for b in record.blockers:
        print(f"  BLOCKER {b}")
    return 0


def _cmd_acquire(args: argparse.Namespace) -> int:
    root = Path(args.root)
    lock, record = load_lock(args.source_id, root), load_rights(args.source_id, root)
    try:
        files = acquire.acquire(lock, source_dir=Path(args.source_dir) if args.source_dir else None)
        check = rights.verify(record, lock, files)
        rights.require_local_inspection(check)
    except (IntegrityError, rights.RightsBlocked) as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        return 1
    for entry in lock.files:
        print(f"verified {entry.sha256} {entry.size_bytes:>8} {files[entry.path]}")
    print(f"quarantine only; canonical admission {check.canonical_admission}")
    return 0


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
    a.add_argument("--root", default=str(EXTERNAL_ROOT))
    a.set_defaults(handler=_cmd_acquire)
