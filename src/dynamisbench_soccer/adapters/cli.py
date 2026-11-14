"""``dbs adapters ...`` commands."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dynamisbench_soccer.adapters.base import AdapterError, all_adapters, get
from dynamisbench_soccer.signals import io


def _cmd_list(args: argparse.Namespace) -> int:
    adapters = all_adapters()
    if args.json:
        print(
            json.dumps(
                {k: a.capabilities.model_dump(mode="json") for k, a in adapters.items()},
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    for name, adapter in sorted(adapters.items()):
        cap = adapter.capabilities
        print(f"{name}@{cap.version}: {cap.kind}, tests={','.join(cap.test_ids)}")
        print(f"    columns: {cap.required_columns}")
    return 0


def _cmd_import(args: argparse.Namespace) -> int:
    try:
        bundle = get(args.adapter).read(Path(args.csv))
    except AdapterError as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        return 1
    parquet, sidecar = io.write(bundle, Path(args.out))
    print(f"wrote {parquet} and {sidecar} ({bundle.meta.n_rows} rows)")
    return 0


def register(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    ad = sub.add_parser("adapters", help="offline device-export import adapters")
    asub = ad.add_subparsers(dest="adapters_command", required=True)
    ls = asub.add_parser("list", help="list adapters and their capability manifests")
    ls.add_argument("--json", action="store_true")
    ls.set_defaults(handler=_cmd_list)
    im = asub.add_parser("import", help="convert an export into a canonical bundle")
    im.add_argument("adapter")
    im.add_argument("csv")
    im.add_argument("out", help="output stem (writes <out>.parquet and <out>.json)")
    im.set_defaults(handler=_cmd_import)
