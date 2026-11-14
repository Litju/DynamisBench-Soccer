"""``dbs signals ...`` commands."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dynamisbench_soccer.signals import io

SCHEMA_DOC = Path("docs/schemas/signal-bundle-meta.v2.json")


def _cmd_schema(args: argparse.Namespace) -> int:
    text = io.json_schema()
    target = Path(args.out)
    if args.write:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(text, encoding="utf-8")
        print(f"wrote {target}")
    elif args.check:
        if not target.is_file() or target.read_text(encoding="utf-8") != text:
            print(f"FAIL {target} out of date; run `dbs signals schema --write`")
            return 1
        print(f"{target} up to date")
    else:
        sys.stdout.write(text)
    return 0


def _cmd_inspect(args: argparse.Namespace) -> int:
    bundle = io.read(Path(args.stem))
    m = bundle.meta
    print(f"{m.bundle_id}: {m.kind} test={m.test_id} device={m.device.device_class} "
          f"rows={m.n_rows} category={m.data_category}")  # fmt: skip
    for c in m.channels:
        values = bundle[c.name]
        print(f"  {c.name:<16} {c.quantity:<12} [{c.unit}] side={c.side} "
              f"min={values.min():.3f} max={values.max():.3f}")  # fmt: skip
    if m.vendor_reported:
        print("  vendor_reported (not reference values):")
        for k, v in sorted(m.vendor_reported.items()):
            print(f"    {k} = {v.value} {v.unit} ({v.definition})")
    return 0


def register(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    sig = sub.add_parser("signals", help="canonical signal bundles")
    ssub = sig.add_subparsers(dest="signals_command", required=True)
    s = ssub.add_parser("schema", help="emit the sidecar JSON Schema")
    s.add_argument("--out", default=str(SCHEMA_DOC))
    mode = s.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    s.set_defaults(handler=_cmd_schema)
    i = ssub.add_parser("inspect", help="verify and summarise a bundle")
    i.add_argument("stem", help="path without extension, or the .parquet/.json path")
    i.set_defaults(handler=_cmd_inspect)
