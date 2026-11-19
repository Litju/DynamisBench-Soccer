"""``dbs agreement run`` writes the analytical cross-device agreement report."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _run(args: argparse.Namespace) -> int:
    from dynamisbench_soccer.agreement.studies import report_json

    try:
        text = report_json()
    except (ValueError, RuntimeError) as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        return 2
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return 0


def register(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    parser = sub.add_parser("agreement", help="cross-device agreement studies (analytical)")
    commands = parser.add_subparsers(dest="agreement_command", required=True)
    run = commands.add_parser("run", help="run every declared non-equivalence study")
    run.add_argument("--out", type=Path)
    run.set_defaults(handler=_run)
