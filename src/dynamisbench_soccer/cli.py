"""``dbs`` command-line interface."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable, Sequence
from pathlib import Path

from dynamisbench_soccer import __version__
from dynamisbench_soccer.envcheck import check_environment

Handler = Callable[[argparse.Namespace], int]


def _cmd_env_check(args: argparse.Namespace) -> int:
    report = check_environment(Path(args.repo_root) if args.repo_root else None)
    for key, value in report.facts.items():
        print(f"{key:>18}: {value}")
    for warning in report.warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    for error in report.errors:
        print(f"ERROR: {error}", file=sys.stderr)
    print("environment OK" if report.ok else "environment NOT canonical")
    return 0 if report.ok else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="dbs", description=__doc__)
    parser.add_argument("--version", action="version", version=f"dbs {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    env = sub.add_parser("env-check", help="verify the canonical WSL2/Linux environment")
    env.add_argument("--repo-root", default=None)
    env.set_defaults(handler=_cmd_env_check)

    _register_optional(sub)
    return parser


def _register_optional(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register subcommands contributed by other modules."""
    from dynamisbench_soccer import commands

    for register in commands.REGISTRARS:
        register(sub)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler: Handler = args.handler
    return handler(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
