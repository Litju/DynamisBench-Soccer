"""``dbs registry ...`` commands."""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

from dynamisbench_soccer.registry.models import load, validate
from dynamisbench_soccer.registry.report import contested_markdown, coverage_lines

MIN_CLAIMS = 60
CONTESTED_DOC = Path("docs/contested-methods.md")


def _cmd_validate(args: argparse.Namespace) -> int:
    reg = load(Path(args.root))
    problems = validate(reg, min_claims=MIN_CLAIMS)
    for p in problems:
        print(f"FAIL {p}", file=sys.stderr)
    status = "OK" if not problems else f"{len(problems)} problem(s)"
    print(f"registry: {len(reg.sources)} sources, {len(reg.claims)} claims, "
          f"{len(reg.gaps)} gap(s): {status}")  # fmt: skip
    return 1 if problems else 0


def _cmd_coverage(args: argparse.Namespace) -> int:
    print("\n".join(coverage_lines(load(Path(args.root)))))
    return 0


def _cmd_contested(args: argparse.Namespace) -> int:
    text = contested_markdown(load(Path(args.root)))
    target = Path(args.doc)
    if args.write:
        target.write_text(text, encoding="utf-8")
        print(f"wrote {target}")
        return 0
    if args.check:
        if not target.is_file() or target.read_text(encoding="utf-8") != text:
            print(f"FAIL {target} is out of date; run `dbs registry contested --write`")
            return 1
        print(f"{target} up to date")
        return 0
    print(text)
    return 0


def _cmd_verify_dois(args: argparse.Namespace) -> int:
    """Network check (not run in CI): every DOI resolves on Crossref with matching author."""
    reg = load(Path(args.root))
    failed = 0
    for src in reg.sources.values():
        if not src.doi:
            continue
        url = "https://api.crossref.org/works/" + urllib.parse.quote(src.doi)
        req = urllib.request.Request(url, headers={"User-Agent": "dbs-registry-check/0.1"})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                msg = json.load(resp)["message"]
            family = ((msg.get("author") or [{}])[0].get("family") or "").lower()
            ok = src.first_author.lower().split()[-1] in family.replace("martin ", "")
        except Exception as exc:
            ok, family = False, f"error: {exc}"
        failed += not ok
        print(f"{'OK  ' if ok else 'FAIL'} {src.id} {src.doi} (crossref author: {family})")
        time.sleep(args.delay)
    return 1 if failed else 0


def register(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    reg = sub.add_parser("registry", help="source-to-claim evidence registry")
    rsub = reg.add_subparsers(dest="registry_command", required=True)
    for name, handler, help_ in (
        ("validate", _cmd_validate, "schema + cross-reference + coverage checks"),
        ("coverage", _cmd_coverage, "claims per v0.1 test and topic"),
        ("verify-dois", _cmd_verify_dois, "re-check every DOI on Crossref (network)"),
    ):
        p = rsub.add_parser(name, help=help_)
        p.add_argument("--root", default="registry")
        if name == "verify-dois":
            p.add_argument("--delay", type=float, default=0.5)
        p.set_defaults(handler=handler)
    c = rsub.add_parser("contested", help="render the contested-methods table")
    c.add_argument("--root", default="registry")
    c.add_argument("--doc", default=str(CONTESTED_DOC))
    mode = c.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    c.set_defaults(handler=_cmd_contested)
