"""Executable load-cell / isometric-dynamometer processing (``dbs load-cell process``)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from dynamisbench_soccer.engines.load_cell import ENGINE, MethodSpec, process, sensitivity
from dynamisbench_soccer.engines.load_cell_acquisition import CellSpec
from dynamisbench_soccer.signals import io


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    method: MethodSpec = Field(default_factory=MethodSpec)
    cell: CellSpec = Field(default_factory=CellSpec)


def _process(args: argparse.Namespace) -> int:
    protected = [args.input, *io.paths_for(args.input), *([args.config] if args.config else [])]
    if args.out is not None and args.out.resolve() in {p.resolve() for p in protected}:
        print("FAIL output path must not overwrite inputs/configuration", file=sys.stderr)
        return 2
    try:
        config = (
            RunConfig.model_validate(json.loads(args.config.read_text()))
            if args.config
            else RunConfig()
        )
        bundle = io.read(args.input)
        result = process(bundle, config.cell, config.method)
        report = result.report()
        report["sensitivity"] = (
            sensitivity(bundle, config.cell, config.method) if args.sensitivity else None
        )
        text = json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
        code = 1 if result.verdict == "reject" else 0
    except (ValueError, OSError, RuntimeError) as exc:
        text = json.dumps({"engine": ENGINE, "verdict": "reject", "error": str(exc)}, indent=2)
        text += "\n"
        code = 2
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return code


def register(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    engine = sub.add_parser(
        "load-cell", help="production load-cell / isometric dynamometer processing"
    )
    commands = engine.add_subparsers(dest="load_cell_command", required=True)
    run = commands.add_parser(
        "process", help="calibrate, check integrity and compute isometric force/torque metrics"
    )
    run.add_argument("input", type=Path, help="signal bundle stem (load_cell/isometric source)")
    run.add_argument("--config", type=Path, help="JSON with 'method' and 'cell' declarations")
    run.add_argument(
        "--sensitivity",
        action="store_true",
        help="add gain/filter/onset sensitivity variants to the report",
    )
    run.add_argument("--out", type=Path, help="JSON metrics, QC, limitations and provenance")
    run.set_defaults(handler=_process)
