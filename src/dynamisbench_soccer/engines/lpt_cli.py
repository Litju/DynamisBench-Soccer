"""Run the LPT/encoder reference engine from a v2 SignalBundle."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from dynamisbench_soccer.engines.lpt import (
    ENGINE,
    MethodSpec,
    PowerSpec,
    SensitivitySpec,
    process,
)
from dynamisbench_soccer.engines.lpt_acquisition import CableSpec, GeometrySpec
from dynamisbench_soccer.signals import io


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    cables: tuple[CableSpec, ...]
    geometry: GeometrySpec
    method: MethodSpec = Field(default_factory=MethodSpec)
    power: PowerSpec | None = None
    sensitivity: SensitivitySpec | None = None


def _process(args: argparse.Namespace) -> int:
    protected = [args.input, *io.paths_for(args.input), args.config]
    try:
        if args.out is not None and any(
            args.out.resolve() == path.resolve()
            or (args.out.exists() and path.exists() and args.out.samefile(path))
            for path in protected
        ):
            raise ValueError("output path must not overwrite inputs or configuration")
        config = RunConfig.model_validate_json(args.config.read_text(encoding="utf-8"))
        report = process(io.read(args.input), **dict(config))
        output = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
        if args.out:
            args.out.write_text(output, encoding="utf-8")
        else:
            sys.stdout.write(output)
        return 1 if report["verdict"] == "reject" else 0
    except (ValueError, OSError, RuntimeError) as exc:
        print(f"{ENGINE}: {exc}", file=sys.stderr)
        return 2


def register(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    engine = sub.add_parser("lpt-encoder", help="reference LPT/encoder acquisition processing")
    commands = engine.add_subparsers(dest="lpt_encoder_command", required=True)
    run = commands.add_parser("process", help="reconstruct cable or vertical attachment kinematics")
    run.add_argument("input", type=Path, help="v2 signal bundle stem")
    run.add_argument("--config", type=Path, required=True, help="JSON acquisition and method spec")
    run.add_argument("--out", type=Path, help="output JSON path (default stdout)")
    run.set_defaults(handler=_process)
