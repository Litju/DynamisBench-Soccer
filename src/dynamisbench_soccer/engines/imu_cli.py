"""Executable IMU processing (``dbs imu process``)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from dynamisbench_soccer.engines.imu import ENGINE, ImuSpec, MethodSpec, process
from dynamisbench_soccer.signals import io


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    spec: ImuSpec = Field(default_factory=ImuSpec)
    method: MethodSpec = Field(default_factory=MethodSpec)


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
        result = process(io.read(args.input), config.spec, config.method)
        text = result.report_json()
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
    engine = sub.add_parser("imu", help="inertial measurement unit processing")
    commands = engine.add_subparsers(dest="imu_command", required=True)
    run = commands.add_parser(
        "process", help="calibrate, check integrity, orient, compensate gravity, detect flight"
    )
    run.add_argument("input", type=Path, help="signal bundle stem (IMU specific force and gyro)")
    run.add_argument("--config", type=Path, help="JSON with 'spec' and 'method' declarations")
    run.add_argument("--out", type=Path, help="JSON summary, events, QC and provenance")
    run.set_defaults(handler=_process)
