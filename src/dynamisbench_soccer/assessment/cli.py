"""Executable assessment commands (``dbs assess ...``)."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from dynamisbench_soccer.assessment import isometric
from dynamisbench_soccer.assessment.jumps import JumpSignal, JumpSpec, assess
from dynamisbench_soccer.engines.force_platform import MethodSpec as ForceMethod
from dynamisbench_soccer.engines.force_platform import process as force_process
from dynamisbench_soccer.engines.force_platform_acquisition import PlateSpec
from dynamisbench_soccer.engines.load_cell import MethodSpec as CellMethod
from dynamisbench_soccer.engines.load_cell_acquisition import CellSpec
from dynamisbench_soccer.signals import io


class JumpConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    jump: JumpSpec
    method: ForceMethod = Field(default_factory=ForceMethod)
    plates: tuple[PlateSpec, ...] = ()


class IsometricConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    isometric: isometric.IsoSpec
    method: CellMethod = Field(default_factory=CellMethod)
    cell: CellSpec = Field(default_factory=CellSpec)


def _emit(args: argparse.Namespace, run: Callable[[], tuple[str, int]]) -> int:
    try:
        text, code = run()
    except (ValueError, OSError, RuntimeError) as exc:
        text = json.dumps({"engine": "assessment", "error": str(exc)}, indent=2) + "\n"
        code = 2
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return code


def _guard(args: argparse.Namespace, inputs: list[Path]) -> bool:
    protected = {p.resolve() for source in inputs for p in (source, *io.paths_for(source))} | {
        args.config.resolve()
    }
    return args.out is None or args.out.resolve() not in protected


def _jump(args: argparse.Namespace) -> int:
    inputs = [Path(name) for name in args.inputs]
    if not _guard(args, [*inputs, *([args.weighing] if args.weighing else [])]):
        print("FAIL output path must not overwrite inputs/configuration", file=sys.stderr)
        return 2

    def run() -> tuple[str, int]:
        config = JumpConfig.model_validate(json.loads(args.config.read_text()))
        bundles = tuple(io.read(p) for p in inputs)
        device = force_process(
            bundles,
            method=config.method,
            plates=config.plates,
            weighing=io.read(args.weighing) if args.weighing else None,
        )
        result = assess(JumpSignal.from_engine(device), config.jump)
        report = result.report()
        report["device"] = {"engine": "force-platform.v1", "verdict": device.verdict}
        text = json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
        return text, 0 if result.eligible and device.verdict != "reject" else 1

    return _emit(args, run)


def _isometric(args: argparse.Namespace) -> int:
    inputs = [Path(name) for name in args.inputs]
    if not _guard(args, inputs):
        print("FAIL output path must not overwrite inputs/configuration", file=sys.stderr)
        return 2

    def run() -> tuple[str, int]:
        config = IsometricConfig.model_validate(json.loads(args.config.read_text()))
        bundles = [io.read(p) for p in inputs]
        results = isometric.run_trials(bundles, config.cell, config.method)
        result = isometric.assess(
            results,
            config.isometric,
            geometry=bundles[0].meta.geometry,
            bundles=bundles,
            cell=config.cell,
            method=config.method,
        )
        return result.report_json(), 0 if result.eligible else 1

    return _emit(args, run)


def register(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    assess_parser = sub.add_parser("assess", help="protocol-level assessment science")
    commands = assess_parser.add_subparsers(dest="assess_command", required=True)
    jump = commands.add_parser("jump", help="CMJ, SJ, SL-CMJ, drop-jump and repeated-hop metrics")
    jump.add_argument("inputs", nargs="+", help="one or two force-plate bundle stems")
    jump.add_argument("--config", type=Path, required=True, help="JSON: jump, method, plates")
    jump.add_argument("--weighing", type=Path, help="independent quiet weighing bundle")
    jump.add_argument("--out", type=Path)
    jump.set_defaults(handler=_jump)
    iso = commands.add_parser("isometric", help="IMTP, hamstring and hip isometric protocols")
    iso.add_argument("inputs", nargs="+", help="one bundle stem per trial")
    iso.add_argument("--config", type=Path, required=True, help="JSON: isometric, method, cell")
    iso.add_argument("--out", type=Path)
    iso.set_defaults(handler=_isometric)
