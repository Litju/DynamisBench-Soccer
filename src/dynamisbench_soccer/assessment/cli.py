"""Executable assessment commands (``dbs assess ...``)."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from dynamisbench_soccer.assessment import cod, isometric, sprint
from dynamisbench_soccer.assessment.jumps import DropEvidence, JumpSignal, JumpSpec, assess
from dynamisbench_soccer.engines.force_platform import MethodSpec as ForceMethod
from dynamisbench_soccer.engines.force_platform import process as force_process
from dynamisbench_soccer.engines.force_platform_acquisition import PlateSpec
from dynamisbench_soccer.engines.load_cell import MethodSpec as CellMethod
from dynamisbench_soccer.engines.load_cell_acquisition import CellSpec
from dynamisbench_soccer.signals import io


class JumpConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    jump: JumpSpec
    drop_evidence: DropEvidence | None = None
    method: ForceMethod = Field(default_factory=ForceMethod)
    plates: tuple[PlateSpec, ...] = ()


class IsometricConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    isometric: isometric.IsoSpec
    method: CellMethod = Field(default_factory=CellMethod)
    cell: CellSpec = Field(default_factory=CellSpec)


class SprintConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    source: sprint.Source
    fv: sprint.FvSpec
    start: sprint.StartSpec = Field(default_factory=sprint.StartSpec)


class CodConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    cod: cod.CodSpec
    start: sprint.StartSpec = Field(default_factory=sprint.StartSpec)


def _trace(path: Path) -> tuple[np.ndarray, np.ndarray]:
    report = json.loads(path.read_text())
    try:
        time = np.asarray(report["time_from_start_s"], dtype=np.float64)
        speed = np.asarray(report["velocity_m_s"], dtype=np.float64)
    except KeyError as exc:
        raise ValueError(f"engine report lacks {exc.args[0]!r}: not a velocity result") from exc
    if time.shape != speed.shape or time.ndim != 1:
        raise ValueError("time and velocity must be equal-length one-dimensional arrays")
    return time, np.abs(speed)


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
        result = assess(
            JumpSignal.from_engine(device), config.jump, drop_evidence=config.drop_evidence
        )
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
            geometries=[b.meta.geometry for b in bundles],
            bundles=bundles,
            cell=config.cell,
            method=config.method,
        )
        return result.report_json(), 0 if result.eligible else 1

    return _emit(args, run)


def _sprint(args: argparse.Namespace) -> int:
    if not _guard(args, [args.input]):
        print("FAIL output path must not overwrite inputs/configuration", file=sys.stderr)
        return 2

    def run() -> tuple[str, int]:
        config = SprintConfig.model_validate(json.loads(args.config.read_text()))
        time, speed = _trace(args.input)
        fit = sprint.fit_velocity_trace(time, speed, config.fv, config.start, source=config.source)
        return sprint.report(fit, source=config.source), 0 if fit.identifiable else 1

    return _emit(args, run)


def _cod(args: argparse.Namespace) -> int:
    if not _guard(args, [args.input]):
        print("FAIL output path must not overwrite inputs/configuration", file=sys.stderr)
        return 2

    def run() -> tuple[str, int]:
        config = CodConfig.model_validate(json.loads(args.config.read_text()))
        time, speed = _trace(args.input)
        result = cod.assess_trace(time, speed, config.cod, config.start)
        return json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n", 0

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
    fv = commands.add_parser("sprint-fv", help="conditional mono-exponential sprint F-V fit")
    fv.add_argument("input", type=Path, help="radar-laser/gnss-lps JSON result with a speed trace")
    fv.add_argument("--config", type=Path, required=True)
    fv.add_argument("--out", type=Path)
    fv.set_defaults(handler=_sprint)
    change = commands.add_parser("cod", help="change-of-direction metrics on a declared path")
    change.add_argument("input", type=Path, help="JSON result with a speed trace")
    change.add_argument("--config", type=Path, required=True)
    change.add_argument("--out", type=Path)
    change.set_defaults(handler=_cod)
