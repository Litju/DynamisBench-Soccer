"""Executable acquisition-to-metric force-platform processing."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from dynamisbench_soccer.adapters.base import get
from dynamisbench_soccer.engines.force_platform import MethodSpec, process
from dynamisbench_soccer.engines.force_platform_acquisition import PlateSpec
from dynamisbench_soccer.qc import spec as qc
from dynamisbench_soccer.signals import io
from dynamisbench_soccer.signals.schema import Side, SignalBundle


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    method: MethodSpec = Field(default_factory=MethodSpec)
    plates: tuple[PlateSpec, ...] = ()
    assignment_reference: dict[int, Side] | None = None


def _read(path: Path, csv: bool) -> SignalBundle:
    return get("generic_forceplate_csv").read(path) if csv else io.read(path)


def _process(args: argparse.Namespace) -> int:
    inputs = [Path(name) for name in args.inputs]
    protected = [
        *inputs,
        *([args.config] if args.config else []),
        *([args.weighing] if args.weighing else []),
        *([args.qc_rules] if args.qc_rules else []),
    ]
    if not args.csv:
        protected.extend(
            p
            for source in [*inputs, *([args.weighing] if args.weighing else [])]
            for p in io.paths_for(source)
        )
    outputs = [p for p in (args.out, args.traces) if p is not None]
    if len({p.resolve() for p in outputs}) != len(outputs) or any(
        out.resolve() in {p.resolve() for p in protected} for out in outputs
    ):
        print(
            "FAIL output paths must be distinct and must not overwrite inputs/configuration",
            file=sys.stderr,
        )
        return 2
    try:
        config = (
            RunConfig.model_validate(json.loads(args.config.read_text()))
            if args.config
            else RunConfig()
        )
        if len(inputs) not in (1, 2):
            raise ValueError("provide one or two acquisition inputs")
        bundles = tuple(_read(path, args.csv) for path in inputs)
        result = process(
            bundles,
            method=config.method,
            plates=config.plates,
            weighing=_read(args.weighing, args.csv) if args.weighing else None,
            assignment_reference=config.assignment_reference,
            rules=qc.load(args.qc_rules) if args.qc_rules else None,
        )
        report = result.report()
        report["trace_output"] = None
        if args.traces:
            columns = {
                "time_s": result.time_s,
                "raw_force_n": result.raw_force_n,
                "force_n": result.force_n,
                "acceleration_m_s2": result.acceleration_m_s2,
                "velocity_m_s": result.velocity_m_s,
                "displacement_m": result.displacement_m,
                "grf_power_w": result.grf_power_w,
                "net_power_w": result.net_power_w,
                "contact": result.contact_state.astype(np.float64),
            }
            columns.update({f"fz_plate{k}_n": v for k, v in sorted(result.plate_force_n.items())})
            if result.cop_m is not None:
                columns.update(
                    {f"cop_{axis}_m": result.cop_m[:, i] for i, axis in enumerate("xyz")}
                )
            args.traces.parent.mkdir(parents=True, exist_ok=True)
            np.savetxt(
                args.traces,
                np.column_stack(list(columns.values())),
                delimiter=",",
                header=",".join(columns),
                comments="",
            )
            report["trace_output"] = {
                "file": str(args.traces),
                "sha256": hashlib.sha256(args.traces.read_bytes()).hexdigest(),
            }
        text = json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
        code = 1 if result.verdict == "reject" else 0
    except (ValueError, OSError, RuntimeError) as exc:
        text = (
            json.dumps(
                {
                    "engine": "force-platform.v1",
                    "verdict": "reject",
                    "error": str(exc),
                    "trace_output": None,
                },
                indent=2,
            )
            + "\n"
        )
        code = 2
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    else:
        sys.stdout.write(text)
    return code


def register(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    engine = sub.add_parser(
        "force-platform", help="production single/dual force-platform processing"
    )
    commands = engine.add_subparsers(dest="force_platform_command", required=True)
    run = commands.add_parser("process", help="calibrate, align, QC and reconstruct measured force")
    run.add_argument(
        "inputs", nargs="+", help="one/two bundle stems, or generic CSV files with --csv"
    )
    run.add_argument(
        "--csv", action="store_true", help="read existing generic force-plate CSV adapter"
    )
    run.add_argument(
        "--config",
        type=Path,
        help="JSON method, plates and optional independent assignment_reference",
    )
    run.add_argument("--weighing", type=Path, help="independent >=1s quiet weighing acquisition")
    run.add_argument(
        "--qc-rules", type=Path, help="explicit QC rule file; packaged rules are the default"
    )
    run.add_argument(
        "--out", type=Path, help="JSON metrics, QC and complete method/provenance report"
    )
    run.add_argument(
        "--traces",
        type=Path,
        help="CSV calibrated/processed force, motion, power and contact traces",
    )
    run.set_defaults(handler=_process)
