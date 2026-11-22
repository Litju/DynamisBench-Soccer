"""Executable Nordic / hip-strength processing (``dbs nordic-hip process``)."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from dynamisbench_soccer.engines.nordic_hip import ENGINE, MethodSpec, NordicSpec, process
from dynamisbench_soccer.signals import io


class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    spec: NordicSpec
    method: MethodSpec = Field(default_factory=MethodSpec)


def _process(args: argparse.Namespace) -> int:
    protected = [args.input, *io.paths_for(args.input), args.config]
    if args.out is not None and args.out.resolve() in {p.resolve() for p in protected}:
        print("FAIL output path must not overwrite inputs/configuration", file=sys.stderr)
        return 2
    try:
        config_bytes = args.config.read_bytes()
        config = RunConfig.model_validate_json(config_bytes)
        bundle = io.read(args.input)
        result = process(bundle, config.spec, config.method)
        report = result.report()
        report["provenance"] = {
            "bundle_id": bundle.meta.bundle_id,
            "data_sha256": bundle.meta.data_sha256,
            "validated_metadata_sha256": hashlib.sha256(
                io.meta_json(bundle.meta).encode()
            ).hexdigest(),
            "configuration_sha256": hashlib.sha256(config_bytes).hexdigest(),
        }
        text = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
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
    engine = sub.add_parser("nordic-hip", help="Nordic hamstring and hip-strength processing")
    commands = engine.add_subparsers(dest="nordic_hip_command", required=True)
    run = commands.add_parser("process", help="admit sides and force basis, compute metrics")
    run.add_argument("input", type=Path, help="signal bundle stem (nordic or hip dynamometer)")
    run.add_argument("--config", type=Path, required=True, help="JSON with 'spec' and 'method'")
    run.add_argument("--out", type=Path, help="JSON metrics, QC, limitations and provenance")
    run.set_defaults(handler=_process)
