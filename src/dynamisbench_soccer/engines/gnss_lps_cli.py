"""CLI registration for RES-700 GNSS and LPS position processing."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
from pydantic import BaseModel

from dynamisbench_soccer.engines.gnss_lps import ProcessingOptions, SpeedThreshold, process
from dynamisbench_soccer.engines.gnss_lps_acquisition import AcquisitionDeclaration
from dynamisbench_soccer.signals import io


def _json_value(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    raise TypeError(f"unsupported result value {type(value).__name__}")


def _process(args: argparse.Namespace) -> int:
    try:
        stem, declaration_path = Path(args.bundle), Path(args.declaration)
        declaration = AcquisitionDeclaration.model_validate_json(declaration_path.read_bytes())
        options = ProcessingOptions(
            position_filter=args.position_filter,
            doppler_filter=args.doppler_filter,
            derivative=args.derivative,
        )
        if args.thresholds is not None:
            thresholds = tuple(
                SpeedThreshold(threshold_m_s=speed, dwell_s=dwell)
                for speed, dwell in args.thresholds
            )
            options = options.model_copy(update={"thresholds": thresholds})
        result = process(io.read(stem), declaration, options)
        content = json.dumps(asdict(result), default=_json_value, allow_nan=False, sort_keys=True)
        if args.out:
            target = Path(args.out)
            if target.resolve() in {
                path.resolve() for path in (*io.paths_for(stem), declaration_path)
            }:
                raise ValueError("output path overlaps an input")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content + "\n", encoding="utf-8")
        else:
            print(content)
    except (OSError, ValueError, io.BundleIntegrityError) as exc:
        print(f"gnss-lps refusal: {exc}", file=sys.stderr)
        return 2
    return 0


def register(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    group = sub.add_parser("gnss-lps", help="native GNSS and LPS reference processing")
    commands = group.add_subparsers(dest="gnss_lps_command", required=True)
    parser = commands.add_parser("process", help="verified v2 position observation to kinematics")
    parser.add_argument("bundle", help="Parquet+JSON bundle stem")
    parser.add_argument("--declaration", required=True, help="acquisition declaration JSON")
    filters = ("none", "boxcar5", "binomial5")
    parser.add_argument("--position-filter", choices=filters, default="none")
    parser.add_argument("--doppler-filter", choices=filters, default="none")
    parser.add_argument(
        "--derivative", choices=("three_point", "five_point"), default="three_point"
    )
    parser.add_argument(
        "--threshold",
        dest="thresholds",
        type=float,
        nargs=2,
        action="append",
        metavar=("SPEED_M_S", "DWELL_S"),
        help="threshold and dwell pair, repeatable",
    )
    parser.add_argument("--out", help="write JSON report to this file")
    parser.set_defaults(handler=_process)
