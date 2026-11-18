"""CLI registration for RES-699 radar and laser processing."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from dynamisbench_soccer.engines.radar_laser import ProcessingOptions, process
from dynamisbench_soccer.engines.radar_laser_acquisition import AcquisitionRefusal, ChannelStage
from dynamisbench_soccer.signals import io


def _process(args: argparse.Namespace) -> int:
    try:
        bundle = io.read(Path(args.bundle))
        options = ProcessingOptions(
            channel_stage=ChannelStage(args.stage),
            filter_method=args.filter_method,
            derivative_method=args.derivative_method,
            start_method=args.start_method,
            correction_method=args.correction_method,
            threshold_m_s=args.threshold_m_s,
            range_change_m=args.range_change_m,
            external_trigger_reference_s=args.external_trigger_reference_s,
            timing_perturbation_s=args.timing_perturbation_s,
            cosine_perturbation_deg=args.cosine_perturbation_deg,
            mounting_distance_behind_start_m=args.mount_distance_m,
        )
        result = process(bundle, options)
        content = json.dumps(asdict(result), sort_keys=True, indent=2, allow_nan=False) + "\n"
        if args.out:
            target = Path(args.out)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        else:
            print(content, end="")
    except (AcquisitionRefusal, OSError, ValueError, io.BundleIntegrityError) as exc:
        print(f"radar-laser refusal: {exc}", file=sys.stderr)
        return 2
    return 0


def register(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    group = sub.add_parser("radar-laser", help="native radar/laser reference processing")
    commands = group.add_subparsers(dest="radar_laser_command", required=True)
    parser = commands.add_parser("process", help="verified v2 bundle -> velocity trace")
    parser.add_argument("bundle", help="Parquet+JSON bundle stem")
    parser.add_argument("--stage", choices=[x.value for x in ChannelStage], required=True)
    parser.add_argument(
        "--filter", dest="filter_method", choices=["none", "boxcar5", "binomial5"], default="none"
    )
    parser.add_argument(
        "--derivative", dest="derivative_method",
        choices=["central", "five_point"], default="five_point"
    )
    parser.add_argument(
        "--start", dest="start_method",
        choices=["record_origin", "velocity_threshold", "range_change", "external_trigger"],
        default="record_origin"
    )
    parser.add_argument(
        "--correction", dest="correction_method",
        choices=["none", "declared_fixed_angle"], default="none"
    )
    parser.add_argument("--threshold-m-s", type=float, default=0.30)
    parser.add_argument("--range-change-m", type=float, default=0.05)
    parser.add_argument("--external-trigger-reference-s", type=float)
    parser.add_argument("--timing-perturbation-s", type=float, default=0.02)
    parser.add_argument("--cosine-perturbation-deg", type=float, default=1.0)
    parser.add_argument("--mount-distance-m", type=float)
    parser.add_argument("--out")
    parser.set_defaults(handler=_process)
