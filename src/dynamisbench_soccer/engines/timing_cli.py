"""Run the event engines on verified canonical bundles and explicit policies."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

from dynamisbench_soccer.engines.timing_gates import GatePolicy, sprint
from dynamisbench_soccer.signals import io


def _run(args: argparse.Namespace) -> int:
    try:
        bundles = [io.read(path) for path in args.bundles]

        def evaluate(path: Path) -> dict[str, object]:
            policy = GatePolicy.model_validate_json(path.read_text(encoding="utf-8"))
            return {
                "policy": policy.model_dump(mode="json"),
                "metrics": asdict(sprint(bundles, policy)),
            }

        result = evaluate(args.policy)
        sensitivity = []
        for path in args.sensitivity:
            try:
                variant = evaluate(path)
            except (OSError, ValueError) as exc:
                variant = {"refusal": str(exc)}
            sensitivity.append({"file": str(path), **variant})
        print(
            json.dumps(
                {
                    "sources": [
                        {
                            "bundle_id": b.meta.bundle_id,
                            "data_sha256": b.meta.data_sha256,
                            "acquisition": b.meta.acquisition.model_dump(mode="json")
                            if b.meta.acquisition is not None
                            else None,
                        }
                        for b in bundles
                    ],
                    **result,
                    "sensitivity": sensitivity,
                },
                indent=2,
                allow_nan=False,
            )
        )
    except (OSError, ValueError, io.BundleIntegrityError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 1
    return 0


def register(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    timing = sub.add_parser("timing", help="reference metrics from source-native discrete events")
    modes = timing.add_subparsers(dest="timing_command", required=True)
    gate = modes.add_parser("gates", help="compute sprint splits and segment-average velocities")
    gate.add_argument("bundles", nargs="+", type=Path, metavar="BUNDLE")
    gate.add_argument("--policy", required=True, type=Path)
    gate.add_argument("--sensitivity", action="append", type=Path, default=[])
    gate.set_defaults(handler=_run)
