"""Run the event engines on verified canonical bundles and explicit policies."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path

from dynamisbench_soccer.engines.contact import ContactPolicy, contact
from dynamisbench_soccer.engines.timing_gates import GatePolicy, sprint
from dynamisbench_soccer.signals import io


def _run(args: argparse.Namespace) -> int:
    try:
        bundles = [io.read(path) for path in args.bundles]

        def evaluate(path: Path) -> dict[str, object]:
            raw = path.read_bytes()
            policy_sha256 = hashlib.sha256(raw).hexdigest()
            if args.timing_command == "gates":
                policy = GatePolicy.model_validate_json(raw)
                return {
                    "policy": policy.model_dump(mode="json"),
                    "policy_sha256": policy_sha256,
                    "metrics": asdict(sprint(bundles, policy)),
                }
            contact_policy = ContactPolicy.model_validate_json(raw)
            return {
                "policy": contact_policy.model_dump(mode="json"),
                "policy_sha256": policy_sha256,
                "metrics": asdict(contact(bundles[0], contact_policy)),
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
                            "validated_metadata_sha256": hashlib.sha256(
                                io.meta_json(b.meta).encode("utf-8")
                            ).hexdigest(),
                            "data_category": b.meta.data_category,
                            "test_id": b.meta.test_id,
                            "protocol": b.meta.protocol.model_dump(mode="json"),
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
    contacts = modes.add_parser("contact", help="pair contact and flight events")
    contacts.add_argument("bundles", nargs=1, type=Path, metavar="BUNDLE")
    for mode in (gate, contacts):
        mode.add_argument("--policy", required=True, type=Path)
        mode.add_argument("--sensitivity", action="append", type=Path, default=[])
        mode.set_defaults(handler=_run)
