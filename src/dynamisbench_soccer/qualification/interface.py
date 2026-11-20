"""The frozen engine-v0.1 software interface snapshot (RES-675).

The snapshot lists every CLI command path with its options and the JSON-schema digests for
registered declaration and method specifications. It freezes the *software interface*
for review; it is independent of, and confers no, qualification. Changing any of it requires
regenerating the snapshot under a new explicit version.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
from typing import Any

INTERFACE_VERSION = "engine-v0.1"


def _walk(parser: argparse.ArgumentParser, prefix: str = "") -> dict[str, list[str]]:
    paths: dict[str, list[str]] = {}
    options = sorted(
        opt
        for action in parser._actions
        if not isinstance(action, argparse._SubParsersAction)
        for opt in action.option_strings
        if opt not in {"-h", "--help"}
    )
    positionals = [
        action.dest
        for action in parser._actions
        if not action.option_strings and not isinstance(action, argparse._SubParsersAction)
    ]
    paths[prefix or "dbs"] = sorted([*options, *(f"<{p}>" for p in positionals)])
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for name, child in sorted(action.choices.items()):
                paths.update(_walk(child, f"{prefix} {name}".strip()))
    return paths


def _digest(model: Any) -> str:
    if hasattr(model, "model_json_schema"):
        payload: Any = model.model_json_schema()
    elif dataclasses.is_dataclass(model):
        payload = {f.name: str(f.type) for f in dataclasses.fields(model)}
    else:
        raise TypeError(f"cannot digest {model!r}")
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode()).hexdigest()


def specifications() -> dict[str, Any]:
    from dynamisbench_soccer.agreement import stats as agreement_stats
    from dynamisbench_soccer.assessment import cod, isometric, jumps, sprint
    from dynamisbench_soccer.engines import (
        contact,
        force_platform,
        gnss_lps,
        imu,
        load_cell,
        lpt,
        nordic_hip,
        radar_laser,
        timing_gates,
    )
    from dynamisbench_soccer.engines import force_platform_acquisition as fpa
    from dynamisbench_soccer.engines import load_cell_acquisition as lca
    from dynamisbench_soccer.qualification import admission

    return {
        "force_platform.MethodSpec": force_platform.MethodSpec,
        "force_platform.PlateSpec": fpa.PlateSpec,
        "load_cell.MethodSpec": load_cell.MethodSpec,
        "load_cell.CellSpec": lca.CellSpec,
        "lpt.MethodSpec": lpt.MethodSpec,
        "lpt.PowerSpec": lpt.PowerSpec,
        "nordic_hip.MethodSpec": nordic_hip.MethodSpec,
        "nordic_hip.NordicSpec": nordic_hip.NordicSpec,
        "imu.MethodSpec": imu.MethodSpec,
        "imu.ImuSpec": imu.ImuSpec,
        "timing_gates.GatePolicy": timing_gates.GatePolicy,
        "contact.ContactPolicy": contact.ContactPolicy,
        "radar_laser.ProcessingOptions": radar_laser.ProcessingOptions,
        "gnss_lps.ProcessingOptions": gnss_lps.ProcessingOptions,
        "assessment.JumpSpec": jumps.JumpSpec,
        "assessment.IsoSpec": isometric.IsoSpec,
        "assessment.FvSpec": sprint.FvSpec,
        "assessment.StartSpec": sprint.StartSpec,
        "assessment.CodSpec": cod.CodSpec,
        "admission.Proposal": admission.Proposal,
        "agreement.PairedAgreement": agreement_stats.PairedAgreement,
    }


def engine_ids() -> dict[str, str]:
    from dynamisbench_soccer.assessment import isometric, jumps, sprint
    from dynamisbench_soccer.engines import imu, load_cell, lpt, nordic_hip

    return {
        "imu": imu.ENGINE,
        "load_cell": load_cell.ENGINE,
        "lpt": lpt.ENGINE,
        "nordic_hip": nordic_hip.ENGINE,
        "jump_assessment": jumps.ENGINE,
        "isometric_assessment": isometric.ENGINE,
        "sprint_assessment": sprint.ENGINE,
    }


def snapshot() -> dict[str, Any]:
    from dynamisbench_soccer.cli import build_parser

    return {
        "interface_version": INTERFACE_VERSION,
        "statement": "software interface freeze for review; not a qualification",
        "engine_ids": engine_ids(),
        "cli": _walk(build_parser()),
        "specifications": {k: _digest(v) for k, v in sorted(specifications().items())},
    }


def snapshot_json() -> str:
    return json.dumps(snapshot(), indent=2, sort_keys=True) + "\n"
