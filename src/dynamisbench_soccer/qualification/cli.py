"""Commands exposing device-parity validation and staged scientific gates."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from dynamisbench_soccer.qualification.ledger import Stage, gate, load, markdown, validate


def _validate(args: argparse.Namespace) -> int:
    model = load(Path(args.ledger))
    problems = validate(model, Path("."))
    for problem in problems:
        print(f"FAIL {problem}", file=sys.stderr)
    print(
        f"qualification: {len(model.subfamilies)} subfamilies; "
        f"{'OK' if not problems else 'INVALID'}"
    )
    return 1 if problems else 0


def _gate(args: argparse.Namespace) -> int:
    model = load(Path(args.ledger))
    result = gate(model, Stage(args.stage))
    print(f"{args.stage}: {result.passed_families}/{result.total_families} families")
    for family, missing in result.missing.items():
        print(f"  {family}: {len(missing)} unmet requirements")
        if args.verbose:
            for detail in missing:
                print(f"    - {detail}")
    return 0 if result.passed else 1


def _ledger(args: argparse.Namespace) -> int:
    model = load(Path(args.ledger))
    problems = validate(model, Path("."))
    if problems:
        for problem in problems:
            print(f"FAIL {problem}", file=sys.stderr)
        return 1
    rendered = markdown(model)
    target = Path(args.doc)
    if args.write:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rendered, encoding="utf-8")
        print(f"wrote {target}")
    elif args.check:
        if not target.is_file() or target.read_text(encoding="utf-8") != rendered:
            print(f"FAIL stale document: {target}", file=sys.stderr)
            return 1
        print(f"{target} up to date")
    else:
        print(rendered)
    return 0


def _states(args: argparse.Namespace) -> int:
    import json
    from dataclasses import asdict

    from dynamisbench_soccer.qualification.state import computed_states, m3_gate

    model = load(Path(args.ledger))
    gate_result = m3_gate(model, Path("."))
    payload = {
        "m3_gate": asdict(gate_result),
        "subfamilies": [asdict(s) for s in computed_states(model)],
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if gate_result.passed else 1


def _interface(args: argparse.Namespace) -> int:
    from dynamisbench_soccer.qualification import interface

    rendered = interface.snapshot_json()
    target = Path(args.file)
    if args.write:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rendered, encoding="utf-8")
        print(f"wrote {target}")
        return 0
    if not target.is_file() or target.read_text(encoding="utf-8") != rendered:
        print(f"FAIL interface differs from the frozen snapshot {target}", file=sys.stderr)
        return 1
    print(f"{target} matches the live interface ({interface.INTERFACE_VERSION})")
    return 0


def _admit(args: argparse.Namespace) -> int:
    import json

    from dynamisbench_soccer.qualification import admission
    from dynamisbench_soccer.registry.models import load as load_registry

    try:
        proposal = admission.load(Path(args.proposal))
    except (ValueError, OSError) as exc:
        print(f"FAIL {exc}", file=sys.stderr)
        return 2
    registry = load_registry(Path("registry"))
    result = admission.evaluate(
        proposal,
        registry_claim_ids={c.id for c in registry.claims},
        registry_gap_subjects={g.subject for g in registry.gaps},
    )
    print(json.dumps(result.report(), indent=2, sort_keys=True))
    return 0 if result.decision == "ADMISSION_READY" else 1


def register(sub: argparse._SubParsersAction) -> None:  # type: ignore[type-arg]
    qual = sub.add_parser("qualification", help="scientific device quality-parity ledger")
    q = qual.add_subparsers(dest="qual_command", required=True)
    valid = q.add_parser("validate", help="verify dimension evidence and constraints")
    valid.add_argument("--ledger", default="qualification/ledger.toml")
    valid.set_defaults(handler=_validate)
    stage = q.add_parser("gate", help="show M3/M5/M8 gate (nonzero while blocked)")
    stage.add_argument("--ledger", default="qualification/ledger.toml")
    stage.add_argument("--stage", choices=[x.value for x in Stage], default="M3")
    stage.add_argument("--verbose", action="store_true")
    stage.set_defaults(handler=_gate)
    report = q.add_parser("ledger", help="generate deterministic ledger report")
    report.add_argument("--ledger", default="qualification/ledger.toml")
    report.add_argument("--doc", default="docs/qualification-ledger.md")
    mode = report.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    report.set_defaults(handler=_ledger)
    admit = q.add_parser("admit", help="evaluate a new-device-family admission proposal")
    admit.add_argument("proposal", help="proposal TOML")
    admit.set_defaults(handler=_admit)
    states = q.add_parser("states", help="computed per-subfamily states and the M3 gate")
    states.add_argument("--ledger", default="qualification/ledger.toml")
    states.set_defaults(handler=_states)
    iface = q.add_parser("interface", help="check or write the frozen engine-v0.1 interface")
    iface.add_argument("--file", default="docs/verification/engine-v0.1-interface.json")
    iface.add_argument("--write", action="store_true")
    iface.set_defaults(handler=_interface)
