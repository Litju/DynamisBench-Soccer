"""Evaluate withheld diagnostic controls and report error counts without disclosing answers."""

from __future__ import annotations

import argparse
import json
import stat
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import Field, JsonValue

from dynamisbench_soccer.m4.grader import ExpectedMethod, check_control
from dynamisbench_soccer.m4.task_registry import Nonnegative, Schema, Text

Category = Literal["positive", "negative", "adversarial", "refusal"]
CATEGORIES: tuple[Category, ...] = ("positive", "negative", "adversarial", "refusal")


class Case(Schema):
    category: Category
    label: Text
    expected_pass: bool
    submission: dict[str, JsonValue]
    reference: ExpectedMethod
    tolerance: Nonnegative


class Controls(Schema):
    cases: tuple[Case, ...] = Field(min_length=1)


def evaluate_controls(controls: Controls) -> dict[str, object]:
    counts = {group: Counter[str]() for group in CATEGORIES}
    findings: Counter[str] = Counter()
    labels = {case.label for case in controls.cases}
    if len(labels) != len(controls.cases):
        raise ValueError("duplicate control label")
    replay_differences = 0
    for case in controls.cases:
        submission = json.dumps(case.submission, sort_keys=True).encode()
        result = check_control(submission, case.reference, case.tolerance)
        replay = check_control(submission, case.reference, case.tolerance)
        replay_differences += result != replay
        findings.update(result)
        summary = counts[case.category]
        summary["total"] += 1
        summary["expected_passes"] += case.expected_pass
        summary["expected_rejections"] += not case.expected_pass
        summary["false_passes"] += not result and not case.expected_pass
        summary["false_rejections"] += bool(result) and case.expected_pass
    enough = all(counts[category]["total"] >= 2 for category in CATEGORIES)
    correct = all(not n["false_passes"] and not n["false_rejections"] for n in counts.values())
    return {
        "classes": counts,
        "findings": dict(sorted(findings.items())),
        "replay_differences": replay_differences,
        "diagnostic_thresholds_met": enough and correct and not replay_differences,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("controls", type=Path)
    path = parser.parse_args().controls
    root = Path(__file__).resolve().parents[1]
    if path.resolve().is_relative_to(root) or stat.S_IMODE(path.stat().st_mode) & 0o077:
        parser.error("controls must be external and owner-private")
    try:
        report = evaluate_controls(Controls.model_validate_json(path.read_bytes()))
    except (ValueError, RecursionError):
        parser.exit(2, "invalid held-out control bundle\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    if not report["diagnostic_thresholds_met"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
