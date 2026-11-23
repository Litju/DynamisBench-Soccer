"""Offline checkpoint execution and host-side provenance for formal tasks."""

from __future__ import annotations

import json
import platform
import subprocess
from dataclasses import dataclass
from pathlib import Path

from dynamisbench_soccer.m4.admission import Admission, _read, evaluate
from dynamisbench_soccer.m4.custody import check_projection, load_private
from dynamisbench_soccer.m4.grader import Outcome, score_checkpoint
from dynamisbench_soccer.m4.key_store import digest
from dynamisbench_soccer.m4.runner import BWRAP, MAX_SOURCE, PYTHON, Execution, execute_payload
from dynamisbench_soccer.m4.task_registry import RegistryTask


@dataclass(frozen=True)
class TrialContext:
    root: Path
    catalog: bytes
    trusted_catalog_sha256: str
    custody_root: Path


@dataclass(frozen=True)
class Attempt:
    checkpoint: str
    candidate_payload_sha256: str
    output_sha256: str
    execution: Execution
    grade: Outcome


@dataclass(frozen=True)
class TrialReport:
    admission: Admission
    task_id: str
    candidate_id: str
    candidate_sha256: str
    engine: str
    grader: str
    seed: int
    environment: dict[str, str]
    attempts: tuple[Attempt, ...]
    terminal_reason: str | None = None
    version: str = "m4.trial.v1"


def run_episode(
    context: TrialContext, task: RegistryTask, candidate_id: str, source: bytes, seed: int
) -> TrialReport:
    root = context.root.resolve(strict=True)
    state = evaluate(root, task, context.catalog, context.trusted_catalog_sha256)
    attempts: list[Attempt] = []
    environment = {
        "platform": platform.platform(),
        "host_python": platform.python_version(),
        "candidate_python": subprocess.check_output([PYTHON, "--version"], env={}).decode().strip(),
        "python_binary_sha256": digest(PYTHON.read_bytes()),
        "sandbox_binary_sha256": digest(BWRAP.read_bytes()),
        "lockfile_sha256": digest(_read(root, "uv.lock")),
    }
    terminal_reason = None
    if state.status != "REFUSED":
        inputs = {artifact.name: _read(root, artifact.path) for artifact in task.artifacts}
        review = state.reviewed_slice
        private = None
        if state.status == "ENGINEERING_ADMITTED" and review is not None:
            try:
                private = load_private(context.custody_root, root, review.answer, state.task_sha256)
            except (OSError, ValueError):
                terminal_reason = "invalid_private_reference"
        if len(source) > MAX_SOURCE:
            terminal_reason = "candidate_source_limit"
        for checkpoint in task.checkpoints:
            if terminal_reason is not None:
                break
            visible = json.loads(task.candidate_payload(checkpoint.id, inputs, seed))
            visible["previous"] = [
                {
                    "checkpoint": attempt.checkpoint,
                    "stdout": attempt.execution.stdout.decode("utf-8", errors="replace"),
                    "outcome": attempt.execution.outcome,
                }
                for attempt in attempts
            ]
            payload = json.dumps(visible, sort_keys=True).encode()
            if len(payload) > MAX_SOURCE:
                terminal_reason = "checkpoint_payload_limit"
                break
            if private is not None and review is not None:
                try:
                    check_projection(payload, private, review.answer)
                except ValueError:
                    terminal_reason = "projection_rejected"
                    break
            execution = execute_payload(payload, source)
            scored = score_checkpoint(
                root,
                task,
                context.catalog,
                context.trusted_catalog_sha256,
                context.custody_root,
                checkpoint.id,
                inputs,
                execution.stdout,
            )
            if (
                state.status != "ENGINEERING_ADMITTED"
                or scored.outcome == "EXCLUDED_FROM_SCORED_BENCHMARK"
            ):
                scored = Outcome(outcome="EXCLUDED_FROM_SCORED_BENCHMARK", findings=scored.findings)
            elif execution.outcome != "graded":
                scored = Outcome(outcome="FAIL", findings=(execution.outcome,))
            attempts.append(
                Attempt(checkpoint.id, digest(payload), digest(execution.stdout), execution, scored)
            )
    return TrialReport(
        state,
        task.task.id,
        candidate_id,
        digest(source),
        task.engine,
        task.grader,
        seed,
        environment,
        tuple(attempts),
        terminal_reason,
    )


# The oracle exists only in controlled tests with external host-side answer custody.
STUBS = {
    "null": b"print('{}')\n",
    "naive": b'print(\'{"value": 1, "unit": "ms"}\')\n',
    "random": (
        b"import json, random, sys\n"
        b"print(json.dumps({'value': random.Random(json.load(sys.stdin)['seed']).random()}))\n"
    ),
}


def run_oracle_control(
    context: TrialContext, task: RegistryTask, independent_candidate: bytes, seed: int
) -> TrialReport:
    """Verify a test-only solver inside the sandbox against host-held answers."""
    report = run_episode(context, task, "oracle", independent_candidate, seed)
    if (
        report.admission.status != "ENGINEERING_ADMITTED"
        or report.terminal_reason is not None
        or len(report.attempts) != len(task.checkpoints)
        or any(
            attempt.grade.outcome != "PASS"
            or attempt.execution.outcome != "graded"
            or not attempt.execution.workspace_removed
            for attempt in report.attempts
        )
    ):
        raise ValueError("oracle control did not pass all admitted checkpoints")
    return report
