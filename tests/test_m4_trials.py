from pathlib import Path

import pytest

from dynamisbench_soccer.m4.key_store import digest
from dynamisbench_soccer.m4.task_registry import load_registry
from dynamisbench_soccer.m4.trials import STUBS, TrialContext, run_episode

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name", ["naive", "null", "random"])
def test_offline_stubs_are_reproducible_and_excluded(name: str, tmp_path: Path) -> None:
    catalog = (ROOT / "benchmark/reviewed-slices.json").read_bytes()
    context = TrialContext(ROOT, catalog, digest(catalog), tmp_path)
    task = load_registry(ROOT / "benchmark/tasks")["elapsed-demo", 1]
    first, second = (run_episode(context, task, name, STUBS[name], seed=17) for _ in range(2))
    assert first.environment == second.environment
    assert first.admission == second.admission
    assert [a.checkpoint for a in first.attempts] == ["initial", "correction"]
    assert [a.output_sha256 for a in first.attempts] == [a.output_sha256 for a in second.attempts]
    assert all(a.execution.workspace_removed for a in first.attempts)
    assert all(a.grade.outcome == "EXCLUDED_FROM_SCORED_BENCHMARK" for a in first.attempts)


def test_future_checkpoint_is_withheld(tmp_path: Path) -> None:
    catalog = (ROOT / "benchmark/reviewed-slices.json").read_bytes()
    context = TrialContext(ROOT, catalog, digest(catalog), tmp_path)
    task = load_registry(ROOT / "benchmark/tasks")["elapsed-demo", 1]
    source = b"""
import json
import sys
data = json.load(sys.stdin)
initial = data['checkpoint']['id'] == 'initial'
assert ('elapsed-correction' in data['inputs']) != initial
assert bool(data['previous']) != initial
print('ok')
"""
    result = run_episode(context, task, "checkpoint-control", source, seed=17)
    assert [attempt.execution.stdout for attempt in result.attempts] == [b"ok\n", b"ok\n"]


def test_large_checkpoint_history_preserves_completed_attempts(tmp_path: Path) -> None:
    catalog = (ROOT / "benchmark/reviewed-slices.json").read_bytes()
    context = TrialContext(ROOT, catalog, digest(catalog), tmp_path)
    task = load_registry(ROOT / "benchmark/tasks")["elapsed-demo", 1]
    result = run_episode(context, task, "large-output", b"print(chr(0)*15000)", seed=17)
    assert result.terminal_reason == "checkpoint_payload_limit"
    assert len(result.attempts) == 1
    assert result.attempts[0].execution.workspace_removed
    assert result.attempts[0].grade.outcome == "EXCLUDED_FROM_SCORED_BENCHMARK"
