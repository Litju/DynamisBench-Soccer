"""DEVELOPMENT_ONLY_NOT_QUALIFIED runner boundary controls."""

import socket
from pathlib import Path

import pytest

from dynamisbench_soccer.m4.contracts import Episode, Task
from dynamisbench_soccer.m4.runner import MAX_SOURCE, execute


@pytest.fixture
def episode():
    return Episode(
        episode_id="isolation",
        seed=72,
        task=Task(
            task_id="test",
            evidence="synthetic",
            prompt="Compute.",
            inputs={},
            estimand="test",
            unit="m",
            method_spec={},
            provenance="test",
        ),
    )


def test_sandbox_hides_host_files_environment_and_network(episode, tmp_path, monkeypatch):
    private = tmp_path / "answer.json"
    private.write_text("unique-answer-canary")
    monkeypatch.setenv("M4_HOST_SECRET", "unique-environment-canary")
    project = Path(__file__).resolve()
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen()
        port = server.getsockname()[1]
        source = f"""
import importlib.util
import json
import os
import pathlib
import socket
import sys
assert json.load(sys.stdin)['task']['status'] == 'DEVELOPMENT_ONLY_NOT_QUALIFIED'
assert set(os.listdir('/')) <= {{'candidate.py', 'lib', 'lib64', 'usr'}}
for name in ({str(private)!r}, {str(project)!r}, '/proc/self/environ', '/etc/passwd'):
    assert not pathlib.Path(name).exists()
assert importlib.util.find_spec('dynamisbench_soccer') is None
assert 'M4_HOST_SECRET' not in os.environ
try:
    socket.create_connection(('127.0.0.1', {port}), timeout=0.2)
except OSError:
    pass
else:
    raise AssertionError('host network reachable')
try:
    os.fork()
except OSError:
    pass
else:
    raise AssertionError('process creation allowed')
try:
    pathlib.Path('/write').write_text('hi')
except OSError:
    pass
else:
    raise AssertionError('sandbox root writable')
print('isolated')
"""
        assert execute(episode, source.encode()) == ("graded", b"isolated\n")


@pytest.mark.parametrize(
    ("source", "outcome"),
    [
        (b"import time; time.sleep(30)", "timeout"),
        (b"while True: pass", "execution_error"),
        (b"print('x'*100000)", "output_limit"),
        (b"x = bytearray(256*1024*1024)", "execution_error"),
    ],
)
def test_trial_enforces_limits(episode, source, outcome):
    assert execute(episode, source) == (outcome, b"")


def test_runner_refuses_oversized_program(episode):
    with pytest.raises(ValueError, match="size limit"):
        execute(episode, b" " * (MAX_SOURCE + 1))
