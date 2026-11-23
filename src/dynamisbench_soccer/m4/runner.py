"""DEVELOPMENT_ONLY_NOT_QUALIFIED offline Linux candidate runner."""

from __future__ import annotations

import os
import signal
import subprocess
import sysconfig
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from dynamisbench_soccer.m4.contracts import Episode, Score, grade
from dynamisbench_soccer.m4.key_store import PrivateKey, digest

MAX_OUTPUT = 16 * 1024
MAX_SOURCE = 64 * 1024
WALL_SECONDS = 3
LauncherResult = Literal["graded", "execution_error", "timeout", "output_limit"]
PYTHON = Path("/usr/bin/python3")
BWRAP = Path("/usr/bin/bwrap")

_LIMITS = """
import resource
import runpy
for kind, limit in (
    (resource.RLIMIT_CPU, 1),
    (resource.RLIMIT_AS, 128 * 1024 * 1024),
    (resource.RLIMIT_FSIZE, 16 * 1024),
    (resource.RLIMIT_NOFILE, 32),
    (resource.RLIMIT_NPROC, 0),
    (resource.RLIMIT_CORE, 0),
):
    resource.setrlimit(kind, (limit, limit))
runpy.run_path("/candidate.py", run_name="__main__")
"""


def _command(candidate: Path) -> list[str]:
    if os.getuid() == 0:
        raise RuntimeError("M4 runner requires an unprivileged host user")
    # shortcut: use Ubuntu/Debian's trusted system Python layout, extend for other hosts on demand.
    stdlib = Path("/usr/lib") / PYTHON.resolve().name
    libraries = Path("/lib") / str(sysconfig.get_config_var("MULTIARCH"))
    loader = Path("/lib64") if Path("/lib64").is_dir() else libraries
    if not all(p.exists() for p in (PYTHON, BWRAP, stdlib, libraries, loader)):
        raise RuntimeError("M4 requires Linux bubblewrap and Debian/Ubuntu system Python")
    command = [
        str(BWRAP),
        "--unshare-all",
        "--unshare-user",
        "--die-with-parent",
        "--new-session",
        "--disable-userns",
        "--cap-drop",
        "ALL",
        "--clearenv",
        "--setenv",
        "LC_ALL",
        "C.UTF-8",
        "--chdir",
        "/",
        "--ro-bind",
        str(PYTHON),
        str(PYTHON),
        "--ro-bind",
        str(stdlib),
        str(stdlib),
        "--ro-bind",
        str(libraries),
        str(libraries),
    ]
    if loader != libraries:
        command += ["--ro-bind", str(loader), str(loader)]
    return [
        *command,
        "--ro-bind",
        str(candidate),
        "/candidate.py",
        "--remount-ro",
        "/",
        "--",
        str(PYTHON),
        "-I",
        "-S",
        "-B",
        "-c",
        _LIMITS,
    ]


@dataclass(frozen=True)
class Execution:
    outcome: LauncherResult
    stdout: bytes
    wall_seconds: float
    stdout_bytes: int
    workspace_removed: bool


def execute_payload(public: bytes, source: bytes) -> Execution:
    """Run only a candidate program and a bounded public stdin payload."""
    if len(source) > MAX_SOURCE:
        raise ValueError("candidate source exceeds size limit")
    if len(public) > MAX_SOURCE:
        raise ValueError("public episode exceeds size limit")
    started = time.monotonic()
    stdout = b""
    with tempfile.TemporaryDirectory(prefix="dbs-m4-") as tmp:
        workspace = Path(tmp)
        path = workspace / "candidate.py"
        path.write_bytes(source)
        with tempfile.TemporaryFile() as payload, tempfile.TemporaryFile() as output:
            payload.write(public)
            payload.seek(0)
            timed_out = False
            with subprocess.Popen(
                _command(path),
                stdin=payload,
                stdout=output,
                stderr=subprocess.DEVNULL,
                env={},
                close_fds=True,
                start_new_session=True,
            ) as child:
                try:
                    code = child.wait(timeout=WALL_SECONDS)
                except subprocess.TimeoutExpired:
                    timed_out = True
                    os.killpg(child.pid, signal.SIGKILL)
                    code = child.wait()
            size = output.seek(0, os.SEEK_END)
            outcome: LauncherResult
            if timed_out:
                outcome = "timeout"
            elif size >= MAX_OUTPUT:
                outcome = "output_limit"
            elif code:
                outcome = "execution_error"
            else:
                outcome = "graded"
                output.seek(0)
                stdout = output.read(MAX_OUTPUT)
    return Execution(outcome, stdout, time.monotonic() - started, size, not workspace.exists())


def execute(episode: Episode, source: bytes) -> tuple[LauncherResult, bytes]:
    """Compatibility entry point for the public RES-737 development demo."""
    attempt = execute_payload(episode.model_dump_json().encode(), source)
    return attempt.outcome, attempt.stdout


def trial(episode: Episode, source: bytes, key: PrivateKey) -> Score:
    if key.episode_sha256 != digest(episode.model_dump_json().encode()):
        raise ValueError("private answer belongs to another episode")
    outcome, output = execute(episode, source)
    if outcome != "graded":
        return Score(episode_id=episode.episode_id, outcome=outcome)
    return grade(
        episode, output, key.answer, atol=key.atol, alternative_formulas=key.alternative_formulas
    )
