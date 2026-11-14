from pathlib import Path

import pytest

from dynamisbench_soccer.cli import main
from dynamisbench_soccer.envcheck import check_environment, is_windows_mount


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("/mnt/c/Users/x/repo", True),
        ("/mnt/e", True),
        ("/mnt/wsl/shared", False),
        ("/home/user/projects/DynamisBench-Soccer", False),
        ("/mnt/data", False),
    ],
)
def test_is_windows_mount(path: str, expected: bool) -> None:
    assert is_windows_mount(path) is expected


def test_current_environment_is_canonical() -> None:
    # CI and local WSL2 must both pass this; it is the guard in executable form.
    report = check_environment(Path(__file__).resolve().parents[1])
    assert report.ok, report.errors


def test_repo_on_windows_mount_is_rejected() -> None:
    report = check_environment(Path("/mnt/c/projects/DynamisBench-Soccer"))
    assert not report.ok
    assert any("Windows mount" in e for e in report.errors)


def test_cli_env_check_exit_code(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["env-check"]) == 0
    assert "environment OK" in capsys.readouterr().out
