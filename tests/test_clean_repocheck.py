import os
import time
from pathlib import Path

import pytest

from dynamisbench_soccer import repocheck
from dynamisbench_soccer.cli import main
from dynamisbench_soccer.data import clean, paths


def _populate(repo: Path) -> dict[str, Path]:
    run = paths.runs_root() / "run-1"
    run.mkdir(parents=True)
    (run / "scores.json").write_text("{}")
    staging = paths.staging_root() / "toy-1-abc"
    staging.mkdir(parents=True)
    outside = repo.parent / "outside.txt"
    outside.write_text("must survive")
    (staging / "link.csv").symlink_to(outside)
    cache = repo / "pkg" / "__pycache__"
    cache.mkdir(parents=True)
    (cache / "m.pyc").write_bytes(b"x")
    (repo / ".venv" / "lib" / "__pycache__").mkdir(parents=True)
    src = repo / "pkg" / "keep.py"
    src.write_text("print(1)\n")
    return {"run": run, "staging": staging, "cache": cache, "outside": outside, "src": src}


def test_dry_run_lists_but_deletes_nothing(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    made = _populate(repo)
    planned = clean.plan(clean.TARGETS, repo)
    planned_paths = {c.path for c in planned}
    assert planned_paths == {made["run"], made["staging"], made["cache"]}
    assert main(["clean", "--dry-run", "--repo-root", str(repo)]) == 0
    assert all(p.exists() for p in made.values())


def test_apply_removes_only_targets_and_never_follows_symlinks(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    made = _populate(repo)
    assert main(["clean", "--repo-root", str(repo)]) == 0
    assert not made["run"].exists()
    assert not made["staging"].exists()
    assert not made["cache"].exists()
    assert made["outside"].read_text() == "must survive"
    assert made["src"].exists()
    assert (repo / ".venv" / "lib" / "__pycache__").exists()  # venv is skipped


def test_target_selection_and_age_filter(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    made = _populate(repo)
    only_ws = clean.plan(["workspaces"], repo)
    assert [c.path for c in only_ws] == [made["staging"]]
    assert clean.plan(["artifacts"], repo, older_than_hours=1) == []
    old = time.time() - 7200
    os.utime(made["run"], (old, old))
    assert [c.path for c in clean.plan(["artifacts"], repo, older_than_hours=1)] == [made["run"]]
    with pytest.raises(ValueError):
        clean.plan(["store"], repo)


def test_store_is_never_a_clean_target(tmp_path: Path) -> None:
    blob = paths.store_root() / "ab" / ("ab" + "0" * 62)
    blob.parent.mkdir(parents=True)
    blob.write_text("data")
    clean.apply(clean.plan(clean.TARGETS, tmp_path))
    assert blob.exists()


def test_repocheck_flags_data_and_large_files(tmp_path: Path) -> None:
    (tmp_path / "tests" / "fixtures").mkdir(parents=True)
    (tmp_path / "tests" / "fixtures" / "ok.csv").write_text("a\n1\n")
    (tmp_path / "trial.csv").write_text("a\n1\n")
    (tmp_path / "x.parquet").write_bytes(b"PAR1")
    (tmp_path / "big.py").write_bytes(b"#" * (repocheck.DEFAULT_MAX_BYTES + 1))
    files = ["tests/fixtures/ok.csv", "trial.csv", "x.parquet", "big.py"]
    flagged = {v.path for v in repocheck.check(tmp_path, files)}
    assert flagged == {"trial.csv", "x.parquet", "big.py"}


def test_repocheck_passes_on_this_repository(repo_root: Path) -> None:
    assert repocheck.check(repo_root) == []
