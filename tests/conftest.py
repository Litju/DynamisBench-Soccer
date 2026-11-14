from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _isolated_dbs_homes(tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch):
    """Never let tests touch the real store, staging or cache directories."""
    base = tmp_path_factory.mktemp("dbs-homes")
    monkeypatch.setenv("DBS_DATA_HOME", str(base / "data"))
    monkeypatch.setenv("DBS_CACHE_HOME", str(base / "cache"))
    return base


@pytest.fixture
def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]
