# Development environment

Canonical decision: [ADR-0001](adr/0001-canonical-environment.md).

## One-time setup (WSL2 Ubuntu)

```bash
# inside WSL2, never from a Windows terminal path
mkdir -p ~/projects && cd ~/projects
git clone https://github.com/Litju/DynamisBench-Soccer
cd DynamisBench-Soccer
curl -LsSf https://astral.sh/uv/install.sh | sh   # if uv is missing
uv sync --frozen                                   # installs Python 3.12 + deps into .venv
uv run dbs env-check
```

Open the folder with **VS Code Remote WSL** (`code .` from the WSL shell). The
committed `.vscode/settings.json` points the interpreter at `.venv/bin/python`.

## Forbidden

| Don't | Why |
| -- | -- |
| Clone or edit under `/mnt/c`, `/mnt/e` | Slow 9P I/O, permission/line-ending drift (rule 2) |
| Run Windows `python.exe`, `pip` or Git for Windows on this repo | Mixed interpreters and caches (rule 11) |
| Point `UV_CACHE_DIR` at a Windows drive | Mixed caches (rule 11) |
| Commit datasets or run outputs | Use manifests and staging (rules 5–6) |
| Add Docker or long-running services | Not justified (rule 7) |

`dbs env-check` detects the first three automatically.

## Everyday commands

```bash
uv run ruff check . && uv run ruff format --check .
uv run mypy
uv run pytest
uv run dbs clean --dry-run    # see docs/data-policy.md
```

## Memory budget (8 GB host)

* Read signals in chunks (Parquet row groups) and never load a whole dataset into
  memory.
* Prefer `numpy` arrays of `float64` per trial. A 10 s, 4-channel, 2000 Hz trial is
  about 0.6 MB, so per-trial processing is cheap. Squad- or season-level work must
  stream over trials.
* Tests that exercise large inputs carry `@pytest.mark.memory_bounded` and assert
  peak RSS where it matters.

## Windows

Windows is used only to run proprietary acquisition software. Exports cross into
WSL2 through the hand-off boundary in `docs/acquisition-boundary.md`. A separate,
non-blocking Windows compatibility check may be added later. It never produces
scientific results.
