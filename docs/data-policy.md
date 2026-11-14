# Data, staging and cleanup policy

Implements ADR-0001 rules 5, 6, 8 and 12 (issue RES-662).

## Principles

1. **No data in Git.** Git holds *manifests* (JSON, `data/manifests/`) describing data:
   source, license, category, allowed usage, and per-file SHA-256 and size. CI runs
   `dbs repo-check`, which fails on data file types (`.parquet`, `.c3d`, `.h5`,
   `.npz`, archives, …), on CSV/TSV/TXT outside `tests/fixtures/`, on fixtures over
   256 KiB, and on any tracked file over 512 KiB.
2. **One physical copy.** Files live once in a content-addressed store
   (`$DBS_DATA_HOME/store/sha256/ab/<sha256>`, read-only). Worktrees, venvs and
   staging directories link to it and never copy it.
3. **Fail closed.** Ingest, fetch and stage re-hash bytes. Any size or hash mismatch
   raises `IntegrityError`, and nothing partial is installed.
4. **Categories never mix.** Every manifest declares one `data_category`:
   `empirical | analytical | synthetic | expert-adjudicated`.
5. **Usage is decided by audit.** `usage` is `task-input`, `cross-validation-only` or
   `excluded`, set by the data audit (`docs/data-audit.md`, RES-666).

## Locations

| Variable | Default | Holds |
| -- | -- | -- |
| `DBS_DATA_HOME` | `$XDG_DATA_HOME/dynamisbench-soccer` (`~/.local/share/...`) | content-addressed store |
| `DBS_CACHE_HOME` | `$XDG_CACHE_HOME/dynamisbench-soccer` (`~/.cache/...`) | `staging/`, `workspaces/`, `runs/` |

Both must be on the Linux filesystem. Never point them at `/mnt/<drive>`.

## Workflow

```bash
# 1. describe a dataset you have locally (hashes every file)
uv run dbs data manifest-create /path/to/export --output data/manifests/<id>.json \
    --dataset-id <id> --version 1.0.0 --title "..." --category empirical \
    --usage cross-validation-only --license CC-BY-4.0 --source-location https://...
# 2. put the bytes into the shared store (verified)
uv run dbs data ingest data/manifests/<id>.json /path/to/export
#    ...or download files that declare a url
uv run dbs data fetch data/manifests/<id>.json
# 3. verify at any time
uv run dbs data verify data/manifests/<id>.json
# 4. stage a temporary read-only view for a run (prints the directory)
uv run dbs data stage data/manifests/<id>.json
```

## Cleanup and retention

`dbs clean [--artifacts] [--workspaces] [--caches] [--older-than-hours H] [--dry-run]`
(no target flag selects all three).

| Target | Removes | Retention rule |
| -- | -- | -- |
| `artifacts` | `$DBS_CACHE_HOME/runs/*`, repo `var/*`, `runs/*` | Keep until results are summarized into a committed report. Then clean, typically `--older-than-hours 168`. |
| `workspaces` | `$DBS_CACHE_HOME/staging/*`, `workspaces/*` | Ephemeral. Clean after every run, or `--older-than-hours 24`. |
| `caches` | `__pycache__`, `.pytest_cache`, `.mypy_cache`, `.ruff_cache`, `dist/`, `build/` in the repo (skips `.git`, `.venv`) | Any time. |

Guarantees (tested in `tests/test_clean_repocheck.py`):

* `--dry-run` deletes nothing.
* Only direct children of the known roots and the named cache directories are
  removed. Symlinks are unlinked, never followed.
* The **store is never cleaned** by `dbs clean`.
* uv's global cache is managed separately with `uv cache prune` (never deleted
  implicitly).

## Memory

Hashing and copying stream in 1 MiB chunks, so dataset size does not affect peak
memory.
