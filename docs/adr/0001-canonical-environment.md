# ADR-0001: Canonical development environment

* **Status:** Accepted (2026-11-13)
* **Issue:** RES-661
* **Decision:** WSL2 Ubuntu primary, Windows for acquisition compatibility, Linux CI for reproducibility.

## Context

DynamisBench-Soccer produces scientific references (signal processing, metrics,
graders) whose results must be reproducible bit-for-bit. Development happens on a
Windows host with WSL2. Some measurement devices only have Windows acquisition
software. Supporting two scientific runtimes would double the verification burden
and invite silent interpreter, path and line-ending drift.

## Decision

DynamisBench-Soccer SHALL use WSL2 Ubuntu as its canonical development, testing
and execution environment.

* Operating system: Linux (WSL2 Ubuntu)
* Primary language: Python 3.12+ (pinned in `.python-version`)
* Dependency management: `uv` (`uv.lock` committed; CI uses `uv sync --frozen`)
* Primary CI: Ubuntu Linux (GitHub Actions)
* Canonical local location: `~/projects/DynamisBench-Soccer`

### Mandatory rules

1. Python tooling, tests, Git operations, dependency management and scientific
   evaluation run inside WSL2.
2. The working Git repository lives on the native Linux filesystem, not under
   `/mnt/c` or `/mnt/e`.
3. Development uses VS Code Remote WSL.
4. Windows-specific device acquisition and proprietary instrument software stay
   isolated behind documented adapters (`docs/acquisition-boundary.md`).
5. Large datasets are never duplicated into Git repositories, worktrees, or multiple
   virtual environments.
6. Datasets are referenced by versioned external manifests with integrity hashes and
   staged into temporary staging directories (`docs/data-policy.md`).
7. No Docker or persistent services unless an actual isolation requirement
   justifies them (record a new ADR if so).
8. Execution is bounded-memory and suitable for an 8 GB host.
9. Linux CI can independently install, build, test and validate the project.
10. Scientific contracts and file formats are portable. Windows compatibility may be
    tested separately; Windows is not the canonical scientific runtime.
11. Windows and Linux Python interpreters, virtual environments, dependency caches
    and Git tooling are never silently mixed.
12. A reproducible cleanup policy covers temporary benchmark artifacts, test
    workspaces and build caches (`dbs clean`).
13. No changes to DynamisSimBench.

## Enforcement

* `dbs env-check` (and `tests/test_envcheck.py`, run in CI) fails on Windows mounts,
  Windows interpreters, and `UV_CACHE_DIR` under `/mnt/<drive>`.
* `.gitattributes` forces LF line endings.
* CI checks that no large or binary data files are committed (RES-662).

## Consequences

* One scientific runtime to verify. Windows acquisition stays an import concern.
* Contributors on Windows must use WSL2. Native Windows Python is unsupported for
  scientific results.
