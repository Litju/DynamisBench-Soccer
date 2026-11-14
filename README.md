# DynamisBench-Soccer

An evaluation benchmark for **AI and LLM agents** doing the scientific work of
**neuromuscular performance testing in association football (soccer)**:

1. **Signal processing** of raw device data. Every supported device family is held to
   the same qualification standard: force plates, LPTs and encoders, load cells and
   isometric dynamometers, Nordic and hip devices, timing gates and contact systems,
   radar and laser, GNSS and LPS, and IMUs (see
   [`docs/m3-architecture.md`](docs/m3-architecture.md)).
2. **Test calculation** of metrics and variables, with explicit formula variants and
   measurement uncertainty.
3. **Soccer-specific interpretation**: match-day (MD±x) timing, fixture congestion,
   exposure, position, sex, and academy maturation.
4. **Application**: individual monitoring, season-long evidence revision, and
   return-to-performance reasoning. Medical clearance always stays with clinicians.

> **Status:** pre-release foundation (roadmap milestones M0–M2). No task is
> scientifically qualified and no model evaluation campaign is authorized.

This project evaluates AI systems. It is **not** a physical simulator, not athlete
prediction software, and not a medical clearance system. It is independent of
DynamisSimBench.

## Quick start (WSL2 Ubuntu / Linux)

```bash
git clone https://github.com/Litju/DynamisBench-Soccer ~/projects/DynamisBench-Soccer
cd ~/projects/DynamisBench-Soccer
uv sync --frozen
uv run dbs env-check          # refuses /mnt/c, /mnt/e and Windows interpreters
uv run pytest
```

See [`docs/environment.md`](docs/environment.md) for the canonical environment and
[`docs/adr/0001-canonical-environment.md`](docs/adr/0001-canonical-environment.md)
for the decision record.

## Repository map

| Path | Contents |
| -- | -- |
| `src/dynamisbench_soccer/` | Python package (`dbs` CLI) |
| `registry/` | Source-to-claim evidence registry (TOML) |
| `ontology/` | Competencies, tests, contexts, coverage matrix, decision boundaries |
| `qc/` | Machine-readable signal and trial QC rules |
| `qualification/` | Device-family qualification ledger (dimensions A–O, computed states) |
| `data/manifests/` | External dataset manifests (hashes only, never data) |
| `docs/` | Environment, policies, specifications, audits |

## Licensing

Code: Apache-2.0. Benchmark content: CC BY 4.0 (see `LICENSE-CONTENT.md`).
