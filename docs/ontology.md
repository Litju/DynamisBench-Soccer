# Ontology v0.1

Source of truth: `ontology/ontology.toml`, `ontology/coverage.toml` (RES-664).
Validate with `uv run dbs ontology validate`. The generated matrix is
[coverage-matrix.md](coverage-matrix.md).

## Epistemic classes

Every statement an evaluated agent makes is graded as one of four classes.

| Class | Meaning | The agent may |
| -- | -- | -- |
| `measured` | Instrument output after calibration and unit conversion | Report with units, sampling and device provenance |
| `derived` | Computed by a declared method | Report with `method_spec` and uncertainty, and compare only like estimands |
| `inference` | Conclusion from derived values plus error and context | State with uncertainty and the registry claims it rests on |
| `unsupported_explanation` | Physiological or mechanistic cause not measured | Only as a labelled hypothesis, never as a finding |

## Estimands

An estimand is named `quantity[.variant][.side]` in dotted snake_case, for example
`jump_height.takeoff_velocity`, `jump_height.flight_time`, `peak_force.net_per_kg`,
`rfd.0_100ms`, `peak_force.left`. **Different estimands are different answers**. A
flight-time jump height submitted for a take-off-velocity task is wrong even if the
number is "close" (claims FP-003, DB-06).

## Competencies

| Id | Competency |
| -- | -- |
| C1 | Signal handling: units, sampling, calibration, drift, multi-plate summation |
| C2 | Event detection: body weight, onset, take-off, landing, phases, contacts |
| C3 | Metric computation for a declared estimand and method |
| C4 | Trial QC and rejection, instrument fault detection |
| C5 | Measurement error: TE, CV, SWC, MDC, ICC form, agreement |
| C6 | Soccer-context interpretation |
| C7 | Longitudinal evidence revision without temporal leakage |
| C8 | Return-to-performance reasoning within clinician authority |

## Tests

Fourteen v0.1 test ids, identical to `registry.models.V01_TESTS`: `cmj`, `sj`,
`slcmj`, `dj`, `rj_10_5`, `imtp`, `nhe`, `hip_add_abd`, `iso_ham`, `sprint_gates`,
`sprint_fv`, `cod_505`, `ift_30_15`, `rsa`.

* `iso_ham` is a **declared registry gap** (`registry/gaps.toml`), so all of its
  matrix cells are extension until a verified source is added.
* `ift_30_15` and `rsa` are **context-only**. They may be core only for C3 (calculation)
  and C6 (context), never as neuromuscular primaries.

## Soccer context dimensions

level, sex, match_day (MD-4 … MD+3), schedule (single match week vs congested),
season_phase, maturation (PHV bands), injury_state (the RTP continuum), and
injury_type (hamstring strain, adductor-related groin pain, late-stage ACL).
Core vs extension status per value is in the coverage matrix.

## How M4 uses this

The task schema (RES-676) will reference competency ids, test ids, estimands, context
values, boundary rule ids and coverage cells. A task may target a core or extension
cell, but v0.1 qualification counts core cells only.
