# Signal-quality and trial-acceptance specification v0.1

Issue RES-668. Rule definitions: `qc/rules.toml`. The generated rule table is
[qc-rules.md](qc-rules.md). Reference evaluators are in
`src/dynamisbench_soccer/qc/checks.py`, and fixtures and fault injectors in
`src/dynamisbench_soccer/qc/faults.py`.

## Purpose

QC decides **whether a signal or trial may be used**, before any metric is computed.
It underpins competency C4 (trial QC and rejection) and the M5 instrument-validity
task family (RES-682). An agent that computes a perfect jump height from a trial that
should have been rejected has failed the task.

## Rule anatomy

| Field | Meaning |
| -- | -- |
| `id` | `QC-<group><nn>`. Groups: S signal, T force-plate trial, N Nordic, H hip/isometric, G gates, R radar |
| `level` | `signal` (instrument or recording problem) or `trial` (protocol or athlete problem) |
| `applies_to` | ontology test ids |
| `severity` | `reject`: unusable. `flag`: usable, but must be reported with the result. `note`: informational |
| `basis` | `evidence`: must cite registry claim ids. `metadata`: a consistency check of declared metadata |
| `params` | **defaults**. A task may override them, and must state the override |
| `rationale` | why the rule exists, linked to claims |

## Outcomes and verdict

Each evaluator returns one of three outcomes:

* `pass`
* `fail`, with a human-readable detail and numeric values
* `not_applicable`, with the reason; for example, there is no time column, or no
  reference session

`verdict()` combines all applicable rules:

* `reject` if any rule with severity `reject` fails
* `accept_with_flags` if only `flag` or `note` rules fail
* otherwise `accept`

## Groups

* **Signal (S01–S08):** zero offset, start-to-end drift, clipping/saturation,
  timestamp gaps and duplicates, sampling-rate adequacy and declared-versus-measured
  mismatch, 50/60 Hz mains interference, dead or unmapped limb channel, and L/R swap
  against a reference session.
* **Force-plate trial (T01–T08):**
  * quiet-stance stability, and at least 1 s of weighing before movement onset
  * a missing flight phase, or multiple flight phases (double contact or step-off)
  * the arm-swing protocol (a metadata rule)
  * incomplete landing, implausible flight time, and countermovement in a squat jump
* **Dynamometry (N01–N02, H01–H03):** NHE break point not captured, NHE lever arm
  missing, squeeze pre-tension, hold duration, and hip angle missing.
* **Sprint (G01–G02, R01–R02):** gate order, plausible split speeds (premature beam
  breaks), radar start captured, and radar acceleration spikes.

## Fixtures and testing

`FIXTURES` maps every rule id to:

* a **negative** fixture, a clean bundle that must give `pass`
* a **positive** fixture, the same bundle with one injected fault, which must give `fail`
* a **reference** bundle, for comparison rules such as QC-S08

`tests/test_qc.py` checks three things:

* Every rule has an evaluator, and every evaluator has a rule.
* Every cited claim exists in the registry.
* Every fixture classifies correctly, and the clean examples are accepted by all
  applicable rules.

## Known limitations (v0.1)

* The thresholds are engineering defaults. They are traceable to the cited claims but
  not individually validated, and M8 expert review may revise them.
* QC-S03 detects saturation as a flat run at the channel maximum. It assumes real
  signals carry sensor noise, and noiseless synthetic plateaus would false-positive.
  For that reason the shape fixtures add noise where plateaus exist.
* QC-S08 needs an earlier session from the same athlete. Without one it returns
  `not_applicable`, and a swap cannot be detected from a single trial.
* Arm swing is checked from declared metadata only (QC-T05). Detecting it from force
  alone is not attempted.
* The isometric hamstring test (`iso_ham`) is a registry gap. QC-H01/H02 apply to it,
  but it has no core tasks.
