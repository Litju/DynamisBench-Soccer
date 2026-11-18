# `shorts-laser`: source pin and rights disposition (RES-720)

**Assessed:** 2026-11-17 · **Records:** [`data/external/shorts-laser/source-lock.json`](../../data/external/shorts-laser/source-lock.json),
[`data/external/shorts-laser/rights.json`](../../data/external/shorts-laser/rights.json)
· **Decision:** **canonical admission PENDING (blocked)**. Local quarantine inspection is permitted under the MIT grant only. No `data/manifests/` entry exists, and nothing enters the canonical store.

## 1. Exact source (Gate 1, verified)

Upstream `mladenjovanovic/shorts`, commit `5f8ce5554b5aca0e857199e0eb50bf66d7089130`.
At assessment, that commit was also the `master` head. Each file was downloaded
from the commit-pinned raw URL. Its size and SHA-256 were computed locally, and its
Git blob SHA-1 was computed locally and matched against the GitHub tree API for
that commit.

| Path | Role | Bytes | SHA-256 | Git blob SHA-1 |
| -- | -- | --: | -- | -- |
| `data-raw/laser-gun-data.csv` | data | 160957 | `52b48d928460a656775e8df2aa1824c84b214040e281d4992754fbac73f3c6ba` | `4423e32f76b7…` |
| `LICENSE` | license | 46 | `73393882eb03…` | `8727fe2ce3e9…` |
| `LICENSE.md` | license | 1076 | `84a0cf8b871c…` | `0aea7661a675…` |
| `DESCRIPTION` | documentation | 1684 | `8c25e7b1e1cd…` | `6476b2522a61…` |
| `R/data-laser-gun.R` | documentation | 1459 | `210d9b26ac46…` | `b7cfbec0da05…` |
| `data-raw/laser-gun-data.R` | transform script (never executed) | 1357 | `048b3be912f9…` | `ed2ac50f2f3f…` |

Full digests are in the lock. `dbs external acquire shorts-laser` re-downloads
(bounded to pinned size + 1 byte) or reads a local copy. It refuses symlinks,
size/SHA-256/blob mismatches and non-pinned URLs, and installs read-only files
under `$DBS_CACHE_HOME/quarantine/shorts-laser/<commit>/`.

## 2. What the source is (from the pinned `R/data-laser-gun.R`)

* One **35 m standing-start sprint by a youth basketball player**. Not soccer, not a
  cohort.
* Noptel Oy CMP3 Distance Sensor, **native 2.56 kHz**. Musclelab v10.232.107.5298
  fitted a **polynomial distance-time model** and resampled it to **1 kHz**.
  Every CSV column (`time,distance,velocity,raw_velocity`) is therefore
  **manufacturer-processed**. The file contains no native range samples, and
  `raw_velocity` is a vendor "unsmoothed" export, not native raw data.
* `data-raw/laser-gun-data.R` builds the *package* object (`velocity - min`, trims
  `time > 0.375` then `> 0.414` → 4,805 rows, dv/dt `raw_acceleration`, 4th-order
  1 Hz Butterworth `butter_acceleration`). These are **package-derived** and are not
  ingested.
* **Not published:** acquisition date, laser mounting height and aiming point,
  calibration records, athlete age, sex or body mass, consent or ethics statement,
  originating study.

## 3. Rights determination (Gate 0)

| Question | Finding | Status |
| -- | -- | -- |
| License of the repository files | MIT, `LICENSE.md` (grant text verified in pinned bytes), `DESCRIPTION: License: MIT + file LICENSE`, copyright holder Mladen Jovanović (`cph`) | verified |
| Dataset-specific terms | none in README, DESCRIPTION, dataset docs or script | none found |
| Rights to the original measurements | not stated; no originating study or data-availability statement located (repo docs and a web search, 2026-11-17) | **pending** |
| Participant consent / ethics | not documented | **pending** |
| Minor | "youth" athlete; minor status cannot be excluded | privacy rule 5 applies |
| Personal identifiers in the CSV | none (4 kinematic columns) | none |
| Redistribution | not allowed (no affirmative dataset-level evidence; possible minor) | `false` |

The plan (RES-719 Gate 0) and the issue both require independent confirmation of
rights to the **original measurement data** before canonical admission. The MIT file
is a software license from the package author and is silent on that, so admission
stays **pending**.

### What *is* done under the MIT grant

The MIT text expressly permits copying and modifying the repository files. The
pipeline therefore:

* keeps a hash-verified, read-only **quarantine** copy outside Git and outside the
  canonical store, never staged;
* publishes only **aggregate** numerical diagnostics and QC values, with attribution
  to `shorts` (no rows, no per-sample exports).

### What unblocks admission

A written statement from the rights holder (package author or the data collector)
covering: ownership of, or licence to, the recording; participant (and, if a minor,
guardian) consent to public research reuse; and permitted uses. After that, update
`rights.json` to `measurement_data_rights=verified`, `participant_consent=documented`,
`canonical_admission=verified`. `RightsRecord` validation enforces exactly these
conditions. The pipeline then emits a manifest-v1 (`usage=cross-validation-only`,
`redistribution_allowed=false` unless separately granted) and uses the unchanged
`ingest/verify/stage` path. An audit section in `docs/data-audit.md` must be added at
the same time.

## 4. Schema-v2 blocker (Gate 4)

Conversion needs **two independent authorities**, and neither can be passed in as a
plain value:

1. **Canonical admission** verified in `rights.json`. An empirical v2 bundle is a
   canonical artifact and is never built from quarantined data.
2. **A session date bound to source evidence.** The only accepted route is an
   `acquisition_date` claim in `source-lock.json`, which must include:
   * an `evidence_path` naming a pinned `acquisition_evidence` file (with its SHA-256,
     size and Git blob SHA-1, at a commit-pinned URL);
   * an `excerpt` that occurs verbatim in those bytes and states the date
     unambiguously (ISO, `6 January 2024` or `January 6, 2024`; numeric `06/01/2024`
     is rejected as ambiguous);
   * `reviewed_by` and `reviewed_on`.

   The converter re-hashes the evidence and re-checks the excerpt itself. There is no
   CLI flag or free-text reference.

The acquisition date is unpublished, so the committed lock has no claim and **v2
conversion stays blocked**, with both blockers listed. Evidence that lives outside the
pinned repository commit would need a reviewed extension of the lock format. See the
[diagnostic report](reports/shorts-laser-diagnostic-report.md) for the full conversion
outcome.

### Test identity

`test_id=sprint_fv` is a **provisional kinematic protocol mapping only, not a soccer
task admission**. A maximal 35 m standing-start sprint with continuous laser distance
matches the acquisition protocol. Only vmax and tau are supportable; force and power
estimands are unsupported (no body mass). Usage stays `cross-validation-only`, and any
task use needs separate review.
