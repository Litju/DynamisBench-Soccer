# DynamisBench-Soccer — External Dataset Admission and Ingestion Plan

**Prepared:** 2026-11-16  
**Scope:** laser sprint, basketball LPS, soccer sprint gates, Hanyang IMUs, U18 soccer GNSS, UWB–OptiTrack  
**Repository contract inspected:** `Litju/DynamisBench-Soccer`, schema-v2 `src/dynamisbench_soccer/signals/schema.py`; dataset manifest schema-v1 `src/dynamisbench_soccer/data/manifest.py`; `docs/data-policy.md` and `docs/data-audit.md`.  
**Decision:** Research candidates identified; **zero new datasets admitted, downloaded, or scientifically qualified by this report**. Rights, privacy and exact-byte checks are source-specific and cannot be inferred from a public link.

## 1. Admission disposition

| ID | Exact upstream files established | Data stage and grain | Rights / privacy evidence | Pre-import decision |
|---|---|---|---|---|
| `shorts-laser` | `mladenjovanovic/shorts/data-raw/laser-gun-data.csv` (160,957 repository bytes; 4 source columns), plus `data/laser_gun_data.rda` processed R package dataset (4,805 × 6) | Real laser-acquired **manufacturer-processed** 1 kHz distance/velocity trace. Original hardware sampled at 2.56 kHz; package script derives/filter accelerations. Not native raw range data | `shorts` repository DESCRIPTION and LICENSE.md are **MIT**; separately sourced sensor/participant data rights are **not explicitly confirmed** | **RIGHTS_REVIEW**, then conditionally cross-validation-only. Record manufacturer and package transforms as different stages |
| `shorts-lps` | `mladenjovanovic/shorts/data-raw/LPS-session.csv` (4,677,409 repository bytes), plus `data/LPS_session.rda` (91,099 × 5) | Real device-provided 20 Hz 2D x/y position and derived device velocity/acceleration; one basketball session | MIT package, no dataset-specific grant independently established; single-individual positional traces | **RIGHTS_AND_SCHEMA_HOLD**; no invented Z axis or soccer test ID |
| `shorts-vescovi` | `mladenjovanovic/shorts/data-raw/vescovi-data.csv` (4,119 bytes; 13 original columns), plus `data/vescovi.rda` (52 × 17 after four derived split columns) | Empirical **split-time summaries**, not raw infrared gate events. 35 soccer + 17 field-hockey women; one selected best sprint per athlete | MIT package, original Vescovi cohort data rights and permissible athlete-level redistribution not independently established; age may include young athletes | **RIGHTS_REVIEW**, then summary-only cross-validation; no synthetic gate timestamps |
| `hanyang-imu` | Zenodo 21712401 `Raw_Data.zip` (69.5 MB; MD5 `02cf77f92e408a9f7f7f0c01c52f95b2`) and optional `Labeled_Data.zip` (32.6 MB; MD5 `af9987e3df7f1d64e09c3ac752d0ca82`) | Three Xsens MTw Awinda sensors, left/right thighs + pelvis, 100 Hz. Original clocks replaced with relative 0.01 s timeline. Labelled archive additionally selects segments and adds labels | Depositor says de-identified; Zenodo record displays a Rights/License heading but the exact license identifier was **not recoverable** through the inspected accessible record. No file-level license independently confirmed | **LICENSE_AND_SCHEMA_HOLD**; preserve raw-vs-labelled distinction, check whether 'linear acceleration' is specific force before any IMU conversion |
| `soccer-gnss-u18` | `kehanabcd/GNSS-positional-data-to-Tactical-analysis/Positional data/U18_6v6_32x24/*.csv` (six player-named files), `Match Info/SSGINFO_6v6_32x24.xlsx`, `Pitch Info/Pitch coordinates.xlsx` | Real U18 Catapult positional traces (reported 10 Hz), plus bouts and pitch geometry. GPS-derived positions are not satellite pseudorange raw data | **No LICENSE file in repository**; filenames expose participant names and players are U18. No adequate source-specific redistribution/consent evidence established | **EXCLUDED / NO DOWNLOAD** until owner/institution explicitly clears rights, minors' privacy and an anonymization route |
| `uwb-optitrack` | Zenodo 3452007 `USMAN_Raza-Dataset_Indoor_Localization_with_Narrowband_Ultra-Wideband_and_Motion_Capture_Systems.zip` (570.0 kB; MD5 `d5221f12a5f093eacc21db797c0ac13a`), plus optional `...-Sample.zip` (23.8 kB; MD5 `da8ad411da3707c52ad45201035733e7`) | Real UWB 2D positions, BLE estimates, UWB anchor-related CIR/PSA/ranges and OptiTrack 2D reference for walk/trolley trials. Not verified as synchronized event timestamps | Zenodo Rights/License identifier **not recoverable** via inspected record; corporate authorship (Toshiba Research Europe), review redistribution terms | **LICENSE_AND_SCHEMA_HOLD**; potential separate cross-validation-only spatial-error benchmark after file and coordinate audit |

**Source versions:** `shorts` Git `master` ref observed at `5f8ce5554b5aca0e857199e0eb50bf66d7089130`; U18 GNSS Git `main` ref observed at `22a63ad72427f63a7b65baa173ab3a6563f0d19f`. Git object/blob SHA-1 is **not** file SHA-256. Zenodo records pinned by record DOI, exact archive names, and published MD5; local SHA-256 remains to be computed from downloaded bytes.

**Primary sources:**
- `shorts` files and license: https://github.com/mladenjovanovic/shorts and https://mladenjovanovic.github.io/shorts/reference/
- Laser stage: https://mladenjovanovic.github.io/shorts/reference/laser_gun_data.html
- LPS field details: https://mladenjovanovic.github.io/shorts/reference/LPS_session.html
- Vescovi details: https://mladenjovanovic.github.io/shorts/reference/vescovi.html
- Hanyang IMU: https://zenodo.org/records/21712401
- U18 GNSS: https://github.com/kehanabcd/GNSS-positional-data-to-Tactical-analysis
- UWB: https://zenodo.org/records/3452007
- Zenodo rights caveat: https://support.zenodo.org/help/en-gb/2-content/21-can-i-get-permission-to-use-a-specific-record

## 2. The existing contracts are binding

**Dataset inventory:** `DatasetManifest` **schema_version 1** is separate from scientific **signal bundle schema_version 2**. A non-EXCLUDED manifest requires exact per-file `sha256` and `size_bytes`; its `usage` must be one of `task-input`, `cross-validation-only`, `excluded`. Current policy finds **no public empirical source admitted as `task-input`**. For any permitted new empirical source, default to *cross-validation-only* after rights and integrity approval. `redistribution_allowed` remains false until positive dataset-level evidence. A provisional research inventory (separate JSON supplied with this plan) is not a validated manifest.

**Storage:** `DBS_DATA_HOME` and `DBS_CACHE_HOME` must point to a Linux filesystem, not `/mnt/<drive>`. Existing `dbs data ingest`, `verify`, `stage` and SHA-256-addressed store are the canonical path; none of these tools determines legal permission. Do not commit raw data, archive, parquet, athlete rows or large fixtures to Git.

**Schema-v2:** strict `BundleMeta` requires registered `test_id` (`V01_TESTS`), real-or-declared session date, canonical SI units, finite float64 columns, explicit instrument acquisition contract for raw GNSS/LPS/IMU/laser and proven frames/clocks. LPS requires a framed **x/y/z** position triplet *per source*; GNSS requires latitude, longitude and position. IMU requires framed XYZ `specific_force` and XYZ `angular_velocity`. Missing calibration must be explicit, never a fabricated calibration certificate. The existing reader checks Parquet SHA-256/schema/counts before processing; use it unchanged.

**Do not bypass schema to admit 2D or nonsoccer data.** `shorts-lps` and UWB have x/y but no asserted Z coordinate; Hanyang's daily-activity trials have no registered soccer `test_id`; dates may be unknown; Vescovi has split summaries, not native gate event records. Those remain quarantined until an **approved, versioned, non-soccer external validation profile** exists. That profile should preserve v2 integrity and mean exactly what it says, rather than silently populating fake Z/date/protocol values. Version compatibility and test-ID governance require separate review; do not rewrite the ontology by stealth.

## 3. Source-specific conversion requirements

### Laser sprint (`shorts-laser`)

- Source CSV columns verified: `time,distance,velocity,raw_velocity`; package `.rda` adds computed `raw_acceleration` and `butter_acceleration`, making six processed columns. Package script filters data, subtracts a velocity offset and applies a 4th-order 1 Hz Butterworth filter to acceleration. Preserve **the source CSV** and the exact software version/algorithm for any reproduction.
- `DeviceClass.LASER`, intended `test_id=sprint_fv` **only if its published protocol actually fits the sprint-FV analysis contract**. Position/distance channel: quantity `distance`, `m`. Do not relabel velocity or acceleration as unfiltered device-native measurements; explicitly mark as derived/vendor-processed and keep original vs transformed waveforms separate.
- 4,805 rows at published processed 1 kHz; validate monotonic time, starting time, intersample Δt and missingness. Unknown recording date, aiming geometry and hardware calibration require honest metadata and may block strict v2 conversion.
- No radar-specific observations or real reference trajectory. Analysis: pipeline sensitivity, reconstruction comparisons, temporal consistency, not empirical laser accuracy certification.

### LPS basketball (`shorts-lps`)

- Expected `time,x,y,velocity,acceleration`, documented 91,099 observations at nominal 20 Hz, one basketball practice. Data-raw CSV is a separate upstream source; R script removes an upstream `clock` field before creating the `.rda` object. Independently check the exact header/column set after permitted acquisition.
- **Documented acceleration unit is written as `m/s`, not `m/s²`**: require clarification/inspection; don't silently correct it. Device velocity/acceleration stay distinct from position-differenced estimates.
- 91,099/20 ≈ 4,555 seconds (~76 min), compared with description ~90 min: inspect gaps and actual first/last timestamps; don't assume uninterrupted 20 Hz.
- Native observations are 2D: **no invented `z=0`** just to pass v2. No approved soccer test identity for basketball. Hold at licensed raw source + quarantined validation preprocessing until an explicit 2D/non-soccer profile is accepted.

### Vescovi gates (`shorts-vescovi`)

- Original CSV has 52 participant rows and 13 columns; package R preprocessing sorts and adds four split-time columns (17 total). The documented gates include 5/10/20/30/35m but the prose methodology omits 30m: preserve protocol ambiguity in metadata.
- Classify as **summary/derived timing metrics**. For a permitted future conversion, one `summary` bundle per pseudonym and session/trial, with units s and source labels, may fit the existing `sprint_gates` ID. Do not create `events` with invented `t_s`/beam interruptions, a fake sampling rate or pseudo-raw start triggers.
- Cohort has potential young participants; no publishable identifiable per-athlete task before rights/consent clearance. No athlete-specific physiological interpretation from these split summaries.

### Hanyang IMUs (`hanyang-imu`)

- Record `Raw_Data.zip` and optional `Labeled_Data.zip` separately; **archive filenames and Zenodo MD5 are verified, archive member filenames and archive SHA-256 are not yet verified**. Unpack only in a controlled scratch directory after rights resolution; reject path traversal/symlinks/zip bombs, inventory all members and their hashes.
- Raw archive contains device orientation, angular velocity, **linear acceleration** and magnetic field. Before v2 mapping, establish if linear acceleration is inertial (gravity removed) or specific force (accelerometer measurement). Do not map acceleration to `Quantity.SPECIFIC_FORCE` without a declared orientation/gravity method.
- Actual timestamps were **replaced with evenly spaced relative time**; do not claim native hardware clock drift, true sampling jitter, synchronization or calibration knowledge. Keep 3 sensor placements and source-specific coordinate frames.
- Separate activity labels and clipped sequences only from `Labeled_Data.zip`. `test_id` compatibility is missing; preserve as external instrument-validation input outside the soccer test ontology until reviewed.

### U18 soccer GNSS (`soccer-gnss-u18`)

- Source code and six player-named positional CSVs plus session and pitch XLSX exist at pinned Git revision. **No license found** and U18 privacy/consent not established. No download, re-hosting, or deriving public individual-player files at this stage.
- If authorized later, obtain de-identified source files and explicit license/consent evidence from a responsible rights holder, then inspect coordinate convention, date/time, 10 Hz claims, sessions/bouts and pitch origin. Never treat lat/long positions as raw GNSS pseudorange or Doppler observations. Link players only through privacy-safe pseudonyms.

### UWB + OptiTrack (`uwb-optitrack`)

- Two Zenodo ZIP filenames and published MD5s are verified; ZIP member inventory, per-column types, internal timestamps, units, coordinate system and exact rights remain unverified. The record describes two scenarios (walking/trolley), UWB/OptiTrack x-y estimates and ranging related channels.
- Separate measured/UWB-estimated positions from OptiTrack-derived reference. Confirm alignment and frame transform before computing planar error; sample index or shared session ID alone is not clock synchronization. Do not claim empirical interchangeability of unrelated sports devices.
- Preserve 2D x/y; do not fabricate 3D LPS signals. Non-soccer experiments remain cross-validation-only if admitted under approved profile.

## 4. Required admission pipeline

### Gate 0 — Authoritative rights and participant consent (BLOCKING)

For **each exact source and version**, record license identifier, license URL or attached license text, licensor/data author authority, permitted copying, transformation, AI benchmark inference, publication and redistributability; preserve attribution and notices. Distinguish MIT `shorts` **software/package** rights from permission to redistribute underlying participant data. Zenodo's default CC BY is **not proof** that these particular records are CC BY: retrieve each record's expanded `rights[]` identifier or depositor evidence. For GNSS U18, get explicit permission and privacy approval, or remain EXCLUDED. If unclear, `rights_status=PENDING`, `redistribution_allowed=false`, no canonical ingestion.

### Gate 1 — Exact immutable source bytes (BLOCKING)

Fetch only permitted, pinned files into offline scratch, no automated execution. For the two Zenodo records, verify published MD5 first, compute independent SHA-256 and byte size. For GitHub, pin commit and compare expected Git blob SHA-1 as additional repository identity, compute **actual SHA-256** on bytes. For archives, inventory members, reject unsafe paths, executable/pickle payloads, symlinks and decompression bombs; hash extracted members. Do not invent checksums in manifest.

### Gate 2 — Source-shape and privacy audit

Record per-file row/column count, field names and units, grain (sample / split / person / session / reference), duplicate IDs/timestamps, missingness, monotonic time, native vs constructed clocks, sample-gap distribution, calibration and coordinate frames, actual population/age, leakage and consent. Separate manufacturer-computed channels and released labels from measured channels. Small bounded profiling first (8 GB RAM).

### Gate 3 — Manifest schema 1 and read-only staging

Only if Gate 0–2 pass: write a `data/manifests/<source_id>.json` conforming to `DatasetManifest` schema 1, with `data_category=empirical`, `usage=cross-validation-only`, **actual** file SHA-256/size, rights, DOI/commit and audit reference. Use `redistribution_allowed=false` absent affirmative rights. The existing store verifies and deduplicates exact bytes, and `dbs data stage` creates a read-only view.

```bash
# Run only after legal and byte-level admission evidence is complete:
uv run dbs data manifest-create <verified_source_dir> \
  --output data/manifests/<dataset_id>.json \
  --dataset-id <dataset_id> --version <pinned_version> \
  --title "<verified_title>" --category empirical \
  --usage cross-validation-only --license <verified_license_id> \
  --source-kind url --source-location <verified_source_url>
# Manually fill rights/privacy/handling/audit_ref, then validate:
uv run dbs data validate data/manifests/<dataset_id>.json
uv run dbs data ingest data/manifests/<dataset_id>.json <verified_source_dir>
uv run dbs data verify data/manifests/<dataset_id>.json
uv run dbs data stage data/manifests/<dataset_id>.json
uv run dbs repo-check
```

### Gate 4 — Approved schema-v2 conversion

A **source-version-specific adapter**, not a global permissive CSV importer, must validate field names/types, SI conversions, exact sample/timestamp and protocol semantics, acquired vs derived estimands, explicit geometry/clock/calibration unknowns and source hash/provenance. Use `SignalBundle` timeseries for valid continuous registered tests; `events` only for real source event times; `summary` only for actual stored summary grain. The v2 `.parquet` and `.json` share a stem; the metadata binds Parquet SHA-256. Run `io.read` and `io.iter_batches` with integrity verification. Hard-block nonconforming 2D, missing date, unregistered tests, nonfinite/dropout imputation and wrong IMU force semantics; escalate to a reviewed contract extension, never covertly weaken the v2 validator.

### Gate 5 — Scoped scientific cross-validation and independent reviewer

Use only dataset-appropriate analyses: laser sprint trace sensitivity, Vescovi split consistency and time/distance modeling, LPS path/velocity consistency, Hanyang frame/sensor interpretation, UWB/OptiTrack planar position agreement. Compare same estimands under declared methods. Record data exclusions, QC flags, uncertainty and known pre-processing. **No v0.1 task-input admission, no J/K promotion, no M3 family verification, no physiological validity claims and no paid model runs.**

## 5. Executable quality/falsifier matrix

| Failure mode | Expected disposition |
|---|---|
| Unverified license, restricted/minor participation | Rights block, no import |
| File SHA-256/size or Zenodo MD5 mismatch | Fetch/ingest fail closed |
| Archive path traversal, pickle/executable members, unsafe decompression | Quarantine/refuse |
| Laser stage silently relabelled `raw` | Reject or explicit provenance failure |
| Vescovi split rows fabricated into raw gate events | Reject |
| LPS/UWB 2D silently padded with invented zero Z | Reject |
| IMU linear acceleration silently treated as specific force | Reject |
| Missing session date replaced with an invented actual date | Reject |
| GPS position labelled pseudorange or Doppler | Reject |
| Participant ID/name leaks to Git or published bundle | Reject |
| Measured timestamps overwritten by nominal rate | Reject; retain true gaps or mark constructed clocks |
| Vendor-derived metrics used as independently verified physical truth | Reject |
| Missing/reordered/duplicated sample and wrong SI conversions | Detect, preserve QC, refuse if unqualified |
| Normal study results escalated to AI scoring reference | Reject |

## 6. Implementation order

1. **Rights and source-byte review:** start `shorts` (3 small files), Zenodo UWB (small) and Hanyang raw archive only after verified rights. Hold the U18 GNSS source outright.
2. **Vescovi / laser first:** most tractable native v2 alignment, provided source rights and mandatory metadata are resolved. A documented, per-athlete summary adapter is acceptable only if rights permit and a real session/protocol identity is available; no fake raw gate events.
3. **Non-soccer 2D and IMU profiles:** independently review explicit validation-only profile/ontology changes rather than weakening v2. Preserve original source files intact in store.
4. **Small reproducible runs:** deterministic selected trials, per-source QA reports, actual values/limits, immutable manifest and source lineage; no large extraction or public data redistribution.
5. **Independent review:** verify conversion semantics, legal/ethical disposition, provenance, no unverified scientific promotion, and exact-head CI of any repository-mutating issues (minimum two meaningful verified commits per Linear issue).

**Important negative disposition:** this is a source admission *plan*, not an authorization to download restricted data, a report that any new manifests are ingested, or a claim that all six sources fit current v2 unchanged.