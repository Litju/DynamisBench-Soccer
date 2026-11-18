# External data: admission pipeline

| Document | What it is |
| -- | -- |
| [Admission plan (RES-719)](DynamisBench_Soccer_external_data_admission_plan_2026-11-16.md) | Gates, falsifier matrix, six-source dispositions |
| [Source inventory (RES-719)](DynamisBench_Soccer_source_inventory_PRE_ADMISSION_2026-11-16.json) | Pre-admission planning inventory (not a manifest) |
| [`shorts-laser` rights disposition (RES-720)](shorts-laser-rights-disposition.md) | Exact source, rights findings, blockers, what unblocks admission |
| [`shorts-laser` diagnostic report (RES-720)](reports/shorts-laser-diagnostic-report.md) | Generated pipeline report from the real pinned bytes ([JSON](reports/shorts-laser-diagnostic-report.json)) |
| [`uwb-optitrack` disposition (RES-721)](uwb-optitrack-disposition.md) | Zenodo 3452007: CC BY 4.0, admitted cross-validation-only; comparability blocked |
| [`uwb-optitrack` comparability report (RES-721)](reports/uwb-optitrack-comparability-report.md) | 2D planar profile v1 and comparability gates; no error metrics ([JSON](reports/uwb-optitrack-comparability-report.json)) |

## Pipeline (`dbs external`, RES-720)

```text
data/external/<id>/source-lock.json  ─┐   commit-pinned URLs; SHA-256, size, Git blob SHA-1
data/external/<id>/rights.json       ─┤   license evidence vs measurement-data rights, consent
                                      ▼
pinned_source → evidence_integrity → rights_verification → data_integrity → ingestion
             → safe_parse → profile
             → schema_v2_conversion → qc → diagnostics → report.{json,md}
```

| Stage | Passes when | Otherwise |
| -- | -- | -- |
| `pinned_source` | lock pins the exact commit and data SHA-256 the adapter was written for | failed |
| `evidence_integrity` | license, documentation and transform-script files only match pinned size, SHA-256 and Git blob SHA-1 (bounded download, no symlinks). **No measurement byte is touched.** | failed, nothing installed |
| `rights_verification` | license text in the pinned bytes contains the recorded grant and holder, **and** the record permits canonical admission | blocked (license verified, admission not permitted) or failed (evidence mismatch) |
| `data_integrity` | runs **only if** `local_inspection_permitted`; measurement data then verified like the evidence | blocked: data bytes are never downloaded, opened or hashed, not even an existing quarantine copy, and the run stops |
| `ingestion` | canonical: manifest-v1 → existing `ingest_local` / `verify` / `stage` | blocked → read-only **quarantine** copy (reached only after `data_integrity` passed) |
| `safe_parse` | hash-bound, exact header, ASCII, finite numbers, bounded size/rows | failed |
| `schema_v2_conversion` | canonical admission verified **and** a lock-pinned, reviewed acquisition-date claim; strict `BundleMeta`; written, read back and streamed through `signals.io` with SHA-256 verification | blocked with each precise blocker; the global schema is never weakened |
| `qc` | source-level QC always, including `SRC-05` distance/velocity time-base consistency (mirrors `sprint_fv` QC rules, params from `qc/rules.toml`); bundle QC when a v2 bundle exists | records the verdict |
| `diagnostics` | pipeline-derived numerical diagnostics | n/a |

```bash
uv run dbs external validate                 # CI: committed locks + rights records
uv run dbs external rights shorts-laser      # disposition and blockers
uv run dbs external acquire shorts-laser     # verified quarantine copy only
uv run dbs external run shorts-laser \
  --report-md docs/external-data/reports/shorts-laser-diagnostic-report.md \
  --report-json docs/external-data/reports/shorts-laser-diagnostic-report.json
# v2 conversion additionally needs verified admission in rights.json AND an
# acquisition_date claim in source-lock.json bound to a pinned acquisition_evidence
# file (see shorts-laser-rights-disposition.md §4). There is no CLI date flag.
```

Bytes live under `$DBS_CACHE_HOME/quarantine/` (or the canonical store once admitted),
never in Git. The report is deterministic for a fixed lockfile and contains only
aggregate values.

### Signal-stage taxonomy

`measured_native` (sensor samples as acquired), `manufacturer_processed` (vendor
software output), `package_derived` (third-party package transforms),
`pipeline_derived` (computed here, for diagnostics only). Stages are never merged or
relabelled. In `shorts-laser`, every CSV column is `manufacturer_processed`, and the
source column `raw_velocity` becomes channel `velocity_vendor_unsmoothed`.

## `shorts-laser` findings (2026-11-17)

* **Blocked:** canonical admission (measurement-data rights, consent, possible minor)
  and v2 conversion (no published session date). Both blockers are recorded, not
  bypassed.
* **Source integrity:** 5,232 rows on an exact 1 ms grid (−0.012 to 5.219 s), no gaps
  or duplicates, distance monotonic at 1 mm resolution. The package's documented
  4,805-row object is reproduced arithmetically from its trim rule.
* **Vendor channels disagree on time base:** d(distance)/dt matches vendor velocity
  to 0.094 m/s RMS only after a **≈ 996 ms shift**. At zero lag the RMS is 2.09 m/s,
  and integrating velocity falls ≈ 7.8 m short of the vendor distance. The cause
  cannot be determined from vendor exports. Joint distance/velocity reconstruction is
  therefore **temporally unqualified** (`SRC-05` fails; tolerance ±5 ms). The shift is
  measured and reported, never applied. Source timestamps are never corrected, and
  joint outputs (derivative residuals, integration, the distance-defined 30 m fit)
  are reported unshifted and labelled. Any converted bundle carries the same
  statement in its provenance.
* **QC:** `velocity_vendor_unsmoothed` fails QC-R02 (56.6 m/s² sample-to-sample).
  Smoothed velocity passes (5.4 m/s²). `SRC-05` (time base) is flagged. Source verdict
  `accept_with_flags`.
* **Model-fit sensitivity:** across four documented preprocessing variants,
  MSS 8.13–8.22 m/s and TAU 1.03–1.07 s. This is a diagnostic of vendor-processed
  velocity, not reference truth.

## Known limitations (independent review, 2026-11-17)

These are deliberately **not** fixed here, because fixing them means changing the
global schema-v2 `BundleMeta` or the lock format, which needs separate review:

* **Temporal qualification and the provisional `sprint_fv` mapping live in bundle
  provenance text and conversion determinations only.** Schema v2 has no structured
  field for either. Bundle QC (`dbs qc run`) has no `SRC-05` equivalent, so a
  converted bundle's QC verdict can read `accept` while source QC flags the time
  base. Consumers must read `source_qc` and the provenance.
* **Evidence outside the pinned repository commit** (for example a rights-holder
  letter) cannot be expressed: lock URLs must be commit-pinned raw URLs of the same
  repository.
* **`--root` overrides** load unreviewed local records. The CLI warns, and the report
  records `records.committed_default_root=false` with an `UNREVIEWED RECORDS` banner.
  Only `data/external/` is reviewed in Git.
* Rights checks are sealed with a per-process key and bound to one source, commit and
  exact source-lock content; the report records the rights-record digest. This defends
  against API misuse (edited copies, reused seals, cross-source or changed-lock checks),
  not against deliberate in-process tampering with module internals. The seal proves
  `rights.verify` ran on a given record. Only the committed `data/external/` records are
  reviewed, and a caller can still hand `verify` a self-written record.
* The acquisition-date check is a fail-closed **syntactic** guard; the named reviewer
  is the semantic backstop. It rejects whole-line contexts containing unparsed
  date-like text, so legitimate lines mentioning e.g. "May", "100 m", "1000 Hz" or a
  clock time are rejected and must be quoted from a cleaner line. Relative wording
  ("the day before", "± a week") is not detected.
* The report embeds Python and numpy versions, so it is byte-reproducible only within
  the same locked environment.
