# Windows acquisition boundary and import adapters

Issue RES-667. Implements ADR-0001 rules 4, 10 and 11.

## The boundary

```
 WINDOWS (acquisition only)                 │  WSL2 UBUNTU (canonical scientific runtime)
                                            │
 vendor software ──► export (CSV)           │
          │                                 │
          ▼                                 │
 hand-off folder  ── copy, never edit ──────┼──► ~/dbs-inbox/<date>/   (Linux filesystem)
 (e.g. C:\dbs-handoff\)                     │        │
                                            │        ▼
                                            │   dbs data manifest-create  (SHA-256 of raw export)
                                            │   dbs data ingest           (read-only store)
                                            │   dbs adapters import       (canonical bundle;
                                            │                              source_sha256 recorded)
```

Rules:

1. **Windows only runs acquisition.** No Windows Python, no Windows venv and no Git for
   Windows touch this repository or its data. Scientific computation never happens on
   the Windows side.
2. **Exports are copied, not edited.** Files are copied from the hand-off folder into a
   Linux-filesystem inbox (`cp /mnt/c/dbs-handoff/... ~/dbs-inbox/...`). Reading from
   `/mnt/c` to copy is the only allowed contact, and nothing is processed in place.
3. **Hash first.** The raw export is hashed into a dataset manifest before any
   conversion. Every canonical bundle records `provenance.source_sha256` of the exact
   bytes it came from, so results trace back to an untouched export.
4. **Vendor numbers are not truth.** Metrics the vendor software computed are imported
   only into `vendor_reported` (with a `vendor.<metric>[unit]:` header line) and are
   never used as reference values (decision boundary DB-07).
5. **Real athletes are pseudonymous.** Exports carry a pseudonym (`subject`), never a
   name. Mapping tables stay off the benchmark machine.

## DBS generic CSV format (version 1)

Vendor software exports differ. Until the owner supplies sample vendor files (default
Q2), exports are first converted, by hand or with a small per-site script, into this
documented format:

```text
# dbs-generic-csv: 1
# test_id: cmj
# data_category: empirical
# subject: ATH-014
# session_id: S-2026-11-14
# date: 2026-11-14
# match_day: MD-3
# protocol_id: cmj.hands_on_hips.v1
# arm_swing: hands_on_hips
# sampling_rate_hz: 1000
# zeroing: yes
# vendor.jump_height[m]: 0.312
fz_left[N],fz_right[N]
368.2,367.9
...
```

* **Metadata:** the `# key: value` lines. Unknown keys are rejected, which catches
  typos such as `sampling_rate` versus `sampling_rate_hz`.
* **Columns:** written as `name[unit]`. Every column needs a unit, and the units allowed
  are `N kN kgf lbf N*m Nm m cm mm km s ms min m/s km/h deg 1`. Values are converted to
  canonical SI, and every conversion is recorded in `provenance.transforms`.
* **Sampling:** either a `time[...]` column (the measured rate is computed, and the
  nominal rate is taken from `sampling_rate_hz` or inferred and recorded) or
  `sampling_rate_hz`. A file with neither is rejected.

## Adapters (v0.1)

`dbs adapters list` prints each capability manifest, and `--json` gives it in machine
form.

| Adapter | Tests | Columns |
| -- | -- | -- |
| `generic_forceplate_csv` | cmj, sj, slcmj, dj, rj_10_5, imtp | `fz` or `fz_left,fz_right` or `fz_plate<k>`, optional `time`, `fx_*`, `fy_*`, `cop_x_*`, `cop_y_*` |
| `generic_dynamometer_csv` | nhe, hip_add_abd, iso_ham | `force` or `force_left,force_right`. hip_add_abd needs `hip_flexion_deg` and `action`. iso_ham needs `knee_angle_deg` |
| `timing_gates_csv` | sprint_gates, cod_505, rsa | exactly `distance[..]`, `time[..]` (events, no sampling rate) |
| `radar_laser_csv` | sprint_fv | `velocity[..]` and/or `distance[..]`, optional `time` |
| `lpt_csv` | cmj, sj | `displacement[..]`, optional `time` |
| `gnss_summary_csv` | context for cmj, nhe, sprint_gates, hip_add_abd, imtp | any `name[unit]` columns, one row per session |

```bash
uv run dbs adapters import generic_forceplate_csv ~/dbs-inbox/2026-11-14/cmj_014_t1.csv \
    "$DBS_CACHE_HOME/workspaces/import/cmj_014_t1"
```

## Conformance

`tests/test_adapters_conformance.py` runs every adapter on CI against the files in
`tests/fixtures/adapters/`:

* **Valid exports, one or more per adapter.** Each must produce canonical units, the
  correct kind and device class, its capability-declared test id, provenance with the
  raw-file SHA-256, and an unchanged round trip through the canonical writer.
* **22 malformed exports**, each rejected with a specific error. The cases include:
  * missing sampling rate, unit mislabels (force in metres, time in metres, velocity
    in newtons), unknown or missing units
  * non-numeric, NaN and ragged rows, and non-monotonic time
  * missing header magic, unknown metadata keys, missing subject, invalid data category
  * a test unsupported by the adapter, missing hip angle or action, and extra columns

Adding a vendor-shaped adapter (VALD-type, Hawkin-type) requires real sample exports,
owner confirmation, a new capability manifest, and valid plus malformed fixtures in
the same suite.
