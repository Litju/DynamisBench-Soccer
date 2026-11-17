# `uwb-optitrack`: source pin and rights disposition (RES-721)

**Assessed:** 2026-11-17 · **Records:** [`source-lock.json`](../../data/external/uwb-optitrack/source-lock.json),
[`rights.json`](../../data/external/uwb-optitrack/rights.json),
[`data/manifests/uwb-optitrack.json`](../../data/manifests/uwb-optitrack.json)

**Decision:** canonical admission is **verified** as **cross-validation-only**.
Redistribution is not used (`false`), although the license would permit it.
Physical and temporal comparability of UWB and OptiTrack is a separate gate, decided
by the planar validation profile.

## 1. Exact source

| Item | Value |
| -- | -- |
| Record | Zenodo 3452007, DOI `10.5281/zenodo.3452007`, published 2019-09-20, open access |
| Creators | Raza, Khan, Kou, Farnham, Premalal, Stanoev, Thompson. All are affiliated with Toshiba Research Europe Limited |
| Archive pinned | `USMAN_Raza-…Motion_Capture_Systems.zip`: 570,029 B. Published MD5 `d5221f12a5f093eacc21db797c0ac13a` matches; SHA-256 `0e26251bff1ddd1f868205c481c63137c4729d480444e60769c3240ba3fc4f38` |
| Members pinned (SHA-256) | `UWB_Walk_dataset.csv`, `UWB_Trolley_dataset.csv`, `BLE_Walk_dataset.csv`, `BLE_Trolley_dataset.csv`, `README_data.md` |
| Members excluded | six `__MACOSX/` AppleDouble resource forks (not data) |
| Not pinned | the `-Sample.zip` archive (23,811 B, MD5 verified). It is a documented subset of the same tables and also holds an editor lock file (`.~lock…#`) |
| Archive safety | no absolute or `..` paths, no links, compression ratio ≤ 5.1, 2.0 MB uncompressed. Members are read in memory and never extracted |

## 2. Rights (Gate 0)

| Question | Finding | Status |
| -- | -- | -- |
| License | `metadata.license.id = "cc-by-4.0"`, read live from `https://zenodo.org/api/records/3452007`. (RES-719 could not recover this from the web page.) | verified |
| Licensor authority | The license was deposited by the data's own creators. Their Toshiba Research Europe affiliation is recorded on the record | verified |
| Metadata stability | A digest of the rights-relevant metadata (id, DOI, title, date, license, access, creators, file keys, sizes and checksums) is pinned in the lock. Re-fetching gave the same digest. Any relicense or replaced file fails `rights.verify_zenodo` | pinned |
| Personal data | Anonymous tag positions, ranges and radio features only. No names, IDs, demographics, biometrics, timestamps or images | none |
| Participant consent | `not_applicable` (**reviewer decision**). The "walking" scenario records only where a tag carried by an unidentified person was. No information about that person exists in the data | judged |
| Attribution | Required by CC BY 4.0. The creators, the DOI and the GLOBECOM 2019 paper must be cited in any report | required |

**Owner may overrule** the `not_applicable` consent judgement. If so, set
`participant_consent` to `not_documented`, add a blocker, and set
`canonical_admission` to `pending`. `RightsRecord` validation will then refuse
admission, and the manifest must be removed.

## 3. What the data are, and are not

* There are **no timestamps** and no sampling rate. UWB and OptiTrack values sit in
  the same CSV row, within `SessionID` 1–5. Row index is the only pairing.
* **Units, coordinate frames, anchor positions and tag/marker reference points are
  undeclared.** The values are consistent with metres in a room of about 5 × 3 m,
  but this is not stated.
* "UWB x/y" are the depositor's localisation estimates (manufacturer- or
  algorithm-processed). "OPT x/y" come from the OptiTrack reference system, a
  separate measurement, not ground truth.
* The data are 2D, with no Z. The project never synthesises one. They are not soccer
  data, so no soccer `test_id` applies.
* The README sample counts (2797 / 3202) include the header row. The data row counts
  are 2796 / 3201.

## 4. Comparability result (Gate 5, measured, never applied)

From the [comparability report](reports/uwb-optitrack-comparability-report.md):

| Gate | Walk (sessions 1–5) | Trolley (sessions 1–5) |
| -- | -- | -- |
| Frame-invariant best row lag (must be 0) | 1, 1, 0, 0, 0 | 3, 3, 3, 4, 3 |
| Lag-aligned similarity scale (must be 1 ± 0.01) | 0.984–1.008 | **0.968–0.975** |
| Rotation (must be ≤ 1°) | **0.76–1.62°** | 0.63–**3.26°** |
| Translation (must be ≤ 0.05) | **0.27–0.30 (mostly in y)** | 0.03–**0.07** |
| Negative-range sentinels | 4 (e.g. −49.13) | 0 |
| Anchor-range geometry (h² ≥ 0) | consistent once sentinels are excluded | **inconsistent (all four anchors h² < 0)** |

**Result:** 0 of 10 sessions are comparable, so **no error metrics are computed.**
The disagreement mixes an unknown device error with a scenario-specific frame,
scale or reference-point mismatch (consistent with the tag and the marker sitting
at different points), and with row misalignment. Neither the lag nor the transform
is applied to "rescue" a metric. A valid comparison would need, from the
depositors:

* timestamps or a stated synchronisation method;
* the common frame definition;
* the tag and marker mounting offsets;
* the units.

**Method note.** The lag is estimated *frame-invariantly*: it is the lag that
minimises the similarity-fit residual. Minimising the raw offset instead confounds
a spatial offset along the path with a time shift. An early exploratory estimate
made that mistake, and the corrected estimator is what is committed and tested.
