# Public dataset audit v0.1

Issue RES-666. Audit date: 2026-11-14. Manifests: `data/manifests/*.json`. Policy:
[data-policy.md](data-policy.md).

## Method

1. **Search.** Zenodo and Figshare were searched through their public APIs for raw
   force-plate, Nordic, sprint and soccer neuromuscular datasets. Candidate records
   were read from the repository APIs (license, files, description), not from search
   snippets.
2. **Download.** Each candidate was downloaded into an isolated scratch directory and
   checked against the repository's published MD5. Its SHA-256 and size were then
   recorded in a manifest.
3. **Inspect safely.** Inspection used `python -I`, `allow_pickle=False` and header-only
   reads. No downloaded file was executed or unpickled.
4. **Ingest test.** Every accepted manifest was ingested into an isolated store and then
   verified, staged and cleaned (`dbs data ingest/verify/stage`, `dbs clean`).
5. **Classify.** Each dataset was assigned `task-input`, `cross-validation-only` or
   `excluded`, with reasons recorded in the manifest (`handling_notes`) and below.

Criteria: license and redistribution, consent and ethics statement,
re-identification risk, completeness of sampling and device metadata, population
relevance, and fitness for signal-processing tasks.

## Outcome

**No public dataset qualifies as `task-input` for v0.1.** v0.1 tasks therefore use
analytical and synthetic signals (M3 RES-669). Empirical data is limited to
cross-validation and plausibility ranges, and data categories are never mixed in
scoring.

| Dataset | Usage | Why |
| -- | -- | -- |
| [white2022-cmj-vgrf-acc](#white2022-cmj-vgrf-acc) | cross-validation-only | Real CMJ vGRF and accelerometer signals, but stored as pickle-only object arrays and with metadata that contradicts the files |
| [holik2025-youth-soccer-plyometrics](#holik2025-youth-soccer-plyometrics) | cross-validation-only | Youth soccer, maturation and repeated attempts; summary metrics only, methods unstated |
| [braking-rfd-cmj-2025](#braking-rfd-cmj-2025) | cross-validation-only | Both jump-height estimands for 206 participants; software-derived values |
| [uwb-optitrack](#uwb-optitrack) | cross-validation-only | CC BY 4.0 UWB/OptiTrack testbed (RES-721, 2026-11-17); comparability not established (time and frame), so no error metrics |
| [jass2026-sampling-article](#jass2026-sampling-article) | excluded | Article deposit, no data |
| [rugby-league-hawkin-2023](#rugby-league-hawkin-2023) | excluded | Vendor summaries, rugby league |
| [badminton-eccentric-2025](#badminton-eccentric-2025) | excluded | Derived analysis tables |
| [climbers-slsj-2026](#climbers-slsj-2026) | excluded | Time-normalised waveforms (no time base) |
| [dvj-vr-cai-2026](#dvj-vr-cai-2026) | excluded | 1.58 GB archive, clinical/VR population; v0.2 candidate |

## Accepted (cross-validation only)

### white2022-cmj-vgrf-acc

Zenodo [19136480](https://zenodo.org/records/19136480) (DOI 10.5281/zenodo.19136480),
CC BY 4.0. It contains vGRF and L5 accelerometer CMJ data derived from White et al.
(2022, PLOS ONE): 67 participants (45 M, 22 F, healthy sports-science students),
663 trials, with and without arm swing. The depositors state written consent and
Swansea University ethics approval.

* **Safety: blocking for direct use.** `grf_signals` and `acc_signals` are ragged
  NumPy **object arrays**, so loading them requires `allow_pickle=True`, and
  unpickling untrusted data can execute code. They must be converted once in an
  isolated sandbox, or a non-pickle export obtained, before any use. Only the
  pickle-free arrays were inspected (take-off indices, ids, jump height, peak power,
  sampling rates).
* **Metadata inconsistencies.**
  * The description says vGRF was downsampled to 250 Hz with a 2,000 ms window, but
    the stored `grf_sampling_rate` is 1000 and take-off indices reach 10,792, which
    suggests full-rate, variable-length trials.
  * Per-file trial counts are 334 (no arms) and 329 (arms), against the documented
    331 and 332. The total of 663 matches.
* **Duplication.** `cmj_dataset_both.npz` duplicates the other two files and is
  deliberately not listed (ADR-0001 rule 5).
* **Use.** Cross-validation of the M3 force-plate engine (take-off velocity and jump
  height) after safe conversion, and realistic QC examples. The depositors already
  removed clipped and miscalibrated trials, which makes this a natural source for
  "would you have rejected this?" checks. Not a soccer population.

### holik2025-youth-soccer-plyometrics

Zenodo [17881835](https://zenodo.org/records/17881835), CC BY 4.0; registered trial
NCT06406764. It covers 45 male adolescent soccer players aged 10–18 (minors) in a
12-week plyometric RCT stratified by maturity offset (Mirwald method, registry
MT-001). Variables: CMJ height, drop-jump contact time and height, 20 m sprint,
L-run, broad and triple jumps, leg stiffness, and maturity offset, with up to three
attempts pre and post.

* **Privacy.** A free-text "Reason for exclusion" column contains health categories
  (Injury, Illness). It **must be dropped** on conversion, because the policy forbids
  free-text medical notes. Squad-level use requires n ≥ 10.
* **Limits.** Summary values only. The device and jump-height method are not stated,
  so the values cannot serve as references for estimand-specific grading (DB-06).
* **Use.** Plausible youth-soccer magnitudes for the M6 scenario generator, examples of
  between-attempt variability for the M3 reliability library, and
  maturation-context tasks.

### braking-rfd-cmj-2025

Zenodo [15525532](https://zenodo.org/records/15525532), CC BY 4.0. It has 206
physically active participants (basketball players and students), with
per-participant software-computed CMJ metrics: bilateral and left-leg jump height
from **take-off velocity and from flight time**, push-off impulses, and eccentric RFD.

* **Use.** A real-data illustration that the two jump-height estimands differ
  (registry FP-003), and plausibility ranges. The values are software-derived, so
  they are never reference truth (DB-07).

### uwb-optitrack

Zenodo 3452007 (RES-721, assessed 2026-11-17).
[Rights disposition](external-data/uwb-optitrack-disposition.md),
[source lock](../data/external/uwb-optitrack/source-lock.json).

* **License:** `cc-by-4.0` in the record metadata, verified live through the Zenodo
  API and pinned by a metadata digest, so a later relicense fails closed. The
  depositors are the seven creators, all at Toshiba Research Europe.
* **Bytes:** a 570,029 B archive. The published MD5 matches, SHA-256 is computed
  locally, and every member is pinned by SHA-256. Six `__MACOSX` entries are
  excluded. Members are read in memory only, never extracted.
* **Content:** UWB/OptiTrack tables (Walk 2,796 rows, Trolley 3,201) and BLE/UWB
  tables. README counts include the header row.
* **Metadata gaps:** no timestamps, sampling rate, units, coordinate frames, anchor
  geometry, or tag/marker reference points.
* **Usage:** cross-validation-only, and only as a negative comparability example. The
  [comparability report](external-data/reports/uwb-optitrack-comparability-report.md)
  finds 0/10 sessions comparable:
  * The integer row lag is non-zero in 7/10 (Walk 1,0,0,1,0; Trolley 3,3,3,3,3).
  * The lag-aligned similarity frame fails in 10/10 (Walk translation 0.27–0.30;
    Trolley scale 0.968–0.975, rotation up to 3.3°).
  * There is a motion-dependent residual in 9/10. In Trolley, the cross-track bias of
    +0.027 to +0.045 shows a mounting offset. In Walk, the along-track bias of −0.028
    to −0.037 is a fractional lag or a mounting offset; the two can't be separated.
  * The ranges contain sentinels and physically impossible short values, and the
    Trolley anchor geometry is inconsistent.

  So no UWB error metric is reported, and OptiTrack is never treated as ground truth.
* **Population:** the Walk scenario is non-identifiable human locomotion: a tag carried
  by an unidentified person. Consent is undocumented and was judged not applicable,
  because no personal data exist. The Trolley scenario has no human subject. Not
  soccer, so no `test_id` applies.

### jass2026-sampling-article
Zenodo [20352639](https://zenodo.org/records/20352639) contains only the article
(PDF/XML), with no data. The article is cited in the registry as `donahue2026`
(claim FP-008). *Audit side-finding:* FP-008 was first attributed to an MJSSM URL.
This audit identified the correct, Crossref-verified source and the registry was
corrected, with FP-008 upgraded from grade C to B.

### rugby-league-hawkin-2023
Figshare [24830790](https://figshare.com/articles/dataset/24830790), CC BY 4.0.
Anonymised CMJ and IMTP summary metrics from Hawkin Dynamics software for one English
Super League rugby league squad (n = 28). It has no raw signal and no soccer context.
Revisit for IMTP plausibility ranges in v0.2.

### badminton-eccentric-2025
Zenodo [15370833](https://zenodo.org/records/15370833), CC BY 4.0. It is mostly
derived analysis tables (PCA, LASSO, ANOVA) for junior badminton athletes, with no
raw signals.

### climbers-slsj-2026
Zenodo [22856905](https://zenodo.org/records/22856905), CC BY 4.0. vGRF is given only as
101-point time-normalised waveforms, with no time base or sampling rate, so it is
unusable for signal-processing tasks.

### dvj-vr-cai-2026
Zenodo [18503500](https://zenodo.org/records/18503500), CC BY 4.0. It is a single
1.58 GB archive covering a clinical (chronic ankle instability) population with
virtual-reality height manipulation. It exceeds the practical v0.1 staging budget
on an 8 GB host. It is a candidate for v0.2 drop-jump cross-validation, through
streaming extraction of selected trials.

## Pending rights review (not admitted)

* `shorts-laser` (RES-720): pinned and hash-verified under `data/external/shorts-laser/`, but
  canonical admission is **pending**. Rights to the original measurement data and participant
  consent are undocumented. See
  [shorts-laser-rights-disposition.md](external-data/shorts-laser-rights-disposition.md).
  No manifest exists until those rights are confirmed.

## Privacy rules for any empirical data

1. **Pseudonyms only.** Bundles carry `subject.pseudonym`, never names, and
   re-identification tables never enter the benchmark machine.
2. **Coarse demographics.** The birth year is the most precise age field allowed, and
   serial numbers are stored only as hashes.
3. **No free-text medical notes.** Injury information enters only as structured,
   clinician-supplied context following the IOC/Fuller definitions (IS-001, IS-002).
4. **Minimum cohort size of 10** for any squad-level task built on empirical data.
5. **Minors.** Youth datasets are used only in aggregate or as plausibility ranges, and
   no individual minor's record becomes a published task.
6. **Licenses are honoured.** All accepted datasets are CC BY 4.0 and are cited in any
   derived task or report. Data is never committed to Git (`dbs repo-check`).

## Gaps found

* No open dataset with **raw Nordic hamstring**, **hip squeeze** or **radar sprint**
  time series was found. These families rely on analytical and synthetic signals.
* No open **soccer** dataset with raw force-plate time series was found.
* These gaps reinforce the M3 analytical-truth strategy and are logged for the v0.2
  data plan.
