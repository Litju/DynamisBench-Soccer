# RES-718 BioSynGen FORCE pilot

The pilot generated one CMJ and one SJ. Both traversed the source adapter and the
production RES-670 processor. Both received a trial-level QC rejection. The
figures below are diagnostics from those runs.

## Authority and data handling

The pin is BioSynGenBenchmark `e74fc7cc188f9fcdea5e745244c21186b8699903`, the
[reviewed cumulative PR #8 head](https://github.com/Litju/BioSynGenBenchmark/pull/8).
Its tree `320b2778e8da159b12d076340346ce1a1cd26b16` matches the accepted landed
commit `efe3cdf53fc89aa5b54aa9a1cb4ceaeaf03ae851`.
The [exact-head qualification run](https://github.com/Litju/BioSynGenBenchmark/actions/runs/42335483000)
passed on Linux and Windows. The cumulative revision includes the reviewed
fail-closed repairs to RES-524. The original PR #2 head has documented verifier
false accepts. The current upstream branch and newer Human V2 development do
not determine this pin.

The generator is Apache-2.0. The exported source declares project-authored
manufactured data and no participant data. The downstream
[dataset manifest](../../data/manifests/biosyngen-res718-force.json) restricts use
to internal cross-validation and records SHA-256 and byte size for the approved
FORCE projection. Retain bulk source, reference, traces, and capture receipts
outside Git. The trusted dataset manifest comes from the reviewed downstream
checkout. A party controlling that manifest and all source authorities can
reissue hashes, so these are integrity checks within the pilot trust boundary.

Generation used Python 3.12.13, seed 718, and default parameters for both
protocols. It produced 46,141,470 corpus bytes in about 28 s, with 97,312,768 bytes
peak RSS. The upstream verifier counted 2/2 source integrity checks, 2/2
model-reference checks, and 2/2 bounded manufactured synchronizations. For
FORCE, maximum mapped-minus-model sample time was 1.22 µs in CMJ and 1.49 µs in
SJ against approximately 20 µs capture-derived bounds. The downstream adapter
and engine ran in a network-isolated mount namespace with only the approved
FORCE projection and runtime files. The protected corpus was inaccessible
there. A separate upstream verifier initially read the reference before
downstream processing, and its result was reproduced after processing. The
downstream comparator opened the protected corpus only after both production
runs completed.

Each case contained 3,501 original samples per independently modeled left and
right vertical force channel, on one shared native nominal 1,000 Hz FORCE
acquisition. There was no resampling or reversal of channel roles. The exact
producing processor is RES-670 `ce6bf6ef54e49981344d62fcded813df11999185` and the
source adapter is RES-718 `dbc31ae6b98cacbd262f8e469b7fb42af15f6f6d`.
The v2 acquisition marks calibration as `ASSUMED_UNCALIBRATED` and retains the
bounded mapping from native FORCE time to the simulated pulse reference.

## Measured diagnostics

Cells give **production diagnostic / protected model reference**. Timing is in
reference-clock coordinates. The first chronological processor flight was
compared with the modeled primary jump, independently of its closeness.

| Quantity | CMJ | SJ | Pilot diagnostic tolerance |
| --- | ---: | ---: | --- |
| Mass (kg) | 74.9769 / 75.0000 | 74.9685 / 75.0000 | 0.20394 kg |
| Body weight (N) | 735.272 / 735.499 | 735.190 / 735.499 | 2 N |
| Start-to-takeoff net impulse (N·s) | 143.814 / 143.176 | 142.163 / 141.480 | 3.75 N·s |
| Takeoff velocity (m/s) | 1.91811 / 1.90901 | 1.89630 / 1.88640 | 0.05 m/s |
| Takeoff (s) | 1.758939 / 1.759000 | 1.357954 / 1.358000 | 3.040 ms |
| Touchdown (s) | 2.135926 / 2.135953 | 1.729941 / 1.729208 | 3.040 ms |
| Flight duration (s) | 0.376987 / 0.376953 | 0.371987 / 0.371208 | 6.080 ms |
| Onset (s) | 1.211958 / 1.203000 | 1.201959 / 1.202000 | Different definitions |
| Whole-record net impulse (N·s) | −63.350 / −0.038 | −56.260 / −0.007 | Clipped segment, no acceptance test |

The quiet left/right means were 365.443/369.830 N for CMJ and 365.453/369.737 N
for SJ. Minima were −5.951/−4.120 N and −5.646/−3.052 N. Both channels
hit 5,000 N, on four rows for CMJ and two for SJ. RES-670 rejected both cases
with `FP-RAILS`, `QC-T04` (two airborne intervals detected), and `QC-T05`
(arm condition undeclared). The additional detected intervals were
2.245922–2.284921 s and 1.840937–1.874936 s. The whole-record impulse deficits
were 63.312 and 56.253 N·s. Clipped post-takeoff data and the rigid impact
comparison prevent interpreting those deficits as calibrated measurement errors.

The 2 N quiet-weight screen covers the declared order of 0.2% vertical gain
(about 1.47 N at 75 kg), offsets, and approximately 0.2 N for three conditional
body-weight standard errors. Converted with the same gravity it is 0.204 kg.
The early impulse screen allowances cover roughly 2 N baseline error across
1.75 s and less after the integration start, plus acquisition and discrete-edge
effects on the unsaturated early interval. The 3.75 N·s scale corresponds to
0.05 m/s at the nominal 75 kg. These are pilot discrepancy screens. Timing
uses the engine's sample interval plus twice the mapping uncertainty
(about 1.040 ms) and two 1 ms model intervals for event bracketing and
threshold/filter response. Flight duration combines both event allowances.
The model comparator obtains the impulse reference from the change in modeled
vertical momentum. Production uses 5-SD observed onset, while model kinetic
onset uses a small latent force departure. The 8.96 ms CMJ difference has no
onset acceptance interpretation.

The body-weight standard errors were 0.06635 N and 0.06626 N under the
processor's independent-quiet-samples assumption. The processor withheld a
complete measurement uncertainty and a velocity/displacement input envelope.
There is no COP, six-axis wrench, impact peak/RFD, real-athlete, or device
validity result. The independent reference is a manufactured physics trajectory.
No trial-level result was admitted, and engineering and scientific qualification
remain unchanged.

LPT was present upstream only, with 3,500 samples per case and a separate
nominal 1 kHz device clock. It observes cable travel at a hip belt, with
positive motion away from the reel and zero at the first acquired encoder
sample. Even the upstream derived inclination result represents attachment
motion under its own geometry. No LPT pairing by row, RES-695 processing,
bar motion, or body COM inference was performed.

## Reproduction

From a separate BioSynGen checkout at the exact pin, write the following
configuration to `$ART/cases.json` in a new private artifact directory.

```json
{"schema_id":"BSMB-CORPUS-CONFIG-1.0","cases":[{"protocol":"cmj","parameters":{}},{"protocol":"sj","parameters":{}}]}
```

```sh
python -m scripts.generate_paired_corpus --config "$ART/cases.json" --protocol all \
  --count 2 --seed 718 --max-seconds 180 --max-bytes 134217728 --out "$ART/corpus"
```

In that checkout, verify the SOURCE and DIAGNOSTICS inventory hashes against
`inventory.json`. For each case call the pinned
`biosyngen_benchmark.paired_source.verify_source(source, capture, bundle_root=corpus)`
with the parsed out-of-band capture record. Require `source == "PASS"` and
`synchronization == "BOUNDED_CAPTURE_MAPPING"`. Take the FORCE-only projection
using `manifest.json` and `streams/force.csv` verbatim, the capture file verbatim
as `capture_authority.json`, and the verifier's result as
`source-verification.json` using `json.dumps(result, indent=2, sort_keys=True) + "\n"`.
Store the four files under `$ART/force-source-only/<case>`. Keep the dataset
manifest fixed. A regenerated input with different bytes fails admission and
needs an independently reviewed manifest rather than a silently refreshed pin.

In DynamisBench-Soccer with Linux Bubblewrap and the frozen Python 3.12 environment:

```sh
uv sync --frozen
scripts/run_biosyngen_force.sh "$ART/force-source-only" "$ART/processing"
```

After both production runs, run the pinned upstream verifier separately.

```sh
python -m scripts.verify_paired_corpus --root "$ART/corpus" --out "$ART/verification.json"
```

Then run the comparator from DynamisBench-Soccer.

```sh
uv run --frozen python scripts/compare_biosyngen_force.py \
  --processing "$ART/processing" --corpus "$ART/corpus" \
  --verification "$ART/verification.json" \
  --dataset data/manifests/biosyngen-res718-force.json --out "$ART/comparison.json"
```

The runner invokes the existing CLI using the recorded [method](method.json):
raw software force retaining upstream analog conditioning, 5-SD onset,
20 N raw contact threshold, trapezoidal integration from quiet stance, and the
declared ±5,000 N input rails. The 0.010–1.012 s weighing window stays within the
declared baseline and provides at least 1 s on the mapped grid. No gain was
reapplied and no arm condition was inferred. Each exit code 1 is preserved
as a diagnostic refusal. The comparator binds the two engine input digests,
versioned bundle/source lineage, bilateral traces and protected trial hashes
before summarizing. CI covers the falsifiers with bounded test doubles.
The genuine corpus stays local.

## Receipt SHA-256

| Record | SHA-256 |
| --- | --- |
| Run receipt | `3a04b5f4d8e870c85f4fd49f32199750200b26157490a773f08fede7a957b1d0` |
| Corpus inventory | `8f7882cfa0174126042af1ae4f7cd5b78d6e3a623161802d78422db326b265ba` |
| Full upstream verifier | `1670dd3a6d3a9efd8dd6d08c06f01c32ec361fe7a31cbbe025bf241f4c35bb92` |
| CMJ protected inventory | `1f7f5b73395d5514b95bcfa4a7a0a8840b825e09cf5618c393b0b2772ae221e4` |
| SJ protected inventory | `1fe42bef8c77b5fbaef1c1ae1404e44aecb3370c5759bf4d6c115e2c63e2ef8d` |
| CMJ production report | `8aa0d83c7a9cfbbf7986541a211bde89531738f29af075137a946e817e048311` |
| SJ production report | `79fc7ea46a1f50d24d86c73dd579b70ae77b3e883bec957e4d27e081361ee88b` |
| Comparator summary | `d02d8868279663d230d8171aa4113c88a5e257e252a143af0fea7da18d810550` |
