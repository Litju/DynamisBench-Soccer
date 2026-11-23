# M4 grader qualification v0.1

**Engineering qualification gate passed for one native LPS reference.** The test suite
uses the unchanged thresholds in [the prespecified protocol](graders-v0.1-protocol.md)
at `4863e9d3134b84291ba0fa13333e7b9874b862c6`. Qualification applies to the tested
M4 infrastructure and the exact engineered slice below.

## Reference and scope

`native-lps-speed` revision 1 binds two synthetic LPS observations to an independently
reviewed external catalog and salted private answer. The clean checkpoint reconstructs
the horizontal speed of the tracked tag at one interior row on a contiguous, unfiltered
25 Hz grid. Production replay uses the three-point derivative and the Euclidean norm
of the horizontal velocity components after native acquisition and QC. The invalid
checkpoint contains a failed signal-valid flag and requires refusal.

The clock is an ideal common measurement clock with zero latency. The field frame is
Z-up with declared synthetic anchors. Reported uncertainty is the deterministic bound
from declared horizontal position-error balls through the three-point derivative.
An independent oracle used the derivative of the prescribed analytical trajectory,
and endpoint errors parallel to motion attain the error bound. The numerical
comparison uses an absolute tolerance of 1e-12 m/s with zero relative tolerance.

The catalog binds `RCP-gnss_lps-lps-{A,B,C,D,E,F,H,I,J,K,M,O}`. RES-700 added
`RCP-gnss_lps-lps-H` for the fixed-grid horizontal tag-speed estimand and
`RCP-gnss_lps-lps-I` for conditional position-error propagation. The independent
review supported both at `cf78f6aae0c47b12547ee6757e50b6d59193a2c2`. DEV-005
supports the distinction between positional and Doppler estimates. G is inapplicable
because the prespecified row uses no protocol event. L is inapplicable because the
task reconstructs a single source. Input hashes, implementation and independent
oracle dependencies, source claims, and verification files are bound by the operator
catalog. The remaining elapsed demonstration is development-only.

## Controls

| Native control class | Expected accepts | Expected rejects | False accepts | False rejects |
| --- | ---: | ---: | ---: | ---: |
| Positive | 2 | 0 | 0/0 | 0/2 |
| Negative | 0 | 3 | 0/3 | 0/0 |
| Adversarial | 0 | 2 | 0/2 | 0/0 |
| Refusal | 1 | 2 | 0/2 | 0/1 |
| Task admission | 1 | 4 | 0/4 | 0/1 |

Empty denominators denote classes with no cases of that disposition. Native grader
controls exercised the correct horizontal estimate and equivalent explanation,
vertical-axis contamination, wrong units, incomplete derivation, a Doppler estimand,
an incompatible method, correct QC refusal, calculation after invalid QC, and
refusal on the eligible observation. The admission controls rejected a wrong
subfamily, stale Policy A receipt binding, changed artifact digest, and untrusted
catalog fingerprint. The key-path sandbox probe passed.

The separate GPT-6.2 Astra control author supplied the analytical observation,
expected results and control design outside the checkout. The prior nine private
comparison diagnostic controls also replayed with zero differences and zero
errors across two positive, two negative, two adversarial and three refusal cases.

## Isolated trials

| Baseline | Passed | Failed |
| --- | ---: | ---: |
| Oracle | 2 | 0 |
| Naive | 0 | 2 |
| Null | 0 | 2 |
| Random | 0 | 2 |

The test oracle reconstructed each result inside bubblewrap from public checkpoint
inputs. Host-side scoring loaded separately withheld answers. Each attempt completed
with a cleanup receipt. The four baselines, all native grading and admission controls,
and the isolation probe replayed with zero differences.
Full transcripts, resource outcomes, environment fingerprints and digests remain
in owner-private custody. An installed-wheel run repeated the same gate outcomes.

## Boundaries

This task supports engineering reconstruction on one synthetic fixed-grid position
track. It establishes no RF ranging or survey accuracy, athlete centre-of-mass
estimate, empirical uncertainty, interchangeability, or broad LPS protocol
qualification. The oracle covers the selected signal-loss case. Its refusal rule
does not generalize to arbitrary native observations.

The canonical full M3 result remains failed at 100/210 verified cells, 110 open,
0/15 engineering-verified subfamilies, and 0/8 passing families. The two narrow
receipts do not change full M3 ratings or empirical evidence. Scientific qualification
and authorization for paid model evaluations remain separate.
