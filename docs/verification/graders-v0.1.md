# M4 grader qualification v0.1

**Gate result: BLOCKED.** Diagnostic comparison thresholds passed on nine
independently authored controls. Engineering-admitted references in the
review catalog: zero. The offline runner has therefore produced no scored
formal benchmark results.

## Controls

The thresholds were committed at
`4863e9d3134b84291ba0fa13333e7b9874b862c6` before execution.
The separate control author, GPT-6.2 Astra, created the reference values
and submissions in an external owner-private file and did not run the
grader. The report publishes counts and category names.

| Class | Cases | Expected passes | Expected rejections | False passes | False rejections |
| --- | ---: | ---: | ---: | ---: | ---: |
| Positive | 2 | 2 | 0 | 0 | 0 |
| Negative | 2 | 0 | 2 | 0 | 0 |
| Adversarial | 2 | 0 | 2 | 0 | 0 |
| Refusal | 3 | 2 | 1 | 0 | 0 |

Refusal controls matched all three independently assigned outcomes.
Repeated evaluation produced zero differences in nine controls.
The grader emitted one finding in each of `eligibility_or_refusal`,
`interpretation_boundary`, `source_citations`, `unit_or_dimension` and
`value`. These counts describe controlled comparisons. They do not
estimate error rates in a deployed benchmark.

## Admission and isolation

The versioned registry has one two-checkpoint analytical demonstration.
The pinned empty catalog returns `DEVELOPMENT_ONLY_NOT_QUALIFIED` for it.
Eleven admission rejection cases exercise untrusted catalog data, changed
method, clock, geometry, uncertainty and estimand, stale input, empirical
and diagnostic source categories, missing QC, and wrong subfamily. The
missing-QC and wrong-subfamily checks share a test function. No expected
positive admission was available, so the positive admission error rate is
undefined.

A fresh Linux wheel environment ran 54 M4 tests. The tests exercised
ephemeral private custody, metadata canaries, host files and sockets,
environment and descriptor isolation, filesystem link handling,
read-only input, execution limits, cleanup and staged correction release.
The formal naive, null and seeded random stubs replayed with stable
output digests and were excluded from scoring. The independent formal
oracle baseline remains unevaluated. RES-737's four public demonstrations
still have four oracle passes and zero for each weaker baseline, solely
as development smoke evidence.

## Outstanding qualification evidence

The formal grader currently replays elapsed-time subtraction. Wider
scientific method families, native event/QC replay, claims expressed solely
in free text, and empirical-device performance remain outside the
demonstrated coverage. Bubblewrap tests exercise the configured Linux
boundary and rely on the host kernel. Runtime usage records wall time and
output size plus enforced limits. CPU time and peak memory consumption
are not measured per attempt.

To satisfy RES-679, an independently reviewed real slice must pass
RES-676's admission evaluator using current Policy A evidence. The
formal runner and grader must then exercise controls on that admitted
slice, including a privileged held-out oracle, invalid method and receipt
controls, and the remaining adversarial cases. Final-head independent
review and Linux CI remain part of acceptance.

The canonical M3 gates remain failed at 100/210 verified computational
cells, 0/15 engineering-verified subfamilies and 0/8 passing families.
Physical-device validity and independent scientific qualification remain
unestablished. M5 scored task authoring stays downstream of RES-679.
