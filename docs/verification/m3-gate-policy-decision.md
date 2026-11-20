# M3 gate policy: owner decision record (RES-675 / RES-692)

Status: **PROPOSED. Not authoritative. No gate semantics, rating or threshold changed.**

## Finding

`ledger.gate(Stage.M3)` and `state.m3_gate()` agree on every ledger tested
(`tests/test_gate_agreement.py`: committed ledger, complete, blocker, partial L,
missing J, unresolved comparison, vendor-only evidence). Both report 0/15 on the
committed ledger and exit 1. No decision contradiction was reproduced.

The real ambiguity is semantic:

| Source | What it implies |
| --- | --- |
| `state.REQUIRED_EXPERIMENT` and the blocker output | Policy B: native audited data, metrology, paired measured agreement (>= 20 subjects) |
| `ledger.validate` VERIFIED rule | Policy A: any of `independent_test`, `analytical_reference`, `empirical_study`, `external_review` suffices for every dimension, including A |
| `m3-architecture.md` QP-08 | State name `ENGINEERING_VERIFIED`, which reads as Policy A |
| `SCIENTIFICALLY_QUALIFIED` | Requires `empirical_study`/`external_review` J evidence plus two reviewers |

Reproduced: a ledger with only `analytical_reference` evidence passes both gates
(`test_analytical_evidence_alone_can_satisfy_the_gate_so_the_verified_flag_is_policy_a`),
while the blocker text for the committed ledger demands empirical experiments for A-O.

## Policies

**A: computational reference readiness.** Requires complete specifications,
executable implementation, independent computational verification under declared
conditions, provenance, and limitations. Empirical device validity is a separate status.
Evidence needed: what the repository holds now, plus completed per-subfamily A-M/O
receipts. Risk: "engineering verified" is read as "device valid".

**B: empirical qualification at M3.** Additionally requires audited native capture,
calibration and timing metrology, adjudicated events, and paired agreement for all
15 subfamilies. Evidence needed: external data and rights clearance not yet obtained
(RES-720/721 pilots only). Risk: M3 and M4 are blocked until measured data exist,
and M3 collapses into M5-M8 scope.

## Recommendation (requires explicit owner authorization)

Report two separate flags per subfamily, `computational_reference_ready` and
`empirical_device_validity`, never one VERIFIED flag. Gate M3 on the first;
keep `SCIENTIFICALLY_QUALIFIED` on the second. Required changes if approved:
split `REQUIRED_EXPERIMENT` into two lists, add per-dimension evidence-class
requirements to `validate`, regenerate the ledger and `states` output, update
QP-08. Acceptance of M3 thresholds, ratings and blockers is untouched until then.

## Original versus current

QP-08 (approved) says M3 completes when every family computes
`ENGINEERING_VERIFIED`. The empirical experiment strings in `state.py` were added
with the computed-state module (RES-675) and were not separately approved as a
scope change. Owner confirmation is needed on which reading is intended.
