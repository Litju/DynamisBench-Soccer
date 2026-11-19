# Admission contract for new device families (RES-702)

`dbs qualification admit <proposal.toml>` evaluates a proposal and prints the decision.
Exit code 0 = `ADMISSION_READY`, 1 = `REJECTED` (problems listed), 2 = unreadable proposal.
It never writes the ledger.

**Admission starts qualification; it confers none.** An admitted candidate begins at
`NOT_STARTED`. Ratings are earned only through ledger evidence under the same validator and
gates as the existing eight families. The eight families and fifteen subfamilies are fixed
constants of the ledger; a proposal that names one of them, collides with one of their
subfamily ids, or tries to request ratings is rejected, and a ledger smuggling an extra family
fails `dbs qualification validate`.

## What a proposal must carry

| Requirement | Why |
| -- | -- |
| family id (snake_case) and `<family>.` subfamily ids | no collision with the qualified scope |
| acquisition paradigm and processing methods valid for it | discrete-event families never use continuous filters |
| an owning `RES-<n>` issue for every dimension A–O | no dimension without an owner |
| registry-backed metrology evidence (claims or declared gaps) | evidence-based, not asserted |
| an independent reference plan including dimension J; reference ≠ the engine; never vendor output | no self-validation |
| non-equivalences against named qualified subfamilies, each with a study plan | no assumed interchangeability |
| data-rights statement (source, license, permitted use) | RES-720/721 rights boundaries apply |
| `scope_impact = "none"` | existing scope cannot change |

The proposal schema is `extra = forbid`; unknown fields are rejected.
A synthetic fixture exercising the contract lives in `tests/fixtures/admission/candidate.toml`.
It is a test fixture, not a proposed device.
