# RES-736 engineering audit disposition

| Finding | Disposition and evidence |
| --- | --- |
| P1-A fractional COD start | Fixed. The COD tests recover constant-speed elapsed time at three fractional offsets and recover distance crossings and entry speed for a linear speed ramp. |
| P1-B partial gates | Fixed. A full-course total requires zero origin in position and elapsed time, strictly increasing positions and times, and an observed finish. Partial, duplicate, reversed and nonfinite inputs are refused. |
| P1-C DJ eligibility | Fixed. The valid DJ control uses a prescribed touchdown velocity independent of the height declaration. The signal must bracket touchdown and show a complete rebound. Standing CMJ, a truncated initial contact, forged velocity on standing force, missing evidence and contradictory height are ineligible. DJ diagnostics remain available. |
| P2-D failed report freshness | Fixed. The checker rejects a freshly rendered report with a failed case and names the failing case. |
| P2-E refusal specificity | Fixed. Matrix cases check exception type and case-specific reason. The GNSS/LPS faults are now bound to the corrupted observations so the case reaches the actual jump check. |
| P2-F execution evidence | Labelled the execution basis of each side. Three studies execute both engines. One-sided and formula-only demonstrations retain their scope. Additional matched two-engine paths remain research backlog. |
| P2-G computed gate | The live test checks consistency with computed states. In-memory controls exercise a structurally valid pass, a blocked transition and refusal without independent reference. Review of measurement-evidence receipts remains outstanding. |

## Empirical research backlog

- Admit native acquisition data only after documented rights and provenance review. Collect clock, latency, calibration, geometry and criterion measurements for each subfamily.
- Collect adjudicated event annotations and real fault classes, and establish device-specific uncertainty and repeatability budgets.
- Pair audited devices under aligned protocols and estimands. Extend synthetic engine comparisons where common observations can be generated and admitted. The analytical force/load-cell, radar/gate and radar/positional examples identify which paths still need independent execution or pairing.
- Prespecify allowable differences and target precision for limits of agreement. Justify sample size for each comparison using independent participants and the replicate design. Treat the existing 20-participant check as a floor.
- Define and review measurement-evidence receipts that bind protocol, source, calibration, analysis and review to each proposed rating. The current ledger can check structure and references. Scientific adequacy requires independent assessment of the actual evidence.
- Resolve laser-data rights before use. Establish time and frame comparability before using the UWB–OptiTrack pilot for agreement. Keep vendor-reported outputs separate from independent criterion measurements.

The generated verification report carries the current computed state. RES-676 remains blocked by RES-675. RES-679 and the M4/M5 scientific gates remain pending.
