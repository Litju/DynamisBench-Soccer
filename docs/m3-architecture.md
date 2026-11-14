# M3 architecture: device parity and assessment separation

M3 evaluates the scientific reference processing substrate for an AI benchmark. It is **not** DynamisSimBench and does not simulate athlete physiology for validity claims.

## Three separate engines

1. **Device-processing engine:** raw acquisition, calibration, geometry, clocks, sensor observations, provenance and uncertainty.
2. **Assessment-science engine:** a specified measurement protocol (CMJ, IMTP, Nordic, VBT, sprint/COD), metric estimands and valid inference conditions.
3. **AI-benchmark engine:** blinded tasks and graders evaluate LLM/agent analysis of those observations in M4/M5; it must never expose oracle answers.

One mathematical truth model cannot describe all assessments. RES-669 must construct separate families of exact physical scenarios: jumping and contact, isometric force histories, eccentric/rigid-body exercise, and planar sprint/COD motion. Each relevant device has its own observation model; the same scenario can serve synchronized devices only when their measured phenomena overlap.

## Stage gates

- **M3 ENGINEERING_VERIFIED:** dimensions A-M and O, independently tested for every instrument subfamily. N is deliberately excluded because it depends on M4/M5.
- **M5 AI_BENCHMARK_QUALIFIED:** M3 plus dimension N for independently scored tasks and qualified graders.
- **M8 SCIENTIFICALLY_QUALIFIED:** M5 plus at least two independently documented qualified expert reviews PER device family.

A-O is one long-term quality contract, not an impossible M3-only requirement. Specified contracts and imported CSV files are not independent validation. NOT_APPLICABLE needs recorded applicability review and cannot mean that evidence was unavailable.

## Measurement paradigms

Discrete contact and timing events require event integrity and chronology, not continuous-signal filtering. LPT cable length is not vertical displacement without geometry. GNSS positional derivative and radar Doppler velocity are different estimands. Commercial device values are not oracles.

Cross-device testing must separate analytical agreement under documented observation models from empirical device interchangeability. The latter needs appropriate paired validation data, specified tolerance and expert review. Unknown interchangeability remains explicitly unknown.

## Scope and ownership

- RES-692: machine-verifiable parity contract, subfamily ledger, staged qualification and test controls.
- RES-693: raw schemas, clocks, frames, per-device calibration and geometries.
- RES-694: validity and uncertainty references, explicit gaps.
- RES-669: exact analytical truth scenarios and observation models.
- RES-670 / RES-695–RES-701: separate, complete device-processing implementations.
- RES-671–RES-674: assessment-specific metrics and reliability.
- RES-703: independently qualified cross-device agreement studies.
- RES-704: later AI task families (dimension N).
- RES-675: independent M3 engineering report.
- M8: actual external scientific qualification.

Current status: **0 of 8** families meets M3. No AI task family or medical clearance process is qualified.
