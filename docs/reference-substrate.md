# RES-669: analytical truth substrate — first implementation slice

This code establishes independent mathematical reference cases. It is **not** a neuromuscular simulator, a validated representation of human physiology, a device calibration certificate or an AI benchmark.

## Cases implemented

- **CMJ/SJ/DJ:** physically consistent piecewise-acceleration trajectories with continuous velocity/displacement, zero GRF during ballistic flight, explicit contact events and known take-off velocity, height and impulse.
- **Isometric force:** exact ramp/hold/release, peak force, time-window mean RFD and integrated external-force impulse.
- **Sprint:** exact exponential velocity, acceleration, displacement and numerically inverted crossing times; theoretical sprint F0, V0, Pmax and RF metrics are intentionally **not** inferred without a declared mechanical model.
- **Ideal observation models:** force plate vertical reaction, vertical IMU specific force in an ideal frame, geometric LPT cable length, load-cell voltage, radar radial velocity, laser displacement and timing-gate event latency.

The observation models test conditional mathematical consistency only. They cannot establish real-vendor agreement, device validity, human landing mechanics or physiological fatigue. They are deliberately isolated from production device/assessment engines. RES-693 raw schema and RES-694 device evidence remain prerequisites for full device-family reference coverage.

## Verification boundary

A clean mathematical fixture is not required to look like a noisy human recording. Legacy QC flat-plateau clipping heuristics may flag an exact analytical waveform; do not change the truth solution to fool a QC heuristic. Controlled corruption and sensor-specific noise belong in separately characterized observation and fault models.

No M3 device-family gate is passed by this initial slice. The work remains RES-669 In Progress pending all approved analytical scenarios, independent verification and subfamily evidence.
