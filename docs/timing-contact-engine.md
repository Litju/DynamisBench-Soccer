# Timing-gate and contact-system reference engine

Use canonical schema-v2 event bundles with the acquisition clock and source preserved.
The command verifies the Parquet hash and reads rows in their recorded order. Timing
uses declared edges and elapsed intervals. The qualification ledger allows discrete
event methods for these devices.

```sh
dbs timing gates sprint-stream-a sprint-stream-b --policy gates.json --sensitivity alternate-gates.json
dbs timing contact jump-stream --policy contact.json --sensitivity alternate-contact.json
```

Each bundle argument is the stem of a Parquet and JSON pair. Use one stem per
native timestamp stream and a common reference clock across gate streams. Direct
Python entry points are `engines.timing_gates.sprint` and `engines.contact.contact`.
The policies are Pydantic models with JSON schemas available through
`GatePolicy.model_json_schema()` and `ContactPolicy.model_json_schema()`.

## Acquisition and timing

| Stream | Columns | Source interpretation |
| -- | -- | -- |
| Gates | `t_s`, `distance_m`, optional `gate_id` and `beam_id` | One raw beam break per row. IDs are integer count columns owned by the timestamp source. A single-beam stream can use the exact declared station position as its ID. Multi-beam stations require native beam IDs. |
| External starts | Same gate-event columns plus `start_kind` | Record the start at its declared position using `gate_id=0`. Use `start_kind=1` for pressure-pad releases or `2` for sound cues. Other rows use `start_kind=0`. When present, `beam_id` is zero on a start row. |
| Contact systems | `t_s`, `contact` | Binary switching events with 1 for contact on and 0 for contact off. The schema-v2 device class `contact_mat` carries all three contact-system types. The policy identifies the mat, pressure pad, or optical contact bar, and the source geometry records its switching definition. |

Legacy distance and cumulative-time exports require an acquisition contract
establishing that the provided timestamps are native clock readings. Analytical
truth bundles are comparison inputs for the test suite. A shared clock uses the
reference clock directly. For wired or wireless clocks, the affine alignment must
provide at least two ordered synchronization pulse pairs bracketing the source
timestamps. Streams with the same native clock ID must agree on its clock and alignment.

Declare the resolution, synchronization latency and its bound for each used clock,
and identify the calibration or protocol in `timing.authority`. Mapped timestamps
are modeled as reference-clock receipts. Subtract positive reference-side latency
from the synchronization pulse capture, then the independent latency of the event
edge. Use zero sync latency when the pulse pairing already accounts for it. Separate
on/off latency for a contact system can move flight and contact times in opposite
directions.

Each estimate reports its nominal value and lower and upper bounds. An edge bound
adds half a native clock tick scaled into reference seconds, alignment and sync
latency error, edge latency error, and any declared event-definition error. This
assumes nearest-tick timestamping. Interval arithmetic allows conservative bounds
even for two edges on the same clock. Use a calibration that covers the trial and
supply bounds in seconds. A duration whose lower bound is nonpositive is refused.
Protocol plausibility checks use nominal durations and velocities. Reported bounds
are scenario bounds without a statistical confidence level.

## Gate policy and sprint metrics

Specify the gate IDs, positions in route order, beam heights, beam types, voting
quorum, start posture and trigger, refractory window, and a protocol-specific
maximum segment-average velocity. This example times the segment from 5 to 10 m.

```json
{
  "timing": {
    "clock_model": "shared",
    "authority": "Example shared timing counter",
    "clocks": {
      "device": {"resolution_s": 0.001, "sync_latency_s": 0, "sync_latency_error_s": 0}
    }
  },
  "gates": [
    {
      "id": 1, "position_m": 5, "beam_type": "single", "quorum": 1, "coincidence_s": 0,
      "beams": [{"id": 1, "height_m": 1, "timing": {"latency_s": 0, "latency_error_s": 0}}]
    },
    {
      "id": 2, "position_m": 10, "beam_type": "single", "quorum": 1, "coincidence_s": 0,
      "beams": [{"id": 1, "height_m": 1, "timing": {"latency_s": 0, "latency_error_s": 0}}]
    }
  ],
  "start": {"mode": "first_gate", "start_position_m": 4.5},
  "start_posture": "standing",
  "debounce_s": 0.003,
  "max_segment_velocity_m_s": 13
}
```

- `first_gate` sets zero at the first accepted gate crossing. Optional
  `start_position_m` records a spatial offset from the first beam.
- `fixed_offset` uses a declared elapsed `offset_s` and `error_s` before the
  first crossing, with optional `start_position_m`. It shifts cumulative times
  while preserving gate-to-gate times and velocities.
- `pressure_pad` and `sound_cue` require one corresponding native start row and
  a `position_m` and `timing` declaration. The initial split measures elapsed time
  from that event. Segment-average velocity begins at the first pair of gates.

A positive refractory window keeps the first accepted edge on a beam and discards
repeats inside the declared window. A coincidence group starts at its first edge
and counts each beam once. The time of the quorum-th distinct edge defines the
crossing. The engine records rows discarded for incomplete votes and refuses zero
or multiple eligible crossings at a required station. A sole plausible false break
can still pass a single beam. Even a successful vote estimates passage under its
declared beam geometry and body-part definition. A gate pair measures distance
divided by crossing-to-crossing time. Instantaneous velocity and COM passage require
other observations.

## Contact policy and jump metrics

The initial row is classified by the protocol as `contact_snapshot`,
`airborne_snapshot`, or an observed `touchdown`. An airborne snapshot can precede
initial landing from a drop. Flight starts at an observed switch-off and ends at
the next switch-on. Ground contact requires an observed preceding touchdown.
The engine refuses incomplete flights, phase order that timing cannot resolve,
and a complete-flight count that differs from the declared count.

```json
{
  "timing": {
    "clock_model": "shared",
    "authority": "Example shared counter and switch-delay calibration",
    "clocks": {
      "device": {"resolution_s": 0.001, "sync_latency_s": 0, "sync_latency_error_s": 0}
    }
  },
  "system": "contact_mat",
  "initial_event": "touchdown",
  "switch_on": {"latency_s": 0.026, "latency_error_s": 0.001},
  "switch_off": {"latency_s": 0.004, "latency_error_s": 0.001},
  "debounce_s": 0.005,
  "expected_flights": 1,
  "flight": {"min_s": 0.15, "max_s": 0.9},
  "ground_contact": {"min_s": 0.04, "max_s": 0.5},
  "ballistic_equal_com_height": true
}
```

Contact debounce uses raw native timestamp differences. A positive refractory
window discards a same-state repeat or a short opposite-state pulse that returns
to the prior state. The window must be shorter than the minimum declared phase.
The pulse rule can move the selected switching edge later when a real transition
bounces before settling. A zero window makes duplicates and unresolved transitions
refusals. A resting contact snapshot censors the start of stance.

Flight-time height uses `g × flight_s² / 8` with standard gravity and requires the
protocol to affirm ballistic flight with equal COM height at takeoff and touchdown.
Comparable posture at both events supports that assumption. For drop jumps and
repeated rebounds the engine can report height divided by the immediately preceding
ground-contact time in m/s. This RSI definition requires both a supported height
and a measured stance. The output provides a reason whenever either derived metric
is refused. Force-platform timing uses force criteria, and IMU timing uses
kinematic detection. Both have distinct event definitions from contact switching.

## Sensitivity, verification and limits

Use repeatable `--sensitivity` policies to rerun the same native rows under a
different start offset, switching latency, debounce window, or beam quorum. Each
result includes its policy, metrics and discarded rows, or its refusal reason.
For body-part and beam-height uncertainty, provide per-beam `definition_error_s`
from calibration or observations. Compare beam votes when the underlying stream
contains the corresponding beams. A physical change in setup requires observations
from the changed geometry. Timing bounds are conditional on the selected event
groups and pairs, so alternate policies are especially relevant near a voting or
debounce threshold.

The test suite compares serialized engine inputs against the independent RES-669
crossing, phase and motion models. It covers missing and duplicated events,
out-of-order native rows, a false arm break, asymmetric switching delays,
a plausible biased wireless split, an extra plausible flight, and early drop
censoring. Linux CI also runs the timing tests from the installed wheel.

Timing and contact remain in progress in the qualification ledger. Empirical
trigger geometry, switching thresholds, clock and latency calibration, assessment
validation, and cross-device agreement remain open under RES-694, RES-671,
RES-673 and RES-703. An isolated plausible false edge or a physically different
landing posture can require another instrument or a protocol observation to detect.
