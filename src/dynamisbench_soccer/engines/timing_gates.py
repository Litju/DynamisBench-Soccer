"""Sprint metrics from declared beam crossings."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Annotated, Literal

from pydantic import Field, model_validator

from dynamisbench_soccer.engines.event_time import (
    Discard,
    EdgeTiming,
    Estimate,
    Event,
    EventRefusal,
    EventStream,
    Nonnegative,
    Policy,
    Positive,
    Timing,
    require_canonical_calibration,
)
from dynamisbench_soccer.signals.schema import DeviceClass, Quantity, SignalBundle


class Beam(Policy):
    id: Annotated[int, Field(gt=0)]
    height_m: Positive
    timing: EdgeTiming


class Gate(Policy):
    id: Annotated[int, Field(gt=0)]
    position_m: float
    beam_type: Literal["single", "dual", "multiple"]
    beams: Annotated[tuple[Beam, ...], Field(min_length=1)]
    quorum: Annotated[int, Field(gt=0)]
    coincidence_s: Nonnegative

    @model_validator(mode="after")
    def _beams(self) -> Gate:
        n = len(self.beams)
        expected = "single" if n == 1 else "dual" if n == 2 else "multiple"
        if self.beam_type != expected or len({beam.id for beam in self.beams}) != n:
            raise ValueError("gate beam type and distinct beam IDs must match the beam inventory")
        if self.quorum > n or (self.quorum > 1 and not self.coincidence_s):
            raise ValueError("beam quorum requires enough beams and a positive coincidence window")
        return self


class FirstGate(Policy):
    mode: Literal["first_gate"]
    start_position_m: float | None = None


class FixedOffset(Policy):
    mode: Literal["fixed_offset"]
    start_position_m: float | None = None
    offset_s: Positive
    error_s: Nonnegative

    @model_validator(mode="after")
    def _offset(self) -> FixedOffset:
        if self.error_s >= self.offset_s:
            raise ValueError("fixed start offset must have a positive lower bound")
        return self


class TriggerStart(Policy):
    mode: Literal["pressure_pad", "sound_cue"]
    position_m: float
    timing: EdgeTiming


class GatePolicy(Policy):
    timing: Timing
    gates: Annotated[tuple[Gate, ...], Field(min_length=2)]
    start: Annotated[FirstGate | FixedOffset | TriggerStart, Field(discriminator="mode")]
    start_posture: Literal["standing", "crouched", "flying"]
    debounce_s: Nonnegative
    max_segment_velocity_m_s: Positive

    @model_validator(mode="after")
    def _route(self) -> GatePolicy:
        if len({g.id for g in self.gates}) != len(self.gates) or any(
            b.position_m <= a.position_m for a, b in pairwise(self.gates)
        ):
            raise ValueError("sprint gates must have distinct IDs and increasing positions")
        position = (
            self.start.position_m
            if isinstance(self.start, TriggerStart)
            else self.start.start_position_m
        )
        if (
            position is not None
            and isinstance(self.start, FirstGate)
            and (position != self.gates[0].position_m)
        ):
            raise ValueError("first-gate start position must equal the first gate")
        if position is not None and position > self.gates[0].position_m:
            raise ValueError("start position must be at or before the first gate")
        return self


_GATE_LIMITATIONS = (
    "A timestamp represents beam interruption by the triggering body part at the configured "
    "height. A limb may interrupt the beam before the torso.",
    "DEV-004: The studied single- and dual-beam methods give different timing observations. "
    "Beam height, trigger and voting settings have no universal correction.",
    "Declared link and trigger delays, clock alignment and nearest-tick resolution condition "
    "the reported timing bounds.",
    "Split speed is surveyed segment distance divided by elapsed beam time. "
    "Instantaneous speed and radar-derived speed have different definitions.",
    "Missing stations, ambiguous beam groups and unresolved event order cause refusal. "
    "Timing bounds are conditional on the selected beam group.",
    "Analytical verification concerns declared event geometry and computation. "
    "Hardware accuracy, course calibration and empirical interchangeability need "
    "measured evidence.",
)


def gate_limitations(policy: GatePolicy | None) -> tuple[str, ...]:
    if policy is None:
        return _GATE_LIMITATIONS
    start = {
        "first_gate": "The first gate defines elapsed time zero; motion before it is unobserved.",
        "fixed_offset": "A declared offset estimates the start-to-first-gate time.",
        "pressure_pad": "Elapsed time starts from the measured and corrected pressure-pad release.",
        "sound_cue": "Elapsed time starts from the measured and corrected sound cue.",
    }[policy.start.mode]
    voting = (
        "Multi-beam voting selects the quorum edge within a declared coincidence window."
        if any(g.beam_type != "single" for g in policy.gates)
        else "Single-beam stations accept one resolved interruption per station."
    )
    refractory = (
        "The declared refractory window can suppress repeated interruptions on the same beam."
        if policy.debounce_s
        else "Zero refractory window applies no beam-repeat suppression."
    )
    return (*_GATE_LIMITATIONS, start, voting, refractory)


@dataclass(frozen=True)
class Split:
    gate_id: int
    position_m: float
    crossing: Event
    cumulative_s: Estimate
    split_s: Estimate | None
    segment_average_velocity_m_s: Estimate | None


@dataclass(frozen=True)
class Sprint:
    estimand: Literal["beam_crossing_segment_average_velocity"]
    start_mode: str
    start_event: Event | None
    splits: tuple[Split, ...]
    discarded: tuple[Discard, ...]
    limitations: tuple[str, ...]


def _crossing(
    gate: Gate, hits: list[tuple[int, Event]], debounce_s: float, discarded: list[Discard]
) -> Event:
    if not hits:
        raise EventRefusal(f"missing gate {gate.id}")
    hits.sort(key=lambda hit: hit[1].time_s.value)
    accepted: list[Event] = []
    seen: dict[int, float] = {}
    group: dict[int, Event] = {}

    def vote() -> None:
        events = tuple(group.values())
        native = tuple(row for hit in events for row in hit.native)
        if len(events) < gate.quorum:
            discarded.append(Discard("incomplete beam quorum", native))
            return
        # The quorum timestamp is the kth edge. Its bounds use the same order statistic.
        k = gate.quorum - 1
        accepted.append(
            Event(
                Estimate(
                    events[k].time_s.value,
                    sorted(hit.time_s.low for hit in events)[k],
                    sorted(hit.time_s.high for hit in events)[k],
                ),
                native,
            )
        )

    for beam, hit in hits:
        time = hit.time_s.value
        if beam in seen and debounce_s > 0 and time - seen[beam] <= debounce_s:
            discarded.append(Discard("refractory beam repeat", hit.native))
            continue
        seen[beam] = time
        if group and time - next(iter(group.values())).time_s.value > gate.coincidence_s:
            vote()
            group.clear()
        if beam in group:
            raise EventRefusal(f"gate {gate.id}: duplicate or ambiguous beam trigger")
        group[beam] = hit
    vote()
    if len(accepted) != 1:
        raise EventRefusal(f"gate {gate.id}: expected one beam quorum, found {len(accepted)}")
    return accepted[0]


def sprint(bundles: Sequence[SignalBundle], policy: GatePolicy) -> Sprint:
    """Reconstruct a straight sprint with a declared trigger and a complete gate route."""
    try:
        policy = GatePolicy.model_validate(policy.model_dump())
    except ValueError as exc:
        raise EventRefusal(f"invalid gate course: {exc}") from exc
    if not bundles or len({b.meta.bundle_id for b in bundles}) != len(bundles):
        raise EventRefusal("provide one or more distinct event bundles for one trial")
    meta = bundles[0].meta
    streams = [EventStream(b, DeviceClass.TIMING_GATES, policy.timing) for b in bundles]
    if any(
        (b.meta.session, b.meta.subject, b.meta.test_id, b.meta.protocol, b.meta.data_category)
        != (meta.session, meta.subject, meta.test_id, meta.protocol, meta.data_category)
        for b in bundles
    ):
        raise EventRefusal("gate streams must describe the same trial and acquisition category")
    if meta.test_id != "sprint_gates" or meta.protocol.start_mode != policy.start_posture:
        raise EventRefusal("sprint assessment and declared start posture must match the source")
    if any(stream.reference != streams[0].reference for stream in streams):
        raise EventRefusal("gate streams do not share a reference clock")
    clocks = {s.clock.id: (s.clock, s.alignment) for s in streams}
    if any(clocks[s.clock.id] != (s.clock, s.alignment) for s in streams):
        raise EventRefusal("conflicting declarations for the same native clock")

    gates = {g.id: g for g in policy.gates}
    positions = {g.position_m: g.id for g in policy.gates}
    hits: dict[int, list[tuple[int, Event]]] = {g.id: [] for g in policy.gates}
    beam_clocks: dict[tuple[int, int], str] = {}
    starts: list[Event] = []
    discarded: list[Discard] = []
    start = policy.start
    for bundle, stream in zip(bundles, streams, strict=True):
        if "distance_m" not in stream.source.channels or (
            bundle.meta.channel("distance_m").quantity is not Quantity.DISTANCE
            or bundle.meta.channel("distance_m").unit != "m"
        ):
            raise EventRefusal("expected source-native gate distances in metres")
        require_canonical_calibration(
            bundle,
            stream,
            {"distance_m": "m", "t_s": "s", "gate_id": "1", "beam_id": "1", "start_kind": "1"},
        )
        explicit_gate = "gate_id" in bundle.data
        explicit_beam = "beam_id" in bundle.data
        for name in ("gate_id", "beam_id", "start_kind"):
            if name in bundle.data and (
                name not in stream.source.channels
                or bundle.meta.channel(name).quantity != Quantity.COUNT
                or bundle.meta.channel(name).unit != "1"
                or not all(float(value).is_integer() for value in bundle[name])
            ):
                raise EventRefusal(f"{name}: expected source-native integer IDs")
        for row, position in enumerate(bundle["distance_m"]):
            gate_id = int(bundle["gate_id"][row]) if explicit_gate else positions.get(position)
            start_kind = bundle["start_kind"][row] if "start_kind" in bundle.data else 0
            if gate_id == 0 and isinstance(start, TriggerStart):
                expected_kind = 1 if start.mode == "pressure_pad" else 2
                if start_kind != expected_kind:
                    raise EventRefusal("missing or mismatched native start trigger type")
                if position != start.position_m or (explicit_beam and bundle["beam_id"][row] != 0):
                    raise EventRefusal("start trigger does not match the declared position or beam")
                starts.append(stream.event(row, start.timing))
                continue
            gate = gates.get(gate_id) if gate_id is not None else None
            if gate is None or gate.position_m != position or start_kind != 0:
                raise EventRefusal(f"unexpected gate ID or position at {stream.bundle_id}:{row}")
            if not explicit_beam and gate.beam_type != "single":
                raise EventRefusal(f"gate {gate.id}: multi-beam events require native beam IDs")
            beam_id = int(bundle["beam_id"][row]) if explicit_beam else gate.beams[0].id
            beam = next((b for b in gate.beams if b.id == beam_id), None)
            if beam is None:
                raise EventRefusal(f"gate {gate.id}: unexpected beam {beam_id}")
            geometry = stream.source.geometry
            required_geometry = {"beam_type", "gate_height_m", "trigger_method"}
            if (
                geometry.beam_type in {None, "unknown"}
                or geometry.gate_height_m is None
                or not (geometry.trigger_method or "").strip()
                or (geometry.trigger_method or "").strip().lower() == "unknown"
                or required_geometry.intersection(geometry.unknown)
            ):
                raise EventRefusal(f"gate {gate.id}: missing source beam geometry or trigger")
            if geometry.beam_type != gate.beam_type or (
                geometry.gate_height_m != min(b.height_m for b in gate.beams)
            ):
                raise EventRefusal(
                    f"gate {gate.id}: beam declaration conflicts with source geometry"
                )
            key = gate.id, beam_id
            if beam_clocks.setdefault(key, stream.clock.id) != stream.clock.id:
                raise EventRefusal(f"gate {gate.id} beam {beam_id}: conflicting timestamp clocks")
            hits[gate.id].append((beam_id, stream.event(row, beam.timing)))

    crossing = [_crossing(g, hits[g.id], policy.debounce_s, discarded) for g in policy.gates]
    first = crossing[0]
    origin: Event | None = first
    offset = Estimate(0, 0, 0)
    if isinstance(start, TriggerStart):
        if len(starts) != 1:
            raise EventRefusal(f"{start.mode}: expected one start trigger, found {len(starts)}")
        origin = starts[0]
        offset = first.time_s.elapsed(origin.time_s, "start to first gate")
    elif isinstance(start, FixedOffset):
        origin = None
        offset = Estimate(
            start.offset_s, start.offset_s - start.error_s, start.offset_s + start.error_s
        )
    splits = [
        Split(
            policy.gates[0].id,
            policy.gates[0].position_m,
            first,
            offset,
            None if isinstance(start, FirstGate) else offset,
            None,
        )
    ]
    for previous_gate, gate, previous, current in zip(
        policy.gates[:-1], policy.gates[1:], crossing[:-1], crossing[1:], strict=True
    ):
        elapsed = current.time_s.elapsed(previous.time_s, f"gate {previous_gate.id} to {gate.id}")
        meters = gate.position_m - previous_gate.position_m
        velocity = Estimate(meters / elapsed.value, meters / elapsed.high, meters / elapsed.low)
        if velocity.value > policy.max_segment_velocity_m_s:
            raise EventRefusal(f"gate {gate.id}: implausible segment-average velocity")
        cumulative = (
            current.time_s.elapsed(origin.time_s, f"start to gate {gate.id}")
            if origin is not None
            else current.time_s.elapsed(first.time_s, f"first to gate {gate.id}").shifted(offset)
        )
        splits.append(Split(gate.id, gate.position_m, current, cumulative, elapsed, velocity))
    return Sprint(
        "beam_crossing_segment_average_velocity",
        start.mode,
        origin,
        tuple(splits),
        tuple(discarded),
        gate_limitations(policy),
    )
