"""Flight and contact phases from a contact system's switching events."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Literal

from pydantic import Field, StrictBool, model_validator

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
)
from dynamisbench_soccer.signals.schema import DeviceClass, SignalBundle

_CONTACT_TESTS = {"cmj", "sj", "slcmj", "dj", "rj_10_5"}
_REBOUND_TESTS = {"dj", "rj_10_5"}
_GRAVITY_M_S2 = 9.80665


class DurationRange(Policy):
    min_s: Positive
    max_s: Positive

    @model_validator(mode="after")
    def _range(self) -> DurationRange:
        if self.max_s <= self.min_s:
            raise ValueError("maximum phase duration must exceed its minimum")
        return self

    def check(self, estimate: Estimate, phase: str) -> None:
        if not self.min_s <= estimate.value <= self.max_s:
            raise EventRefusal(f"{phase} duration outside declared protocol limits")


class ContactPolicy(Policy):
    timing: Timing
    system: Literal["contact_mat", "pressure_pad", "optical_contact_bar"]
    initial_event: Literal["contact_snapshot", "airborne_snapshot", "touchdown"]
    switch_on: EdgeTiming
    switch_off: EdgeTiming
    debounce_s: Nonnegative
    expected_flights: Annotated[int, Field(gt=0)]
    flight: DurationRange
    ground_contact: DurationRange
    ballistic_equal_com_height: StrictBool

    @model_validator(mode="after")
    def _refractory(self) -> ContactPolicy:
        if self.debounce_s >= min(self.flight.min_s, self.ground_contact.min_s):
            raise ValueError("contact debounce must be shorter than the shortest declared phase")
        return self


@dataclass(frozen=True)
class Flight:
    takeoff: Event
    touchdown: Event
    flight_s: Estimate
    preceding_touchdown: Event | None
    ground_contact_s: Estimate | None
    flight_time_height_m: Estimate | None
    flight_time_height_refusal: str | None
    flight_height_rsi_m_s: Estimate | None
    rsi_refusal: str | None


@dataclass(frozen=True)
class ContactResult:
    estimand: Literal["contact_system_switch_flight_time"]
    system: str
    trigger_definition: str
    initial_event: Event
    flights: tuple[Flight, ...]
    discarded: tuple[Discard, ...]


def _stable_rows(
    stream: EventStream, states: tuple[bool, ...], policy: ContactPolicy
) -> tuple[list[int], list[Discard]]:
    discarded = []

    def discard(reason: str, *rows: int) -> None:
        discarded.append(Discard(reason, tuple(stream.native(row) for row in rows)))

    edges = [0]
    for i in range(1, len(states)):
        previous = edges[-1]
        if states[i] != states[previous]:
            edges.append(i)
        elif (
            policy.debounce_s > 0 and stream.times[i] - stream.times[previous] <= policy.debounce_s
        ):
            discard("refractory contact repeat", i)
        else:
            raise EventRefusal(f"repeated contact state without an intervening edge at row {i}")

    stable = [0]
    index = 1
    while index < len(edges):
        row = edges[index]
        if (
            policy.debounce_s > 0
            and index + 1 < len(edges)
            and stream.times[edges[index + 1]] - stream.times[row] <= policy.debounce_s
        ):
            discard("contact switching bounce", row, edges[index + 1])
            index += 2
        else:
            stable.append(row)
            index += 1
    return stable, discarded


def contact(bundle: SignalBundle, policy: ContactPolicy) -> ContactResult:
    """Pair measured transitions and bound each estimand using its actual endpoints."""
    stream = EventStream(bundle, DeviceClass.CONTACT_MAT, policy.timing)
    trigger = stream.source.geometry.trigger_method
    if not trigger or trigger.strip().lower() in {"", "unknown"}:
        raise EventRefusal("missing contact switching definition")
    test_id = bundle.meta.test_id
    if test_id not in _CONTACT_TESTS:
        raise EventRefusal("expected a contact jump or rebound protocol")
    if test_id != "rj_10_5" and policy.expected_flights != 1:
        raise EventRefusal("this jump protocol requires exactly one flight")
    states = tuple(map(bool, bundle["contact"]))
    if states[0] != (policy.initial_event != "airborne_snapshot"):
        raise EventRefusal("initial contact state contradicts the declared capture")
    stable, discarded = _stable_rows(stream, states, policy)
    if not states[stable[-1]]:
        raise EventRefusal("missing touchdown for the final flight")

    initial = stream.event(
        0,
        policy.switch_on
        if policy.initial_event == "touchdown"
        else EdgeTiming(latency_s=0, latency_error_s=0),
    )
    previous = initial
    preceding_touchdown = initial if policy.initial_event == "touchdown" else None
    takeoff = None
    stance = None
    flights = []
    for row in stable[1:]:
        on = states[row]
        event = stream.event(row, policy.switch_on if on else policy.switch_off)
        interval = event.time_s.elapsed(previous.time_s, f"contact transition at row {row}")
        previous = event
        if not on:
            takeoff = event
            stance = interval if preceding_touchdown is not None else None
            if stance is not None:
                policy.ground_contact.check(stance, "ground contact")
            continue
        if takeoff is not None:
            policy.flight.check(interval, "flight")
            height = None
            height_refusal: str | None = (
                "equal COM height at takeoff and touchdown must be affirmed"
            )
            if policy.ballistic_equal_com_height:
                k = _GRAVITY_M_S2 / 8
                height = Estimate(
                    k * interval.value * interval.value,
                    k * interval.low * interval.low,
                    k * interval.high * interval.high,
                )
                height_refusal = None
            rsi = None
            if test_id not in _REBOUND_TESTS:
                rsi_refusal = "height/contact RSI requires a rebound protocol"
            elif stance is None:
                rsi_refusal = "missing measured touchdown before takeoff"
            elif height is None:
                rsi_refusal = "height/contact RSI requires a valid flight-time height estimate"
            else:
                rsi = Estimate(
                    height.value / stance.value, height.low / stance.high, height.high / stance.low
                )
                rsi_refusal = None
            flights.append(
                Flight(
                    takeoff,
                    event,
                    interval,
                    preceding_touchdown,
                    stance,
                    height,
                    height_refusal,
                    rsi,
                    rsi_refusal,
                )
            )
            takeoff = None
        preceding_touchdown = event

    if len(flights) != policy.expected_flights:
        raise EventRefusal(
            f"expected {policy.expected_flights} complete flights, found {len(flights)}"
        )
    return ContactResult(
        "contact_system_switch_flight_time",
        policy.system,
        trigger,
        initial,
        tuple(flights),
        tuple(discarded),
    )
