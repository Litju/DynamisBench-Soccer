"""Independent RES-669 gate observations and adversarial native acquisitions."""

import json

import numpy as np
import pytest

from dynamisbench_soccer.cli import main
from dynamisbench_soccer.data.manifest import DataCategory
from dynamisbench_soccer.engines.event_time import ClockTiming, EdgeTiming, EventRefusal, Timing
from dynamisbench_soccer.engines.timing_gates import (
    Beam,
    FirstGate,
    FixedOffset,
    Gate,
    GatePolicy,
    TriggerStart,
    sprint,
)
from dynamisbench_soccer.reference.clocks import DeviceClock
from dynamisbench_soccer.reference.exporters import export_case
from dynamisbench_soccer.reference.faults import EventFault, EventKind, inject
from dynamisbench_soccer.reference.motion_models import MonoExponentialSprint
from dynamisbench_soccer.reference.observations import spherical_gate_trigger_s
from dynamisbench_soccer.signals import io
from dynamisbench_soccer.signals.schema import BundleMeta, SignalBundle

EDGE = EdgeTiming(latency_s=0, latency_error_s=0)
CLOCK = ClockTiming(resolution_s=0.0001, sync_latency_s=0, sync_latency_error_s=0)
DISTANCES = (5.0, 10.0, 20.0, 30.0)


def gate_policy(edge=EDGE, **updates) -> GatePolicy:
    return GatePolicy.model_validate(
        {
            "timing": Timing(clock_model="wired", clocks={"device": CLOCK}, authority="RES-669"),
            "gates": [
                Gate(
                    id=i,
                    position_m=d,
                    beam_type="single",
                    beams=(Beam(id=1, height_m=1, timing=edge),),
                    quorum=1,
                    coincidence_s=0,
                )
                for i, d in enumerate(DISTANCES, start=1)
            ],
            "start": FirstGate(mode="first_gate"),
            "start_posture": "standing",
            "debounce_s": 0.0,
            "max_segment_velocity_m_s": 13,
            **updates,
        }
    )


def gate_stream(base, rows, *, name=None, clock="device", geometry=None):
    names = ("t_s", "distance_m", "gate_id", "beam_id", "start_kind")[: len(rows[0])]
    data = dict(zip(names, np.array(rows, dtype=np.float64).T, strict=True))
    meta = base.meta.model_dump()
    meta["n_rows"] = len(rows)
    if name is not None:
        meta["bundle_id"] = name
    for extra in names[2:]:
        meta["channels"].append({"name": extra, "quantity": "count", "unit": "1"})
    source = meta["acquisition"]["sources"][0]
    source["channels"] = list(names)
    if geometry is not None:
        source["geometry"].update(geometry)
    source["clock"] = clock
    meta["acquisition"]["clocks"][1]["id"] = clock
    meta["acquisition"]["alignments"][0]["clock"] = clock
    return SignalBundle(BundleMeta.model_validate(meta), data)


def test_splits_use_beam_times_and_segment_average_after_wheel_compatible_round_trip(tmp_path):
    case = export_case("timing_contact.gates", clock=DeviceClock(0.32, 812), latency_s=0.018)
    case.write(tmp_path)
    source = io.read(tmp_path / "observation")
    truth = io.read(tmp_path / "truth")
    result = sprint([source], gate_policy(EdgeTiming(latency_s=0.018, latency_error_s=0.001)))
    assert not result.discarded
    assert result.splits[0].cumulative_s.value == 0
    assert result.splits[0].segment_average_velocity_m_s is None
    for i, split in enumerate(result.splits[1:], 1):
        elapsed = truth["time"][i] - truth["time"][i - 1]
        assert split.cumulative_s.value == pytest.approx(truth["time"][i] - truth["time"][0])
        assert split.split_s.value == pytest.approx(elapsed)
        velocity = (DISTANCES[i] - DISTANCES[i - 1]) / elapsed
        assert split.segment_average_velocity_m_s.value == pytest.approx(velocity)
        assert (
            split.segment_average_velocity_m_s.low
            < velocity
            < split.segment_average_velocity_m_s.high
        )
        assert abs(velocity - truth["velocity"][i]) > 0.02
        assert split.crossing.native[0].native_s == source["t_s"][i]
    wrong_event = MonoExponentialSprint().gate_time_s(10) - MonoExponentialSprint().gate_time_s(5)
    assert abs(result.splits[1].split_s.value - wrong_event) > 0.001


@pytest.mark.parametrize(
    ("fault", "message"),
    [
        (EventKind.MISSING, "missing gate 2"),
        (EventKind.OUT_OF_ORDER, "out-of-order native"),
        (EventKind.DUPLICATE, "duplicate or ambiguous"),
    ],
)
def test_adversarial_reference_faults_are_refused(fault, message):
    source = export_case("timing_contact.gates").observation
    with pytest.raises(EventRefusal, match=message):
        sprint([inject(source, EventFault(fault, index=1))], gate_policy())


def test_declared_refractory_removes_duplicate_but_does_not_select_between_crossings():
    source = export_case("timing_contact.gates").observation
    duplicated = inject(source, EventFault(EventKind.DUPLICATE, index=1))
    result = sprint([duplicated], gate_policy(debounce_s=0.003))
    assert result.discarded[0].reason == "refractory beam repeat"
    assert result.discarded[0].native[0].row == 2
    assert result.splits[1].split_s == sprint([source], gate_policy()).splits[1].split_s
    delayed = gate_stream(
        source, sorted([*zip(source["t_s"], DISTANCES, strict=True), (source["t_s"][1] - 0.08, 10)])
    )
    with pytest.raises(EventRefusal, match="expected one beam quorum, found 2"):
        sprint([delayed], gate_policy(debounce_s=0.003))


def test_dual_beam_vote_rejects_early_arm_and_changes_event_definition():
    model = MonoExponentialSprint()
    reference = export_case("timing_contact.gates")
    rows = []
    for i, (t, d) in enumerate(zip(reference.truth["time"], DISTANCES, strict=True), 1):
        rows.append((t, d, i, 1))
        if i == 2:
            second = spherical_gate_trigger_s(
                model, d, beam_height_m=1.08, center_height_m=1, radius_m=0.1
            )
            arm = spherical_gate_trigger_s(
                model, d, beam_height_m=1, center_height_m=1, radius_m=0.1, forward_offset_m=0.50
            )
            rows.extend([(arm, d, i, 1), (second, d, i, 2)])
    # Only the 10 m station has two beams. Each exporter keeps its native row order.
    single = gate_stream(
        reference.observation, [r for r in rows if r[2] != 2], name="single-beam-stations"
    )
    dual = gate_stream(
        reference.observation,
        sorted(r for r in rows if r[2] == 2),
        name="dual-beam-station",
        geometry={"beam_type": "dual"},
    )
    gates = list(gate_policy().gates)
    gates[1] = Gate(
        id=2,
        position_m=10,
        beam_type="dual",
        quorum=2,
        coincidence_s=0.015,
        beams=(Beam(id=1, height_m=1, timing=EDGE), Beam(id=2, height_m=1.08, timing=EDGE)),
    )
    result = sprint([single, dual], gate_policy(gates=gates))
    assert result.splits[1].crossing.time_s.value == pytest.approx(second)
    assert result.discarded[0].reason == "incomplete beam quorum"
    assert result.discarded[0].native[0].native_s == pytest.approx(arm)
    assert abs(result.splits[1].crossing.time_s.value - arm) > 0.05
    gates[1] = gates[1].model_copy(update={"quorum": 1})
    with pytest.raises(EventRefusal, match="found 2"):
        sprint([single, dual], gate_policy(gates=gates))


def test_wireless_sync_and_latency_are_separate_and_require_authority():
    early = export_case("timing_contact.gates", clock=DeviceClock(0.45, 340), latency_s=0.01)
    late = export_case("timing_contact.gates", clock=DeviceClock(-0.37, -560), latency_s=0.07)
    a = gate_stream(
        early.observation,
        list(zip(early.observation["t_s"][:2], DISTANCES[:2], strict=True)),
        name="radio-a",
        clock="a",
    )
    b = gate_stream(
        late.observation,
        list(zip(late.observation["t_s"][2:], DISTANCES[2:], strict=True)),
        name="radio-b",
        clock="b",
    )
    policy = gate_policy(
        EdgeTiming(latency_s=0.01, latency_error_s=0),
        timing=Timing(
            clock_model="wireless",
            authority="pulse calibration",
            clocks={
                "a": CLOCK,
                "b": ClockTiming(
                    resolution_s=0.0001, sync_latency_s=0.06, sync_latency_error_s=0.001
                ),
            },
        ),
    )
    expected = early.truth["time"][2] - early.truth["time"][1]
    y = sprint([a, b], policy).splits[2].split_s
    assert y.value == pytest.approx(expected)
    assert y.low < expected < y.high
    biased = policy.timing.model_copy(update={"clocks": {"a": CLOCK, "b": CLOCK}})
    wrong = sprint([a, b], policy.model_copy(update={"timing": biased})).splits[2].split_s
    assert wrong.value == pytest.approx(expected + 0.06)
    assert wrong.low > expected
    no_b = biased.model_copy(update={"clocks": {"a": CLOCK}})
    with pytest.raises(EventRefusal, match="missing resolution"):
        sprint([a, b], policy.model_copy(update={"timing": no_b}))
    b.meta.acquisition.alignments[0].sync_events.pop()
    with pytest.raises(EventRefusal, match="bracketing"):
        sprint([a, b], policy)


@pytest.mark.parametrize(("mode", "kind"), [("pressure_pad", 1), ("sound_cue", 2)])
def test_external_start_requires_matching_native_trigger_without_first_velocity(mode, kind):
    case = export_case("timing_contact.gates")
    rows = [
        (0.04, 0, 0, 0, kind),
        *[
            (t, d, i, 1, 0)
            for i, (t, d) in enumerate(zip(case.observation["t_s"], DISTANCES, strict=True), 1)
        ],
    ]
    policy = gate_policy(
        start=TriggerStart(
            mode=mode, position_m=0, timing=EdgeTiming(latency_s=0.04, latency_error_s=0)
        )
    )
    result = sprint([gate_stream(case.observation, rows)], policy)
    assert result.splits[0].split_s.value == pytest.approx(case.truth["time"][0])
    assert result.splits[0].segment_average_velocity_m_s is None
    with pytest.raises(EventRefusal, match="found 0"):
        sprint([gate_stream(case.observation, rows[1:])], policy)
    rows[0] = (0.04, 0, 0, 0, 3 - kind)
    with pytest.raises(EventRefusal, match="mismatched native start"):
        sprint([gate_stream(case.observation, rows)], policy)


def test_impossible_route_even_with_monotonic_rows_and_plausible_total():
    case = export_case("timing_contact.gates").observation
    times = list(case["t_s"])
    order = [(times[0], 5), (times[1], 20), (times[2], 10), (times[3], 30)]
    with pytest.raises(EventRefusal, match="out of order"):
        sprint([gate_stream(case, order)], gate_policy())
    fast = [(times[0], 5), (times[0] + 0.1, 10), (times[2], 20), (times[3], 30)]
    with pytest.raises(EventRefusal, match="implausible"):
        sprint([gate_stream(case, fast)], gate_policy())
    with pytest.raises(EventRefusal, match="unresolved"):
        sprint(
            [case],
            gate_policy(
                timing=Timing(
                    clock_model="wired",
                    authority="coarse clock",
                    clocks={
                        "device": CLOCK.model_copy(update={"resolution_s": 3}),
                    },
                )
            ),
        )
    for update, message in [
        ({"acquisition": None}, "acquisition and clock authority"),
        ({"data_category": DataCategory.ANALYTICAL}, "analytical truth"),
    ]:
        with pytest.raises(EventRefusal, match=message):
            sprint([SignalBundle(case.meta.model_copy(update=update), case.data)], gate_policy())


def test_fixed_offset_sensitivity_via_cli(tmp_path, capsys):
    path = tmp_path / "events"
    io.write(export_case("timing_contact.gates").observation, path)
    baseline = tmp_path / "first.json"
    baseline.write_text(gate_policy().model_dump_json())
    alternative = tmp_path / "fixed.json"
    alternative.write_text(
        gate_policy(
            start=FixedOffset(mode="fixed_offset", offset_s=0.2, error_s=0.01),
        ).model_dump_json()
    )
    bad = tmp_path / "missing.json"
    assert (
        main(
            [
                "timing",
                "gates",
                str(path),
                "--policy",
                str(baseline),
                "--sensitivity",
                str(alternative),
                "--sensitivity",
                str(bad),
            ]
        )
        == 0
    )
    result = json.loads(capsys.readouterr().out)
    original = result["metrics"]["splits"][1]
    changed = result["sensitivity"][0]["metrics"]["splits"][1]
    assert changed["cumulative_s"]["value"] == pytest.approx(
        original["cumulative_s"]["value"] + 0.2
    )
    assert changed["segment_average_velocity_m_s"] == original["segment_average_velocity_m_s"]
    assert "refusal" in result["sensitivity"][1]
    assert main(["timing", "gates", str(path), "--policy", str(bad)]) == 1
    assert capsys.readouterr().err.startswith("REFUSED:")


def test_multi_beam_quorum_and_beam_height_sensitivity_on_clean_events():
    case = export_case("timing_contact.gates")
    model = MonoExponentialSprint()
    heights = (1.0, 1.06, 1.08)
    events = [
        (
            spherical_gate_trigger_s(
                model, 10, beam_height_m=height, center_height_m=1, radius_m=0.1
            ),
            10,
            2,
            i,
        )
        for i, height in enumerate(heights, start=1)
    ]
    gates = list(gate_policy().gates)
    gates[1] = Gate(
        id=2,
        position_m=10,
        beam_type="multiple",
        coincidence_s=0.015,
        quorum=1,
        beams=tuple(Beam(id=i, height_m=h, timing=EDGE) for i, h in enumerate(heights, 1)),
    )
    sources = [
        gate_stream(
            case.observation,
            [
                (t, d, i, 1)
                for i, (t, d) in enumerate(zip(case.observation["t_s"], DISTANCES, strict=True), 1)
                if i != 2
            ],
            name="single",
        ),
        gate_stream(
            case.observation, events, name="three-beam", geometry={"beam_type": "multiple"}
        ),
    ]
    first = sprint(sources, gate_policy(gates=gates)).splits[1]
    assert first.crossing.time_s.value == pytest.approx(events[0][0])
    gates[1] = gates[1].model_copy(update={"quorum": 3})
    all_beams = sprint(sources, gate_policy(gates=gates)).splits[1]
    assert all_beams.crossing.time_s.value == pytest.approx(events[-1][0])
    assert all_beams.split_s.value - first.split_s.value == pytest.approx(
        events[-1][0] - events[0][0]
    )
    uncertain = (
        gates[1]
        .beams[-1]
        .model_copy(
            update={"timing": EDGE.model_copy(update={"definition_error_s": 0.002})},
        )
    )
    gates[1] = gates[1].model_copy(update={"beams": (*gates[1].beams[:-1], uncertain)})
    sensitivity = sprint(sources, gate_policy(gates=gates)).splits[1].split_s
    assert sensitivity.value == all_beams.split_s.value
    assert sensitivity.high == pytest.approx(all_beams.split_s.high + 0.002)


@pytest.mark.parametrize("sync_error_s", [0, 0.0015])
def test_wireless_split_uncertainty_matches_independent_endpoint_budget(sync_error_s):
    case = export_case("timing_contact.gates", clock=DeviceClock(0.32, 800), latency_s=0.018)
    latency = EdgeTiming(latency_s=0.008, latency_error_s=0.0007, definition_error_s=0.0002)
    timing = Timing(
        clock_model="wireless",
        authority="synthetic pulse and beam calibration",
        clocks={
            "device": ClockTiming(
                resolution_s=0.0005, sync_latency_s=0.01, sync_latency_error_s=sync_error_s
            )
        },
    )
    result = sprint([case.observation], gate_policy(latency, timing=timing))
    # An 800 ppm affine map scales one half-count to 0.2502 ms.
    # Endpoint errors can have opposite signs, so the two radii add.
    radius = 2 * (0.0002502 + 1e-12 + sync_error_s + 0.0007 + 0.0002)
    assert not result.discarded
    for i, split in enumerate(result.splits[1:], 1):
        elapsed = case.truth["time"][i] - case.truth["time"][i - 1]
        distance = DISTANCES[i] - DISTANCES[i - 1]
        assert (split.split_s.value, split.split_s.low, split.split_s.high) == pytest.approx(
            (elapsed, elapsed - radius, elapsed + radius), rel=0, abs=1e-12
        )
        expected_speed = (
            distance / elapsed,
            distance / (elapsed + radius),
            distance / (elapsed - radius),
        )
        speed = split.segment_average_velocity_m_s
        assert (speed.value, speed.low, speed.high) == pytest.approx(
            expected_speed, rel=0, abs=1e-11
        )


def test_gate_timing_uncertainty_refuses_unresolved_beam_order():
    case = export_case("timing_contact.gates")
    assert len(sprint([case.observation], gate_policy()).splits) == 4
    with pytest.raises(EventRefusal, match="out of order or unresolved"):
        sprint([case.observation], gate_policy(EdgeTiming(latency_s=0, latency_error_s=0.5)))
