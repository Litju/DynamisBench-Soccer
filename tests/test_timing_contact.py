"""Independent contact observations, censored phases, and adverse switching events."""

import json

import numpy as np
import pytest

from dynamisbench_soccer.cli import main
from dynamisbench_soccer.engines.contact import ContactPolicy, DurationRange, contact
from dynamisbench_soccer.engines.event_time import ClockTiming, EdgeTiming, EventRefusal, Timing
from dynamisbench_soccer.reference.clocks import DeviceClock
from dynamisbench_soccer.reference.exporters import export_case
from dynamisbench_soccer.reference.faults import EventFault, EventKind, inject
from dynamisbench_soccer.reference.kinematics import cmj, dj, hops, sj
from dynamisbench_soccer.reference.spatial_observations import contact_events
from dynamisbench_soccer.signals import io
from dynamisbench_soccer.signals.schema import BundleMeta, SignalBundle

EDGE = EdgeTiming(latency_s=0, latency_error_s=0)
CLOCK = ClockTiming(resolution_s=0.001, sync_latency_s=0, sync_latency_error_s=0)


def policy(**updates) -> ContactPolicy:
    return ContactPolicy.model_validate(
        {
            "timing": Timing(clock_model="wired", clocks={"device": CLOCK}, authority="RES-669"),
            "system": "contact_mat",
            "initial_event": "contact_snapshot",
            "switch_on": EDGE,
            "switch_off": EDGE,
            "debounce_s": 0.0,
            "expected_flights": 1,
            "flight": DurationRange(min_s=0.15, max_s=0.9),
            "ground_contact": DurationRange(min_s=0.04, max_s=0.5),
            "ballistic_equal_com_height": True,
            **updates,
        }
    )


def stream(source, rows, *, shared=False):
    data = dict(zip(("t_s", "contact"), np.array(rows, dtype=np.float64).T, strict=True))
    meta = source.meta.model_dump()
    meta["n_rows"] = len(rows)
    if shared:
        acquisition = meta["acquisition"]
        acquisition["sources"][0]["clock"] = acquisition["reference_clock"]
        acquisition["clocks"] = acquisition["clocks"][:1]
        acquisition["alignments"] = []
    return SignalBundle(BundleMeta.model_validate(meta), data)


@pytest.mark.parametrize("motion", [cmj(), sj()])
def test_standing_jumps_report_flight_height_but_no_contact_phase_or_rsi(motion):
    case = export_case(
        "timing_contact.contact", motion, clock=DeviceClock(0.45, -800), latency_s=0.02
    )
    latency = EdgeTiming(latency_s=0.02, latency_error_s=0.002)
    result = contact(case.observation, policy(switch_on=latency, switch_off=latency))
    [jump] = result.flights
    takeoff, landing = motion.flight_intervals()[0]
    assert jump.takeoff.time_s.value == pytest.approx(takeoff)
    assert jump.touchdown.time_s.value == pytest.approx(landing)
    assert jump.flight_s.value == pytest.approx(landing - takeoff)
    speed = motion.state(takeoff).velocity_m_s
    height = speed * speed / (2 * motion.gravity_m_s2)
    assert jump.flight_time_height_m.value == pytest.approx(height)
    assert jump.flight_time_height_m.low < height < jump.flight_time_height_m.high
    assert jump.ground_contact_s is None
    assert jump.flight_height_rsi_m_s is None
    assert "rebound protocol" in jump.rsi_refusal


@pytest.mark.parametrize("system", ["contact_mat", "pressure_pad", "optical_contact_bar"])
def test_asymmetric_switch_latency_and_quantized_shared_clock(system):
    motion = dj()
    case = export_case("timing_contact.contact", motion)
    events = contact_events(motion, on_latency_s=0.026, off_latency_s=0.004)
    rows = [
        (round((0.026 if i == 0 else native) * 1000) / 1000, on)
        for i, (native, on) in enumerate(events)
    ]
    source = stream(case.observation, rows, shared=True)
    chosen = policy(
        timing=Timing(
            clock_model="shared",
            clocks={"reference": CLOCK},
            authority="shared counter and switching calibration",
        ),
        system=system,
        initial_event="touchdown",
        switch_on=EdgeTiming(latency_s=0.026, latency_error_s=0.001),
        switch_off=EdgeTiming(latency_s=0.004, latency_error_s=0.0005),
    )
    result = contact(source, chosen)
    [jump] = result.flights
    takeoff, landing = motion.flight_intervals()[0]
    speed = motion.state(takeoff).velocity_m_s
    height = speed * speed / (2 * motion.gravity_m_s2)
    rsi = height / takeoff
    assert result.system == system
    assert jump.flight_s.value == pytest.approx(landing - takeoff, abs=0.001)
    assert jump.ground_contact_s.value == pytest.approx(takeoff, abs=0.001)
    assert jump.flight_height_rsi_m_s.low < rsi < jump.flight_height_rsi_m_s.high
    assert jump.flight_height_rsi_m_s.value == pytest.approx(rsi, abs=0.005)
    [biased] = contact(
        source, chosen.model_copy(update={"switch_on": EDGE, "switch_off": EDGE})
    ).flights
    assert biased.flight_s.value > (landing - takeoff) + 0.02
    assert biased.ground_contact_s.high < takeoff
    assert biased.flight_height_rsi_m_s.low > rsi


def test_repeated_hops_keep_the_unobserved_initial_stance_and_posture_out_of_rsi():
    motion = hops(3)
    source = export_case("timing_contact.contact", motion).observation
    result = contact(source, policy(expected_flights=3, ballistic_equal_com_height=False))
    for jump, (takeoff, landing) in zip(result.flights, motion.flight_intervals(), strict=True):
        assert jump.flight_s.value == pytest.approx(landing - takeoff)
        assert jump.flight_time_height_m is None
        assert "equal COM height" in jump.flight_time_height_refusal
        assert jump.flight_height_rsi_m_s is None
    assert result.flights[0].ground_contact_s is None
    assert "missing measured touchdown" in result.flights[0].rsi_refusal
    assert result.flights[1].ground_contact_s.value == pytest.approx(0.20)
    assert "flight-time height" in result.flights[1].rsi_refusal


def test_airborne_capture_excludes_initial_fall_from_rebound_flight():
    motion = dj()
    case = export_case("timing_contact.contact", motion)
    source = stream(
        case.observation,
        [(0, 0), *[(t + 0.24, on) for t, on in contact_events(motion)]],
        shared=True,
    )
    result = contact(
        source,
        policy(
            initial_event="airborne_snapshot",
            timing=Timing(
                clock_model="shared", clocks={"reference": CLOCK}, authority="shared pulse"
            ),
        ),
    )
    [jump] = result.flights
    assert jump.preceding_touchdown.time_s.value == pytest.approx(0.24)
    assert jump.takeoff.time_s.value == pytest.approx(0.24 + motion.takeoff_s())
    assert jump.ground_contact_s.value == pytest.approx(motion.takeoff_s())


@pytest.mark.parametrize(
    ("kind", "index", "message"),
    [
        (EventKind.MISSING, 2, "missing touchdown"),
        (EventKind.MISSING, 1, "repeated contact state"),
        (EventKind.DUPLICATE, 1, "repeated contact state"),
        (EventKind.OUT_OF_ORDER, 1, "out-of-order native"),
    ],
)
def test_refuses_independent_contact_event_faults(kind, index, message):
    observed = export_case("timing_contact.contact").observation
    with pytest.raises(EventRefusal, match=message):
        contact(inject(observed, EventFault(kind, index=index)), policy())


def test_declared_contact_bounce_and_same_state_repeat_are_auditable():
    observed = export_case("timing_contact.contact").observation
    rows = list(zip(observed["t_s"], observed["contact"], strict=True))
    takeoff = rows[1][0]
    dirty = stream(
        observed,
        [rows[0], rows[1], (takeoff, 0), (takeoff + 0.10, 1), (takeoff + 0.104, 0), rows[2]],
    )
    cleaned = contact(dirty, policy(debounce_s=0.005))
    assert [x.reason for x in cleaned.discarded] == [
        "refractory contact repeat",
        "contact switching bounce",
    ]
    assert [(x.native[0].row, x.native[-1].row) for x in cleaned.discarded] == [(2, 2), (3, 4)]
    assert cleaned.flights[0].flight_s == contact(observed, policy()).flights[0].flight_s
    with pytest.raises(EventRefusal, match="repeated contact state"):
        contact(dirty, policy())
    with pytest.raises(ValueError, match="shorter than the shortest"):
        policy(debounce_s=0.05)


def test_plausible_but_extra_flight_and_wrong_device_are_refused():
    observed = export_case("timing_contact.contact").observation
    rows = list(zip(observed["t_s"], observed["contact"], strict=True))
    extra = stream(observed, [rows[0], (0.5, 0), (0.9, 1), *rows[1:]])
    with pytest.raises(EventRefusal, match="expected 1 complete flights, found 2"):
        contact(extra, policy(ground_contact=DurationRange(min_s=0.04, max_s=1.0)))
    with pytest.raises(EventRefusal, match="source-native contact_mat"):
        contact(export_case("force_platform.single").observation, policy())
    with pytest.raises(EventRefusal, match="source-native contact_mat"):
        contact(export_case("imu.inertial").observation, policy())
    with pytest.raises(EventRefusal, match="out of order or unresolved"):
        contact(observed, policy(switch_on=EdgeTiming(latency_s=0.7, latency_error_s=0)))


def test_contact_cli_reports_sensitivity_and_refuses_tampered_parquet(tmp_path, capsys):
    base = tmp_path / "base.json"
    base.write_text(policy().model_dump_json())
    variant = tmp_path / "variant.json"
    variant.write_text(
        policy(
            switch_on=EdgeTiming(latency_s=0.02, latency_error_s=0),
            ballistic_equal_com_height=False,
        ).model_dump_json()
    )
    path = tmp_path / "raw"
    parquet, _ = io.write(export_case("timing_contact.contact").observation, path)
    args = ["timing", "contact", str(path), "--policy", str(base), "--sensitivity", str(variant)]
    assert main(args) == 0
    data = json.loads(capsys.readouterr().out)
    nominal = data["metrics"]["flights"][0]
    alternative = data["sensitivity"][0]["metrics"]["flights"][0]
    assert alternative["flight_s"]["value"] == pytest.approx(nominal["flight_s"]["value"] - 0.02)
    assert alternative["flight_time_height_m"] is None
    parquet.write_bytes(parquet.read_bytes() + b"invalid")
    assert main(args) == 1
    assert "REFUSED:" in capsys.readouterr().err
