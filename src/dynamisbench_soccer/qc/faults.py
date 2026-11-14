"""Deterministic fault injectors and the per-rule QC fixture table.

Every QC rule has at least one **negative** fixture (a clean bundle the rule must
pass) and one **positive** fixture (a faulted bundle the rule must fail). The
injectors are reusable for M5 instrument-validity tasks (RES-682).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from dynamisbench_soccer.signals import examples
from dynamisbench_soccer.signals.schema import (
    BundleMeta,
    ChannelSpec,
    FloatArray,
    Geometry,
    Quantity,
    Sampling,
    SignalBundle,
)

Builder = Callable[[], SignalBundle]


def _with(
    bundle: SignalBundle, data: dict[str, FloatArray] | None = None, **meta: object
) -> SignalBundle:
    new_data = data if data is not None else dict(bundle.data)
    n = len(next(iter(new_data.values())))
    new_meta = bundle.meta.model_copy(update={"n_rows": n, **meta})
    BundleMeta.model_validate(new_meta.model_dump())  # re-validate after edits
    return SignalBundle(new_meta, new_data)


def _split(total: FloatArray, left_share: float = 0.52) -> dict[str, FloatArray]:
    left = total * left_share
    return {"fz_left": left, "fz_right": total - left}


def cmj(fs: float = 1000.0) -> SignalBundle:
    return examples.cmj_dual_plate(fs=fs)


def cmj_from_total(total: FloatArray, fs: float = 1000.0, **meta: object) -> SignalBundle:
    base = examples.cmj_dual_plate(fs=fs)
    return _with(base, _split(total), **meta)


def sj() -> SignalBundle:
    base = examples.cmj_dual_plate()
    return _with(
        base,
        _split(examples.sj_total_force()),
        test_id="sj",
        protocol=base.meta.protocol.model_copy(update={"protocol_id": "sj.hands_on_hips.v1"}),
    )


# --- signal-level faults ------------------------------------------------------------------
def offset(b: SignalBundle, newtons_per_plate: float) -> SignalBundle:
    return _with(b, {k: v + newtons_per_plate for k, v in b.data.items()})


def drift(b: SignalBundle, total_newtons: float) -> SignalBundle:
    n = b.meta.n_rows
    ramp = np.linspace(0.0, total_newtons / len(b.data), n)
    return _with(b, {k: v + ramp for k, v in b.data.items()})


def clip(b: SignalBundle, fraction_of_max: float = 0.8) -> SignalBundle:
    return _with(b, {k: np.minimum(v, fraction_of_max * v.max()) for k, v in b.data.items()})


def with_time_gap(b: SignalBundle, drop: int = 3) -> SignalBundle:
    t = b.time()
    keep = np.ones(len(t), dtype=bool)
    keep[len(t) // 2 : len(t) // 2 + drop] = False
    data = {"time": t[keep], **{k: v[keep] for k, v in b.data.items()}}
    channels = [ChannelSpec(name="time", quantity=Quantity.TIME, unit="s"), *b.meta.channels]
    return _with(b, data, channels=channels)


def with_time(b: SignalBundle) -> SignalBundle:
    data = {"time": b.time(), **b.data}
    channels = [ChannelSpec(name="time", quantity=Quantity.TIME, unit="s"), *b.meta.channels]
    return _with(b, data, channels=channels)


def resample_label(
    b: SignalBundle, nominal_hz: float, measured_hz: float | None = None
) -> SignalBundle:
    return _with(b, sampling=Sampling(nominal_hz=nominal_hz, measured_hz=measured_hz))


def mains(b: SignalBundle, amplitude_n: float = 5.0, freq: float = 50.0) -> SignalBundle:
    hum = amplitude_n * np.sin(2 * np.pi * freq * b.time())
    return _with(b, {k: v + hum / len(b.data) for k, v in b.data.items()})


def dead_channel(b: SignalBundle, name: str) -> SignalBundle:
    data = dict(b.data)
    data[name] = np.zeros_like(data[name])
    return _with(b, data)


def swap_lr(b: SignalBundle) -> SignalBundle:
    names = [c.name for c in b.meta.channels]
    left = next(n for n in names if n.endswith("left"))
    right = next(n for n in names if n.endswith("right"))
    data = dict(b.data)
    data[left], data[right] = b.data[right], b.data[left]
    return _with(b, data)


# --- trial-level faults -------------------------------------------------------------------
def sway(b: SignalBundle, amplitude_n: float = 30.0, freq: float = 1.5) -> SignalBundle:
    t = b.time()
    wobble = np.where(t < 1.4, amplitude_n * np.sin(2 * np.pi * freq * t), 0.0)
    return _with(b, {k: v + wobble / len(b.data) for k, v in b.data.items()})


def short_weighing(fs: float = 1000.0) -> SignalBundle:
    total = examples.cmj_total_force(fs)
    return cmj_from_total(total[round(1.0 * fs) :], fs)  # only 0.5 s quiet stance remains


def no_flight() -> SignalBundle:
    total = examples.cmj_total_force()
    bw = total[0]
    return cmj_from_total(np.maximum(total, 0.3 * bw))


def double_contact(fs: float = 1000.0) -> SignalBundle:
    total = examples.cmj_total_force(fs).copy()
    land = int(np.flatnonzero(total[: round(3.0 * fs)] == 0)[-1]) + 1
    total[land + round(0.10 * fs) : land + round(0.25 * fs)] = 0.0  # second airborne phase
    return cmj_from_total(total, fs)


def arm_swing_violation(b: SignalBundle) -> SignalBundle:
    return _with(b, protocol=b.meta.protocol.model_copy(update={"arm_swing": "free_arms"}))


def truncated_after_landing(fs: float = 1000.0) -> SignalBundle:
    total = examples.cmj_total_force(fs)
    land = int(np.flatnonzero(total == 0)[-1]) + 1
    return cmj_from_total(total[: land + round(0.2 * fs)], fs)


def long_flight(fs: float = 1000.0) -> SignalBundle:
    total = examples.cmj_total_force(fs)
    i = int(np.flatnonzero(total == 0)[0])
    stretched = np.concatenate([total[:i], np.zeros(round(0.8 * fs)), total[i:]])
    return cmj_from_total(stretched, fs)


def sj_with_countermovement() -> SignalBundle:
    base = sj()
    return _with(base, _split(examples.cmj_total_force()))


def nhe_truncated() -> SignalBundle:
    b = examples.nhe()
    cut = round(3.9 * b.meta.sampling.nominal_hz) if b.meta.sampling else 0  # stop before peak
    return _with(b, {k: v[:cut] for k, v in b.data.items()})


def no_lever_arm(b: SignalBundle) -> SignalBundle:
    return _with(b, geometry=Geometry(knee_angle_deg=90.0))


def pre_tension(b: SignalBundle, newtons: float = 60.0) -> SignalBundle:
    return _with(b, {k: v + newtons for k, v in b.data.items()})


def short_hold() -> SignalBundle:
    b = examples.hip_squeeze()
    fs = b.meta.sampling.nominal_hz if b.meta.sampling else 100.0
    t = np.arange(b.meta.n_rows) / fs
    pulse = np.exp(-(((t - 2.0) / 0.25) ** 2))
    return _with(b, {"force_left": 5 + 290 * pulse, "force_right": 5 + 270 * pulse})


def no_hip_angle(b: SignalBundle) -> SignalBundle:
    return _with(b, geometry=Geometry())


def gates_swapped() -> SignalBundle:
    b = examples.sprint_gates()
    t = b["t_s"].copy()
    t[2], t[3] = t[3], t[2]
    return _with(b, {"distance_m": b["distance_m"], "t_s": t})


def gates_early_break() -> SignalBundle:
    b = examples.sprint_gates()
    t = b["t_s"].copy()
    t[2] = 1.06  # arm breaks the 10 m beam just after the 5 m gate
    return _with(b, {"distance_m": b["distance_m"], "t_s": t})


def radar_moving_start() -> SignalBundle:
    b = examples.sprint_radar()
    return _with(b, {"velocity": b["velocity"] + 2.0})


def radar_spikes() -> SignalBundle:
    b = examples.sprint_radar()
    v = b["velocity"].copy()
    v[100] += 3.0  # limb reflection
    return _with(b, {"velocity": v})


@dataclass(frozen=True)
class Fixture:
    negative: Builder
    positive: Builder
    reference: Builder | None = None  # for rules comparing against a reference session


FIXTURES: dict[str, Fixture] = {
    "QC-S01": Fixture(cmj, lambda: offset(cmj(), 25.0)),
    "QC-S02": Fixture(cmj, lambda: drift(cmj(), 60.0)),
    "QC-S03": Fixture(cmj, lambda: clip(cmj())),
    "QC-S04": Fixture(lambda: with_time(cmj()), lambda: with_time_gap(cmj())),
    "QC-S05": Fixture(cmj, lambda: examples.cmj_dual_plate(fs=200.0)),
    "QC-S06": Fixture(cmj, lambda: mains(cmj())),
    "QC-S07": Fixture(cmj, lambda: dead_channel(cmj(), "fz_right")),
    "QC-S08": Fixture(
        lambda: examples.cmj_dual_plate(left_share=0.60),
        lambda: examples.cmj_dual_plate(left_share=0.40),
        reference=lambda: examples.cmj_dual_plate(left_share=0.60),
    ),
    "QC-T01": Fixture(cmj, lambda: sway(cmj())),
    "QC-T02": Fixture(cmj, short_weighing),
    "QC-T03": Fixture(cmj, no_flight),
    "QC-T04": Fixture(cmj, double_contact),
    "QC-T05": Fixture(cmj, lambda: arm_swing_violation(cmj())),
    "QC-T06": Fixture(cmj, truncated_after_landing),
    "QC-T07": Fixture(cmj, long_flight),
    "QC-T08": Fixture(sj, sj_with_countermovement),
    "QC-N01": Fixture(examples.nhe, nhe_truncated),
    "QC-N02": Fixture(examples.nhe, lambda: no_lever_arm(examples.nhe())),
    "QC-H01": Fixture(examples.hip_squeeze, lambda: pre_tension(examples.hip_squeeze())),
    "QC-H02": Fixture(examples.hip_squeeze, short_hold),
    "QC-H03": Fixture(examples.hip_squeeze, lambda: no_hip_angle(examples.hip_squeeze())),
    "QC-G01": Fixture(examples.sprint_gates, gates_swapped),
    "QC-G02": Fixture(examples.sprint_gates, gates_early_break),
    "QC-R01": Fixture(examples.sprint_radar, radar_moving_start),
    "QC-R02": Fixture(examples.sprint_radar, radar_spikes),
}
