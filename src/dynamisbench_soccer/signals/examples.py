"""Small deterministic example bundles, one per channel type (RES-665 fixtures).

These are **shape fixtures** for format, adapter and QC tests. They are labelled
``synthetic`` but are *not* physically consistent: for example, the CMJ impulse does
not match its flight time. They must never be used to verify metric calculations.
Physically exact analytical generators arrive in M3 (RES-669).
"""

from __future__ import annotations

import numpy as np

from dynamisbench_soccer.data.manifest import DataCategory
from dynamisbench_soccer.signals.schema import (
    BundleKind,
    BundleMeta,
    ChannelSpec,
    Device,
    DeviceClass,
    FloatArray,
    Geometry,
    Protocol,
    Provenance,
    Quantity,
    Sampling,
    Session,
    Side,
    SignalBundle,
    Subject,
    VendorValue,
    Zeroing,
)

G = 9.80665
_PROV = Provenance(created_by="examples@0.1", transforms=["synthetic shape fixture"])
_SUBJECT = Subject(pseudonym="ATH-001", body_mass_kg=75.0, sex="male")


def _session(trial: int = 1, md: str | None = "MD-3") -> Session:
    return Session(session_id="S-2026-11-14", date="2026-11-14", match_day=md, trial_index=trial)


def _halfcos(n: int, start: float, end: float) -> FloatArray:
    x = np.linspace(0.0, np.pi, n, dtype=np.float64)
    return start + (end - start) * (1 - np.cos(x)) / 2


def cmj_total_force(fs: float = 1000.0, mass: float = 75.0) -> FloatArray:
    """Piecewise CMJ-like vertical force (N): quiet, unweight, push, flight, land, quiet."""
    bw = mass * G

    def n(sec: float) -> int:
        return round(sec * fs)

    parts = [
        np.full(n(1.5), bw),
        _halfcos(n(0.30), bw, 0.40 * bw),
        _halfcos(n(0.20), 0.40 * bw, 2.20 * bw),
        _halfcos(n(0.20), 2.20 * bw, 0.0),
        np.zeros(n(0.50)),
        _halfcos(n(0.05), 0.0, 3.50 * bw),
        _halfcos(n(0.25), 3.50 * bw, bw),
        np.full(n(0.50), bw),
    ]
    return np.concatenate(parts).astype(np.float64)


def sj_total_force(fs: float = 1000.0, mass: float = 75.0) -> FloatArray:
    """Piecewise squat-jump-like force (N): static squat, push (no dip), flight, land."""
    bw = mass * G

    def n(sec: float) -> int:
        return round(sec * fs)

    parts = [
        np.full(n(1.5), bw),
        _halfcos(n(0.25), bw, 2.10 * bw),
        _halfcos(n(0.15), 2.10 * bw, 0.0),
        np.zeros(n(0.45)),
        _halfcos(n(0.05), 0.0, 3.20 * bw),
        _halfcos(n(0.25), 3.20 * bw, bw),
        np.full(n(0.50), bw),
    ]
    return np.concatenate(parts).astype(np.float64)


def cmj_dual_plate(fs: float = 1000.0, left_share: float = 0.52) -> SignalBundle:
    total = cmj_total_force(fs)
    left = total * left_share
    right = total - left
    meta = BundleMeta(
        bundle_id="example-cmj-dual",
        kind=BundleKind.TIMESERIES,
        test_id="cmj",
        data_category=DataCategory.SYNTHETIC,
        subject=_SUBJECT,
        session=_session(),
        device=Device(device_class=DeviceClass.FORCE_PLATE, model="generic-dual-plate"),
        protocol=Protocol(protocol_id="cmj.hands_on_hips.v1", arm_swing="hands_on_hips"),
        sampling=Sampling(nominal_hz=fs),
        channels=[
            ChannelSpec(
                name="fz_left", quantity=Quantity.FORCE, unit="N", side=Side.LEFT, plate=1, axis="z"
            ),
            ChannelSpec(
                name="fz_right",
                quantity=Quantity.FORCE,
                unit="N",
                side=Side.RIGHT,
                plate=2,
                axis="z",
            ),
        ],
        n_rows=len(total),
        zeroing=Zeroing(performed=True, method="unloaded plate before trial"),
        geometry=Geometry(
            plate_layout="dual_side_by_side", summed_force_policy="sum_fz_all_plates"
        ),
        provenance=_PROV,
        vendor_reported={
            "jump_height": VendorValue(value=0.31, unit="m", definition="unknown (vendor)")
        },
    )
    return SignalBundle(meta, {"fz_left": left, "fz_right": right})


def nhe(fs: float = 100.0) -> SignalBundle:
    n = round(6 * fs)
    t = np.arange(n) / fs
    ramp = np.clip(t / 4.0, 0, 1)
    shape = np.where(t < 4.0, ramp, np.maximum(0.0, 1 - (t - 4.0) / 0.6))
    left = (60 + 320 * shape).astype(np.float64)
    right = (60 + 300 * shape).astype(np.float64)
    meta = BundleMeta(
        bundle_id="example-nhe",
        kind=BundleKind.TIMESERIES,
        test_id="nhe",
        data_category=DataCategory.SYNTHETIC,
        subject=_SUBJECT,
        session=_session(md="MD+2"),
        device=Device(device_class=DeviceClass.NORDIC_DYNAMOMETER),
        protocol=Protocol(protocol_id="nhe.bilateral.v1"),
        sampling=Sampling(nominal_hz=fs),
        channels=[
            ChannelSpec(name="force_left", quantity=Quantity.FORCE, unit="N", side=Side.LEFT),
            ChannelSpec(name="force_right", quantity=Quantity.FORCE, unit="N", side=Side.RIGHT),
        ],
        n_rows=n,
        geometry=Geometry(lever_arm_m=0.42, knee_angle_deg=90.0),
        provenance=_PROV,
    )
    return SignalBundle(meta, {"force_left": left, "force_right": right})


def hip_squeeze(fs: float = 100.0, action: str = "adduction") -> SignalBundle:
    n = round(5 * fs)
    t = np.arange(n) / fs
    effort = np.where((t > 0.8) & (t < 4.0), 1.0, 0.0)
    smooth = np.convolve(effort, np.ones(20) / 20, mode="same")
    rng = np.random.default_rng(7)  # sensor noise so the plateau is not flat (cf. QC-S03)
    left = (5 + 290 * smooth + rng.normal(0, 0.5, n)).astype(np.float64)
    right = (5 + 270 * smooth + rng.normal(0, 0.5, n)).astype(np.float64)
    meta = BundleMeta(
        bundle_id=f"example-hip-{action}",
        kind=BundleKind.TIMESERIES,
        test_id="hip_add_abd",
        data_category=DataCategory.SYNTHETIC,
        subject=_SUBJECT,
        session=_session(),
        device=Device(device_class=DeviceClass.HIP_DYNAMOMETER),
        protocol=Protocol(protocol_id="hip.squeeze.supine.v1", action=action),
        sampling=Sampling(nominal_hz=fs),
        channels=[
            ChannelSpec(name="force_left", quantity=Quantity.FORCE, unit="N", side=Side.LEFT),
            ChannelSpec(name="force_right", quantity=Quantity.FORCE, unit="N", side=Side.RIGHT),
        ],
        n_rows=n,
        geometry=Geometry(hip_flexion_deg=45.0),
        provenance=_PROV,
    )
    return SignalBundle(meta, {"force_left": left, "force_right": right})


def sprint_gates() -> SignalBundle:
    distances = np.array([0.0, 5.0, 10.0, 20.0, 30.0])
    times = np.array([0.0, 1.05, 1.80, 3.05, 4.22])
    meta = BundleMeta(
        bundle_id="example-sprint-gates",
        kind=BundleKind.EVENTS,
        test_id="sprint_gates",
        data_category=DataCategory.SYNTHETIC,
        subject=_SUBJECT,
        session=_session(md="MD-4"),
        device=Device(device_class=DeviceClass.TIMING_GATES),
        protocol=Protocol(protocol_id="sprint.30m.v1", start_mode="standing"),
        channels=[
            ChannelSpec(name="distance_m", quantity=Quantity.DISTANCE, unit="m"),
            ChannelSpec(name="t_s", quantity=Quantity.TIME, unit="s"),
        ],
        n_rows=len(distances),
        geometry=Geometry(gate_height_m=1.0),
        provenance=_PROV,
    )
    return SignalBundle(meta, {"distance_m": distances, "t_s": times})


def sprint_radar(fs: float = 47.0, vmax: float = 9.2, tau: float = 1.25) -> SignalBundle:
    n = round(5 * fs)
    t = np.arange(n) / fs
    v = (vmax * (1 - np.exp(-t / tau))).astype(np.float64)
    meta = BundleMeta(
        bundle_id="example-sprint-radar",
        kind=BundleKind.TIMESERIES,
        test_id="sprint_fv",
        data_category=DataCategory.SYNTHETIC,
        subject=_SUBJECT,
        session=_session(md="MD-4"),
        device=Device(device_class=DeviceClass.RADAR_LASER),
        protocol=Protocol(protocol_id="sprint.radar.30m.v1", start_mode="crouched"),
        sampling=Sampling(nominal_hz=fs),
        channels=[ChannelSpec(name="velocity", quantity=Quantity.VELOCITY, unit="m/s")],
        n_rows=n,
        provenance=_PROV,
    )
    return SignalBundle(meta, {"velocity": v})


def lpt(fs: float = 500.0) -> SignalBundle:
    n = round(2 * fs)
    t = np.arange(n) / fs
    disp = (0.35 * np.sin(np.pi * np.clip(t / 2.0, 0, 1))).astype(np.float64)
    meta = BundleMeta(
        bundle_id="example-lpt",
        kind=BundleKind.TIMESERIES,
        test_id="cmj",
        data_category=DataCategory.SYNTHETIC,
        subject=_SUBJECT,
        session=_session(),
        device=Device(device_class=DeviceClass.LPT),
        protocol=Protocol(protocol_id="cmj.lpt.v1", arm_swing="hands_on_hips"),
        sampling=Sampling(nominal_hz=fs),
        channels=[ChannelSpec(name="displacement", quantity=Quantity.DISPLACEMENT, unit="m")],
        n_rows=n,
        provenance=_PROV,
    )
    return SignalBundle(meta, {"displacement": disp})


def gnss_summary() -> SignalBundle:
    meta = BundleMeta(
        bundle_id="example-gnss-summary",
        kind=BundleKind.SUMMARY,
        test_id="cmj",  # the session the context belongs to
        data_category=DataCategory.SYNTHETIC,
        subject=_SUBJECT,
        session=_session(md="MD"),
        device=Device(device_class=DeviceClass.GNSS_IMU),
        protocol=Protocol(protocol_id="match.gnss.summary.v1"),
        channels=[
            ChannelSpec(name="duration", quantity=Quantity.TIME, unit="s"),
            ChannelSpec(name="total_distance", quantity=Quantity.DISTANCE, unit="m"),
            ChannelSpec(
                name="hsr_distance",
                quantity=Quantity.DISTANCE,
                unit="m",
                description="distance above a declared speed threshold",
            ),
            ChannelSpec(name="sprint_count", quantity=Quantity.COUNT, unit="1"),
        ],
        n_rows=1,
        provenance=_PROV,
    )
    data = {
        "duration": np.array([5400.0]),
        "total_distance": np.array([10850.0]),
        "hsr_distance": np.array([720.0]),
        "sprint_count": np.array([24.0]),
    }
    return SignalBundle(meta, data)


ALL = {
    "cmj_dual_plate": cmj_dual_plate,
    "nhe": nhe,
    "hip_squeeze": hip_squeeze,
    "sprint_gates": sprint_gates,
    "sprint_radar": sprint_radar,
    "lpt": lpt,
    "gnss_summary": gnss_summary,
}
