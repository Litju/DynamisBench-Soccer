"""Reference evaluators for ``qc/rules.toml``: one function per rule id."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np

from dynamisbench_soccer.qc.spec import Outcome, Status
from dynamisbench_soccer.signals.schema import (
    BundleKind,
    FloatArray,
    Quantity,
    Side,
    SignalBundle,
)

Params = Mapping[str, float]
Evaluator = Callable[[SignalBundle, Params, SignalBundle | None], Outcome]
EVALUATORS: dict[str, Evaluator] = {}

PASS = Outcome(Status.PASS)


def _na(reason: str) -> Outcome:
    return Outcome(Status.NOT_APPLICABLE, reason)


def _fail(detail: str, **values: float) -> Outcome:
    return Outcome(Status.FAIL, detail, values)


def _ok(**values: float) -> Outcome:
    return Outcome(Status.PASS, "", values)


def rule(rule_id: str) -> Callable[[Evaluator], Evaluator]:
    def deco(fn: Evaluator) -> Evaluator:
        EVALUATORS[rule_id] = fn
        return fn

    return deco


# --- helpers ------------------------------------------------------------------------------
def _fs(b: SignalBundle) -> float | None:
    return b.meta.sampling.nominal_hz if b.meta.sampling else None


def _limb_forces(b: SignalBundle) -> dict[Side, FloatArray]:
    out: dict[Side, FloatArray] = {}
    for c in b.meta.channels:
        if (
            c.quantity is Quantity.FORCE
            and c.axis in (None, "z")
            and c.side in (Side.LEFT, Side.RIGHT)
        ):
            out[c.side] = out.get(c.side, np.zeros(b.meta.n_rows)) + b[c.name]
    return out


def _total(b: SignalBundle) -> FloatArray:
    if any(c.quantity is Quantity.FORCE and c.axis == "z" for c in b.meta.channels):
        return b.total_vertical_force()
    total = np.zeros(b.meta.n_rows, dtype=np.float64)
    for c in b.meta.channels:
        if c.quantity is Quantity.FORCE:
            total += b[c.name]
    return total


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous True runs as (start, end_exclusive)."""
    padded = np.concatenate([[False], mask.astype(bool), [False]])
    edges = np.flatnonzero(np.diff(padded.astype(np.int8)))
    return [(int(s), int(e)) for s, e in zip(edges[::2], edges[1::2], strict=True)]


def _weigh(b: SignalBundle, seconds: float) -> FloatArray | None:
    fs = _fs(b)
    if fs is None:
        return None
    n = round(seconds * fs)
    total = _total(b)
    return total[:n] if len(total) >= n else None


def _airborne(b: SignalBundle, threshold_n: float, min_s: float) -> list[tuple[int, int]]:
    fs = _fs(b)
    assert fs is not None
    min_n = max(1, round(min_s * fs))
    return [(s, e) for s, e in runs(_total(b) < threshold_n) if e - s >= min_n]


def _left_share(b: SignalBundle) -> float | None:
    limbs = _limb_forces(b)
    if set(limbs) != {Side.LEFT, Side.RIGHT}:
        return None
    total = limbs[Side.LEFT] + limbs[Side.RIGHT]
    loaded = total >= 0.5 * total.max()
    denom = float(total[loaded].sum())
    return float(limbs[Side.LEFT][loaded].sum()) / denom if denom > 0 else None


# --- signal level -------------------------------------------------------------------------
@rule("QC-S01")
def zero_offset(b: SignalBundle, p: Params, _: SignalBundle | None) -> Outcome:
    fs, weigh = _fs(b), _weigh(b, 1.0)
    if fs is None or weigh is None:
        return _na("no sampling or record shorter than 1 s")
    bw = float(np.mean(weigh))
    total = _total(b)
    trim = round(p["edge_trim_s"] * fs)
    segments = [
        (s, e) for s, e in runs(total < p["airborne_fraction_bw"] * bw) if e - s > 2 * trim + 2
    ]
    if not segments:
        return _na("no unloaded segment to measure offset")
    offsets = [float(np.median(total[s + trim : e - trim])) for s, e in segments]
    worst = max(offsets, key=abs)
    if abs(worst) > p["max_offset_n"]:
        return _fail(
            f"unloaded force {worst:.1f} N exceeds +/-{p['max_offset_n']:g} N", offset_n=worst
        )
    return _ok(offset_n=worst)


@rule("QC-S02")
def baseline_drift(b: SignalBundle, p: Params, _: SignalBundle | None) -> Outcome:
    fs = _fs(b)
    if fs is None:
        return _na("no sampling")
    n = round(p["window_s"] * fs)
    total = _total(b)
    if len(total) < 3 * n:
        return _na("record too short")
    first, last = float(np.mean(total[:n])), float(np.mean(total[-n:]))
    if last < 0.5 * first:
        return _na("record does not end in quiet stance")
    rel = abs(last - first) / first
    if rel > p["max_relative_difference"]:
        return _fail(f"start/end system weight differ by {rel:.1%}", relative_difference=rel)
    return _ok(relative_difference=rel)


@rule("QC-S03")
def clipping(b: SignalBundle, p: Params, _: SignalBundle | None) -> Outcome:
    if b.meta.kind is not BundleKind.TIMESERIES:
        return _na("not a timeseries")
    min_run = int(p["min_run"])
    for c in b.meta.channels:
        if c.quantity in (Quantity.TIME,):
            continue
        values = b[c.name]
        top = values.max()
        if top <= 0:
            continue
        longest = max((e - s for s, e in runs(values == top)), default=0)
        if longest >= min_run:
            return _fail(
                f"{c.name}: {longest} consecutive samples at maximum {top:.2f}",
                run_length=float(longest),
            )
    return PASS


@rule("QC-S04")
def timestamp_integrity(b: SignalBundle, p: Params, _: SignalBundle | None) -> Outcome:
    fs = _fs(b)
    if "time" not in b.data or fs is None:
        return _na("no device timestamps")
    dt = np.diff(b["time"])
    if np.any(dt <= 0):
        return _fail("duplicated or out-of-order timestamps", n_bad=float(np.sum(dt <= 0)))
    gaps = dt > p["max_gap_factor"] / fs
    if np.any(gaps):
        return _fail(
            f"{int(gaps.sum())} gap(s) in timestamps; max dt {dt.max() * 1000:.2f} ms",
            n_gaps=float(gaps.sum()),
        )
    return PASS


@rule("QC-S05")
def sampling_rate_adequacy(b: SignalBundle, p: Params, _: SignalBundle | None) -> Outcome:
    s = b.meta.sampling
    if s is None:
        return _na("no sampling")
    minimum = p["rfd_min_hz"] if b.meta.test_id == "imtp" else p["min_hz"]
    if s.nominal_hz < minimum:
        return _fail(
            f"{s.nominal_hz:g} Hz below minimum {minimum:g} Hz for {b.meta.test_id}",
            nominal_hz=s.nominal_hz,
        )
    if s.measured_hz is not None:
        mismatch = abs(s.measured_hz - s.nominal_hz) / s.nominal_hz
        if mismatch > p["max_rate_mismatch"]:
            return _fail(
                f"measured {s.measured_hz:.2f} Hz vs declared {s.nominal_hz:g} Hz",
                mismatch=mismatch,
            )
    return _ok(nominal_hz=s.nominal_hz)


@rule("QC-S06")
def mains_interference(b: SignalBundle, p: Params, _: SignalBundle | None) -> Outcome:
    fs, window = _fs(b), _weigh(b, p["window_s"])
    if fs is None or window is None or fs < 150:
        return _na("needs >= 150 Hz and a full quiet window")
    x = window - window.mean()
    ac_power = float(np.sum(x**2))
    if ac_power <= 1e-12:
        return _ok(fraction=0.0)
    t = np.arange(len(x)) / fs
    best_fraction, best_amp, best_f = 0.0, 0.0, 0.0
    for f in (50.0, 60.0):
        coef = 2.0 / len(x) * np.abs(np.sum(x * np.exp(-2j * np.pi * f * t)))
        fraction = (coef**2 / 2.0) * len(x) / ac_power
        if fraction > best_fraction:
            best_fraction, best_amp, best_f = float(fraction), float(coef), f
    if best_fraction > p["max_fraction"] and best_amp > p["min_amplitude_n"]:
        return _fail(
            f"{best_f:g} Hz component {best_amp:.1f} N ({best_fraction:.0%} of AC power)",
            fraction=best_fraction,
            amplitude_n=best_amp,
        )
    return _ok(fraction=best_fraction)


@rule("QC-S07")
def dead_or_unmapped_channel(b: SignalBundle, p: Params, _: SignalBundle | None) -> Outcome:
    share = _left_share(b)
    if share is None:
        return _na("needs left and right force channels")
    low = min(share, 1 - share)
    if low < p["min_share"]:
        return _fail(f"one limb channel carries {low:.1%} of load", min_share=low)
    return _ok(left_share=share)


@rule("QC-S08")
def left_right_swap_vs_reference(b: SignalBundle, p: Params, ref: SignalBundle | None) -> Outcome:
    if ref is None:
        return _na("no reference session supplied")
    share, ref_share = _left_share(b), _left_share(ref)
    if share is None or ref_share is None:
        return _na("needs left and right force channels in both bundles")
    change = abs(share - ref_share)
    mirrored = abs(share - (1 - ref_share)) < change
    if change > p["min_share_change"] and mirrored:
        return _fail(
            f"left share {share:.1%} mirrors reference {ref_share:.1%}: likely L/R swap",
            left_share=share,
            reference_left_share=ref_share,
        )
    return _ok(left_share=share, reference_left_share=ref_share)


# --- trial level: force-plate jumps ---------------------------------------------------------
@rule("QC-T01")
def quiet_stance_stability(b: SignalBundle, p: Params, _: SignalBundle | None) -> Outcome:
    window = _weigh(b, p["weigh_s"])
    if window is None:
        return _fail("record shorter than the weighing window")
    sd = float(np.std(window))
    if sd > p["max_sd_n"]:
        return _fail(f"weighing-window SD {sd:.1f} N > {p['max_sd_n']:g} N", sd_n=sd)
    return _ok(sd_n=sd)


@rule("QC-T02")
def insufficient_weighing_before_onset(
    b: SignalBundle, p: Params, _: SignalBundle | None
) -> Outcome:
    fs = _fs(b)
    half = _weigh(b, p["weigh_s"] / 2)
    if fs is None or half is None:
        return _fail("record shorter than the weighing window")
    bw, sd = float(np.mean(half)), float(np.std(half))
    threshold = max(p["onset_sd_multiple"] * sd, p["min_onset_threshold_n"])
    moving = np.flatnonzero(np.abs(_total(b) - bw) > threshold)
    if len(moving) == 0:
        return _na("no movement detected")
    onset_s = moving[0] / fs
    if onset_s < p["weigh_s"]:
        return _fail(
            f"movement at {onset_s:.2f} s, before {p['weigh_s']:g} s of quiet stance",
            onset_s=onset_s,
        )
    return _ok(onset_s=onset_s)


@rule("QC-T03")
def no_flight_phase(b: SignalBundle, p: Params, _: SignalBundle | None) -> Outcome:
    if _fs(b) is None:
        return _na("no sampling")
    if not _airborne(b, p["flight_threshold_n"], p["min_flight_s"]):
        return _fail("no airborne phase found")
    return PASS


@rule("QC-T04")
def multiple_flight_phases(b: SignalBundle, p: Params, _: SignalBundle | None) -> Outcome:
    if _fs(b) is None:
        return _na("no sampling")
    n = len(_airborne(b, p["flight_threshold_n"], p["min_segment_s"]))
    if n > 1:
        return _fail(f"{n} airborne phases (double contact or step-off)", n_flights=float(n))
    return _ok(n_flights=float(n))


@rule("QC-T05")
def arm_swing_protocol(b: SignalBundle, p: Params, _: SignalBundle | None) -> Outcome:
    proto = b.meta.protocol
    if proto.arm_swing == "not_applicable":
        return _fail("arm condition not declared for a jump trial")
    if "hands_on_hips" in proto.protocol_id and proto.arm_swing != "hands_on_hips":
        return _fail(f"protocol {proto.protocol_id} requires hands on hips, got {proto.arm_swing}")
    if "free_arms" in proto.protocol_id and proto.arm_swing != "free_arms":
        return _fail(f"protocol {proto.protocol_id} requires free arms, got {proto.arm_swing}")
    return PASS


@rule("QC-T06")
def incomplete_landing(b: SignalBundle, p: Params, _: SignalBundle | None) -> Outcome:
    fs = _fs(b)
    if fs is None:
        return _na("no sampling")
    flights = _airborne(b, p["flight_threshold_n"], 0.03)
    if not flights:
        return _na("no flight phase")
    end = flights[-1][1]
    if end >= b.meta.n_rows:
        return _fail("recording ends during flight")
    post = (b.meta.n_rows - end) / fs
    if post < p["min_post_landing_s"]:
        return _fail(f"only {post:.2f} s recorded after landing", post_landing_s=post)
    return _ok(post_landing_s=post)


@rule("QC-T07")
def flight_time_plausibility(b: SignalBundle, p: Params, _: SignalBundle | None) -> Outcome:
    fs = _fs(b)
    if fs is None:
        return _na("no sampling")
    flights = _airborne(b, p["flight_threshold_n"], 0.03)
    if not flights:
        return _na("no flight phase")
    longest = max(e - s for s, e in flights) / fs
    if not p["min_flight_s"] <= longest <= p["max_flight_s"]:
        return _fail(
            f"flight time {longest:.3f} s outside [{p['min_flight_s']:g}, {p['max_flight_s']:g}] s",
            flight_s=longest,
        )
    return _ok(flight_s=longest)


@rule("QC-T08")
def squat_jump_countermovement(b: SignalBundle, p: Params, _: SignalBundle | None) -> Outcome:
    fs, window = _fs(b), _weigh(b, p["weigh_s"])
    if fs is None or window is None:
        return _na("no sampling or record too short")
    flights = _airborne(b, 30.0, 0.1)
    if not flights:
        return _na("no flight phase")
    bw = float(np.mean(window))
    pre = _total(b)[len(window) : flights[0][0]]
    # ignore the final unloading to take-off: only look before peak propulsive force
    pre = pre[: int(np.argmax(pre))] if len(pre) else pre
    dip = (bw - float(pre.min())) / bw if len(pre) else 0.0
    if dip > p["max_dip_fraction_bw"]:
        return _fail(
            f"force fell {dip:.1%} below system weight before propulsion", dip_fraction=dip
        )
    return _ok(dip_fraction=dip)


# --- trial level: dynamometry -------------------------------------------------------------
@rule("QC-N01")
def nhe_no_break_point(b: SignalBundle, p: Params, _: SignalBundle | None) -> Outcome:
    n = b.meta.n_rows
    for side, values in _limb_forces(b).items() or {Side.BILATERAL: _total(b)}.items():
        if int(np.argmax(values)) >= (1 - p["end_fraction"]) * (n - 1):
            return _fail(f"{side} force still rising at end of record (no break point)")
    return PASS


@rule("QC-N02")
def nhe_lever_arm_missing(b: SignalBundle, p: Params, _: SignalBundle | None) -> Outcome:
    if b.meta.geometry.lever_arm_m is None:
        return _fail("lever arm not recorded; torque cannot be computed")
    return PASS


@rule("QC-H01")
def squeeze_pre_tension(b: SignalBundle, p: Params, _: SignalBundle | None) -> Outcome:
    window = _weigh(b, p["baseline_s"])
    if window is None:
        return _na("record too short")
    total = _total(b)
    frac = float(np.mean(window)) / float(total.max())
    if frac > p["max_fraction_of_peak"]:
        return _fail(f"baseline force is {frac:.0%} of peak", baseline_fraction=frac)
    return _ok(baseline_fraction=frac)


@rule("QC-H02")
def squeeze_hold_duration(b: SignalBundle, p: Params, _: SignalBundle | None) -> Outcome:
    fs = _fs(b)
    if fs is None:
        return _na("no sampling")
    total = _total(b)
    near = runs(total >= p["near_peak_fraction"] * total.max())
    hold = max((e - s for s, e in near), default=0) / fs
    if hold < p["min_hold_s"]:
        return _fail(f"near-peak hold {hold:.2f} s < {p['min_hold_s']:g} s", hold_s=hold)
    return _ok(hold_s=hold)


@rule("QC-H03")
def squeeze_position_missing(b: SignalBundle, p: Params, _: SignalBundle | None) -> Outcome:
    if b.meta.geometry.hip_flexion_deg is None:
        return _fail("hip flexion angle not recorded")
    return PASS


# --- trial level: sprint ------------------------------------------------------------------
def _gates(b: SignalBundle) -> tuple[FloatArray, FloatArray] | None:
    if b.meta.kind is not BundleKind.EVENTS:
        return None
    return b["distance_m"], b["t_s"]


@rule("QC-G01")
def gates_monotonic(b: SignalBundle, p: Params, _: SignalBundle | None) -> Outcome:
    g = _gates(b)
    if g is None:
        return _na("not a gate event bundle")
    if np.any(np.diff(g[0]) <= 0) or np.any(np.diff(g[1]) <= 0):
        return _fail("gate distances or times are not strictly increasing")
    return PASS


@rule("QC-G02")
def gates_split_speed_plausible(b: SignalBundle, p: Params, _: SignalBundle | None) -> Outcome:
    g = _gates(b)
    if g is None:
        return _na("not a gate event bundle")
    dd, dt = np.diff(g[0]), np.diff(g[1])
    with np.errstate(divide="ignore", invalid="ignore"):
        speeds = np.where(dt > 0, dd / dt, np.inf)
    bad = (speeds < p["min_speed_m_s"]) | (speeds > p["max_speed_m_s"])
    if np.any(bad):
        i = int(np.flatnonzero(bad)[0])
        return _fail(
            f"segment {g[0][i]:g}-{g[0][i + 1]:g} m speed {speeds[i]:.2f} m/s implausible",
            speed_m_s=float(speeds[i]),
        )
    return _ok(max_speed_m_s=float(speeds.max()))


@rule("QC-R01")
def radar_start_captured(b: SignalBundle, p: Params, _: SignalBundle | None) -> Outcome:
    fs = _fs(b)
    if fs is None or "velocity" not in b.data:
        return _na("no velocity channel")
    n = max(1, round(p["start_window_s"] * fs))
    v0 = float(np.median(b["velocity"][:n]))
    if v0 > p["max_start_speed_m_s"]:
        return _fail(f"athlete already moving at {v0:.2f} m/s at record start", start_speed=v0)
    return _ok(start_speed=v0)


@rule("QC-R02")
def radar_acceleration_spikes(b: SignalBundle, p: Params, _: SignalBundle | None) -> Outcome:
    fs = _fs(b)
    if fs is None or "velocity" not in b.data:
        return _na("no velocity channel")
    accel = float(np.max(np.abs(np.diff(b["velocity"])))) * fs
    if accel > p["max_abs_accel_m_s2"]:
        return _fail(f"sample-to-sample acceleration {accel:.1f} m/s^2", max_accel=accel)
    return _ok(max_accel=accel)
