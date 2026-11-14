"""v0.1 generic CSV adapters (owner default Q2: no vendor-shaped adapters yet)."""

from __future__ import annotations

import re

from dynamisbench_soccer.adapters.base import Adapter, AdapterError, Capabilities, register
from dynamisbench_soccer.adapters.csvformat import Column, ParsedCsv
from dynamisbench_soccer.signals.schema import (
    BundleKind,
    ChannelSpec,
    DeviceClass,
    FloatArray,
    Geometry,
    Quantity,
    Side,
    SignalBundle,
)

_FORCE_COL = re.compile(r"^(f[xyz])(?:_(left|right)|_plate(\d))?$")
_COP_COL = re.compile(r"^cop_([xy])(?:_(left|right)|_plate(\d))?$")


def _expect(col: Column, quantity: Quantity) -> None:
    if col.quantity is not quantity:
        raise AdapterError(
            f"unit mislabel: column {col.name} has unit {col.unit!r} ({col.quantity}), "
            f"expected a {quantity} unit"
        )


def _time_channel(parsed: ParsedCsv) -> tuple[list[ChannelSpec], dict[str, FloatArray]]:
    col = parsed.column("time")
    if col is None:
        return [], {}
    _expect(col, Quantity.TIME)
    return [
        ChannelSpec(name="time", quantity=Quantity.TIME, unit="s", description="device timestamps")
    ], {"time": col.values}


class GenericForcePlateCsv(Adapter):
    capabilities = Capabilities(
        name="generic_forceplate_csv",
        version="0.1",
        device_classes=(DeviceClass.FORCE_PLATE,),
        test_ids=("cmj", "sj", "slcmj", "dj", "rj_10_5", "imtp"),
        kind=BundleKind.TIMESERIES,
        required_columns=(
            "fz | fz_left,fz_right | fz_plate<k>...; optional time, fx_*, fy_*, cop_x_*, cop_y_*"
        ),
        optional_metadata=("plate<k>_side", "arm_swing", "drop_height_m"),
        notes="Multi-plate files get summed_force_policy=sum_fz_all_plates.",
    )
    extra_keys = frozenset({"arm_swing", "drop_height_m", *(f"plate{k}_side" for k in range(1, 5))})

    def convert(self, parsed: ParsedCsv, test_id: str) -> SignalBundle:
        transforms: list[str] = []
        sampling = self.sampling(parsed, transforms)
        channels, data = _time_channel(parsed)
        plates: set[int] = set()
        for col in parsed.columns:
            if col.name == "time":
                continue
            fm, cm = _FORCE_COL.match(col.name), _COP_COL.match(col.name)
            if not (fm or cm):
                raise AdapterError(f"unexpected force-plate column {col.name!r}")
            match = fm or cm
            assert match is not None
            side_txt, plate_txt = match.group(2), match.group(3)
            if side_txt:
                plate, side = (1 if side_txt == "left" else 2), Side(side_txt)
            else:
                plate = int(plate_txt) if plate_txt else 1
                side = Side(parsed.meta.get(f"plate{plate}_side", "bilateral"))
            plates.add(plate)
            if fm:
                _expect(col, Quantity.FORCE)
                spec = ChannelSpec(
                    name=col.name,
                    quantity=Quantity.FORCE,
                    unit="N",
                    side=side,
                    plate=plate,
                    axis=fm.group(1)[1],
                )
            else:
                _expect(col, Quantity.DISTANCE)
                assert cm is not None
                spec = ChannelSpec(
                    name=col.name,
                    quantity=Quantity.POSITION,
                    unit="m",
                    side=side,
                    plate=plate,
                    axis=cm.group(1),
                )
            channels.append(spec)
            data[col.name] = col.values
        if not any(c.quantity is Quantity.FORCE and c.axis == "z" for c in channels):
            raise AdapterError("no vertical force (fz*) column")
        geometry = Geometry(
            plate_layout="single" if len(plates) == 1 else "dual_side_by_side",
            summed_force_policy="sum_fz_all_plates" if len(plates) > 1 else None,
        )
        extra: dict[str, object] = {"arm_swing": parsed.meta.get("arm_swing", "not_applicable")}
        if (h := self._float(parsed, "drop_height_m")) is not None:
            extra["drop_height_m"] = h
        return self.build(
            parsed,
            test_id,
            BundleKind.TIMESERIES,
            DeviceClass.FORCE_PLATE,
            channels,
            data,
            transforms,
            sampling=sampling,
            geometry=geometry,
            protocol_extra=extra,
        )


class GenericDynamometerCsv(Adapter):
    capabilities = Capabilities(
        name="generic_dynamometer_csv",
        version="0.1",
        device_classes=(DeviceClass.NORDIC_DYNAMOMETER, DeviceClass.HIP_DYNAMOMETER),
        test_ids=("nhe", "hip_add_abd", "iso_ham"),
        kind=BundleKind.TIMESERIES,
        required_columns="force | force_left,force_right; optional time",
        optional_metadata=("lever_arm_m", "knee_angle_deg", "hip_flexion_deg", "action"),
        notes="hip_add_abd requires hip_flexion_deg and action; iso_ham requires knee_angle_deg.",
    )
    extra_keys = frozenset({"lever_arm_m", "knee_angle_deg", "hip_flexion_deg", "action"})

    def convert(self, parsed: ParsedCsv, test_id: str) -> SignalBundle:
        transforms: list[str] = []
        sampling = self.sampling(parsed, transforms)
        channels, data = _time_channel(parsed)
        for col in parsed.columns:
            if col.name == "time":
                continue
            if col.name not in {"force", "force_left", "force_right"}:
                raise AdapterError(f"unexpected dynamometer column {col.name!r}")
            _expect(col, Quantity.FORCE)
            side = {"force": Side.BILATERAL, "force_left": Side.LEFT, "force_right": Side.RIGHT}
            channels.append(
                ChannelSpec(name=col.name, quantity=Quantity.FORCE, unit="N", side=side[col.name])
            )
            data[col.name] = col.values
        if len(channels) == (1 if "time" in data else 0):
            raise AdapterError("no force column")
        geometry = Geometry(
            lever_arm_m=self._float(parsed, "lever_arm_m"),
            knee_angle_deg=self._float(parsed, "knee_angle_deg"),
            hip_flexion_deg=self._float(parsed, "hip_flexion_deg"),
        )
        action = parsed.meta.get("action")
        if test_id == "hip_add_abd":
            if geometry.hip_flexion_deg is None:
                raise AdapterError("hip_add_abd requires '# hip_flexion_deg'")
            if action not in {"adduction", "abduction"}:
                raise AdapterError("hip_add_abd requires '# action: adduction|abduction'")
        if test_id == "iso_ham" and geometry.knee_angle_deg is None:
            raise AdapterError("iso_ham requires '# knee_angle_deg'")
        device = DeviceClass.NORDIC_DYNAMOMETER if test_id == "nhe" else DeviceClass.HIP_DYNAMOMETER
        return self.build(
            parsed,
            test_id,
            BundleKind.TIMESERIES,
            device,
            channels,
            data,
            transforms,
            sampling=sampling,
            geometry=geometry,
            protocol_extra={"action": action},
        )


class TimingGatesCsv(Adapter):
    capabilities = Capabilities(
        name="timing_gates_csv",
        version="0.1",
        device_classes=(DeviceClass.TIMING_GATES,),
        test_ids=("sprint_gates", "cod_505", "rsa"),
        kind=BundleKind.EVENTS,
        required_columns="distance[length unit], time[time unit]",
        optional_metadata=("start_mode", "gate_height_m"),
    )
    extra_keys = frozenset({"start_mode", "gate_height_m"})

    def convert(self, parsed: ParsedCsv, test_id: str) -> SignalBundle:
        if "sampling_rate_hz" in parsed.meta:
            raise AdapterError("timing-gate exports are events; sampling_rate_hz is not allowed")
        dist, time = parsed.column("distance"), parsed.column("time")
        if dist is None or time is None or len(parsed.columns) != 2:
            raise AdapterError("timing-gate CSV needs exactly 'distance[..]' and 'time[..]'")
        _expect(dist, Quantity.DISTANCE)
        _expect(time, Quantity.TIME)
        channels = [
            ChannelSpec(name="distance_m", quantity=Quantity.DISTANCE, unit="m"),
            ChannelSpec(name="t_s", quantity=Quantity.TIME, unit="s"),
        ]
        geometry = Geometry(gate_height_m=self._float(parsed, "gate_height_m"))
        return self.build(
            parsed,
            test_id,
            BundleKind.EVENTS,
            DeviceClass.TIMING_GATES,
            channels,
            {"distance_m": dist.values, "t_s": time.values},
            ["renamed distance->distance_m, time->t_s"],
            geometry=geometry,
            protocol_extra={"start_mode": parsed.meta.get("start_mode", "not_applicable")},
        )


class VelocityCsv(Adapter):
    capabilities = Capabilities(
        name="radar_laser_csv",
        version="0.1",
        device_classes=(DeviceClass.RADAR_LASER,),
        test_ids=("sprint_fv",),
        kind=BundleKind.TIMESERIES,
        required_columns="velocity[speed unit] or distance[length unit]; optional time",
        optional_metadata=("start_mode",),
    )
    extra_keys = frozenset({"start_mode"})

    def convert(self, parsed: ParsedCsv, test_id: str) -> SignalBundle:
        transforms: list[str] = []
        sampling = self.sampling(parsed, transforms)
        channels, data = _time_channel(parsed)
        for col in parsed.columns:
            if col.name == "time":
                continue
            if col.name == "velocity":
                _expect(col, Quantity.VELOCITY)
                channels.append(
                    ChannelSpec(name="velocity", quantity=Quantity.VELOCITY, unit="m/s")
                )
            elif col.name == "distance":
                _expect(col, Quantity.DISTANCE)
                channels.append(ChannelSpec(name="distance", quantity=Quantity.DISTANCE, unit="m"))
            else:
                raise AdapterError(f"unexpected radar/laser column {col.name!r}")
            data[col.name] = col.values
        if not set(data) - {"time"}:
            raise AdapterError("radar/laser CSV needs velocity or distance")
        return self.build(
            parsed,
            test_id,
            BundleKind.TIMESERIES,
            DeviceClass.RADAR_LASER,
            channels,
            data,
            transforms,
            sampling=sampling,
            protocol_extra={"start_mode": parsed.meta.get("start_mode", "not_applicable")},
        )


class LptCsv(Adapter):
    capabilities = Capabilities(
        name="lpt_csv",
        version="0.1",
        device_classes=(DeviceClass.LPT,),
        test_ids=("cmj", "sj"),
        kind=BundleKind.TIMESERIES,
        required_columns="displacement[length unit]; optional time",
        optional_metadata=("arm_swing",),
    )
    extra_keys = frozenset({"arm_swing"})

    def convert(self, parsed: ParsedCsv, test_id: str) -> SignalBundle:
        transforms: list[str] = []
        sampling = self.sampling(parsed, transforms)
        channels, data = _time_channel(parsed)
        disp = parsed.column("displacement")
        extra_cols = {c.name for c in parsed.columns} - {"time", "displacement"}
        if disp is None or extra_cols:
            raise AdapterError("LPT CSV needs exactly 'displacement[..]' (+ optional time)")
        _expect(disp, Quantity.DISTANCE)
        channels.append(ChannelSpec(name="displacement", quantity=Quantity.DISPLACEMENT, unit="m"))
        data["displacement"] = disp.values
        return self.build(
            parsed,
            test_id,
            BundleKind.TIMESERIES,
            DeviceClass.LPT,
            channels,
            data,
            transforms,
            sampling=sampling,
            protocol_extra={"arm_swing": parsed.meta.get("arm_swing", "not_applicable")},
        )


class GnssSummaryCsv(Adapter):
    capabilities = Capabilities(
        name="gnss_summary_csv",
        version="0.1",
        device_classes=(DeviceClass.GNSS_IMU,),
        test_ids=("cmj", "nhe", "sprint_gates", "hip_add_abd", "imtp"),
        kind=BundleKind.SUMMARY,
        required_columns="any name[unit] columns (one row per session or drill)",
        notes="Context only; test_id names the testing session the load summary belongs to.",
    )
    extra_keys = frozenset({"speed_threshold_m_s"})

    def convert(self, parsed: ParsedCsv, test_id: str) -> SignalBundle:
        if "sampling_rate_hz" in parsed.meta:
            raise AdapterError("summary exports must not declare sampling_rate_hz")
        channels, data = [], {}
        canonical = {
            Quantity.DISTANCE: "m",
            Quantity.TIME: "s",
            Quantity.VELOCITY: "m/s",
            Quantity.COUNT: "1",
            Quantity.FORCE: "N",
        }
        for col in parsed.columns:
            if col.quantity not in canonical:
                raise AdapterError(f"unsupported summary quantity for {col.name}: {col.quantity}")
            channels.append(
                ChannelSpec(name=col.name, quantity=col.quantity, unit=canonical[col.quantity])
            )
            data[col.name] = col.values
        transforms = []
        if "speed_threshold_m_s" in parsed.meta:
            transforms.append(f"speed threshold {parsed.meta['speed_threshold_m_s']} m/s")
        return self.build(
            parsed, test_id, BundleKind.SUMMARY, DeviceClass.GNSS_IMU, channels, data, transforms
        )


for _adapter in (
    GenericForcePlateCsv(),
    GenericDynamometerCsv(),
    TimingGatesCsv(),
    VelocityCsv(),
    LptCsv(),
    GnssSummaryCsv(),
):
    register(_adapter)
