"""RES-693 contracts; examples are shapes, never scientific reference data."""

from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from dynamisbench_soccer.signals import examples, io
from dynamisbench_soccer.signals.schema import (
    CANONICAL_UNIT,
    Acquisition,
    AcquisitionSource,
    Alignment,
    BundleKind,
    BundleMeta,
    ChannelSpec,
    Clock,
    Device,
    DeviceClass,
    Frame,
    Geometry,
    Quantity,
    SignalBundle,
)

CLASSES = [c for c in DeviceClass if c not in {DeviceClass.GNSS_IMU, DeviceClass.MANUAL_PROTOCOL}]


def shape(cls: DeviceClass) -> SignalBundle:
    raw = examples.lpt().meta.model_dump(mode="json")
    geometry = Geometry(unknown=list(Geometry.model_fields))
    specs: list[tuple[str, Quantity, str | None]]
    if cls is DeviceClass.IMU:
        specs = [
            (f"{q.value}_{a}", q, a)
            for q in (Quantity.SPECIFIC_FORCE, Quantity.ANGULAR_VELOCITY)
            for a in ("x", "y", "z")
        ]
    elif cls is DeviceClass.LPS:
        specs = [(f"position_{a}", Quantity.POSITION, a) for a in ("x", "y", "z")]
    elif cls is DeviceClass.GNSS:
        specs = [
            ("latitude", Quantity.LATITUDE, None),
            ("longitude", Quantity.LONGITUDE, None),
            ("height", Quantity.POSITION, "z"),
        ]
    elif cls is DeviceClass.ENCODER:
        specs = [("counts", Quantity.COUNT, None)]
    elif cls is DeviceClass.CONTACT_MAT:
        specs = [("t_s", Quantity.TIME, None), ("contact", Quantity.DIMENSIONLESS, None)]
    elif cls is DeviceClass.TIMING_GATES:
        specs = [("t_s", Quantity.TIME, None), ("distance_m", Quantity.DISTANCE, None)]
    else:
        q = (
            Quantity.DISPLACEMENT
            if cls is DeviceClass.LPT
            else Quantity.DISTANCE
            if cls is DeviceClass.LASER
            else Quantity.VELOCITY
            if cls in {DeviceClass.RADAR, DeviceClass.RADAR_LASER}
            else Quantity.FORCE
        )
        specs = [("value", q, "z" if cls is DeviceClass.FORCE_PLATE else None)]
    channels = [
        ChannelSpec(
            name=n,
            quantity=q,
            axis=a,
            unit=CANONICAL_UNIT[q],
            frame="sensor",
            plate=1 if cls is DeviceClass.FORCE_PLATE else None,
        )
        for n, q, a in specs
    ]
    device = Device(device_class=cls)
    acq = Acquisition(
        clocks=[Clock(id="clock", source="device", epoch="trial start")],
        reference_clock="clock",
        sources=[
            AcquisitionSource(
                id="device",
                device=device,
                clock="clock",
                channels=[c.name for c in channels],
                geometry=geometry,
                calibration_unknown_reason="Shape fixture; no physical calibration",
                raw_format="synthetic",
            )
        ],
        frames=[Frame(id="sensor", kind="sensor", definition="shape fixture axes")],
    )
    event = cls in {DeviceClass.TIMING_GATES, DeviceClass.CONTACT_MAT}
    raw.update(
        device=device.model_dump(),
        channels=[c.model_dump() for c in channels],
        n_rows=3,
        kind=BundleKind.EVENTS if event else BundleKind.TIMESERIES,
        sampling=None if event else {"nominal_hz": 100},
        acquisition=acq.model_dump(),
    )
    meta = BundleMeta.model_validate(raw)
    return SignalBundle(
        meta,
        {
            c.name: np.array([0.0, 1.0, 2.0]) if c.name != "contact" else np.array([1.0, 0.0, 1.0])
            for c in channels
        },
    )


@pytest.mark.parametrize("cls", CLASSES)
def test_each_device_shape_roundtrip(cls: DeviceClass, tmp_path: Path) -> None:
    b = shape(cls)
    io.write(b, tmp_path / cls.value)
    assert io.read(tmp_path / cls.value).meta == b.meta


def test_clock_alignment_and_rotation_validation() -> None:
    with pytest.raises(ValidationError, match="rotation"):
        Frame(
            id="s",
            kind="sensor",
            definition="bad",
            parent="field",
            rotation_to_parent=((1, 0, 0), (0, 1, 0), (0, 0, -1)),
        )
    with pytest.raises(ValidationError, match="sync event"):
        Alignment(
            clock="a",
            reference_clock="b",
            method="trigger",
            offset_s=1,
            uncertainty_s=0,
            sync_events=[(0, 2)],
        )
    a = shape(DeviceClass.LPT).meta.acquisition
    assert a is not None
    raw = a.model_dump()
    raw["clocks"].append({"id": "second", "source": "host", "epoch": "trial"})
    with pytest.raises(ValidationError, match="requires alignment"):
        Acquisition.model_validate(raw)
    raw["alignments"] = [
        dict(
            clock="second",
            reference_clock="clock",
            method="shared trigger",
            offset_s=1,
            uncertainty_s=0,
            sync_events=[(0, 1)],
        )
    ]
    assert Acquisition.model_validate(raw)


def test_missing_raw_channels_and_ambiguous_geometry_reject() -> None:
    raw = shape(DeviceClass.IMU).meta.model_dump()
    raw["channels"][0]["quantity"] = Quantity.FORCE
    raw["channels"][0]["unit"] = "N"
    with pytest.raises(ValidationError, match="xyz"):
        BundleMeta.model_validate(raw)
    source = shape(DeviceClass.ENCODER).meta.acquisition.sources[0]  # type: ignore[union-attr]
    raw = source.model_dump()
    raw["geometry"] = {}
    with pytest.raises(ValidationError, match="encoder_counts_per_m"):
        AcquisitionSource.model_validate(raw)


def test_v1_migration_roundtrip_preserves_samples_hash_and_unknowns(tmp_path: Path) -> None:
    import json
    import hashlib

    fixture = Path(__file__).parent / "fixtures/signals/v1_meta.json"
    raw = json.loads(fixture.read_text())
    original = json.loads(fixture.read_text())
    p, sidecar = io.write(examples.cmj_dual_plate(), tmp_path / "v1")
    assert raw["data_sha256"] == hashlib.sha256(p.read_bytes()).hexdigest()
    sidecar.write_text(json.dumps(raw))
    migrated = io.read(tmp_path / "v1")
    assert migrated.meta.schema_version == 2
    assert migrated.meta.acquisition is None
    assert "Migrated v1" in migrated.meta.acquisition_unknown_reason
    assert io.migrate(raw) == io.migrate(io.migrate(raw))
    assert raw == original
    p2, j2 = io.write(migrated, tmp_path / "v2")
    assert p.read_bytes() == p2.read_bytes()
    assert io.read(tmp_path / "v2").meta == migrated.meta
    assert j2.read_text() == io.meta_json(migrated.meta)


@pytest.mark.parametrize("version", [True, 0, -1, 3, "1"])
def test_invalid_versions_rejected(version: object) -> None:
    with pytest.raises(ValueError):
        io.migrate({"schema_version": version})
