"""RES-718 versioned adapter for the reviewed BioSynGen FORCE-only pilot."""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
from pathlib import Path
from typing import Any

import numpy as np

from dynamisbench_soccer.adapters.base import AdapterError
from dynamisbench_soccer.data.manifest import DataCategory, DatasetManifest, Usage
from dynamisbench_soccer.signals import io as bundle_io
from dynamisbench_soccer.signals.schema import (
    Acquisition,
    AcquisitionSource,
    Alignment,
    BundleKind,
    BundleMeta,
    ChannelSpec,
    Clock,
    Device,
    DeviceClass,
    FloatArray,
    Frame,
    Geometry,
    Protocol,
    Provenance,
    Quantity,
    Sampling,
    Session,
    Side,
    SignalBundle,
    Subject,
)

ADAPTER = "biosyngen_force@1.0"
UPSTREAM = "e74fc7cc188f9fcdea5e745244c21186b8699903"
DATASET = "biosyngen-res718-force"
CASES = {"trial-00000718-0000": "cmj", "trial-00000718-0001": "sj"}
HEADER = ("sample_index", "time_s", "force_left_N", "force_right_N")
FILES = {"manifest.json", "streams/force.csv", "capture_authority.json", "source-verification.json"}
CALIBRATION = "ASSUMED_UNCALIBRATED"


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AdapterError(message)


def _files(root: Path, dataset: DatasetManifest) -> dict[str, bytes]:
    _require(
        (dataset.dataset_id, dataset.version, dataset.data_category, dataset.usage)
        == (DATASET, UPSTREAM, DataCategory.SYNTHETIC, Usage.CROSS_VALIDATION_ONLY)
        and dataset.source.location
        == f"https://github.com/Litju/BioSynGenBenchmark/tree/{UPSTREAM}"
        and dataset.source.kind == "generated"
        and not dataset.redistribution_allowed,
        "unsupported or unpinned pilot dataset",
    )
    approved = {f.path: f for f in dataset.files}
    _require(
        set(approved) == {f"{case}/{name}" for case in CASES for name in FILES},
        "pilot dataset inventory differs",
    )
    _require(
        root.name in CASES and root.is_dir() and not root.is_symlink(),
        "expected a known FORCE-only case directory",
    )
    inventory = {file.relative_to(root).as_posix(): file for file in root.rglob("*")}
    _require(
        set(inventory) == FILES | {"streams"}
        and inventory["streams"].is_dir()
        and not inventory["streams"].is_symlink(),
        "unexpected case inventory; SOURCE-only projection required",
    )
    result = {}
    for name in sorted(FILES):
        candidate = inventory[name]
        _require(
            candidate.is_file() and not candidate.is_symlink() and candidate.stat().st_nlink == 1,
            f"{name}: unsafe or linked source file",
        )
        entry = approved[f"{root.name}/{name}"]
        _require(
            0 < entry.size_bytes <= 1_048_576 and candidate.stat().st_size == entry.size_bytes,
            f"{name}: size mismatch",
        )
        with candidate.open("rb") as stream:
            data = stream.read(entry.size_bytes + 1)
        _require(hashlib.sha256(data).hexdigest() == entry.sha256, f"{name}: SHA-256 mismatch")
        result[name] = data
    return result


def _force(data: bytes) -> FloatArray:
    reader = csv.reader(io.StringIO(data.decode("utf-8"), newline=""))
    _require(tuple(next(reader, [])) == HEADER, "FORCE columns or units differ")
    rows = []
    for i, row in enumerate(reader):
        _require(len(row) == 4 and row[0] == str(i), "missing, duplicated or invalid sample index")
        rows.append([float(s) for s in row[1:]])
    values = np.asarray(rows, dtype=np.float64)
    _require(len(rows) >= 3 and bool(np.isfinite(values).all()), "invalid FORCE samples")
    _require(
        bool(np.allclose(np.diff(values[:, 0]), 0.001, rtol=0, atol=1e-10)),
        "FORCE timestamps must be strictly native nominal 1 kHz without gaps",
    )
    return values


def _alignment(
    source: dict[str, Any],
    notes: dict[str, Any],
    capture: dict[str, Any],
    verified: dict[str, Any],
    time: FloatArray,
) -> Alignment:
    _require(
        verified["source"] == "PASS"
        and verified["synchronization"] == "BOUNDED_CAPTURE_MAPPING"
        and verified["clock_admission_contract"] == "BSMB-CLOCK-ADMISSION-1.0"
        and verified["nominal_rates_hz"] == {"force": 1000, "lpt": 1000}
        and verified["declared_jitter_bound_s"] == 0
        and verified["dropped_sample_counts"]["force"] == 0
        and capture["contract_id"] == "BSMB-FOUR-EDGE-CAPTURE-1.0"
        and capture["reference_clock_id"] == "external-pulse-clock"
        and {k: v for k, v in capture.items() if k != "source_manifest_sha256"}
        == notes["sync_capture"]
        and set(capture["devices"]) == {"force", "lpt"}
        and all(capture["devices"][k]["clock_id"] == k for k in ("force", "lpt"))
        and capture["fit_edge_indices"] == [0, 3]
        and capture["held_out_edge_indices"] == [1, 2],
        "qualified, distinct clock capture authority required",
    )
    by_name = source["synchronization"]["stream_mappings"]
    maps = verified["stream_mappings"]
    _require(
        set(by_name) == set(maps) == {"force", "lpt"}
        and all(
            by_name[k]["drift_scale"] == maps[k]["drift_scale"]
            and by_name[k]["offset_s"] == maps[k]["offset_s"]
            for k in maps
        ),
        "source and verified clock mappings differ",
    )
    native = np.asarray(capture["devices"]["force"]["edge_native_times_s"], dtype=float)
    reference = np.asarray(capture["reference_edge_times_s"], dtype=float)
    scale, offset = maps["force"]["drift_scale"], maps["force"]["offset_s"]
    bound = verified["worst_case_mapping_error_s"]["force"]
    capture_error = capture["capture_error_bound_s"]
    _require(
        native.shape == reference.shape == (4,)
        and bool(np.all(np.diff(native) > 0) and np.all(np.diff(reference) > 0))
        and 0.999 <= scale <= 1.001
        and 0 < capture_error <= bound < 0.0005
        and native[0] - capture_error <= time[0] < time[-1] <= native[-1] + capture_error
        and bool(np.all(np.abs(native * scale + offset - reference) <= bound)),
        "invalid or unbracketed FORCE clock mapping",
    )
    return Alignment(
        clock="force",
        reference_clock="external-pulse-clock",
        method="BSMB four-edge capture; upstream bounded affine fit and held-out checks",
        offset_s=offset,
        drift_ppm=(scale - 1) * 1e6,
        uncertainty_s=bound,
        sync_events=list(zip(native.tolist(), reference.tolist(), strict=True)),
    )


def _convert(case: str, files: dict[str, bytes]) -> SignalBundle:
    source = json.loads(files["manifest.json"])
    capture = json.loads(files["capture_authority.json"])
    verified = json.loads(files["source-verification.json"])
    notes = json.loads(source["operator_notes"])
    spec = source["force_stream"]
    force_sha = hashlib.sha256(files["streams/force.csv"]).hexdigest()
    source_sha = hashlib.sha256(files["manifest.json"]).hexdigest()
    _require(
        all(source[k] == case for k in ("acquisition_id", "bundle_id", "trial_id"))
        and capture["source_manifest_sha256"] == source_sha
        and capture["recording_hashes"]["force"] == force_sha,
        "trial identity or independent capture hashes differ",
    )
    _require(
        source["profile_id"] == "BSMB-PAIRED-CABLE-ANGLE-1.1"
        and source["profile_version"] == "1.1"
        and notes["generator_contract_id"] == "BSMB-GENERATOR-1.1"
        and notes["acquisition_profile"] == "BSMB-FORCE1000-LPT1000-1.0"
        and spec["path"] == "streams/force.csv"
        and tuple(spec["columns"]) == HEADER
        and spec["unit"] == "N"
        and spec["frame_id"] == "LAB_VERTICAL_Z_UP"
        and spec["positive_direction"] == "UP"
        and spec["processing"]["state"] == "RAW_UNFILTERED_DECLARED"
        and spec["calibration"]["evidence_reference"]
        == "ASSUMED_MANUFACTURED_PARAMETERS_NOT_EMPIRICAL_CALIBRATION",
        "expected assumed-uncalibrated, source-stage vertical FORCE in N",
    )
    front = notes["force_front_end"]
    _require(front["adc_range"] == [2500, 5000, 1500], "unexpected FORCE ADC rails")
    for side in ("left", "right"):
        gain = np.asarray(front[side]["gain"], dtype=float)
        _require(
            gain.shape == (3, 3)
            and bool(np.isfinite(gain).all())
            and bool(np.max(np.abs(np.diag(gain) - 1)) <= 0.05)
            and bool(np.max(np.abs(gain - np.diag(np.diag(gain)))) <= 0.01),
            "FORCE gain outside narrow pilot admission",
        )
    timebase = spec["timebase"]
    _require(
        (timebase["clock_id"], timebase["authority"], timebase["nominal_sampling_rate_hz"])
        == ("force", "TIMESTAMP", 1000),
        "invalid native FORCE timebase declaration",
    )
    values = _force(files["streams/force.csv"])
    time = values[:, 0]
    link = _alignment(source, notes, capture, verified, time)
    frame = Frame(id="lab", kind="field", definition="BioSynGen lab vertical z positive up")
    geometry = Geometry(plate_layout="dual_side_by_side", summed_force_policy="sum_fz_all_plates")
    device = Device(
        device_class=DeviceClass.FORCE_PLATE,
        manufacturer=spec["left_plate"]["manufacturer"],
        model=spec["left_plate"]["model"],
    )
    channels = [
        ChannelSpec(name="time", quantity=Quantity.TIME, unit="s"),
        ChannelSpec(
            name="fz_left",
            quantity=Quantity.FORCE,
            unit="N",
            side=Side.LEFT,
            axis="z",
            plate=1,
            frame=frame.id,
        ),
        ChannelSpec(
            name="fz_right",
            quantity=Quantity.FORCE,
            unit="N",
            side=Side.RIGHT,
            axis="z",
            plate=2,
            frame=frame.id,
        ),
    ]
    acquisition = Acquisition(
        clocks=[
            Clock(id="force", source="device", epoch=f"{case} native FORCE acquisition"),
            Clock(
                id=link.reference_clock, source="host", epoch=f"{case} simulated external pulses"
            ),
        ],
        reference_clock=link.reference_clock,
        frames=[frame],
        alignments=[link],
        sources=[
            AcquisitionSource(
                id="force",
                device=device,
                clock="force",
                channels=[c.name for c in channels],
                geometry=geometry,
                calibration_unknown_reason=CALIBRATION,
                raw_format="BSMB dual vertical force CSV; analog acquisition stage retained",
                source_sha256=force_sha,
            )
        ],
    )
    test = CASES[case]
    meta = BundleMeta(
        bundle_id=case,
        kind=BundleKind.TIMESERIES,
        test_id=test,
        data_category=DataCategory.SYNTHETIC,
        subject=Subject(pseudonym="BSMB-SYNTHETIC"),
        session=Session(session_id=case, date=source["created_at_utc"][:10]),
        device=device,
        protocol=Protocol(protocol_id=f"bsmb.{test}.force1000"),
        sampling=Sampling(
            nominal_hz=1000, measured_hz=1 / float(np.median(np.diff(time))), t0_s=float(time[0])
        ),
        channels=channels,
        n_rows=len(time),
        geometry=geometry,
        acquisition=acquisition,
        acquisition_unknown_reason=None,
        provenance=Provenance(
            created_by=f"adapter:{ADAPTER}",
            source_sha256=force_sha,
            transforms=[
                f"upstream={UPSTREAM}",
                f"source_manifest_sha256={source_sha}",
                f"capture_authority_sha256={hashlib.sha256(files['capture_authority.json']).hexdigest()}",
                f"source_verification_sha256={hashlib.sha256(files['source-verification.json']).hexdigest()}",
                "native timestamps retained; timebase alignment deferred to processor",
            ],
        ),
    )
    return SignalBundle(meta, dict(zip((c.name for c in channels), values.T, strict=True)))


def convert(root: Path, dataset: DatasetManifest) -> SignalBundle:
    try:
        return _convert(root.name, _files(root, dataset))
    except (KeyError, IndexError, AttributeError, TypeError) as exc:
        raise AdapterError("invalid SOURCE, capture or verification metadata") from exc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        bundle = convert(args.case, DatasetManifest.load(args.dataset))
        args.out_dir.mkdir()
        bundle_io.write(bundle, args.out_dir / "force")
    except (OSError, ValueError) as exc:
        parser.exit(2, f"{ADAPTER}: {exc}\n")


if __name__ == "__main__":
    main()
