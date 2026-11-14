"""Deterministic Parquet + JSON-sidecar I/O for :class:`SignalBundle`.

Determinism contract (tested byte-for-byte): for a fixed lockfile, writing the same
bundle twice, or writing a bundle that was just read, produces identical bytes for
both files. To achieve this, compression, dictionary encoding, statistics and
row-group size are fixed, and the sidecar is emitted with sorted keys.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from dynamisbench_soccer.signals.schema import SCHEMA_VERSION, BundleMeta, FloatArray, SignalBundle

ROW_GROUP_ROWS = 65_536
_WRITE_OPTIONS: dict[str, Any] = {
    "compression": "zstd",
    "compression_level": 3,
    "use_dictionary": False,
    "write_statistics": False,
    "row_group_size": ROW_GROUP_ROWS,
    "version": "2.6",
    "data_page_version": "1.0",
}

Migration = Callable[[dict[str, Any]], dict[str, Any]]


def _v1_to_v2(raw: dict[str, Any]) -> dict[str, Any]:
    migrated = copy.deepcopy(raw)
    migrated["schema_version"] = 2
    migrated["acquisition_unknown_reason"] = (
        "Migrated v1: clocks, synchronization, frames and device geometry not recorded"
    )
    migrated["provenance"]["transforms"] = [
        *migrated["provenance"].get("transforms", []),
        "schema migration v1->v2; no signal changes",
    ]
    return migrated


MIGRATIONS: dict[int, Migration] = {1: _v1_to_v2}


class BundleIntegrityError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def paths_for(stem: Path) -> tuple[Path, Path]:
    stem = stem.with_suffix("") if stem.suffix in {".parquet", ".json"} else stem
    return stem.with_suffix(".parquet"), stem.with_suffix(".json")


def meta_json(meta: BundleMeta) -> str:
    return json.dumps(meta.model_dump(mode="json"), indent=2, sort_keys=True) + "\n"


def write(bundle: SignalBundle, stem: Path) -> tuple[Path, Path]:
    """Write ``<stem>.parquet`` and ``<stem>.json``; returns both paths."""
    parquet_path, json_path = paths_for(stem)
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {name: pa.array(values, type=pa.float64()) for name, values in bundle.data.items()}
    )
    pq.write_table(table, parquet_path, **_WRITE_OPTIONS)
    meta = bundle.meta.model_copy(update={"data_sha256": _sha256(parquet_path)})
    json_path.write_text(meta_json(meta), encoding="utf-8")
    bundle.meta = meta
    return parquet_path, json_path


def migrate(raw: dict[str, Any]) -> dict[str, Any]:
    """Upgrade a raw sidecar dict to the current schema version."""
    version = raw.get("schema_version")
    if type(version) is not int:
        raise ValueError("sidecar has no integer schema_version")
    if version > SCHEMA_VERSION:
        raise ValueError(
            f"sidecar schema_version {version} is newer than supported {SCHEMA_VERSION}"
        )
    while version < SCHEMA_VERSION:
        if version not in MIGRATIONS:
            raise ValueError(f"unsupported sidecar schema_version {version}")
        raw = MIGRATIONS[version](raw)
        if raw.get("schema_version") != version + 1:
            raise ValueError("migration must advance exactly one schema version")
        version += 1
    return raw


def read_meta(stem: Path) -> BundleMeta:
    _, json_path = paths_for(stem)
    raw = json.loads(json_path.read_text(encoding="utf-8"))
    return BundleMeta.model_validate(migrate(raw))


def _open_parquet(stem: Path, verify: bool = True) -> tuple[BundleMeta, pq.ParquetFile]:
    parquet_path, _ = paths_for(stem)
    meta = read_meta(stem)
    if verify:
        if meta.data_sha256 is None:
            raise BundleIntegrityError("sidecar has no data_sha256")
        if _sha256(parquet_path) != meta.data_sha256:
            raise BundleIntegrityError(f"{parquet_path.name}: sha256 mismatch")
    pf = pq.ParquetFile(parquet_path)
    names = [c.name for c in meta.channels]
    if pf.schema_arrow.names != names or pf.metadata.num_rows != meta.n_rows:
        raise BundleIntegrityError("Parquet columns/row count disagree with sidecar")
    if any(field.type != pa.float64() for field in pf.schema_arrow):
        raise BundleIntegrityError("Parquet channels must be stored as float64")
    return meta, pf


def _arrays(table: pa.Table | pa.RecordBatch) -> dict[str, FloatArray]:
    return {
        name: np.ascontiguousarray(table.column(i).to_numpy(), dtype=np.float64)
        for i, name in enumerate(table.schema.names)
    }


def read(stem: Path, verify: bool = True) -> SignalBundle:
    """Read validated canonical storage; hash checking may be explicitly disabled."""
    meta, pf = _open_parquet(stem, verify)
    return SignalBundle(meta, _arrays(pf.read()))


def iter_batches(
    stem: Path, columns: Sequence[str] | None = None, batch_rows: int = ROW_GROUP_ROWS
) -> Iterator[dict[str, FloatArray]]:
    """Verify identity/storage, then validate full batches before projecting columns."""
    if batch_rows <= 0:
        raise ValueError("batch_rows must be positive")
    meta, pf = _open_parquet(stem)
    if columns is not None and (
        len(set(columns)) != len(columns) or set(columns) - set(pf.schema_arrow.names)
    ):
        raise ValueError("requested columns must be unique known channels")
    for batch in pf.iter_batches(batch_size=batch_rows):
        data = _arrays(batch)
        SignalBundle(meta.model_copy(update={"n_rows": batch.num_rows}), data)
        yield data if columns is None else {name: data[name] for name in columns}


def json_schema() -> str:
    return json.dumps(BundleMeta.model_json_schema(), indent=2, sort_keys=True) + "\n"
