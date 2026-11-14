"""RES-705: both readers enforce integrity and canonical storage before use."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from dynamisbench_soccer.signals import examples, io


def rewrite(stem: Path, table: pa.Table) -> None:
    p, j = io.paths_for(stem)
    pq.write_table(table, p)
    raw = json.loads(j.read_text())
    raw["data_sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
    j.write_text(json.dumps(raw))


@pytest.mark.parametrize("reader", [io.read, lambda p: list(io.iter_batches(p))])
@pytest.mark.parametrize("fault", ["hash", "dtype", "columns", "rows", "null", "nan"])
def test_reader_rejects_corrupt_storage(tmp_path: Path, reader, fault: str) -> None:  # type: ignore[no-untyped-def]
    stem = tmp_path / "bundle"
    p, _ = io.write(examples.nhe(), stem)
    if fault == "hash":
        p.write_bytes(p.read_bytes() + b"corruption")
    else:
        table = pq.read_table(p)
        if fault == "dtype":
            table = table.cast(pa.schema([(name, pa.float32()) for name in table.column_names]))
        elif fault == "columns":
            table = table.rename_columns(["wrong", "other"])
        elif fault == "rows":
            table = table.slice(0, 2)
        else:
            values = table.column(0).to_pylist()
            values[0] = None if fault == "null" else np.nan
            table = table.set_column(0, table.column_names[0], pa.array(values, type=pa.float64()))
        rewrite(stem, table)
    with pytest.raises((io.BundleIntegrityError, ValueError)):
        reader(stem)


def test_stream_checks_unselected_channels_and_missing_metadata(tmp_path: Path) -> None:
    stem = tmp_path / "bundle"
    p, j = io.write(examples.nhe(), stem)
    table = pq.read_table(p)
    values = table.column(1).to_pylist()
    values[0] = np.nan
    rewrite(stem, table.set_column(1, table.column_names[1], pa.array(values, type=pa.float64())))
    with pytest.raises(ValueError, match="non-finite"):
        list(io.iter_batches(stem, columns=["force_left"]))
    j.unlink()
    with pytest.raises(FileNotFoundError):
        list(io.iter_batches(stem))
