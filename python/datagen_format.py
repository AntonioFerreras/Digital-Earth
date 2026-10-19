"""MLP training-data files: binary files written by `earth datagen`, and text.

Binary layout, little-endian, mirrored by `src/assets/datagen_format.zig`:

    magic "EDGN" | version u32 | record_count u32 | field_count u32 (= 7)
    record_count x f32[7]: u v w z R G B

`earth datagen` writes RGB in the renderer's radiance scale. Text data from the
original Taichi data_prep.py (and `.edgn` files converted from it) is
`export_weights.TAICHI_OUTPUT_SCALE` times darker; nothing in a file records
which scale it holds.

Binary files carry the `SUFFIX` extension. Text files hold one record per line
as seven whitespace-separated floats.
"""

from pathlib import Path

import numpy as np

MAGIC = b"EDGN"
SUFFIX = ".edgn"
VERSION = 1
FIELD_COUNT = 7
HEADER = np.dtype([("magic", "S4"), ("version", "<u4"), ("record_count", "<u4"), ("field_count", "<u4")])


def read_records(path: Path) -> np.ndarray:
    """All records of one binary file as a float32 array of shape (record_count, 7)."""
    header = np.fromfile(path, dtype=HEADER, count=1)[0]
    if header["magic"] != MAGIC:
        raise ValueError(f"{path}: not an EDGN file")
    if header["version"] != VERSION:
        raise ValueError(f"{path}: unsupported EDGN version {header['version']}")
    if header["field_count"] != FIELD_COUNT:
        raise ValueError(f"{path}: expected {FIELD_COUNT} fields, got {header['field_count']}")
    record_count = int(header["record_count"])
    records = np.fromfile(path, dtype="<f4", offset=HEADER.itemsize)
    if records.size != record_count * FIELD_COUNT:
        raise ValueError(f"{path}: header says {record_count} records, file holds {records.size / FIELD_COUNT}")
    return records.reshape(record_count, FIELD_COUNT)


def read_text(path: Path) -> np.ndarray:
    """All records of one text file as a float32 array of shape (line_count, 7)."""
    # Parsing to float64 first rounds each value exactly as Python's float() followed by a float32 cast.
    records = np.loadtxt(path, dtype=np.float64, ndmin=2).astype(np.float32)
    if records.shape[1] != FIELD_COUNT:
        raise ValueError(f"{path}: expected {FIELD_COUNT} floats per line, got {records.shape[1]}")
    return records


def write_records(path: Path, records: np.ndarray) -> None:
    """Writes float32 records of shape (record_count, 7) as one binary file."""
    assert records.dtype == np.float32
    assert records.ndim == 2
    assert records.shape[1] == FIELD_COUNT
    assert records.shape[0] <= np.iinfo(np.uint32).max
    header = np.array([(MAGIC, VERSION, records.shape[0], FIELD_COUNT)], dtype=HEADER)
    with path.open("wb") as file:
        file.write(header.tobytes())
        file.write(records.astype("<f4", copy=False).tobytes())
