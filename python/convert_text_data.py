"""Convert text MLP training data to binary `.edgn` files, one per source file.

Every text file in *--source_dir* (no suffix or `.txt`, one `u v w z R G B`
record per line) becomes `<out_dir>/<stem>.edgn`; `train.py` loads those far
faster than the text. Each output is read back and a sample of its rows is
checked bit for bit against the source lines parsed by Python's float().

Usage:
    python python/convert_text_data.py --source_dir ./uvwz_rgb_data \
                                       --out_dir cache/training/uvwz_rgb
"""

import argparse
import os
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from tqdm import tqdm

import datagen_format


def convert(source: Path, target: Path, verify_row_count: int) -> int:
    """Converts one text file and returns its record count."""
    datagen_format.write_records(target, datagen_format.read_text(source))
    records = datagen_format.read_records(target)
    lines = source.read_text().splitlines()
    assert records.shape[0] == len(lines)

    rng = np.random.default_rng(zlib.crc32(source.name.encode()))
    for row in rng.choice(len(lines), size=min(verify_row_count, len(lines)), replace=False):
        expected = np.array([float(value) for value in lines[row].split()], dtype=np.float32)
        if not np.array_equal(records[row].view(np.uint32), expected.view(np.uint32)):
            raise ValueError(f"{target} row {row}: {records[row]} != {source} line {row + 1}: {expected}")
    return records.shape[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source_dir", type=Path, default=Path("uvwz_rgb_data"), help="directory of text data files")
    parser.add_argument("--out_dir", type=Path, default=Path("cache/training/uvwz_rgb"), help="directory for .edgn files")
    parser.add_argument("--workers", type=int, default=os.cpu_count(), help="parallel conversion processes")
    parser.add_argument("--verify_rows", type=int, default=64, help="rows per file checked against float()")
    args = parser.parse_args()

    sources = sorted(path for path in args.source_dir.iterdir() if path.is_file() and path.suffix in {"", ".txt"})
    if not sources:
        raise SystemExit(f"no text data files in {args.source_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    targets = [args.out_dir / (source.stem + datagen_format.SUFFIX) for source in sources]
    assert len(set(targets)) == len(sources), "two source files share a stem"

    start = time.perf_counter()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        counts = pool.map(convert, sources, targets, [args.verify_rows] * len(sources))
        record_count = sum(tqdm(counts, total=len(sources), desc="Converting"))
    elapsed_s = time.perf_counter() - start
    size_bytes = sum(target.stat().st_size for target in targets)
    print(f"wrote {len(targets)} files, {record_count:,} records, {size_bytes / 2**30:.2f} GiB "
          f"to {args.out_dir} in {elapsed_s:.1f} s")


if __name__ == "__main__":
    main()
