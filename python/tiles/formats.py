"""Reader and writer of the `.tiles` files of the streamed Earth.

One file per layer and cube face (docs/plans/20261020-streamed-earth-tiles.md):
a 32-byte header, a 16-byte entry per tile of levels 0 to `last_level`, level
by level and each level in Morton order of (x, y), then the payloads.
Everything is little-endian. Rows of a decoded tile run along v from its
smallest v; columns along u.

Run the round-trip self-test: pixi run -e py python python/tiles/formats.py
"""

import io
import os
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import zstandard
from PIL import Image

from cube import HEIGHT_SIZE, SURFACE_SIZE, TILE_INTERVALS, BORDER

MAGIC = b"ETIL"
VERSION = 1
LAYERS = {"height": 0, "surface": 1}
CODECS = {"height": 0, "surface": 1}
HEADER = struct.Struct("<4s7I")
ENTRY = struct.Struct("<QIhh")
ZSTD_LEVEL = 19
JPEG_QUALITY = 90


def morton(x, y):
    """Bit k of x goes to bit 2k, bit k of y to bit 2k + 1."""
    code = 0
    for bit in range(16):
        code |= ((x >> bit) & 1) << (2 * bit) | ((y >> bit) & 1) << (2 * bit + 1)
    return code


def entry_count(last_level):
    return (4 ** (last_level + 1) - 1) // 3


def entry_index(level, x, y):
    return (4**level - 1) // 3 + morton(x, y)


def tile_keys(last_level):
    """(level, x, y) of every entry, in file order."""
    keys = []
    for level in range(last_level + 1):
        side = 1 << level
        level_keys = [(level, x, y) for y in range(side) for x in range(side)]
        keys += sorted(level_keys, key=lambda key: morton(key[1], key[2]))
    return keys


def med_predict(h):
    """MED (LOCO-I) prediction of every sample of int32 `h` from its left, up,
    and up-left neighbours, missing ones taken as 0; "up" is the previous row."""
    padded = np.pad(h, ((1, 0), (1, 0)))
    a, b, c = padded[1:, :-1], padded[:-1, 1:], padded[:-1, :-1]
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    return np.where(c >= hi, lo, np.where(c <= lo, hi, a + b - c))


def encode_height(height, water):
    """Height payload of int16 metres and uint8 water fraction, both bordered squares."""
    assert height.shape == water.shape == (HEIGHT_SIZE, HEIGHT_SIZE)
    assert height.dtype == np.int16 and water.dtype == np.uint8
    h = height.astype(np.int32)
    d = (h - med_predict(h)).astype(np.int16)
    z = ((d.astype(np.int32) << 1) ^ (d.astype(np.int32) >> 15)).astype(np.uint16)
    planes = (z & 0xFF).astype(np.uint8).tobytes() + (z >> 8).astype(np.uint8).tobytes() + water.tobytes()
    return zstandard.ZstdCompressor(level=ZSTD_LEVEL).compress(planes)


def decode_height(payload):
    """(height int16, water uint8) of a height payload."""
    planes = np.frombuffer(zstandard.ZstdDecompressor().decompress(payload), dtype=np.uint8)
    count = HEIGHT_SIZE * HEIGHT_SIZE
    assert planes.size == 3 * count
    z = planes[:count].astype(np.int32) | planes[count:2 * count].astype(np.int32) << 8
    d = ((z >> 1) ^ -(z & 1)).reshape(HEIGHT_SIZE, HEIGHT_SIZE)
    # Each sample depends on its left, up, and up-left neighbours, all on the
    # previous anti-diagonal or the one before, so a diagonal decodes at once.
    h = np.zeros((HEIGHT_SIZE + 1, HEIGHT_SIZE + 1), dtype=np.int32)
    for k in range(2 * HEIGHT_SIZE - 1):
        j = np.arange(max(0, k - HEIGHT_SIZE + 1), min(k, HEIGHT_SIZE - 1) + 1)
        i = k - j
        a, b, c = h[j + 1, i], h[j, i + 1], h[j, i]
        lo, hi = np.minimum(a, b), np.maximum(a, b)
        prediction = np.where(c >= hi, lo, np.where(c <= lo, hi, a + b - c))
        h[j + 1, i + 1] = (prediction + d[j, i] + 32768) % 65536 - 32768
    water = planes[2 * count:].reshape(HEIGHT_SIZE, HEIGHT_SIZE)
    return h[1:, 1:].astype(np.int16), water.copy()


def encode_surface(rgb):
    """Surface payload of a bordered uint8 sRGB square: baseline JPEG, 4:4:4."""
    assert rgb.shape == (SURFACE_SIZE, SURFACE_SIZE, 3) and rgb.dtype == np.uint8
    out = io.BytesIO()
    Image.fromarray(rgb, "RGB").save(out, format="JPEG", quality=JPEG_QUALITY, subsampling=0, progressive=False)
    return out.getvalue()


def decode_surface(payload):
    rgb = np.asarray(Image.open(io.BytesIO(payload)).convert("RGB"))
    assert rgb.shape == (SURFACE_SIZE, SURFACE_SIZE, 3)
    return rgb


@dataclass
class Tile:
    """A finished tile: its payload (empty when absent) and height bounds."""
    payload: bytes
    min: int = 0
    max: int = 0


def write_tiles(path, layer, face, last_level, tiles):
    """Writes `tiles` ({(level, x, y): Tile}) as one file, atomically.

    Keys not in `tiles` are absent; identical payloads are stored once.
    """
    keys = tile_keys(last_level)
    entries = []
    offsets = {}
    blobs = []
    offset = HEADER.size + ENTRY.size * len(keys)
    for key in keys:
        tile = tiles.get(key)
        if tile is None or not tile.payload:
            entries.append(ENTRY.pack(0, 0, 0, 0))
            continue
        if tile.payload not in offsets:
            offsets[tile.payload] = offset
            blobs.append(tile.payload)
            offset += len(tile.payload)
        entries.append(ENTRY.pack(offsets[tile.payload], len(tile.payload), tile.min, tile.max))
    header = HEADER.pack(MAGIC, VERSION, LAYERS[layer], face, TILE_INTERVALS, BORDER, last_level, CODECS[layer])
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with open(temporary, "wb") as file:
        file.write(header)
        file.write(b"".join(entries))
        for blob in blobs:
            file.write(blob)
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)


class TileFile:
    """A `.tiles` file read whole into memory."""

    def __init__(self, path):
        self.data = Path(path).read_bytes()
        magic, version, layer, self.face, intervals, border, self.last_level, codec = HEADER.unpack_from(self.data)
        assert magic == MAGIC and version == VERSION, f"{path}: not a version {VERSION} tile file"
        assert intervals == TILE_INTERVALS and border == BORDER
        self.layer = {value: name for name, value in LAYERS.items()}[layer]
        assert codec == CODECS[self.layer]
        self.entries = [ENTRY.unpack_from(self.data, HEADER.size + ENTRY.size * index)
                        for index in range(entry_count(self.last_level))]

    def tile(self, level, x, y):
        """The Tile at (level, x, y), with an empty payload when absent."""
        if level > self.last_level:
            return Tile(b"")
        offset, size, low, high = self.entries[entry_index(level, x, y)]
        return Tile(self.data[offset:offset + size], low, high)

    def present(self):
        """{(level, x, y): Tile} of every present tile."""
        tiles = {}
        for key in tile_keys(self.last_level):
            tile = self.tile(*key)
            if tile.payload:
                tiles[key] = tile
        return tiles

    def decode(self, level, x, y):
        """The decoded tile ((height, water) or rgb), or None when absent."""
        payload = self.tile(level, x, y).payload
        if not payload:
            return None
        return decode_height(payload) if self.layer == "height" else decode_surface(payload)


def self_test():
    import tempfile

    rng = np.random.default_rng(1)
    assert [morton(x, y) for y in range(2) for x in range(2)] == [0, 1, 2, 3]
    assert morton(5, 3) == 0b011011 and entry_index(2, 1, 0) == 6 and entry_count(9) == 349525
    assert [key for key in tile_keys(1)] == [(0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 0, 1), (1, 1, 1)]

    smooth = np.cumsum(np.cumsum(rng.integers(-3, 4, (HEIGHT_SIZE, HEIGHT_SIZE)), 0), 1)
    extremes = rng.choice(np.array([-32768, 32767, 0, -1], dtype=np.int16), (HEIGHT_SIZE, HEIGHT_SIZE))
    for height in (smooth.astype(np.int16), extremes, rng.integers(-32768, 32768, (HEIGHT_SIZE, HEIGHT_SIZE)).astype(np.int16)):
        water = rng.integers(0, 256, height.shape).astype(np.uint8)
        decoded, decoded_water = decode_height(encode_height(height, water))
        assert np.array_equal(decoded, height) and np.array_equal(decoded_water, water)

    gradient = np.linspace(0, 255, SURFACE_SIZE)
    rgb = np.stack(np.broadcast_arrays(gradient[None, :], gradient[:, None], 128.0), -1).astype(np.uint8)
    payload = encode_surface(rgb)
    assert payload[:2] == b"\xff\xd8" and b"\xff\xc0" in payload, "not a baseline JPEG"
    assert np.abs(decode_surface(payload).astype(int) - rgb).max() <= 4

    tiles = {(0, 0, 0): Tile(encode_height(smooth.astype(np.int16), np.zeros(smooth.shape, np.uint8)), -5, 9),
             (2, 3, 1): Tile(b"same", 1, 2), (2, 0, 3): Tile(b"same", 1, 2), (1, 1, 1): Tile(b"other", -3, 3)}
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "face4.tiles"
        write_tiles(path, "height", 4, 2, tiles)
        file = TileFile(path)
        assert file.face == 4 and file.last_level == 2 and file.layer == "height"
        assert file.present() == tiles
        assert file.entries[entry_index(2, 3, 1)][0] == file.entries[entry_index(2, 0, 3)][0], "duplicates not shared"
        assert file.tile(1, 0, 0).payload == b"" and file.tile(5, 0, 0).payload == b""
        assert np.array_equal(file.decode(0, 0, 0)[0], smooth.astype(np.int16))
    print("formats: round trips pass")


if __name__ == "__main__":
    self_test()
