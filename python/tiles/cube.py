"""The equal-angle cube sphere of the streamed Earth tiles.

The reference implementation of the mapping in
docs/plans/20261020-streamed-earth-tiles.md: the shaders are tested against it
through python/goldens.py, so any change here is a format change.

The world frame is the renderer's: +Y north, Greenwich on -X, 90 E on +Z.
Each face's s axis points east along its equator row, so u grows eastward on
the four equatorial faces.
"""

import numpy as np

# Rows: face normal, s axis, t axis.
FACES = np.array([
    [[1, 0, 0], [0, 0, -1], [0, 1, 0]],
    [[-1, 0, 0], [0, 0, 1], [0, 1, 0]],
    [[0, 1, 0], [1, 0, 0], [0, 0, -1]],
    [[0, -1, 0], [1, 0, 0], [0, 0, 1]],
    [[0, 0, 1], [1, 0, 0], [0, 1, 0]],
    [[0, 0, -1], [-1, 0, 0], [0, 1, 0]],
], dtype=np.float64)

TILE_INTERVALS = 192
BORDER = 2
HEIGHT_SIZE = TILE_INTERVALS + 1 + 2 * BORDER
SURFACE_SIZE = TILE_INTERVALS + 2 * BORDER


def direction_to_face_uv(n):
    """Returns (face, u, v) of directions `n` (..., 3), not necessarily unit.

    The face is the axis of largest |n|; on an exact tie the first of x, y, z
    wins (np.argmax), so a port must break ties the same way.
    """
    n = np.asarray(n, dtype=np.float64)
    axis = np.argmax(np.abs(n), axis=-1)
    major = np.take_along_axis(n, axis[..., None], axis=-1)[..., 0]
    face = 2 * axis + (major < 0)
    s_axis, t_axis = FACES[face, 1], FACES[face, 2]
    a = np.sum(n * s_axis, axis=-1) / np.abs(major)
    b = np.sum(n * t_axis, axis=-1) / np.abs(major)
    u = (4.0 / np.pi * np.arctan(a) + 1.0) / 2.0
    v = (4.0 / np.pi * np.arctan(b) + 1.0) / 2.0
    return face, u, v


def face_uv_to_direction(face, u, v):
    """Returns unit directions (..., 3) of face coordinates.

    (u, v) outside [0, 1] continue the face's own warp past its edge, which is
    where the tile borders are sampled.
    """
    face = np.asarray(face)
    a = np.tan(np.pi / 4.0 * (2.0 * np.asarray(u, dtype=np.float64) - 1.0))
    b = np.tan(np.pi / 4.0 * (2.0 * np.asarray(v, dtype=np.float64) - 1.0))
    d = FACES[face, 0] + a[..., None] * FACES[face, 1] + b[..., None] * FACES[face, 2]
    return d / np.linalg.norm(d, axis=-1, keepdims=True)


def direction_to_lonlat(n):
    """Returns longitude and latitude in degrees (east and north positive) of unit `n`."""
    n = np.asarray(n, dtype=np.float64)
    lon = np.degrees(np.arctan2(n[..., 2], -n[..., 0]))
    lat = np.degrees(np.arcsin(np.clip(n[..., 1], -1.0, 1.0)))
    return lon, lat


def lonlat_to_direction(lon, lat):
    lon, lat = np.radians(lon), np.radians(lat)
    return np.stack([-np.cos(lat) * np.cos(lon), np.sin(lat), np.cos(lat) * np.sin(lon)], axis=-1)


def height_coords(level, x):
    """Face coordinate of each bordered height sample column (or row, from y).

    Vertex-centred: index BORDER is the tile's lower edge and neighbours share
    their edge samples.
    """
    i = np.arange(HEIGHT_SIZE, dtype=np.float64)
    return (x + (i - BORDER) / TILE_INTERVALS) / 2.0**level


def surface_coords(level, x):
    """Face coordinate of each bordered surface texel centre column (or row, from y)."""
    i = np.arange(SURFACE_SIZE, dtype=np.float64)
    return (x + (i - BORDER + 0.5) / TILE_INTERVALS) / 2.0**level


def tile_directions(coords, face, level, x, y):
    """Unit directions (rows by v, columns by u) of a tile's samples; `coords` is
    height_coords or surface_coords."""
    u = coords(level, x)
    v = coords(level, y)
    uu, vv = np.meshgrid(u, v)
    return face_uv_to_direction(np.full(uu.shape, face), uu, vv)


def sample_spacing(level):
    """Angle in radians between neighbouring samples at a face centre, the
    largest on the face."""
    return np.pi / 2.0 / (TILE_INTERVALS * 2**level)
