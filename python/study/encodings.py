"""Encodings of uvwz in [0, 1]^4 and the decoder heads behind them.

Each encoding computes exactly what its shader in shaders/neural/ evaluates
(fourier.slang, hashgrid.slang, kplanes.slang, lut4d.slang), in float32, so a
trained model ships without re-deriving anything: same feature order, hash,
per-axis resolution schedule, plane layout, and interpolation convention.
"""

import math

import torch
from torch import nn
from torch.nn import functional

AXES = 4
# hashgrid.slang's HASH_PRIMES; products wrap at 32 bits.
HASH_PRIMES = (1, 2654435761, 805459861, 3674653429)
# Coarsest and finest cells per axis (nu, mu_s, mu, r): mu holds the horizon
# discontinuity, nu only the smooth phase functions.
HASH_CELLS_MIN = (4, 8, 8, 4)
HASH_CELLS_MAX = (32, 512, 1024, 64)
# K-Planes vertices per axis relative to a scale's base resolution, halved.
KPLANES_AXIS_SCALE = (1, 2, 4, 2)
KPLANES_SCALES = 3
PLANE_AXES = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


class GatherWeighted(torch.autograd.Function):
    """sum_k weight[n, k] * table[index[n, k]]; the backward scatters with atomics
    (index_add_) instead of autograd's sort-based index_put_."""

    @staticmethod
    def forward(ctx, table: torch.Tensor, index: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(index, weight)
        ctx.table_shape = table.shape
        return torch.einsum("nkf,nk->nf", table[index], weight)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        index, weight = ctx.saved_tensors
        grad_table = torch.zeros(ctx.table_shape, dtype=grad.dtype, device=grad.device)
        contributions = weight[:, :, None] * grad[:, None, :]
        grad_table.index_add_(0, index.reshape(-1), contributions.reshape(-1, ctx.table_shape[1]))
        return grad_table, None, None


def corner_select(per_axis: torch.Tensor, combine) -> torch.Tensor:
    """Combines per-axis values [..., 4, 2] (lower, upper vertex) over the 16 corners
    of a 4D cell into [..., 16], corner bit a selecting axis a's upper vertex."""
    assert per_axis.shape[-2:] == (AXES, 2)
    x, y, z, w = (per_axis[..., a, :] for a in range(AXES))
    result = combine(combine(w[..., :, None, None, None], z[..., None, :, None, None]),
                     combine(y[..., None, None, :, None], x[..., None, None, None, :]))
    return result.flatten(-4)


class GridLevels(nn.Module):
    """Quadrilinear lookups in a stack of 4D vertex lattices sharing one table.

    Level l has `cells[l]` cells per axis; vertex v of a level sits at v / cells.
    A level whose vertex count fits `table_size` is stored densely (x fastest),
    else hashed into `table_size` entries, as in hashgrid.slang.
    """

    def __init__(self, cells: list[tuple[int, ...]], table_size: int, features: int, init_scale: float):
        super().__init__()
        assert len(cells) > 0
        assert table_size & (table_size - 1) == 0
        first, hashed, strides = [], [], []
        entry_count = 0
        for level_cells in cells:
            assert len(level_cells) == AXES and min(level_cells) >= 1
            vertices = [n + 1 for n in level_cells]
            vertex_count = math.prod(vertices)
            first.append(entry_count)
            hashed.append(vertex_count > table_size)
            strides.append([1, vertices[0], vertices[0] * vertices[1], vertices[0] * vertices[1] * vertices[2]])
            entry_count += table_size if hashed[-1] else vertex_count
        assert entry_count < 2**31
        self.table = nn.Parameter(torch.empty(entry_count, features).uniform_(-init_scale, init_scale))
        self.register_buffer("cells", torch.tensor(cells, dtype=torch.float32), persistent=False)
        self.register_buffer("first", torch.tensor(first, dtype=torch.int64), persistent=False)
        self.register_buffer("hashed", torch.tensor(hashed), persistent=False)
        self.register_buffer("strides", torch.tensor(strides, dtype=torch.int64), persistent=False)
        self.register_buffer("primes", torch.tensor(HASH_PRIMES, dtype=torch.int64), persistent=False)
        self.cells_list = [tuple(level_cells) for level_cells in cells]
        self.table_size = table_size
        self.level_count = len(cells)
        self.features = features
        self.output_features = self.level_count * features

    def corners(self, uvwz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Table indices and weights [B * L, 16] of every level's 16 cell corners."""
        assert uvwz.dim() == 2 and uvwz.shape[1] == AXES
        position = uvwz.clamp(0.0, 1.0)[:, None, :] * self.cells  # [B, L, 4]
        base = torch.minimum(position.floor(), self.cells - 1)
        fraction = position - base
        vertex = base.long()[..., None] + torch.arange(2, device=uvwz.device)  # [B, L, 4, 2]
        hashed_terms = (vertex * self.primes[:, None]) & (self.table_size - 1)
        hashed_index = corner_select(hashed_terms, torch.bitwise_xor)
        dense_index = corner_select(vertex * self.strides[:, :, None], torch.add)
        index = torch.where(self.hashed[:, None], hashed_index, dense_index) + self.first[:, None]
        weight = corner_select(torch.stack([1.0 - fraction, fraction], dim=-1), torch.mul)
        return index.reshape(-1, 16), weight.reshape(-1, 16)

    def forward(self, uvwz: torch.Tensor) -> torch.Tensor:
        index, weight = self.corners(uvwz)
        return GatherWeighted.apply(self.table, index, weight).reshape(uvwz.shape[0], self.output_features)


def hash_cells(levels: int, cells_max: tuple[int, ...]) -> list[tuple[int, ...]]:
    """Cells per axis of each level: geometric from HASH_CELLS_MIN to `cells_max`."""
    assert levels >= 2
    assert len(cells_max) == AXES
    cells = []
    for level in range(levels):
        t = level / (levels - 1)
        cells.append(tuple(int(math.floor(low * (high / low) ** t + 1e-9))
                           for low, high in zip(HASH_CELLS_MIN, cells_max)))
    return cells


def hash_grid(levels: int, log2_table: int, cells_max: tuple[int, ...]) -> GridLevels:
    """Instant-NGP multiresolution hash encoding: two features per level, initialized
    as Mueller et al. 2022 do."""
    return GridLevels(hash_cells(levels, cells_max), 2**log2_table, features=2, init_scale=1e-4)


def dense_grid(vertices: tuple[int, ...]) -> GridLevels:
    """A dense RGB table with vertex i of an axis at i / (n - 1), as lut4d.slang samples it."""
    return GridLevels([tuple(n - 1 for n in vertices)], 2**31, features=3, init_scale=0.0)


# bruneton.slang: the uvwz scattering_uvwz reaches at the ends of each unit range.
V_LOW = 0.5 / 4096
Z_LOW = 0.5 / 4096
W_GROUND_LOW = 0.25 / 2048


class BrunetonGrid(nn.Module):
    """A dense RGB table in atmosphere/lut4d.slang's vertex convention: vertex k of an
    axis at fraction k / (N - 1) of the uvwz range scattering_uvwz reaches, and mu as
    two halves of N / 2 vertices (ground rays horizon to nadir, then sky rays zenith
    to horizon) that interpolation never crosses. Table rows are u fastest, as the
    datagen bake orders its records."""

    def __init__(self, vertices: tuple[int, ...]):
        super().__init__()
        assert len(vertices) == AXES
        assert vertices[2] % 2 == 0
        assert min(vertices) >= 2
        self.vertices = tuple(vertices)
        self.table = nn.Parameter(torch.zeros(math.prod(vertices), 3))
        n = vertices
        self.strides = (1, n[0], n[0] * n[1], n[0] * n[1] * n[2])
        self.level_count = 1
        self.output_features = 3

    def corners(self, uvwz: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Table indices and weights [B, 16] of the 16 cell corners."""
        assert uvwz.dim() == 2 and uvwz.shape[1] == AXES
        n = self.vertices
        half = n[2] // 2

        def fraction(u: torch.Tensor, low: float, high: float) -> torch.Tensor:
            return ((u - low) / (high - low)).clamp(0.0, 1.0)

        sky = uvwz[:, 2] >= 0.5
        mu = torch.where(sky, half + fraction(uvwz[:, 2], 0.5 + W_GROUND_LOW, 1.0 - W_GROUND_LOW) * (half - 1),
                         fraction(uvwz[:, 2], W_GROUND_LOW, 0.5 - W_GROUND_LOW) * (half - 1))
        position = torch.stack([fraction(uvwz[:, 0], 0.0, 1.0) * (n[0] - 1),
                                fraction(uvwz[:, 1], V_LOW, 1.0 - V_LOW) * (n[1] - 1),
                                mu,
                                fraction(uvwz[:, 3], Z_LOW, 1.0 - Z_LOW) * (n[3] - 1)], dim=1)
        base_max = torch.tensor([n[0] - 2, n[1] - 2, 0, n[3] - 2], dtype=position.dtype, device=position.device)
        base_max = base_max.expand_as(position).clone()
        base_max[:, 2] = torch.where(sky, 2 * half - 2, half - 2).to(position.dtype)
        base = torch.minimum(position.floor(), base_max)
        fraction_in_cell = position - base
        vertex = base.long()[..., None] + torch.arange(2, device=uvwz.device)  # [B, 4, 2]
        strides = torch.tensor(self.strides, device=uvwz.device)
        index = corner_select(vertex * strides[:, None], torch.add)
        weight = corner_select(torch.stack([1.0 - fraction_in_cell, fraction_in_cell], dim=-1), torch.mul)
        return index, weight

    def forward(self, uvwz: torch.Tensor) -> torch.Tensor:
        index, weight = self.corners(uvwz)
        return GatherWeighted.apply(self.table, index, weight)


class Fourier(nn.Module):
    """The raw coordinates, then per octave k and axis a sin and cos of
    2 pi frac(uvwz[a] * frequency[a] * 2^k), interleaved as fourier.slang writes them."""

    def __init__(self, octaves: int, frequency: tuple[float, ...] = (0.5, 0.5, 0.5, 0.5)):
        super().__init__()
        assert octaves >= 1
        assert len(frequency) == AXES
        self.octaves = octaves
        self.frequency_list = frequency
        self.register_buffer("frequency", torch.tensor(frequency, dtype=torch.float32), persistent=False)
        self.output_features = AXES + 2 * AXES * octaves

    def forward(self, uvwz: torch.Tensor) -> torch.Tensor:
        cycles = uvwz * self.frequency
        parts = [uvwz]
        for k in range(self.octaves):
            phase = 2.0 * math.pi * torch.frac(cycles * float(2**k))
            parts.append(torch.stack([torch.sin(phase), torch.cos(phase)], dim=-1).flatten(1))
        return torch.cat(parts, dim=1)


class KPlanes(nn.Module):
    """K-Planes (Fridovich-Keil et al. 2023) in 4D: per scale, six planes over the axis
    pairs with F features, bilinear with aligned corners (vertex i at i / (n - 1)),
    multiplied feature-wise; scales concatenate."""

    def __init__(self, features: int, base: int, init_low: float, init_high: float):
        super().__init__()
        assert features % 4 == 0
        self.features = features
        self.output_features = KPLANES_SCALES * features
        self.base = base
        self.vertices = [tuple((base << s) * axis_scale // 2 for axis_scale in KPLANES_AXIS_SCALE)
                         for s in range(KPLANES_SCALES)]
        # Each plane is [1, F, n_b, n_a]: rows along the second axis, as grid_sample's (x, y) = (a, b).
        self.planes = nn.ParameterList(
            nn.Parameter(torch.empty(1, features, vertices[b], vertices[a]).uniform_(init_low, init_high))
            for vertices in self.vertices for a, b in PLANE_AXES)

    def forward(self, uvwz: torch.Tensor) -> torch.Tensor:
        grid = 2.0 * uvwz.clamp(0.0, 1.0) - 1.0
        scales = []
        for s in range(KPLANES_SCALES):
            product = None
            for p, (a, b) in enumerate(PLANE_AXES):
                plane = self.planes[s * len(PLANE_AXES) + p]
                sampled = functional.grid_sample(plane, grid[None, :, None, [a, b]], mode="bilinear",
                                                 padding_mode="border", align_corners=True)
                sampled = sampled[0, :, :, 0].T  # [B, F]
                product = sampled if product is None else product * sampled
            scales.append(product)
        return torch.cat(scales, dim=1)


class Raw(nn.Module):
    output_features = AXES

    def forward(self, uvwz: torch.Tensor) -> torch.Tensor:
        return uvwz


ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU}


def head(in_features: int, width: int, hidden: int, activation: str) -> nn.Sequential:
    """in_features -> hidden layers of `width` -> 3, as mlp.slang's Mlp; hidden = 0 is linear."""
    assert hidden >= 0
    layers: list[nn.Module] = []
    previous = in_features
    for _ in range(hidden):
        layers += [nn.Linear(previous, width), ACTIVATIONS[activation]()]
        previous = width
    layers.append(nn.Linear(previous, 3))
    return nn.Sequential(*layers)


class Model(nn.Module):
    """Optional u_mu rotation, encoding, optional head, output activation.

    Without a head (dense grids) the encoding's features are the RGB output."""

    def __init__(self, encoding: nn.Module, head: nn.Sequential | None, softplus: bool, mu_rotation: bool):
        super().__init__()
        self.encoding = encoding
        self.head = head
        self.softplus = softplus
        self.mu_rotation = mu_rotation

    def encoding_input(self, uvwz: torch.Tensor) -> torch.Tensor:
        if self.mu_rotation:
            return torch.cat([uvwz[:, :2], torch.frac(uvwz[:, 2:3] + 0.5), uvwz[:, 3:]], dim=1)
        return uvwz

    def forward(self, uvwz: torch.Tensor) -> torch.Tensor:
        features = self.encoding(self.encoding_input(uvwz))
        rgb = features if self.head is None else self.head(features)
        return functional.softplus(rgb) if self.softplus else rgb


def build(spec: dict) -> Model:
    """The model a run spec describes.

    `mu_rotation` maps u_mu to frac(u_mu + 0.5): the zenith/nadir discontinuity at
    u_mu = 0.5 (rays that miss the ground vs rays that hit it) moves onto the domain
    edges, where no interpolation crosses it, and the two horizon ends meet in the middle."""
    encoding_spec = spec["encoding"]
    kind = encoding_spec["kind"]
    if kind == "raw":
        encoding: nn.Module = Raw()
    elif kind == "fourier":
        encoding = Fourier(encoding_spec["octaves"])
    elif kind == "hash":
        encoding = hash_grid(encoding_spec["levels"], encoding_spec["log2_table"],
                             tuple(encoding_spec.get("cells_max", HASH_CELLS_MAX)))
    elif kind == "kplanes":
        encoding = KPlanes(encoding_spec["features"], encoding_spec["base"],
                           encoding_spec["init_low"], encoding_spec["init_high"])
    elif kind == "bruneton":
        assert not spec.get("mu_rotation", False), "the table splits mu in halves itself"
        encoding = BrunetonGrid(tuple(encoding_spec["vertices"]))
    else:
        assert kind == "dense", kind
        encoding = dense_grid(tuple(encoding_spec["vertices"]))
    decoder = None
    if kind not in ("dense", "bruneton"):
        head_spec = spec["head"]
        decoder = head(encoding.output_features, head_spec["width"], head_spec["hidden"], head_spec["activation"])
    assert spec["output"] in ("identity", "softplus")
    return Model(encoding, decoder, spec["output"] == "softplus", spec.get("mu_rotation", False))


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
