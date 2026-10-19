// Instant-NGP multiresolution hash encoding (Mueller et al. 2022) in 4D: L
// levels of two features each, quadrilinear over the 16 corners of the cell.
//
// Level l reads two uint4 of `hash_levels`: [2l] the cell count per axis (a
// lattice of cells + 1 vertices), [2l + 1] = (first entry, mask, hashed, 0).
// A dense level indexes vertex v as v.x + (n.x + 1) * (v.y + (n.y + 1) *
// (v.z + (n.z + 1) * v.w)); a hashed one as the XOR of v times the primes
// below, AND mask (2^T - 1). Entry e of `hash_table` is a feature pair, two
// halves in one 4-byte word, low half first.

module hashgrid;

import neural.encoding;

public StructuredBuffer<uint4> hash_levels;
public ByteAddressBuffer hash_table;

static const uint4 HASH_PRIMES = uint4(1u, 2654435761u, 805459861u, 3674653429u);

float2 feature_pair(uint entry)
{
    uint word = hash_table.Load(4 * entry);
    return float2(f16tof32(word), f16tof32(word >> 16));
}

public struct HashGrid<let L : int> : IEncoding
{
    public static const int FEATURES = 2 * L;

    public static void encode(float4 uvwz, out float features[FEATURES])
    {
        float4 x = saturate(uvwz);
        [ForceUnroll]
        for (int l = 0; l < L; l++)
        {
            uint4 cells = hash_levels[2 * l];
            uint4 info = hash_levels[2 * l + 1];
            float4 position = x * float4(cells);
            uint4 base = min(uint4(position), cells - 1);
            float4 w = position - float4(base);
            uint4 stride = uint4(1, cells.x + 1, (cells.x + 1) * (cells.y + 1), (cells.x + 1) * (cells.y + 1) * (cells.z + 1));
            float2 sum = float2(0.0);
            [ForceUnroll]
            for (uint corner = 0; corner < 16; corner++)
            {
                uint4 offset = uint4(corner, corner >> 1, corner >> 2, corner >> 3) & 1;
                uint4 vertex = base + offset;
                float4 axis_weight = select(offset != 0, w, 1.0 - w);
                float weight = axis_weight.x * axis_weight.y * axis_weight.z * axis_weight.w;
                uint4 hashed = vertex * HASH_PRIMES;
                uint index = info.z != 0 ? (hashed.x ^ hashed.y ^ hashed.z ^ hashed.w) & info.y : dot(vertex, stride);
                sum += weight * feature_pair(info.x + index);
            }
            features[2 * l] = sum.x;
            features[2 * l + 1] = sum.y;
        }
    }
}
