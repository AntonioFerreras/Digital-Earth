// K-Planes (Fridovich-Keil et al. 2023) in 4D: per scale, six planes over the
// axis pairs (0,1) (0,2) (0,3) (1,2) (1,3) (2,3), each a lattice of
// n[a] x n[b] vertices with F features, bilinear with aligned corners (vertex
// i sits at i / (n - 1)). The six samples multiply feature-wise and the
// KPLANES_SCALES scales concatenate: features [s * F, (s + 1) * F) come from
// scale s.
//
// Scale s is the Texture2DArray `kplanes_scale<s>` of side n_max, its largest
// vertex count, with `kplanes_vertices[s]` vertices per axis; layer
// plane * F / 4 + q holds features 4q..4q+3 with u along the plane's first
// axis. The sampler's bilinear filter interpolates, at the texture unit's
// fixed-point weight precision.

module kplanes;

import neural.encoding;
import shared.bindings;

public static const int KPLANES_SCALES = 3;
public uniform uint4 kplanes_vertices[KPLANES_SCALES];
public Texture2DArray<float4> kplanes_scale0;
public Texture2DArray<float4> kplanes_scale1;
public Texture2DArray<float4> kplanes_scale2;

static const uint PLANES = 6;
static const uint2 PLANE_AXES[PLANES] = { uint2(0, 1), uint2(0, 2), uint2(0, 3), uint2(1, 2), uint2(1, 3), uint2(2, 3) };

float4 sample_scale(uint scale, float3 uv_layer)
{
    if (scale == 0)
        return kplanes_scale0.SampleLevel(linear_clamp, uv_layer, 0.0);
    if (scale == 1)
        return kplanes_scale1.SampleLevel(linear_clamp, uv_layer, 0.0);
    return kplanes_scale2.SampleLevel(linear_clamp, uv_layer, 0.0);
}

public struct KPlanes<let F : int> : IEncoding
{
    public static const int FEATURES = KPLANES_SCALES * F;

    public static void encode(float4 uvwz, out float features[FEATURES])
    {
        static_assert(F % 4 == 0, "textures pack four features per texel");
        float4 x = saturate(uvwz);
        [ForceUnroll]
        for (int s = 0; s < KPLANES_SCALES; s++)
        {
            uint4 vertices = kplanes_vertices[s];
            float4 t = x * float4(vertices - 1);
            float size = float(max(max(vertices.x, vertices.y), max(vertices.z, vertices.w)));
            float product[F];
            [ForceUnroll]
            for (int f = 0; f < F; f++)
                product[f] = 1.0;
            [ForceUnroll]
            for (uint plane = 0; plane < PLANES; plane++)
            {
                uint2 axes = PLANE_AXES[plane];
                float2 uv = (float2(t[axes.x], t[axes.y]) + 0.5) / size;
                [ForceUnroll]
                for (int q = 0; q < F / 4; q++)
                {
                    float4 v = sample_scale(s, float3(uv, plane * (F / 4) + q));
                    [ForceUnroll]
                    for (int c = 0; c < 4; c++)
                        product[4 * q + c] *= v[c];
                }
            }
            [ForceUnroll]
            for (int f = 0; f < F; f++)
                features[s * F + f] = product[f];
        }
    }
}
