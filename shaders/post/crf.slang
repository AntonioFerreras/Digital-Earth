// Film and sensor camera response functions (LUT/camera_response_functions).
// The CRF texture is 1024 x count rgba32f, one response per row, sampled
// bilinearly with clamp addressing.

module crf;

static const float CRF_RESOLUTION = 1024.0;

public struct CrfCoords
{
    // Per-channel horizontal texture coordinate.
    public float3 u;
    // Row centre of the selected response.
    public float v;
};

// The half-texel offset maps 0 onto the first texel centre; 1 lands half a
// texel past the last centre and is pulled back onto it.
public CrfCoords crf_coords(float3 display, uint index, uint count)
{
    const float offset = 0.5 / CRF_RESOLUTION;
    CrfCoords coords;
    coords.u = min(saturate(display) + offset, 1.0 - offset);
    coords.v = (float(index) + 0.5) / float(count);
    return coords;
}

public float3 camera_response(Texture2D<float4> responses, SamplerState linear_clamp, float3 display, uint index, uint count)
{
    CrfCoords c = crf_coords(display, index, count);
    return saturate(float3(
        responses.SampleLevel(linear_clamp, float2(c.u.x, c.v), 0.0).x,
        responses.SampleLevel(linear_clamp, float2(c.u.y, c.v), 0.0).y,
        responses.SampleLevel(linear_clamp, float2(c.u.z, c.v), 0.0).z));
}
