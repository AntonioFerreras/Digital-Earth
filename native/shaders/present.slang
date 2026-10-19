// Presentation: the rendered image stretched over the swapchain, then the ImGui draw lists.
// Both pipelines share these globals; clip space is y-up on every backend.

Texture2D image;
SamplerState linear_sampler;
// ImGui display coordinates to clip space: xy scale, zw translate.
uniform float4 imgui_transform;

struct BlitVarying
{
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

// One triangle covering the viewport; uv (0, 0) maps the image's first row to the top edge.
[shader("vertex")]
BlitVarying blit_vertex(uint id: SV_VertexID)
{
    const float2 uv = float2((id << 1) & 2, id & 2);
    BlitVarying result;
    result.position = float4(uv.x * 2 - 1, 1 - uv.y * 2, 0, 1);
    result.uv = uv;
    return result;
}

[shader("fragment")]
float4 blit_fragment(BlitVarying input) : SV_Target
{
    return image.Sample(linear_sampler, input.uv);
}

struct ImGuiVertex
{
    float2 position : POSITION;
    float2 uv : TEXCOORD0;
    float4 color : COLOR;
};

struct ImGuiVarying
{
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
    float4 color : COLOR;
};

[shader("vertex")]
ImGuiVarying imgui_vertex(ImGuiVertex input)
{
    ImGuiVarying result;
    result.position = float4(input.position * imgui_transform.xy + imgui_transform.zw, 0, 1);
    result.uv = input.uv;
    result.color = input.color;
    return result;
}

[shader("fragment")]
float4 imgui_fragment(ImGuiVarying input) : SV_Target
{
    return input.color * image.Sample(linear_sampler, input.uv);
}
