// The global parameters of every render-mode kernel and the per-pixel helpers
// built on them. A mode kernel imports this module instead of declaring its own
// camera, scene, accumulation, or LUT parameters; src/render/ModeKernel.zig
// binds every global below by name on each dispatch.
//
// Pixel convention: `pixel` counts from the bottom-left corner, as the Taichi
// renderer did, so `accum` row 0 is the bottom of the image. post/display.slang
// is the one place that flips rows into the top-down display image.

module bindings;

import common.rng;
import shared.uniforms;

// Every mode kernel runs MODE_GROUP_SIZE x MODE_GROUP_SIZE threads per group,
// one thread per pixel.
public static const uint MODE_GROUP_SIZE = 8;

public ConstantBuffer<FrameUniforms> frame;
public ConstantBuffer<SceneUniforms> scene;
// Sum of the linear Rec.709 samples since the last reset; `w` is unused.
[format("rgba32f")]
public RWTexture2D<float4> accum;
// LUT/CIE.dat, see spectral/cie.slang.
public StructuredBuffer<float> cie;
// LUT/srgb2spec.dat widened to f32, see spectral/uplift.slang.
public StructuredBuffer<float> srgb_to_spectrum_basis;
// LUT/ozone_cross_section.dat and LUT/ozone_profile.dat, see atmosphere/media.slang.
public StructuredBuffer<float> ozone_cross_section;
public StructuredBuffer<float> ozone_profile;
// The column densities of atmosphere/scattering_luts.slang.
public Texture2D<float4> column_density_lut;
public SamplerState linear_clamp;
// Earth maps wrap around the globe in u and clamp at the poles in v.
public SamplerState linear_repeat_u;

// One sample of the pixel a mode thread owns: `valid` is false for the threads of
// the last group row or column that fall outside the image.
public struct PixelSample
{
    public uint2 pixel;
    public bool valid;
    // Keyed by the frame index since the reset plus the seed; dimension 0 is the
    // pixel jitter.
    public Rng rng;

    public __init(uint2 thread)
    {
        pixel = thread;
        valid = all(thread < frame.frame.yz);
        rng = Rng(thread, frame.frame.x + frame.frame.w);
    }

    // Unit direction of the camera ray through the jittered pixel. The image
    // plane at unit distance spans camera_forward.w vertically either side of
    // the centre and camera_right.w times that horizontally.
    [mutating]
    public float3 camera_ray_direction()
    {
        float2 jitter = rng.next4().xy;
        float2 plane = 2.0 * (float2(pixel) + jitter) / float2(frame.frame.yz) - 1.0;
        float half_height = frame.camera_forward.w;
        float half_width = half_height * frame.camera_right.w;
        return normalize(frame.camera_forward.xyz + plane.x * half_width * frame.camera_right.xyz +
                         plane.y * half_height * frame.camera_up.xyz);
    }

    // The angle one pixel spans at the image centre.
    public float pixel_spread()
    {
        return 2.0 * frame.camera_forward.w / float(frame.frame.z);
    }

    // Adds this frame's linear Rec.709 sample; frame 0 overwrites, so a reset needs no clear.
    public void accumulate(float3 rgb)
    {
        float3 sum = frame.frame.x == 0 ? float3(0.0) : accum[pixel].rgb;
        accum[pixel] = float4(sum + rgb, 1.0);
    }
};
