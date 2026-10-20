// A ray's footprint for choosing texture levels (Akenine-Moller et al. 2019, "Texture
// Level of Detail Strategies for Real-Time Ray Tracing"): its width where it starts and
// the angle it widens by.

module ray_cone;

// What a bounce off the ground and a scatter in the air add to the spread: both send
// light into wide lobes, so later lookups want coarse tiles.
public static const float SURFACE_BOUNCE_SPREAD = 0.5;
public static const float VOLUME_SCATTER_SPREAD = 1.0;

public struct RayCone
{
    // Metres.
    public float width;
    // Radians.
    public float spread;

    // The width `t` metres along the ray.
    public float width_at(float t)
    {
        return width + spread * t;
    }

    // Moves the cone's start `t` metres along the ray and widens it by `added_spread`.
    [mutating]
    public void advance(float t, float added_spread)
    {
        width = width_at(t);
        spread += added_spread;
    }
};
