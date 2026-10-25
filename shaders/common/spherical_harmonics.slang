// Real spherical harmonics through order 2, and through order 3 (Sloan 2008, "Stupid
// Spherical Harmonics Tricks", appendix A2), orthonormal over the sphere.

module spherical_harmonics;

public static const uint SH_COUNT = 9;
// Through order 3.
public static const uint SH3_COUNT = 16;

public float[SH_COUNT] sh_basis(float3 dir)
{
    float x = dir.x;
    float y = dir.y;
    float z = dir.z;
    return {
        0.2820948,
        0.4886025 * y,
        0.4886025 * z,
        0.4886025 * x,
        1.0925484 * x * y,
        1.0925484 * y * z,
        0.3153916 * (3.0 * z * z - 1.0),
        1.0925484 * x * z,
        0.5462742 * (x * x - y * y),
    };
}

public float[SH3_COUNT] sh3_basis(float3 dir)
{
    float x = dir.x;
    float y = dir.y;
    float z = dir.z;
    float order2[SH_COUNT] = sh_basis(dir);
    return {
        order2[0], order2[1], order2[2], order2[3], order2[4], order2[5], order2[6], order2[7], order2[8],
        0.5900436 * y * (3.0 * x * x - y * y),
        2.8906114 * x * y * z,
        0.4570458 * y * (5.0 * z * z - 1.0),
        0.3731763 * z * (5.0 * z * z - 3.0),
        0.4570458 * x * (5.0 * z * z - 1.0),
        1.4453057 * z * (x * x - y * y),
        0.5900436 * x * (x * x - 3.0 * y * y),
    };
}
