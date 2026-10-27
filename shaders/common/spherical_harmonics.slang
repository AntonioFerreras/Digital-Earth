// Real spherical harmonics, orthonormal over the sphere, at index l (l + 1) + m, by
// recurrence through order 5.

module spherical_harmonics;

public static const uint SH2_COUNT = 9;
public static const uint SH3_COUNT = 16;
public static const uint SH5_COUNT = 36;

// sqrt((2 l + 1) / 4 pi (l - m)! / (l + m)!), and sqrt 2 more for m > 0, at index
// l (l + 1) / 2 + m.
static const float SH_SCALE[21] = {
    0.2820948, 0.4886025, 0.4886025, 0.6307831, 0.3641828, 0.1820914, 0.7463527,
    0.3046972, 0.0963537, 0.0393362, 0.8462844, 0.2676186, 0.0630783, 0.0168584,
    0.0059603, 0.9356026, 0.2415715, 0.0456527, 0.0093188, 0.0021965, 0.0006946,
};

// The associated Legendre polynomials' recurrence in l with sin^m theta divided out,
// which the real and imaginary parts of (x + i y)^m supply with cos and sin of m phi,
// so that no angle is ever taken.
public float[(ORDER + 1) * (ORDER + 1)] sh_basis<let ORDER : uint>(float3 dir)
{
    float basis[(ORDER + 1) * (ORDER + 1)];
    float z = dir.z;
    float re = 1.0;
    float im = 0.0;
    float diagonal = 1.0;
    for (uint m = 0; m <= ORDER; m++)
    {
        float previous = 0.0;
        float current = 0.0;
        for (uint l = m; l <= ORDER; l++)
        {
            float p = l == m ? diagonal : (float(2 * l - 1) * z * current - float(l + m - 1) * previous) / float(l - m);
            previous = current;
            current = p;
            float value = SH_SCALE[l * (l + 1) / 2 + m] * p;
            basis[l * (l + 1) + m] = value * re;
            if (m > 0)
                basis[l * (l + 1) - m] = value * im;
        }
        float rotated = re * dir.x - im * dir.y;
        im = re * dir.y + im * dir.x;
        re = rotated;
        diagonal *= float(2 * m + 1);
    }
    return basis;
}
