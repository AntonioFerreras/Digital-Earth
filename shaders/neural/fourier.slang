// NeRF-style frequency encoding: the raw coordinates, then per octave k and
// axis a the pair sin(2 pi phase), cos(2 pi phase) with phase =
// uvwz[a] * fourier_frequency[a] * 2^k, at features 4 + 8k + 2a and 4 + 8k + 2a + 1.

module fourier;

import common.constants;
import neural.encoding;

// The per-axis octave-0 frequency in cycles per unit.
public uniform float4 fourier_frequency;

public struct Fourier<let L : int> : IEncoding
{
    public static const int FEATURES = 4 + 8 * L;

    public static void encode(float4 uvwz, out float features[FEATURES])
    {
        float4 cycles = uvwz * fourier_frequency;
        [ForceUnroll]
        for (int a = 0; a < 4; a++)
            features[a] = uvwz[a];
        [ForceUnroll]
        for (int k = 0; k < L; k++)
        {
            // Scaling by 2^k is exact, so the reduction to one period loses
            // nothing and sin and cos see arguments below 2 pi at every octave.
            float4 phase = 2.0 * PI * frac(cycles * float(1 << k));
            float4 s, c;
            sincos(phase, s, c);
            [ForceUnroll]
            for (int a = 0; a < 4; a++)
            {
                features[4 + 8 * k + 2 * a] = s[a];
                features[4 + 8 * k + 2 * a + 1] = c[a];
            }
        }
    }
}
