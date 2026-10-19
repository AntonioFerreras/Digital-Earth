// The input encoding in front of a network: uvwz in [0, 1]^4 to FEATURES floats.

module encoding;

public interface IEncoding
{
    static const int FEATURES;
    static void encode(float4 uvwz, out float features[FEATURES]);
}

// uvwz itself, for networks that take the raw coordinates.
public struct Raw : IEncoding
{
    public static const int FEATURES = 4;

    public static void encode(float4 uvwz, out float features[FEATURES])
    {
        [ForceUnroll]
        for (int a = 0; a < 4; a++)
            features[a] = uvwz[a];
    }
}
