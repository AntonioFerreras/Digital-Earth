// Scene-wide constants. Lengths are metres measured from the planet centre,
// wavelengths are nanometres, and extinction coefficients are per metre.

module constants;

public static const float PI = 3.14159265358979;
public static const float FLOAT_MAX = 3.402823466e38;

public static const float PLANET_RADIUS = 6371e3;
public static const float ATMOSPHERE_HEIGHT = 110e3;
public static const float ATMOSPHERE_RADIUS = PLANET_RADIUS + ATMOSPHERE_HEIGHT;

public static const float CLOUD_BASE_RADIUS = PLANET_RADIUS + 4000.0;
public static const float CLOUD_THICKNESS = 6000.0;
public static const float CLOUD_TOP_RADIUS = CLOUD_BASE_RADIUS + CLOUD_THICKNESS;

// Solar radius over the Earth-Sun distance.
public static const float SUN_ANGULAR_RADIUS = 6.95e8 / 1.4959e11;
public static const float SUN_TEMPERATURE = 5778.0;
public static const float NIGHT_LIGHT_TEMPERATURE = 2700.0;

// Spectral domain: 441 one-nanometre bins from 390 nm to 830 nm inclusive.
public static const float WAVELENGTH_MIN = 390.0;
public static const uint WAVELENGTH_COUNT = 441;

// Where `wavelength` falls in a table with one entry per nanometre from 390 nm:
// the entry below and the weight of the one above, clamped to [390, 830] nm.
public struct WavelengthTablePosition
{
    public uint index;
    public float weight;

    public __init(float wavelength)
    {
        float x = clamp(wavelength - WAVELENGTH_MIN, 0.0, float(WAVELENGTH_COUNT - 1));
        index = min(uint(x), WAVELENGTH_COUNT - 2);
        weight = x - float(index);
    }
};
