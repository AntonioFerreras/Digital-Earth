import tests.harness;
import spectral.cie;
import spectral.planck;
import spectral.uplift;

// lut: LUT/CIE.dat as f32.
[shader("compute")]
[numthreads(GROUP_SIZE, 1, 1)]
void test_sample_wavelength(uint3 id: SV_DispatchThreadID)
{
    Case c = Case(id.x, 1, 5);
    if (!c.valid)
        return;
    WavelengthSample s = sample_wavelength(lut, inputs[c.input_base]);
    c.store(0, float4(s.wavelength, s.cmf));
    outputs[c.output_base + 4] = s.rcp_pdf;
}

// lut: LUT/CIE.dat as f32.
[shader("compute")]
[numthreads(GROUP_SIZE, 1, 1)]
void test_cie_cmf(uint3 id: SV_DispatchThreadID)
{
    Case c = Case(id.x, 1, 3);
    if (!c.valid)
        return;
    c.store(0, cie_cmf(lut, inputs[c.input_base]));
}

[shader("compute")]
[numthreads(GROUP_SIZE, 1, 1)]
void test_xyz_to_rec709(uint3 id: SV_DispatchThreadID)
{
    Case c = Case(id.x, 3, 3);
    if (!c.valid)
        return;
    c.store(0, xyz_to_rec709(c.load<3>(0)));
}

// lut: LUT/srgb2spec.dat converted from f16 to f32.
[shader("compute")]
[numthreads(GROUP_SIZE, 1, 1)]
void test_srgb_to_spectrum(uint3 id: SV_DispatchThreadID)
{
    Case c = Case(id.x, 4, 1);
    if (!c.valid)
        return;
    outputs[c.output_base] = srgb_to_spectrum(lut, c.load<3>(0), inputs[c.input_base + 3]);
}

[shader("compute")]
[numthreads(GROUP_SIZE, 1, 1)]
void test_planck(uint3 id: SV_DispatchThreadID)
{
    Case c = Case(id.x, 2, 1);
    if (!c.valid)
        return;
    outputs[c.output_base] = planck(inputs[c.input_base], inputs[c.input_base + 1]);
}
