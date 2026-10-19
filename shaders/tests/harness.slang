// Golden-test I/O: case i reads `input_stride` floats at inputs[i * input_stride]
// and writes `output_stride` floats at outputs[i * output_stride] for
// i < case_count; the host dispatches ceil(case_count / GROUP_SIZE) groups.
// `lut` carries a set's lookup table when it has one.

module harness;

public StructuredBuffer<float> inputs;
public RWStructuredBuffer<float> outputs;
public StructuredBuffer<float> lut;
// Metal has no buffer-length query, so the host passes the count.
public uniform uint case_count;

public static const uint GROUP_SIZE = 64;

public struct Case
{
    public uint input_base;
    public uint output_base;
    public bool valid;

    public __init(uint index, uint input_stride, uint output_stride)
    {
        valid = index < case_count;
        input_base = index * input_stride;
        output_base = index * output_stride;
    }

    public vector<float, N> load<let N : int>(uint offset)
    {
        vector<float, N> v;
        for (int k = 0; k < N; k++)
            v[k] = inputs[input_base + offset + k];
        return v;
    }

    public void store<let N : int>(uint offset, vector<float, N> v)
    {
        for (int k = 0; k < N; k++)
            outputs[output_base + offset + k] = v[k];
    }
};
