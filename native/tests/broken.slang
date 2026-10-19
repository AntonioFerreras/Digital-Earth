// Fixture: fails to compile, for the diagnostics test.

[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    undeclared_function(id);
}
