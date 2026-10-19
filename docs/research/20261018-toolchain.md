# Toolchain: Zig + Slang + slang-rhi + pixi (as of 2026-10-18)

## Versions

| Component | Pin | Notes |
| --- | --- | --- |
| Zig | 0.17.0 (conda-forge `zig`, main label) | released 2026-10-01. `@cImport` removed; `b.addTranslateC` deprecated in favor of the `translate_c` package; `b.build_root` -> `b.root`. 0.16 moved I/O onto `std.Io` (`main(init: std.process.Init)`) |
| slang-rhi | commit SHA `d825f483` (main, 2026-10-18) | no tags, no ABI guarantee: headers and binaries from one revision |
| Slang | fetched by slang-rhi (`SLANG_RHI_FETCH_SLANG_VERSION`, 2026.17.1 at that SHA) | latest is v2026.20 (2026-10-17); conda-forge `shader-slang` is stale (2026.14) |
| CMake / Ninja | conda-forge `cmake`, `ninja` | slang-rhi needs CMake >= 3.25, C++20 |
| C++ compiler | conda-forge `cxx-compiler` | |
| SDL3 | conda-forge `sdl3` 3.4.18 | native handles via `SDL_GetWindowProperties` |
| Dear ImGui | 1.93.0 via dear_bindings v0.26 (`dcimgui`) | C API generated for Zig |
| Vulkan | `libvulkan-loader`, `libvulkan-headers`, `vulkan-validation-layers`, `vulkan-tools` 1.4.357.0 | `vulkan-loader`/`vulkan-headers` are old names. ICD from the system driver |

## slang-rhi API (namespace `rhi`, `include/slang-rhi.h`)

- COM-style interfaces (`ISlangUnknown`, `Result`, `ComPtr`); only C exports
  are `rhiGetInstance()` / `rhiDestroyInstance()`.
- Device: `getRHI()->createDevice(DeviceDesc{deviceType, enableValidation,
  debugCallback, slang{searchPaths, ...}, ...})`;
  `DeviceType { Default, D3D11, D3D12, Vulkan, Metal, CPU, CUDA, WGPU }`.
- Shaders: `device->getSlangSession` -> `loadModule` ->
  `findEntryPointByName` -> `createCompositeComponentType` ->
  `createShaderProgram(ShaderProgramDesc)` -> `createComputePipeline`.
- Buffers: `BufferDesc{size, elementSize, format, memoryType, usage}`,
  `MemoryType {DeviceLocal, Upload, ReadBack}`; `createBuffer`,
  `readBuffer`, `ICommandEncoder::uploadBufferData`.
- Textures: `TextureDesc{type, size, arrayLength, mipCount, format, usage}`;
  `uploadTextureData`, `readTexture`; `createSampler(SamplerDesc)`.
- Binding: `ShaderCursor(rootObject)["name"]`, `.setData`, `.setBinding`.
- Commands: `getQueue(QueueType::Graphics)` -> `createCommandEncoder` ->
  `beginComputePass` -> `bindPipeline` (returns root `IShaderObject*`) ->
  `dispatchCompute` -> `end` -> `finish` -> `submit`; `waitOnHost`.
- Surface: `createSurface(WindowHandle::fromXlibWindow | fromHwnd |
  fromNSWindow)`, `configure`, `acquireNextImage`, `present`. No Wayland
  handle: run SDL with the X11 video driver (XWayland).
- Build: `SLANG_RHI_BUILD_SHARED` (default OFF), `SLANG_RHI_BUILD_TESTS`,
  `SLANG_RHI_BUILD_EXAMPLES`, `SLANG_RHI_ENABLE_{VULKAN,METAL,D3D12,CUDA,...}`,
  `SLANG_RHI_FETCH_SLANG`.

## ABI

Zig's `c++` uses its own libc++; a GCC-built slang-rhi uses libstdc++. Two C++
runtimes in one process are fragile, and Metal needs Objective-C++. Therefore
the C++ shim, slang-rhi, and ImGui are built by one CMake project into one
shared library exporting only `extern "C"`; Zig links that library and never
compiles C++.

## References

- https://github.com/shader-slang/slang-rhi
- https://github.com/shader-slang/slang/releases/tag/v2026.20
- https://ziglang.org/download/0.17.0/release-notes.html
- https://github.com/dearimgui/dear_bindings/releases
- sgl's ImGui renderer (Apache-2.0 WITH LLVM-exception):
  https://github.com/shader-slang/slangpy/blob/main/src/sgl/ui/ui.cpp
