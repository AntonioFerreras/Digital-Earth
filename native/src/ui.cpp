#include "internal.h"

#include <imgui.h>
#include <imgui_impl_sdl3.h>
#include <slang-rhi/shader-cursor.h>

#include <SDL3/SDL_events.h>

#include <algorithm>
#include <array>
#include <bit>

using rhi::ComPtr;

struct en_surface {
    en_device *device;
    ComPtr<rhi::ISurface> surface;
    rhi::Format format;
    ComPtr<rhi::IRenderPipeline> blit;
    ComPtr<rhi::IRenderPipeline> imgui;
    ComPtr<rhi::ISampler> linear;
    // ImGui geometry, grown geometrically and reused while the sizes are stable.
    ComPtr<rhi::IBuffer> vertices;
    ComPtr<rhi::IBuffer> indices;
    uint32_t width = 0;
    uint32_t height = 0;
};

namespace {

static_assert(sizeof(ImDrawIdx) == 2, "index format below is Uint16");

// ImGui supports one context per process, so this is process-wide too.
bool imgui_frame_begun = false;

ComPtr<rhi::IRenderPipeline> make_pipeline(rhi::IDevice &device, const char *vertex,
                                           const char *fragment, rhi::IInputLayout *input_layout,
                                           const rhi::ColorTargetDesc &target) {
    const char *entries[] = {vertex, fragment};
    ComPtr<rhi::IShaderProgram> program = en::link_program(device, "present", entries);
    rhi::RenderPipelineDesc desc;
    desc.program = program;
    desc.inputLayout = input_layout;
    desc.targets = &target;
    desc.targetCount = 1;
    desc.depthStencil.depthWriteEnable = false;
    desc.rasterizer.scissorEnable = true;
    desc.label = vertex;
    ComPtr<rhi::IRenderPipeline> pipeline;
    en::check(device.createRenderPipeline(desc, pipeline.writeRef()),
              std::string("createRenderPipeline(") + vertex + ")");
    return pipeline;
}

void grow(rhi::IDevice &device, ComPtr<rhi::IBuffer> &buffer, size_t bytes,
          rhi::BufferUsage usage, const char *label) {
    if (buffer && buffer->getDesc().size >= bytes)
        return;
    rhi::BufferDesc desc;
    desc.size = std::bit_ceil(std::max<size_t>(bytes, 64 * 1024));
    desc.usage = usage | rhi::BufferUsage::CopyDestination;
    desc.label = label;
    // Dropping the old buffer is safe: slang-rhi defers destruction past in-flight frames.
    en::check(device.createBuffer(desc, nullptr, buffer.writeRef()), "createBuffer");
}

void upload_rect(rhi::ICommandEncoder &encoder, rhi::ITexture &texture, ImTextureData &tex,
                 const ImTextureRect &rect) {
    const rhi::SubresourceData data = {tex.GetPixelsAt(rect.x, rect.y), size_t(tex.GetPitch()),
                                       size_t(tex.GetPitch()) * rect.h};
    en::check(encoder.uploadTextureData(&texture, {0, 1, 0, 1}, {rect.x, rect.y, 0},
                                        {rect.w, rect.h, 1}, &data, 1),
              "uploadTextureData");
}

// Honors ImGui 1.92+ texture requests. ImTextureID is an en_texture*, so hosts pass their own
// textures to ImGui::Image the same way.
void update_texture(rhi::IDevice &device, rhi::ICommandEncoder &encoder, ImTextureData &tex) {
    switch (tex.Status) {
    case ImTextureStatus_WantCreate: {
        IM_ASSERT(tex.Format == ImTextureFormat_RGBA32);
        rhi::TextureDesc desc;
        desc.size = {uint32_t(tex.Width), uint32_t(tex.Height), 1};
        desc.format = rhi::Format::RGBA8Unorm;
        desc.usage = rhi::TextureUsage::ShaderResource | rhi::TextureUsage::CopyDestination;
        desc.label = "imgui";
        auto texture = std::make_unique<en_texture>();
        en::check(device.createTexture(desc, nullptr, texture->texture.writeRef()),
                  "createTexture(imgui)");
        upload_rect(encoder, *texture->texture, tex,
                    {0, 0, uint16_t(tex.Width), uint16_t(tex.Height)});
        tex.SetTexID(ImTextureID(texture.release()));
        tex.SetStatus(ImTextureStatus_OK);
        return;
    }
    case ImTextureStatus_WantUpdates:
        upload_rect(encoder, *reinterpret_cast<en_texture *>(tex.TexID)->texture, tex,
                    tex.UpdateRect);
        tex.SetStatus(ImTextureStatus_OK);
        return;
    case ImTextureStatus_WantDestroy:
        // Destruction is deferred by slang-rhi until in-flight frames retire.
        delete reinterpret_cast<en_texture *>(tex.TexID);
        tex.SetTexID(ImTextureID_Invalid);
        tex.SetStatus(ImTextureStatus_Destroyed);
        return;
    default: return;
    }
}

// Records everything outside the render pass: texture requests and geometry uploads.
void prepare_imgui(en_surface &surface, rhi::ICommandEncoder &encoder, ImDrawData &draw) {
    rhi::IDevice &device = *surface.device->device;
    if (draw.Textures)
        for (ImTextureData *tex : *draw.Textures)
            update_texture(device, encoder, *tex);
    if (draw.TotalVtxCount == 0)
        return;
    grow(device, surface.vertices, draw.TotalVtxCount * sizeof(ImDrawVert),
         rhi::BufferUsage::VertexBuffer, "imgui vertices");
    grow(device, surface.indices, draw.TotalIdxCount * sizeof(ImDrawIdx),
         rhi::BufferUsage::IndexBuffer, "imgui indices");
    size_t vertex_offset = 0, index_offset = 0;
    for (const ImDrawList *list : draw.CmdLists) {
        const size_t vertex_bytes = list->VtxBuffer.size_in_bytes();
        const size_t index_bytes = list->IdxBuffer.size_in_bytes();
        en::check(encoder.uploadBufferData(surface.vertices, vertex_offset, vertex_bytes,
                                           list->VtxBuffer.Data),
                  "uploadBufferData(imgui vertices)");
        en::check(encoder.uploadBufferData(surface.indices, index_offset, index_bytes,
                                           list->IdxBuffer.Data),
                  "uploadBufferData(imgui indices)");
        vertex_offset += vertex_bytes;
        index_offset += index_bytes;
    }
}

void draw_imgui(en_surface &surface, rhi::IRenderPassEncoder &pass, ImDrawData &draw) {
    const ImVec2 scale = {2.0f / draw.DisplaySize.x, -2.0f / draw.DisplaySize.y};
    const float transform[4] = {scale.x, scale.y, -1.0f - draw.DisplayPos.x * scale.x,
                                1.0f - draw.DisplayPos.y * scale.y};
    const ImVec2 clip_scale = draw.FramebufferScale;

    rhi::RenderState state;
    state.viewports[0] = rhi::Viewport::fromSize(float(surface.width), float(surface.height));
    state.viewportCount = 1;
    state.scissorRectCount = 1;
    state.vertexBuffers[0] = surface.vertices;
    state.vertexBufferCount = 1;
    state.indexBuffer = surface.indices;
    state.indexFormat = rhi::IndexFormat::Uint16;

    ImTextureID bound = ImTextureID_Invalid;
    uint32_t list_vertex = 0, list_index = 0;
    for (const ImDrawList *list : draw.CmdLists) {
        for (const ImDrawCmd &cmd : list->CmdBuffer) {
            if (cmd.UserCallback) {
                if (cmd.UserCallback != ImDrawCallback_ResetRenderState)
                    cmd.UserCallback(list, &cmd);
                continue;
            }
            const float x0 = std::max((cmd.ClipRect.x - draw.DisplayPos.x) * clip_scale.x, 0.0f);
            const float y0 = std::max((cmd.ClipRect.y - draw.DisplayPos.y) * clip_scale.y, 0.0f);
            const float x1 = std::min((cmd.ClipRect.z - draw.DisplayPos.x) * clip_scale.x,
                                      float(surface.width));
            const float y1 = std::min((cmd.ClipRect.w - draw.DisplayPos.y) * clip_scale.y,
                                      float(surface.height));
            if (x1 <= x0 || y1 <= y0)
                continue;
            // A new texture needs a new root object, which only bindPipeline hands out.
            if (cmd.GetTexID() != bound) {
                bound = cmd.GetTexID();
                rhi::ShaderCursor root(pass.bindPipeline(surface.imgui));
                en::check(root["image"].setBinding(
                              rhi::Binding(reinterpret_cast<en_texture *>(bound)->texture)),
                          "bind imgui texture");
                en::check(root["linear_sampler"].setBinding(rhi::Binding(surface.linear)),
                          "bind sampler");
                en::check(root["imgui_transform"].setData(transform, sizeof transform),
                          "bind imgui_transform");
            }
            state.scissorRects[0] = {uint32_t(x0), uint32_t(y0), uint32_t(x1), uint32_t(y1)};
            pass.setRenderState(state);
            rhi::DrawArguments args;
            args.vertexCount = cmd.ElemCount;
            args.startIndexLocation = list_index + cmd.IdxOffset;
            args.startVertexLocation = list_vertex + cmd.VtxOffset;
            pass.drawIndexed(args);
        }
        list_vertex += uint32_t(list->VtxBuffer.Size);
        list_index += uint32_t(list->IdxBuffer.Size);
    }
}

} // namespace

extern "C" {

en_status en_surface_create(en_device *device, const en_window_handle *window, en_surface **out) {
    return en::guard([&] {
        auto &dev = en::deref(device, "device");
        const auto &w = en::deref(window, "window");
        en::deref(out, "out") = nullptr;

        rhi::WindowHandle handle;
        switch (w.kind) {
        case EN_WINDOW_XLIB:
            handle = rhi::WindowHandle::fromXlibWindow(w.display, uint32_t(w.window));
            break;
        case EN_WINDOW_WIN32:
            handle = rhi::WindowHandle::fromHwnd(reinterpret_cast<void *>(w.window));
            break;
        case EN_WINDOW_COCOA:
            handle = rhi::WindowHandle::fromNSWindow(reinterpret_cast<void *>(w.window));
            break;
        default: en::raise(EN_ERR_INVALID_ARGUMENT, "unknown en_window_kind");
        }
        auto surface = std::make_unique<en_surface>(&dev);
        en::check(dev.device->createSurface(handle, surface->surface.writeRef()), "createSurface");

        // The image and ImGui colors are already display-encoded, so the target must not be sRGB.
        const rhi::SurfaceInfo &info = surface->surface->getInfo();
        const std::span formats(info.formats, info.formatCount);
        auto found = std::ranges::find_first_of(
            formats, std::array{rhi::Format::BGRA8Unorm, rhi::Format::RGBA8Unorm});
        if (found == formats.end())
            en::raise(EN_ERR_UNSUPPORTED, "surface offers no BGRA8/RGBA8 UNORM format");
        surface->format = *found;

        rhi::IDevice &d = *dev.device;
        rhi::ColorTargetDesc opaque;
        opaque.format = surface->format;
        surface->blit = make_pipeline(d, "blit_vertex", "blit_fragment", nullptr, opaque);

        const rhi::InputElementDesc elements[] = {
            {"POSITION", 0, rhi::Format::RG32Float, offsetof(ImDrawVert, pos), 0},
            {"TEXCOORD", 0, rhi::Format::RG32Float, offsetof(ImDrawVert, uv), 0},
            {"COLOR", 0, rhi::Format::RGBA8Unorm, offsetof(ImDrawVert, col), 0},
        };
        ComPtr<rhi::IInputLayout> input_layout;
        en::check(d.createInputLayout(sizeof(ImDrawVert), elements, std::size(elements),
                                      input_layout.writeRef()),
                  "createInputLayout");
        rhi::ColorTargetDesc blended = opaque;
        blended.enableBlend = true;
        blended.color = {rhi::BlendFactor::SrcAlpha, rhi::BlendFactor::InvSrcAlpha};
        blended.alpha = {rhi::BlendFactor::One, rhi::BlendFactor::InvSrcAlpha};
        surface->imgui = make_pipeline(d, "imgui_vertex", "imgui_fragment", input_layout, blended);

        rhi::SamplerDesc sampler;
        sampler.mipFilter = rhi::TextureFilteringMode::Point;
        sampler.addressU = sampler.addressV = sampler.addressW =
            rhi::TextureAddressingMode::ClampToEdge;
        en::check(d.createSampler(sampler, surface->linear.writeRef()), "createSampler");
        *out = surface.release();
    });
}

void en_surface_destroy(en_surface *surface) { delete surface; }

en_status en_surface_configure(en_surface *surface, uint32_t width, uint32_t height,
                               uint32_t vsync) {
    return en::guard([&] {
        auto &s = en::deref(surface, "surface");
        s.width = width;
        s.height = height;
        // A minimized window has no extent; en_present reports out-of-date until it grows again.
        if (width == 0 || height == 0) {
            en::check(s.surface->unconfigure(), "unconfigure");
            return;
        }
        rhi::SurfaceConfig config;
        config.format = s.format;
        config.width = width;
        config.height = height;
        config.vsync = vsync != 0;
        en::check(s.surface->configure(config), "configure surface");
    });
}

en_status en_present(en_device *device, en_surface *surface, en_texture *image) {
    return en::guard([&] {
        auto &dev = en::deref(device, "device");
        auto &s = en::deref(surface, "surface");

        ImDrawData *draw = nullptr;
        if (imgui_frame_begun) {
            ImGui::Render();
            imgui_frame_begun = false;
            draw = ImGui::GetDrawData();
        }

        ComPtr<rhi::ICommandEncoder> encoder;
        en::check(dev.queue->createCommandEncoder(encoder.writeRef()), "createCommandEncoder");
        if (draw)
            prepare_imgui(s, *encoder, *draw);

        // slang-rhi folds VK_ERROR_OUT_OF_DATE_KHR into a generic acquire failure and hides
        // VK_SUBOPTIMAL_KHR, so any acquire failure asks the host to reconfigure.
        ComPtr<rhi::ITexture> target;
        if (s.width == 0 || s.height == 0 ||
            SLANG_FAILED(s.surface->acquireNextImage(target.writeRef()))) {
            en::submit(dev, *encoder); // ImGui considers its texture requests honored.
            en::raise(EN_ERR_SURFACE_OUT_OF_DATE, "surface must be reconfigured");
        }

        rhi::RenderPassColorAttachment color;
        color.view = target->getDefaultView();
        color.loadOp = image ? rhi::LoadOp::DontCare : rhi::LoadOp::Clear;
        rhi::RenderPassDesc pass_desc;
        pass_desc.colorAttachments = &color;
        pass_desc.colorAttachmentCount = 1;
        rhi::IRenderPassEncoder *pass = encoder->beginRenderPass(pass_desc);
        {
            const en::PassScope scope{pass};
            if (image) {
                rhi::ShaderCursor root(pass->bindPipeline(s.blit));
                en::check(root["image"].setBinding(rhi::Binding(image->texture)), "bind image");
                en::check(root["linear_sampler"].setBinding(rhi::Binding(s.linear)),
                          "bind sampler");
                rhi::RenderState state;
                state.viewports[0] = rhi::Viewport::fromSize(float(s.width), float(s.height));
                state.viewportCount = 1;
                state.scissorRects[0] = rhi::ScissorRect::fromSize(s.width, s.height);
                state.scissorRectCount = 1;
                pass->setRenderState(state);
                rhi::DrawArguments args;
                args.vertexCount = 3;
                pass->draw(args);
            }
            if (draw && draw->TotalVtxCount > 0)
                draw_imgui(s, *pass, *draw);
        }
        en::submit(dev, *encoder);
        en::check(s.surface->present(), "present");
    });
}

en_status en_ui_init(en_device *device, en_surface *surface, void *sdl_window) {
    return en::guard([&] {
        auto &dev = en::deref(device, "device");
        if (en::deref(surface, "surface").device != &dev)
            en::raise(EN_ERR_INVALID_ARGUMENT, "surface belongs to another device");
        if (!sdl_window)
            en::raise(EN_ERR_INVALID_ARGUMENT, "sdl_window is NULL");
        if (ImGui::GetCurrentContext())
            en::raise(EN_ERR_INVALID_ARGUMENT, "ImGui is already initialized");

        ImGui::CreateContext();
        if (!ImGui_ImplSDL3_InitForOther(static_cast<SDL_Window *>(sdl_window))) {
            ImGui::DestroyContext();
            en::raise(EN_ERR_INTERNAL, "ImGui_ImplSDL3_InitForOther failed");
        }
        ImGuiIO &io = ImGui::GetIO();
        io.BackendRendererName = "earth_native";
        io.BackendFlags |= ImGuiBackendFlags_RendererHasVtxOffset |
                           ImGuiBackendFlags_RendererHasTextures;
    });
}

void en_ui_shutdown(void) {
    if (!ImGui::GetCurrentContext())
        return;
    ImGui_ImplSDL3_Shutdown();
    for (ImTextureData *tex : ImGui::GetPlatformIO().Textures) {
        if (tex->RefCount == 1 && tex->TexID != ImTextureID_Invalid) {
            delete reinterpret_cast<en_texture *>(tex->TexID);
            tex->SetTexID(ImTextureID_Invalid);
            tex->SetStatus(ImTextureStatus_Destroyed);
        }
    }
    ImGuiIO &io = ImGui::GetIO();
    io.BackendRendererName = nullptr;
    io.BackendFlags &= ~(ImGuiBackendFlags_RendererHasVtxOffset |
                         ImGuiBackendFlags_RendererHasTextures);
    ImGui::DestroyContext();
    imgui_frame_begun = false;
}

void en_ui_process_event(const void *sdl_event) {
    if (ImGui::GetCurrentContext() && sdl_event)
        ImGui_ImplSDL3_ProcessEvent(static_cast<const SDL_Event *>(sdl_event));
}

void en_ui_new_frame(void) {
    if (!ImGui::GetCurrentContext())
        return;
    // A frame the host never presented must still be closed before ImGui starts the next one.
    if (imgui_frame_begun)
        ImGui::EndFrame();
    ImGui_ImplSDL3_NewFrame();
    ImGui::NewFrame();
    imgui_frame_begun = true;
}

} // extern "C"
