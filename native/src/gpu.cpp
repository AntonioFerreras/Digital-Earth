#include "internal.h"

#include <slang-rhi/shader-cursor.h>

#include <algorithm>
#include <bit>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <format>
#include <fstream>
#include <random>
#include <vector>

using rhi::ComPtr;

namespace {

thread_local std::string last_error;

class DebugCallback final : public rhi::IDebugCallback {
  public:
    SLANG_NO_THROW void SLANG_MCALL handleMessage(rhi::DebugMessageType type,
                                                  rhi::DebugMessageSource source,
                                                  const char *message) override {
        if (type == rhi::DebugMessageType::Info)
            return;
        static constexpr const char *sources[] = {"layer", "driver", "slang"};
        const bool error = type == rhi::DebugMessageType::Error;
        std::fprintf(stderr, "[earth_native] %s %s: %s\n", sources[int(source)],
                     error ? "error" : "warning", message);
        if (error)
            en::rhi_error = message;
    }
};

DebugCallback debug_callback;

std::string blob_text(ISlangBlob *blob) {
    return blob ? std::string(static_cast<const char *>(blob->getBufferPointer()),
                              blob->getBufferSize())
                : std::string();
}

rhi::Format format(en_format f) {
    switch (f) {
    case EN_FORMAT_R8_UNORM: return rhi::Format::R8Unorm;
    case EN_FORMAT_RGBA8_UNORM: return rhi::Format::RGBA8Unorm;
    case EN_FORMAT_R16_FLOAT: return rhi::Format::R16Float;
    case EN_FORMAT_RGBA16_FLOAT: return rhi::Format::RGBA16Float;
    case EN_FORMAT_R32_FLOAT: return rhi::Format::R32Float;
    case EN_FORMAT_RGBA32_FLOAT: return rhi::Format::RGBA32Float;
    }
    en::raise(EN_ERR_INVALID_ARGUMENT, "unknown en_format " + std::to_string(int(f)));
}

rhi::BufferUsage buffer_usage(uint32_t bits) {
    rhi::BufferUsage usage = rhi::BufferUsage::None;
    if (bits & EN_BUFFER_CONSTANT) usage |= rhi::BufferUsage::ConstantBuffer;
    if (bits & EN_BUFFER_SHADER_RESOURCE) usage |= rhi::BufferUsage::ShaderResource;
    if (bits & EN_BUFFER_UNORDERED_ACCESS) usage |= rhi::BufferUsage::UnorderedAccess;
    if (bits & EN_BUFFER_COPY_SOURCE) usage |= rhi::BufferUsage::CopySource;
    if (bits & EN_BUFFER_COPY_DESTINATION) usage |= rhi::BufferUsage::CopyDestination;
    return usage;
}

rhi::TextureUsage texture_usage(uint32_t bits) {
    rhi::TextureUsage usage = rhi::TextureUsage::None;
    if (bits & EN_TEXTURE_SHADER_RESOURCE) usage |= rhi::TextureUsage::ShaderResource;
    if (bits & EN_TEXTURE_UNORDERED_ACCESS) usage |= rhi::TextureUsage::UnorderedAccess;
    if (bits & EN_TEXTURE_COPY_SOURCE) usage |= rhi::TextureUsage::CopySource;
    if (bits & EN_TEXTURE_COPY_DESTINATION) usage |= rhi::TextureUsage::CopyDestination;
    return usage;
}

void require_usage(bool present, const char *what) {
    if (!present)
        en::raise(EN_ERR_INVALID_ARGUMENT, std::string("resource was created without ") + what);
}

// Records with `record` into a fresh encoder and submits without waiting.
template <class Record> void record_and_submit(en_device &device, Record &&record) {
    ComPtr<rhi::ICommandEncoder> encoder;
    en::check(device.queue->createCommandEncoder(encoder.writeRef()), "createCommandEncoder");
    record(*encoder);
    en::submit(device, *encoder);
}

// en_param is opaque to the host; inside it is a root-object offset plus what binding it accepts.
struct Param {
    rhi::ShaderOffset offset;
    uint32_t size : 28; // uniform bytes for EN_PARAM_DATA
    uint32_t kind : 3;  // en_param_kind
    // Data lives in the sub-object of a ConstantBuffer/ParameterBlock global at `offset`.
    uint32_t in_sub_object : 1;
};
static_assert(sizeof(Param) == sizeof(en_param) && alignof(Param) <= alignof(en_param));

} // namespace

struct en_buffer {
    ComPtr<rhi::IBuffer> buffer;
};

struct en_sampler {
    ComPtr<rhi::ISampler> sampler;
};

struct en_kernel {
    ComPtr<rhi::IShaderProgram> program;
    ComPtr<rhi::IComputePipeline> pipeline;
    // Never bound: it only resolves parameter paths, whose offsets hold for every root object
    // of this program, including the one each dispatch binds.
    ComPtr<rhi::IShaderObject> layout_root;
    uint32_t group_size[3];
};

struct en_cmd {
    en_device *device;
    ComPtr<rhi::ICommandEncoder> encoder;
};

struct en_fence {
    ComPtr<rhi::IFence> fence;
};

namespace en {

void raise(en_status status, std::string message) { throw Error{status, std::move(message)}; }

en_status fail(en_status status, std::string message) {
    last_error = std::move(message);
    return status;
}

void submit(en_device &device, rhi::ICommandEncoder &encoder) {
    ComPtr<rhi::ICommandBuffer> commands;
    check(encoder.finish(commands.writeRef()), "finish");
    check(device.queue->submit(commands), "submit");
}

void check(rhi::Result result, std::string_view what) {
    if (SLANG_SUCCEEDED(result))
        return;
    en_status status = EN_ERR_INTERNAL;
    switch (result) {
    case SLANG_E_INVALID_ARG: status = EN_ERR_INVALID_ARGUMENT; break;
    case SLANG_E_OUT_OF_MEMORY: status = EN_ERR_OUT_OF_MEMORY; break;
    case SLANG_E_NOT_FOUND: status = EN_ERR_NOT_FOUND; break;
    case SLANG_E_NOT_AVAILABLE:
    case SLANG_E_NOT_IMPLEMENTED: status = EN_ERR_UNSUPPORTED; break;
    }
    // Backends report VkResult names only through the debug callback.
    if (rhi_error.find("DEVICE_LOST") != std::string::npos)
        status = EN_ERR_DEVICE_LOST;
    char code[16];
    std::snprintf(code, sizeof code, "0x%08" PRIX32, uint32_t(result));
    std::string message = std::string(what) + " failed (" + code + ")";
    if (!rhi_error.empty())
        message += ": " + rhi_error;
    raise(status, std::move(message));
}

SlangResult FileCache::queryInterface(const SlangUUID &uuid, void **out) {
    if (uuid == ISlangUnknown::getTypeGuid() || uuid == rhi::IPersistentCache::getTypeGuid()) {
        addRef();
        *out = static_cast<rhi::IPersistentCache *>(this);
        return SLANG_OK;
    }
    *out = nullptr;
    return SLANG_E_NO_INTERFACE;
}

std::filesystem::path FileCache::entry_path(ISlangBlob *key) const {
    uint64_t hash = 0xcbf29ce484222325ull; // FNV-1a; collisions are caught by the stored key.
    auto bytes = static_cast<const uint8_t *>(key->getBufferPointer());
    for (size_t i = 0; i < key->getBufferSize(); ++i)
        hash = (hash ^ bytes[i]) * 0x100000001b3ull;
    char name[32];
    std::snprintf(name, sizeof name, "%016" PRIx64 ".bin", hash);
    return dir_ / name;
}

// Entry layout: u64 key size, key bytes, data bytes. Written to a unique temporary and renamed,
// so concurrent writers and readers only ever observe complete entries.
rhi::Result FileCache::writeCache(ISlangBlob *key, ISlangBlob *data) {
    try {
        const auto path = entry_path(key);
        auto tmp = path;
        static const uint64_t process_salt = std::random_device{}();
        static std::atomic<uint64_t> counter;
        tmp += std::format(".{:x}.{}.tmp", process_salt, counter++);
        {
            std::ofstream file(tmp, std::ios::binary | std::ios::trunc);
            const uint64_t key_size = key->getBufferSize();
            file.write(reinterpret_cast<const char *>(&key_size), sizeof key_size);
            file.write(static_cast<const char *>(key->getBufferPointer()),
                       std::streamsize(key_size));
            file.write(static_cast<const char *>(data->getBufferPointer()),
                       std::streamsize(data->getBufferSize()));
            if (!file)
                return SLANG_FAIL;
        }
        std::filesystem::rename(tmp, path);
        return SLANG_OK;
    } catch (...) {
        return SLANG_FAIL;
    }
}

rhi::Result FileCache::queryCache(ISlangBlob *key, ISlangBlob **out_data) {
    *out_data = nullptr;
    try {
        std::ifstream file(entry_path(key), std::ios::binary | std::ios::ate);
        if (!file)
            return SLANG_E_NOT_FOUND;
        const uint64_t file_size = uint64_t(file.tellg());
        const uint64_t key_size = key->getBufferSize();
        uint64_t stored_key_size = 0;
        file.seekg(0);
        file.read(reinterpret_cast<char *>(&stored_key_size), sizeof stored_key_size);
        if (!file || stored_key_size != key_size || file_size < sizeof(uint64_t) + key_size)
            return SLANG_E_NOT_FOUND;
        std::vector<char> contents(file_size - sizeof(uint64_t));
        file.read(contents.data(), std::streamsize(contents.size()));
        if (!file || std::memcmp(contents.data(), key->getBufferPointer(), key_size) != 0)
            return SLANG_E_NOT_FOUND;
        return rhi::getRHI()->createBlob(contents.data() + key_size, contents.size() - key_size,
                                         out_data);
    } catch (...) {
        return SLANG_FAIL;
    }
}

ComPtr<rhi::IShaderProgram> link_program(rhi::IDevice &device, const char *module,
                                         std::span<const char *const> entries,
                                         std::span<const char *const> generic_args,
                                         slang::ProgramLayout **out_layout) {
    ComPtr<slang::ISession> session;
    check(device.getSlangSession(session.writeRef()), "getSlangSession");
    ComPtr<ISlangBlob> diagnostics;
    slang::IModule *loaded = session->loadModule(module, diagnostics.writeRef());
    const std::string context = std::string("module '") + module + "'";
    if (!loaded)
        raise(EN_ERR_SHADER_COMPILE, context + ":\n" + blob_text(diagnostics));
    if (diagnostics)
        std::fprintf(stderr, "[earth_native] %s:\n%s", context.c_str(),
                     blob_text(diagnostics).c_str());

    std::vector<slang::SpecializationArg> specialization;
    for (const char *arg : generic_args)
        specialization.push_back(slang::SpecializationArg::fromExpr(arg));
    std::vector<ComPtr<slang::IComponentType>> entry_points;
    std::vector<slang::IComponentType *> components{loaded};
    for (const char *entry : entries) {
        ComPtr<slang::IEntryPoint> entry_point;
        if (SLANG_FAILED(loaded->findEntryPointByName(entry, entry_point.writeRef())))
            raise(EN_ERR_NOT_FOUND, context + " has no entry point '" + entry +
                                        "' (it needs a [shader(...)] attribute)");
        ComPtr<slang::IComponentType> component(entry_point);
        if (!specialization.empty() &&
            SLANG_FAILED(entry_point->specialize(specialization.data(),
                                                 SlangInt(specialization.size()),
                                                 component.writeRef(), diagnostics.writeRef())))
            raise(EN_ERR_SHADER_COMPILE,
                  context + " failed to specialize '" + entry + "':\n" + blob_text(diagnostics));
        components.push_back(component);
        entry_points.push_back(std::move(component));
    }

    ComPtr<slang::IComponentType> composite, linked;
    if (SLANG_FAILED(session->createCompositeComponentType(components.data(),
                                                           SlangInt(components.size()),
                                                           composite.writeRef(),
                                                           diagnostics.writeRef())) ||
        SLANG_FAILED(composite->link(linked.writeRef(), diagnostics.writeRef())))
        raise(EN_ERR_SHADER_COMPILE, context + " failed to link:\n" + blob_text(diagnostics));

    rhi::ShaderProgramDesc desc;
    desc.slangGlobalScope = linked;
    desc.label = module;
    ComPtr<rhi::IShaderProgram> program;
    if (SLANG_FAILED(device.createShaderProgram(desc, program.writeRef(), diagnostics.writeRef())))
        raise(EN_ERR_SHADER_COMPILE,
              context + " failed to compile:\n" + blob_text(diagnostics) + rhi_error);
    if (out_layout)
        *out_layout = linked->getLayout(); // The program keeps `linked` alive.
    return program;
}

} // namespace en

namespace {

// Resolves a path to a global parameter (or a field or element of one) of the kernel's program.
rhi::ShaderCursor global_cursor(const en_kernel &kernel, const char *path) {
    rhi::ShaderCursor cursor = rhi::ShaderCursor(kernel.layout_root).getPath(path);
    if (!cursor.isValid())
        en::raise(EN_ERR_NOT_FOUND, std::string("no shader parameter '") + path + "'");
    // Paths into entry-point parameters or through a ConstantBuffer land on another object, whose
    // offsets the dispatch-time root object cannot address.
    if (cursor.m_baseObject != kernel.layout_root.get())
        en::raise(EN_ERR_UNSUPPORTED,
                  std::format("'{}' is not a global parameter or a field of a global struct; "
                              "kernels take global parameters only, and ConstantBuffer globals "
                              "are bound as a whole",
                              path));
    return cursor;
}

bool is_constant_block(slang::TypeLayoutReflection *type) {
    const auto kind = type->getKind();
    return kind == slang::TypeReflection::Kind::ConstantBuffer ||
           kind == slang::TypeReflection::Kind::ParameterBlock;
}

Param classify(slang::TypeLayoutReflection *type, const char *path) {
    using Kind = slang::TypeReflection::Kind;
    Param param{};
    auto data = [&](size_t size) {
        if (size >= (1u << 28))
            en::raise(EN_ERR_UNSUPPORTED, std::format("'{}' is too large", path));
        param.kind = EN_PARAM_DATA;
        param.size = uint32_t(size);
    };
    switch (type->getKind()) {
    case Kind::ConstantBuffer:
    case Kind::ParameterBlock:
        data(type->getElementTypeLayout()->getSize());
        param.in_sub_object = 1;
        break;
    case Kind::SamplerState: param.kind = EN_PARAM_SAMPLER; break;
    case Kind::Resource: {
        const bool writable = type->getResourceAccess() != SLANG_RESOURCE_ACCESS_READ;
        switch (type->getResourceShape() & SLANG_RESOURCE_BASE_SHAPE_MASK) {
        case SLANG_STRUCTURED_BUFFER:
        case SLANG_BYTE_ADDRESS_BUFFER:
        case SLANG_TEXTURE_BUFFER:
            param.kind = writable ? EN_PARAM_RW_BUFFER : EN_PARAM_BUFFER;
            break;
        case SLANG_TEXTURE_1D:
        case SLANG_TEXTURE_2D:
        case SLANG_TEXTURE_3D:
        case SLANG_TEXTURE_CUBE:
            param.kind = writable ? EN_PARAM_RW_TEXTURE : EN_PARAM_TEXTURE;
            break;
        default:
            en::raise(EN_ERR_UNSUPPORTED,
                      std::format("'{}' has an unsupported resource type", path));
        }
        break;
    }
    case Kind::Array:
        if (type->getElementTypeLayout()->getKind() == Kind::Resource ||
            type->getElementTypeLayout()->getKind() == Kind::SamplerState)
            en::raise(EN_ERR_UNSUPPORTED,
                      std::format("'{}' is a resource array; look up its elements as path[i]",
                                  path));
        data(type->getSize());
        break;
    default: data(type->getSize()); break;
    }
    return param;
}

void bind(rhi::IShaderObject &root, const en_binding &binding) {
    const Param param = std::bit_cast<Param>(binding.param);
    auto expect = [&](bool ok, const char *what) {
        if (!ok)
            en::raise(EN_ERR_INVALID_ARGUMENT,
                      std::string("binding kind does not match parameter kind; expected ") + what);
    };
    rhi::Binding resource;
    switch (binding.kind) {
    case EN_BIND_DATA:
        expect(param.kind == EN_PARAM_DATA, "data");
        // A shorter write would leave the rest of the parameter undefined.
        if (!binding.data || binding.size != param.size)
            en::raise(EN_ERR_INVALID_ARGUMENT,
                      std::format("data binding of {} bytes for a parameter of {} bytes",
                                  binding.size, uint32_t(param.size)));
        if (param.in_sub_object) {
            ComPtr<rhi::IShaderObject> block;
            en::check(root.getObject(param.offset, block.writeRef()), "getObject");
            en::check(block->setData(rhi::ShaderOffset{}, binding.data, binding.size), "setData");
        } else {
            en::check(root.setData(param.offset, binding.data, binding.size), "setData");
        }
        return;
    case EN_BIND_BUFFER:
        expect(param.kind == EN_PARAM_BUFFER || param.kind == EN_PARAM_RW_BUFFER, "a buffer");
        resource = rhi::Binding(en::deref(binding.buffer, "binding.buffer").buffer);
        break;
    case EN_BIND_TEXTURE:
        expect(param.kind == EN_PARAM_TEXTURE || param.kind == EN_PARAM_RW_TEXTURE, "a texture");
        resource = rhi::Binding(en::deref(binding.texture, "binding.texture").texture);
        break;
    case EN_BIND_SAMPLER:
        expect(param.kind == EN_PARAM_SAMPLER, "a sampler");
        resource = rhi::Binding(en::deref(binding.sampler, "binding.sampler").sampler);
        break;
    default:
        en::raise(EN_ERR_INVALID_ARGUMENT,
                  std::format("unknown en_binding_kind {}", int(binding.kind)));
    }
    en::check(root.setBinding(param.offset, resource), "setBinding");
}

} // namespace

extern "C" {

const char *en_last_error(void) { return last_error.c_str(); }

en_status en_device_create(const en_device_desc *desc, en_device **out) {
    return en::guard([&] {
        const auto &d = en::deref(desc, "desc");
        en::deref(out, "out") = nullptr;

        static constexpr rhi::DeviceType types[] = {rhi::DeviceType::Default,
                                                    rhi::DeviceType::Vulkan, rhi::DeviceType::Metal,
                                                    rhi::DeviceType::D3D12, rhi::DeviceType::CUDA};
        if (uint32_t(d.backend) >= std::size(types))
            en::raise(EN_ERR_INVALID_ARGUMENT, "unknown en_backend");
        const rhi::DeviceType type = types[d.backend];
        if (type != rhi::DeviceType::Default && !rhi::getRHI()->isDeviceTypeSupported(type))
            en::raise(EN_ERR_UNSUPPORTED, std::string(rhi::getRHI()->getDeviceTypeName(type)) +
                                              " is not available in this build or on this host");

        auto device = std::make_unique<en_device>();
        if (d.shader_cache_path) {
            std::filesystem::create_directories(d.shader_cache_path);
            device->cache = std::make_unique<en::FileCache>(d.shader_cache_path);
        }

        // Debug layer options are process-global and only settable while no device is alive,
        // so a second concurrent device inherits the first one's choice.
        rhi::DebugLayerOptions layers;
        layers.coreValidation = d.enable_validation != 0;
        rhi::getRHI()->setDebugLayerOptions(layers);

        rhi::DeviceDesc rd;
        rd.deviceType = type;
        rd.enableValidation = d.enable_validation != 0;
        rd.debugCallback = &debug_callback;
        rd.slang.searchPaths = d.shader_search_paths;
        rd.slang.searchPathCount = d.shader_search_path_count;
        rd.persistentShaderCache = device->cache.get();
        rd.persistentPipelineCache = device->cache.get();
        en::check(rhi::getRHI()->createDevice(rd, device->device.writeRef()), "createDevice");
        en::check(device->device->getQueue(rhi::QueueType::Graphics, device->queue.writeRef()),
                  "getQueue");

        const rhi::DeviceInfo &info = device->device->getInfo();
        en_device_info &out_info = device->info;
        out_info = {};
        switch (info.deviceType) {
        case rhi::DeviceType::Vulkan: out_info.backend = EN_BACKEND_VULKAN; break;
        case rhi::DeviceType::Metal: out_info.backend = EN_BACKEND_METAL; break;
        case rhi::DeviceType::D3D12: out_info.backend = EN_BACKEND_D3D12; break;
        case rhi::DeviceType::CUDA: out_info.backend = EN_BACKEND_CUDA; break;
        default: out_info.backend = EN_BACKEND_DEFAULT; break;
        }
        std::snprintf(out_info.adapter_name, sizeof out_info.adapter_name, "%s",
                      info.adapterName ? info.adapterName : "");
        out_info.max_texture_dimension_2d = info.limits.maxTextureDimension2D;
        out_info.max_texture_dimension_3d = info.limits.maxTextureDimension3D;
        out_info.max_texture_array_layers = info.limits.maxTextureLayers;
        // ShaderSample means linear-filterable sampling on Vulkan; Metal reports it for all
        // sampleable formats, which matches Apple GPUs filtering rgba32f.
        rhi::FormatSupport support = rhi::FormatSupport::None;
        en::check(device->device->getFormatSupport(rhi::Format::RGBA32Float, &support),
                  "getFormatSupport");
        out_info.rgba32f_filterable = is_set(support, rhi::FormatSupport::ShaderSample);
        *out = device.release();
    });
}

void en_device_destroy(en_device *device) { delete device; }

void en_device_get_info(const en_device *device, en_device_info *out) {
    if (device && out)
        *out = device->info;
}

en_status en_device_wait_idle(en_device *device) {
    return en::guard(
        [&] { en::check(en::deref(device, "device").queue->waitOnHost(), "waitOnHost"); });
}

en_status en_buffer_create(en_device *device, const en_buffer_desc *desc, const void *initial_data,
                           en_buffer **out) {
    return en::guard([&] {
        auto &dev = en::deref(device, "device");
        const auto &d = en::deref(desc, "desc");
        en::deref(out, "out") = nullptr;
        if (d.size == 0)
            en::raise(EN_ERR_INVALID_ARGUMENT, "buffer size is 0");
        static constexpr rhi::MemoryType memory[] = {
            rhi::MemoryType::DeviceLocal, rhi::MemoryType::Upload, rhi::MemoryType::ReadBack};
        if (uint32_t(d.memory) >= std::size(memory))
            en::raise(EN_ERR_INVALID_ARGUMENT, "unknown en_memory");

        rhi::BufferDesc bd;
        bd.size = d.size;
        bd.elementSize = d.element_size;
        bd.memoryType = memory[d.memory];
        bd.usage = buffer_usage(d.usage);
        bd.label = d.label;
        auto buffer = std::make_unique<en_buffer>();
        en::check(dev.device->createBuffer(bd, initial_data, buffer->buffer.writeRef()),
                  "createBuffer");
        *out = buffer.release();
    });
}

void en_buffer_destroy(en_buffer *buffer) { delete buffer; }

en_status en_buffer_upload(en_device *device, en_buffer *buffer, uint64_t offset, const void *data,
                           uint64_t size) {
    return en::guard([&] {
        auto &dev = en::deref(device, "device");
        rhi::IBuffer &buf = *en::deref(buffer, "buffer").buffer;
        const rhi::BufferDesc &desc = buf.getDesc();
        if (!data || offset > desc.size || size > desc.size - offset)
            en::raise(EN_ERR_INVALID_ARGUMENT, "upload range outside the buffer");
        // Upload memory is host-visible: write through a mapping, ordered by the host.
        if (desc.memoryType == rhi::MemoryType::Upload) {
            void *mapped = nullptr;
            en::check(dev.device->mapBuffer(&buf, rhi::CpuAccessMode::Write, &mapped), "mapBuffer");
            std::memcpy(static_cast<uint8_t *>(mapped) + offset, data, size);
            en::check(dev.device->unmapBuffer(&buf), "unmapBuffer");
            return;
        }
        require_usage(is_set(desc.usage, rhi::BufferUsage::CopyDestination),
                      "EN_BUFFER_COPY_DESTINATION");
        record_and_submit(dev, [&](rhi::ICommandEncoder &encoder) {
            en::check(encoder.uploadBufferData(&buf, offset, size, data), "uploadBufferData");
        });
    });
}

en_status en_buffer_read(en_device *device, en_buffer *buffer, uint64_t offset, void *dst,
                         uint64_t size) {
    return en::guard([&] {
        auto &dev = en::deref(device, "device");
        rhi::IBuffer &buf = *en::deref(buffer, "buffer").buffer;
        const rhi::BufferDesc &desc = buf.getDesc();
        if (!dst || offset > desc.size || size > desc.size - offset)
            en::raise(EN_ERR_INVALID_ARGUMENT, "read range outside the buffer");
        if (desc.memoryType == rhi::MemoryType::ReadBack) {
            en::check(dev.queue->waitOnHost(), "waitOnHost");
            void *mapped = nullptr;
            en::check(dev.device->mapBuffer(&buf, rhi::CpuAccessMode::Read, &mapped), "mapBuffer");
            std::memcpy(dst, static_cast<const uint8_t *>(mapped) + offset, size);
            en::check(dev.device->unmapBuffer(&buf), "unmapBuffer");
            return;
        }
        require_usage(is_set(desc.usage, rhi::BufferUsage::CopySource), "EN_BUFFER_COPY_SOURCE");
        en::check(dev.device->readBuffer(&buf, offset, size, dst), "readBuffer");
    });
}

en_status en_buffer_map(en_device *device, en_buffer *buffer, void **out) {
    return en::guard([&] {
        auto &dev = en::deref(device, "device");
        rhi::IBuffer &buf = *en::deref(buffer, "buffer").buffer;
        auto &mapped = en::deref(out, "out");
        mapped = nullptr;
        const rhi::MemoryType memory = buf.getDesc().memoryType;
        if (memory == rhi::MemoryType::DeviceLocal)
            en::raise(EN_ERR_INVALID_ARGUMENT, "device-local buffers cannot be mapped");
        const auto mode = memory == rhi::MemoryType::Upload ? rhi::CpuAccessMode::Write
                                                            : rhi::CpuAccessMode::Read;
        en::check(dev.device->mapBuffer(&buf, mode, &mapped), "mapBuffer");
    });
}

en_status en_buffer_unmap(en_device *device, en_buffer *buffer) {
    return en::guard([&] {
        en::check(en::deref(device, "device")
                      .device->unmapBuffer(en::deref(buffer, "buffer").buffer),
                  "unmapBuffer");
    });
}

en_status en_texture_create(en_device *device, const en_texture_desc *desc, en_texture **out) {
    return en::guard([&] {
        auto &dev = en::deref(device, "device");
        const auto &d = en::deref(desc, "desc");
        en::deref(out, "out") = nullptr;
        if (d.width == 0 || d.height == 0 || d.depth_or_layers == 0)
            en::raise(EN_ERR_INVALID_ARGUMENT, "texture extent has a zero dimension");

        rhi::TextureDesc td;
        switch (d.kind) {
        case EN_TEXTURE_2D:
            if (d.depth_or_layers != 1)
                en::raise(EN_ERR_INVALID_ARGUMENT, "EN_TEXTURE_2D needs depth_or_layers == 1");
            td.type = rhi::TextureType::Texture2D;
            break;
        case EN_TEXTURE_2D_ARRAY:
            td.type = rhi::TextureType::Texture2DArray;
            td.arrayLength = d.depth_or_layers;
            break;
        case EN_TEXTURE_3D:
            td.type = rhi::TextureType::Texture3D;
            td.size.depth = d.depth_or_layers;
            break;
        default: en::raise(EN_ERR_INVALID_ARGUMENT, "unknown en_texture_kind");
        }
        td.size.width = d.width;
        td.size.height = d.height;
        td.format = format(d.format);
        td.usage = texture_usage(d.usage);
        td.label = d.label;
        auto texture = std::make_unique<en_texture>();
        en::check(dev.device->createTexture(td, nullptr, texture->texture.writeRef()),
                  "createTexture");
        *out = texture.release();
    });
}

void en_texture_destroy(en_texture *texture) { delete texture; }

en_status en_texture_upload(en_device *device, en_texture *texture, const en_texture_region *region,
                            const void *data, uint64_t row_pitch) {
    return en::guard([&] {
        auto &dev = en::deref(device, "device");
        rhi::ITexture &tex = *en::deref(texture, "texture").texture;
        const auto &r = en::deref(region, "region");
        const rhi::TextureDesc &desc = tex.getDesc();
        require_usage(is_set(desc.usage, rhi::TextureUsage::CopyDestination),
                      "EN_TEXTURE_COPY_DESTINATION");
        const bool layered = desc.type == rhi::TextureType::Texture2DArray;
        const uint64_t depth_limit = layered ? desc.arrayLength : desc.size.depth;
        if (!data || r.width == 0 || r.height == 0 || r.depth == 0 ||
            uint64_t(r.x) + r.width > desc.size.width ||
            uint64_t(r.y) + r.height > desc.size.height || uint64_t(r.z) + r.depth > depth_limit)
            en::raise(EN_ERR_INVALID_ARGUMENT, "upload region outside the texture");
        if (row_pitch < uint64_t(r.width) * rhi::getFormatInfo(desc.format).blockSizeInBytes)
            en::raise(EN_ERR_INVALID_ARGUMENT, "row_pitch is smaller than one row of the region");

        const uint64_t slice_pitch = row_pitch * r.height;
        const auto *bytes = static_cast<const uint8_t *>(data);
        // Array layers are separate subresources; 3D slices are one subresource with a depth.
        std::vector<rhi::SubresourceData> subresources;
        for (uint32_t i = 0; i < (layered ? r.depth : 1); ++i)
            subresources.push_back({bytes + i * slice_pitch, row_pitch, slice_pitch});
        const rhi::SubresourceRange range = {layered ? r.z : 0, uint32_t(subresources.size()), 0,
                                             1};
        const rhi::Offset3D offset(r.x, r.y, layered ? 0 : r.z);
        const rhi::Extent3D extent = {r.width, r.height, layered ? 1 : r.depth};
        record_and_submit(dev, [&](rhi::ICommandEncoder &encoder) {
            en::check(encoder.uploadTextureData(&tex, range, offset, extent, subresources.data(),
                                                uint32_t(subresources.size())),
                      "uploadTextureData");
        });
    });
}

en_status en_texture_read(en_device *device, en_texture *texture, uint32_t layer, void *dst,
                          uint64_t dst_size) {
    return en::guard([&] {
        auto &dev = en::deref(device, "device");
        rhi::ITexture &tex = *en::deref(texture, "texture").texture;
        const rhi::TextureDesc &desc = tex.getDesc();
        require_usage(is_set(desc.usage, rhi::TextureUsage::CopySource), "EN_TEXTURE_COPY_SOURCE");
        const bool volume = desc.type == rhi::TextureType::Texture3D;
        if (layer >= (volume ? desc.size.depth : desc.arrayLength))
            en::raise(EN_ERR_INVALID_ARGUMENT, "layer outside the texture");
        const uint64_t row_size =
            uint64_t(desc.size.width) * rhi::getFormatInfo(desc.format).blockSizeInBytes;
        if (!dst || dst_size < row_size * desc.size.height)
            en::raise(EN_ERR_INVALID_ARGUMENT, "dst is smaller than one tightly packed layer");

        rhi::SubresourceLayout layout;
        en::check(tex.getSubresourceLayout(0, &layout), "getSubresourceLayout");
        const uint64_t staging_size = layout.rowPitch * desc.size.height;
        rhi::BufferDesc bd;
        bd.size = staging_size;
        bd.memoryType = rhi::MemoryType::ReadBack;
        bd.usage = rhi::BufferUsage::CopyDestination;
        ComPtr<rhi::IBuffer> staging;
        en::check(dev.device->createBuffer(bd, nullptr, staging.writeRef()), "createBuffer");
        record_and_submit(dev, [&](rhi::ICommandEncoder &encoder) {
            encoder.copyTextureToBuffer(staging, 0, staging_size, layout.rowPitch, &tex,
                                        volume ? 0 : layer, 0, {0, 0, volume ? layer : 0},
                                        {desc.size.width, desc.size.height, 1});
        });
        en::check(dev.queue->waitOnHost(), "waitOnHost");

        void *mapped = nullptr;
        en::check(dev.device->mapBuffer(staging, rhi::CpuAccessMode::Read, &mapped), "mapBuffer");
        for (uint32_t y = 0; y < desc.size.height; ++y)
            std::memcpy(static_cast<uint8_t *>(dst) + y * row_size,
                        static_cast<const uint8_t *>(mapped) + y * layout.rowPitch, row_size);
        en::check(dev.device->unmapBuffer(staging), "unmapBuffer");
    });
}

en_status en_sampler_create(en_device *device, const en_sampler_desc *desc, en_sampler **out) {
    return en::guard([&] {
        auto &dev = en::deref(device, "device");
        const auto &d = en::deref(desc, "desc");
        en::deref(out, "out") = nullptr;
        auto address = [](en_address a) {
            return a == EN_ADDRESS_CLAMP ? rhi::TextureAddressingMode::ClampToEdge
                                         : rhi::TextureAddressingMode::Wrap;
        };
        const auto filter = d.filter == EN_FILTER_LINEAR ? rhi::TextureFilteringMode::Linear
                                                         : rhi::TextureFilteringMode::Point;
        rhi::SamplerDesc sd;
        sd.minFilter = sd.magFilter = filter;
        sd.mipFilter = rhi::TextureFilteringMode::Point;
        sd.addressU = address(d.address_u);
        sd.addressV = address(d.address_v);
        sd.addressW = address(d.address_w);
        auto sampler = std::make_unique<en_sampler>();
        en::check(dev.device->createSampler(sd, sampler->sampler.writeRef()), "createSampler");
        *out = sampler.release();
    });
}

void en_sampler_destroy(en_sampler *sampler) { delete sampler; }

en_status en_kernel_create(en_device *device, const char *module, const char *entry,
                           const char *const *generic_args, uint32_t generic_arg_count,
                           en_kernel **out) {
    return en::guard([&] {
        auto &dev = en::deref(device, "device");
        en::deref(module, "module");
        en::deref(entry, "entry");
        en::deref(out, "out") = nullptr;
        if (generic_arg_count && !generic_args)
            en::raise(EN_ERR_INVALID_ARGUMENT, "generic_args is NULL");
        const std::span args(generic_args, generic_arg_count);
        for (const char *arg : args)
            en::deref(arg, "generic_args[i]");

        auto kernel = std::make_unique<en_kernel>();
        slang::ProgramLayout *layout = nullptr;
        const char *entries[] = {entry};
        kernel->program = en::link_program(*dev.device, module, entries, args, &layout);
        slang::EntryPointReflection *entry_point = layout->getEntryPointByIndex(0);
        if (entry_point->getStage() != SLANG_STAGE_COMPUTE)
            en::raise(EN_ERR_INVALID_ARGUMENT,
                      std::format("'{}' is not a compute entry point", entry));
        SlangUInt group_size[3] = {};
        entry_point->getComputeThreadGroupSize(3, group_size);
        std::copy(std::begin(group_size), std::end(group_size), kernel->group_size);

        rhi::ComputePipelineDesc pd;
        pd.program = kernel->program;
        pd.label = entry;
        en::check(dev.device->createComputePipeline(pd, kernel->pipeline.writeRef()),
                  std::string("createComputePipeline(") + module + "::" + entry + ")");
        en::check(
            dev.device->createRootShaderObject(kernel->program, kernel->layout_root.writeRef()),
            "createRootShaderObject");
        *out = kernel.release();
    });
}

void en_kernel_destroy(en_kernel *kernel) { delete kernel; }

void en_kernel_thread_group_size(const en_kernel *kernel, uint32_t out[3]) {
    for (int i = 0; i < 3; ++i)
        out[i] = kernel ? kernel->group_size[i] : 0;
}

en_status en_kernel_find_param(const en_kernel *kernel, const char *path, en_param_info *out) {
    return en::guard([&] {
        const auto &k = en::deref(kernel, "kernel");
        auto &info = en::deref(out, "out");
        en::deref(path, "path");
        const rhi::ShaderCursor cursor = global_cursor(k, path);
        Param param = classify(cursor.getTypeLayout(), path);
        param.offset = cursor.m_offset;
        info.param = std::bit_cast<en_param>(param);
        info.kind = en_param_kind(param.kind);
        info.size = param.kind == EN_PARAM_DATA ? param.size : 0;
    });
}

en_status en_kernel_find_field(const en_kernel *kernel, const char *path, const char *field,
                               uint64_t *offset, uint64_t *size) {
    return en::guard([&] {
        const auto &k = en::deref(kernel, "kernel");
        en::deref(path, "path");
        en::deref(field, "field");
        auto &out_offset = en::deref(offset, "offset");
        auto &out_size = en::deref(size, "size");
        slang::TypeLayoutReflection *type = global_cursor(k, path).getTypeLayout();
        if (is_constant_block(type))
            type = type->getElementTypeLayout();
        if (type->getKind() != slang::TypeReflection::Kind::Struct)
            en::raise(EN_ERR_INVALID_ARGUMENT, std::format("'{}' is not a struct", path));
        const SlangInt index = type->findFieldIndexByName(field);
        if (index < 0)
            en::raise(EN_ERR_NOT_FOUND, std::format("'{}' has no field '{}'", path, field));
        slang::VariableLayoutReflection *layout = type->getFieldByIndex(unsigned(index));
        out_offset = layout->getOffset();
        out_size = layout->getTypeLayout()->getSize();
    });
}

en_status en_fence_create(en_device *device, en_fence **out) {
    return en::guard([&] {
        auto &dev = en::deref(device, "device");
        en::deref(out, "out") = nullptr;
        auto fence = std::make_unique<en_fence>();
        en::check(dev.device->createFence(rhi::FenceDesc{}, fence->fence.writeRef()),
                  "createFence");
        *out = fence.release();
    });
}

void en_fence_destroy(en_fence *fence) { delete fence; }

en_status en_fence_value(en_fence *fence, uint64_t *out) {
    return en::guard([&] {
        en::check(en::deref(fence, "fence").fence->getCurrentValue(&en::deref(out, "out")),
                  "getCurrentValue");
    });
}

en_status en_fence_wait(en_device *device, en_fence *fence, uint64_t value) {
    return en::guard([&] {
        rhi::IFence *native = en::deref(fence, "fence").fence;
        en::check(en::deref(device, "device")
                      .device->waitForFences(1, &native, &value, true, rhi::kTimeoutInfinite),
                  "waitForFences");
    });
}

en_status en_cmd_begin(en_device *device, en_cmd **out) {
    return en::guard([&] {
        auto &dev = en::deref(device, "device");
        en::deref(out, "out") = nullptr;
        auto cmd = std::make_unique<en_cmd>(&dev);
        en::check(dev.queue->createCommandEncoder(cmd->encoder.writeRef()), "createCommandEncoder");
        *out = cmd.release();
    });
}

// slang-rhi tracks resource states per encoder and emits a barrier for every UAV -> UAV use, so
// each dispatch observes the writes of the previous ones without an explicit barrier here.
en_status en_cmd_dispatch(en_cmd *cmd, en_kernel *kernel, const en_binding *bindings,
                          uint32_t binding_count, uint32_t groups_x, uint32_t groups_y,
                          uint32_t groups_z) {
    return en::guard([&] {
        auto &c = en::deref(cmd, "cmd");
        auto &k = en::deref(kernel, "kernel");
        if (binding_count && !bindings)
            en::raise(EN_ERR_INVALID_ARGUMENT, "bindings is NULL");
        rhi::IComputePassEncoder *pass = c.encoder->beginComputePass();
        const en::PassScope scope{pass};
        rhi::IShaderObject *root = pass->bindPipeline(k.pipeline);
        for (const en_binding &binding : std::span(bindings, binding_count))
            bind(*root, binding);
        pass->dispatchCompute(groups_x, groups_y, groups_z);
    });
}

en_status en_cmd_copy_buffer(en_cmd *cmd, en_buffer *dst, uint64_t dst_offset, en_buffer *src,
                             uint64_t src_offset, uint64_t size) {
    return en::guard([&] {
        auto &c = en::deref(cmd, "cmd");
        rhi::IBuffer &to = *en::deref(dst, "dst").buffer;
        rhi::IBuffer &from = *en::deref(src, "src").buffer;
        require_usage(is_set(to.getDesc().usage, rhi::BufferUsage::CopyDestination),
                      "EN_BUFFER_COPY_DESTINATION");
        require_usage(is_set(from.getDesc().usage, rhi::BufferUsage::CopySource),
                      "EN_BUFFER_COPY_SOURCE");
        if (size == 0 || dst_offset > to.getDesc().size || size > to.getDesc().size - dst_offset ||
            src_offset > from.getDesc().size || size > from.getDesc().size - src_offset)
            en::raise(EN_ERR_INVALID_ARGUMENT, "copy range outside a buffer");
        c.encoder->copyBuffer(&to, dst_offset, &from, src_offset, size);
    });
}

en_status en_cmd_copy_buffer_to_texture(en_cmd *cmd, en_texture *dst,
                                        const en_texture_region *region, en_buffer *src,
                                        uint64_t src_offset, uint64_t row_pitch) {
    return en::guard([&] {
        auto &c = en::deref(cmd, "cmd");
        rhi::ITexture &tex = *en::deref(dst, "dst").texture;
        rhi::IBuffer &from = *en::deref(src, "src").buffer;
        const auto &r = en::deref(region, "region");
        const rhi::TextureDesc &desc = tex.getDesc();
        require_usage(is_set(desc.usage, rhi::TextureUsage::CopyDestination),
                      "EN_TEXTURE_COPY_DESTINATION");
        require_usage(is_set(from.getDesc().usage, rhi::BufferUsage::CopySource),
                      "EN_BUFFER_COPY_SOURCE");
        const bool layered = desc.type == rhi::TextureType::Texture2DArray;
        if (!layered && desc.type != rhi::TextureType::Texture2D)
            en::raise(EN_ERR_INVALID_ARGUMENT, "copies go to 2D textures and 2D arrays only");
        if (r.width == 0 || r.height == 0 || r.depth != 1 ||
            uint64_t(r.x) + r.width > desc.size.width ||
            uint64_t(r.y) + r.height > desc.size.height ||
            r.z >= (layered ? desc.arrayLength : 1))
            en::raise(EN_ERR_INVALID_ARGUMENT, "copy region outside the texture");
        if (row_pitch % EN_COPY_ROW_PITCH_ALIGNMENT || src_offset % EN_COPY_OFFSET_ALIGNMENT)
            en::raise(EN_ERR_INVALID_ARGUMENT, "row_pitch or src_offset is misaligned");
        if (row_pitch < uint64_t(r.width) * rhi::getFormatInfo(desc.format).blockSizeInBytes)
            en::raise(EN_ERR_INVALID_ARGUMENT, "row_pitch is smaller than one row of the region");
        const uint64_t size = row_pitch * r.height;
        if (src_offset > from.getDesc().size || size > from.getDesc().size - src_offset)
            en::raise(EN_ERR_INVALID_ARGUMENT, "copy source outside the buffer");
        c.encoder->copyBufferToTexture(&tex, layered ? r.z : 0, 0, {r.x, r.y, 0}, &from,
                                       src_offset, size, row_pitch, {r.width, r.height, 1});
    });
}

en_status en_cmd_submit(en_cmd *cmd, en_fence *fence, uint64_t value) {
    const std::unique_ptr<en_cmd> owned(cmd);
    return en::guard([&] {
        auto &c = en::deref(owned.get(), "cmd");
        if (!fence) {
            en::submit(*c.device, *c.encoder);
            return;
        }
        ComPtr<rhi::ICommandBuffer> commands;
        en::check(c.encoder->finish(commands.writeRef()), "finish");
        rhi::ICommandBuffer *buffers[] = {commands};
        rhi::IFence *fences[] = {fence->fence};
        rhi::SubmitDesc submit;
        submit.commandBuffers = buffers;
        submit.commandBufferCount = 1;
        submit.signalFences = fences;
        submit.signalFenceValues = &value;
        submit.signalFenceCount = 1;
        en::check(c.device->queue->submit(submit), "submit");
    });
}

} // extern "C"
