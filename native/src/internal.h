// Shim-private definitions shared by gpu.cpp and ui.cpp. Never included by the host.
#pragma once

#include "earth_native.h"

#include <slang-rhi.h>

#include <atomic>
#include <filesystem>
#include <memory>
#include <new>
#include <span>
#include <string>
#include <string_view>

namespace en {

// Thrown inside the shim and converted to en_status by guard() at the C boundary.
struct Error {
    en_status status;
    std::string message;
};

[[noreturn]] void raise(en_status status, std::string message);

// Raises on failure, naming `what` and attaching the last error slang-rhi reported on this thread.
void check(rhi::Result result, std::string_view what);

en_status fail(en_status status, std::string message);

// Last error slang-rhi reported through the debug callback on this thread; reset per C call.
inline thread_local std::string rhi_error;

template <class Body> en_status guard(Body &&body) noexcept {
    rhi_error.clear();
    try {
        body();
        return EN_OK;
    } catch (Error &e) {
        return fail(e.status, std::move(e.message));
    } catch (const std::bad_alloc &) {
        return fail(EN_ERR_OUT_OF_MEMORY, "host out of memory");
    } catch (const std::exception &e) {
        return fail(EN_ERR_INTERNAL, e.what());
    } catch (...) {
        return fail(EN_ERR_INTERNAL, "unknown C++ exception");
    }
}

template <class T> T &deref(T *handle, const char *name) {
    if (!handle)
        raise(EN_ERR_INVALID_ARGUMENT, std::string(name) + " is NULL");
    return *handle;
}

// Persistent shader and pipeline cache with one file per key. Owned by en_device, which outlives
// every slang-rhi reference to it, so the reference count never deletes.
class FileCache final : public rhi::IPersistentCache {
  public:
    explicit FileCache(std::filesystem::path dir) : dir_(std::move(dir)) {}

    SLANG_NO_THROW SlangResult SLANG_MCALL queryInterface(const SlangUUID &uuid,
                                                          void **out) override;
    SLANG_NO_THROW uint32_t SLANG_MCALL addRef() override { return ++refs_; }
    SLANG_NO_THROW uint32_t SLANG_MCALL release() override { return --refs_; }
    SLANG_NO_THROW rhi::Result SLANG_MCALL writeCache(ISlangBlob *key, ISlangBlob *data) override;
    SLANG_NO_THROW rhi::Result SLANG_MCALL queryCache(ISlangBlob *key,
                                                      ISlangBlob **out_data) override;

  private:
    std::filesystem::path entry_path(ISlangBlob *key) const;

    std::filesystem::path dir_;
    std::atomic<uint32_t> refs_{1};
};

// Loads `module` through the device's search paths and links it with `entries`, each
// specialized with `generic_args` when there are any; compiler diagnostics become the error
// message.
rhi::ComPtr<rhi::IShaderProgram> link_program(rhi::IDevice &device, const char *module,
                                              std::span<const char *const> entries,
                                              std::span<const char *const> generic_args = {},
                                              slang::ProgramLayout **out_layout = nullptr);

// Finishes `encoder` and submits it to the device queue without waiting.
void submit(en_device &device, rhi::ICommandEncoder &encoder);

// Ends a pass when its scope exits, including by exception, so the encoder stays usable.
struct PassScope {
    rhi::IPassEncoder *pass;
    ~PassScope() { pass->end(); }
};

} // namespace en

struct en_device {
    std::unique_ptr<en::FileCache> cache; // Declared first so it is destroyed after `device`.
    rhi::ComPtr<rhi::IDevice> device;
    rhi::ComPtr<rhi::ICommandQueue> queue;
    en_device_info info;
};

struct en_texture {
    rhi::ComPtr<rhi::ITexture> texture;
};
