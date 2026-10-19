/*
 * C ABI of libearth_native: GPU compute through slang-rhi, presentation, and
 * Dear ImGui. Only this header crosses the library boundary; no C++ type,
 * exception, or allocation ownership does.
 *
 * Conventions:
 * - Functions returning en_status report detail through en_last_error().
 * - Handles are owned by the caller and released with the matching _destroy;
 *   destroying NULL is a no-op. A device outlives every handle created from it.
 * - Blocking calls are marked; everything else only records or enqueues work.
 */
#ifndef EARTH_NATIVE_H
#define EARTH_NATIVE_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#if defined(EN_BUILDING)
#define EN_API __declspec(dllexport)
#else
#define EN_API __declspec(dllimport)
#endif
#else
#define EN_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef enum en_status {
    EN_OK = 0,
    EN_ERR_INVALID_ARGUMENT = 1,
    EN_ERR_UNSUPPORTED = 2,
    EN_ERR_OUT_OF_MEMORY = 3,
    EN_ERR_SHADER_COMPILE = 4,
    EN_ERR_NOT_FOUND = 5,
    EN_ERR_DEVICE_LOST = 6,
    EN_ERR_SURFACE_OUT_OF_DATE = 7,
    EN_ERR_INTERNAL = 8,
} en_status;

/* Message for the last failed call on this thread; valid until the next call. */
EN_API const char *en_last_error(void);

typedef struct en_device en_device;
typedef struct en_buffer en_buffer;
typedef struct en_texture en_texture;
typedef struct en_sampler en_sampler;
typedef struct en_kernel en_kernel;
typedef struct en_cmd en_cmd;
typedef struct en_surface en_surface;

/* ---------------------------------------------------------------- device */

typedef enum en_backend {
    EN_BACKEND_DEFAULT = 0,
    EN_BACKEND_VULKAN = 1,
    EN_BACKEND_METAL = 2,
    EN_BACKEND_D3D12 = 3,
    EN_BACKEND_CUDA = 4,
} en_backend;

typedef struct en_device_desc {
    en_backend backend;
    uint32_t enable_validation;
    const char *const *shader_search_paths;
    uint32_t shader_search_path_count;
    /* Directory for the persistent shader cache; NULL disables it. */
    const char *shader_cache_path;
} en_device_desc;

typedef struct en_device_info {
    en_backend backend;
    char adapter_name[256];
    uint32_t max_texture_dimension_2d;
    uint32_t max_texture_dimension_3d;
    uint32_t max_texture_array_layers;
    /* Nonzero when rgba32f textures support linear filtering. */
    uint32_t rgba32f_filterable;
} en_device_info;

EN_API en_status en_device_create(const en_device_desc *desc, en_device **out);
EN_API void en_device_destroy(en_device *device);
EN_API void en_device_get_info(const en_device *device, en_device_info *out);
/* Blocking: waits for all submitted work. */
EN_API en_status en_device_wait_idle(en_device *device);

/* ---------------------------------------------------------------- buffers */

typedef enum en_memory {
    EN_MEMORY_DEVICE = 0,
    EN_MEMORY_UPLOAD = 1,
    EN_MEMORY_READBACK = 2,
} en_memory;

enum {
    EN_BUFFER_CONSTANT = 1u << 0,
    EN_BUFFER_SHADER_RESOURCE = 1u << 1,
    EN_BUFFER_UNORDERED_ACCESS = 1u << 2,
    EN_BUFFER_COPY_SOURCE = 1u << 3,
    EN_BUFFER_COPY_DESTINATION = 1u << 4,
};

typedef struct en_buffer_desc {
    uint64_t size;
    /* Stride for structured buffers; 0 for raw or constant buffers. */
    uint32_t element_size;
    uint32_t usage;
    en_memory memory;
    const char *label;
} en_buffer_desc;

/* initial_data may be NULL; otherwise it holds desc->size bytes. */
EN_API en_status en_buffer_create(en_device *device, const en_buffer_desc *desc,
                                  const void *initial_data, en_buffer **out);
EN_API void en_buffer_destroy(en_buffer *buffer);
EN_API en_status en_buffer_upload(en_device *device, en_buffer *buffer, uint64_t offset,
                                  const void *data, uint64_t size);
/* Blocking. */
EN_API en_status en_buffer_read(en_device *device, en_buffer *buffer, uint64_t offset,
                                void *dst, uint64_t size);

/* ---------------------------------------------------------------- textures */

typedef enum en_texture_kind {
    EN_TEXTURE_2D = 0,
    EN_TEXTURE_2D_ARRAY = 1,
    EN_TEXTURE_3D = 2,
} en_texture_kind;

typedef enum en_format {
    EN_FORMAT_R8_UNORM = 0,
    EN_FORMAT_RGBA8_UNORM = 1,
    EN_FORMAT_R16_FLOAT = 2,
    EN_FORMAT_RGBA16_FLOAT = 3,
    EN_FORMAT_R32_FLOAT = 4,
    EN_FORMAT_RGBA32_FLOAT = 5,
} en_format;

enum {
    EN_TEXTURE_SHADER_RESOURCE = 1u << 0,
    EN_TEXTURE_UNORDERED_ACCESS = 1u << 1,
    EN_TEXTURE_COPY_SOURCE = 1u << 2,
    EN_TEXTURE_COPY_DESTINATION = 1u << 3,
};

typedef struct en_texture_desc {
    en_texture_kind kind;
    en_format format;
    uint32_t width;
    uint32_t height;
    /* Depth for EN_TEXTURE_3D, layer count for EN_TEXTURE_2D_ARRAY, else 1. */
    uint32_t depth_or_layers;
    uint32_t usage;
    const char *label;
} en_texture_desc;

/* z is the depth offset for 3D textures and the first layer for arrays. */
typedef struct en_texture_region {
    uint32_t x, y, z;
    uint32_t width, height, depth;
} en_texture_region;

EN_API en_status en_texture_create(en_device *device, const en_texture_desc *desc,
                                   en_texture **out);
EN_API void en_texture_destroy(en_texture *texture);
/* Rows are row_pitch bytes apart; slices (or layers) row_pitch * region->height. */
EN_API en_status en_texture_upload(en_device *device, en_texture *texture,
                                   const en_texture_region *region, const void *data,
                                   uint64_t row_pitch);
/* Blocking. Reads one 2D layer (or 3D slice) tightly packed into dst. */
EN_API en_status en_texture_read(en_device *device, en_texture *texture, uint32_t layer,
                                 void *dst, uint64_t dst_size);

/* ---------------------------------------------------------------- samplers */

typedef enum en_filter {
    EN_FILTER_NEAREST = 0,
    EN_FILTER_LINEAR = 1,
} en_filter;

typedef enum en_address {
    EN_ADDRESS_REPEAT = 0,
    EN_ADDRESS_CLAMP = 1,
} en_address;

typedef struct en_sampler_desc {
    en_filter filter;
    en_address address_u;
    en_address address_v;
    en_address address_w;
} en_sampler_desc;

EN_API en_status en_sampler_create(en_device *device, const en_sampler_desc *desc,
                                   en_sampler **out);
EN_API void en_sampler_destroy(en_sampler *sampler);

/* ---------------------------------------------------------------- kernels */

/* Opaque location of a global shader parameter; valid only for its kernel. */
typedef struct en_param {
    uint64_t opaque[2];
} en_param;

typedef enum en_param_kind {
    EN_PARAM_DATA = 0,
    EN_PARAM_BUFFER = 1,
    EN_PARAM_RW_BUFFER = 2,
    EN_PARAM_TEXTURE = 3,
    EN_PARAM_RW_TEXTURE = 4,
    EN_PARAM_SAMPLER = 5,
} en_param_kind;

typedef struct en_param_info {
    en_param param;
    en_param_kind kind;
    /* Byte size of the uniform data for EN_PARAM_DATA, else 0. */
    uint64_t size;
} en_param_info;

/* Compiles entry `entry` of Slang module `module` (resolved via the search paths). A generic
 * entry takes one Slang expression per generic parameter in `generic_args`, in declaration
 * order, e.g. "KPlanes<8>" or "16"; a plain entry takes none (NULL, 0). Compiler
 * diagnostics are reported through en_last_error(). */
EN_API en_status en_kernel_create(en_device *device, const char *module, const char *entry,
                                  const char *const *generic_args, uint32_t generic_arg_count,
                                  en_kernel **out);
EN_API void en_kernel_destroy(en_kernel *kernel);
EN_API void en_kernel_thread_group_size(const en_kernel *kernel, uint32_t out[3]);
/* path names a global parameter, e.g. "frame" or "earth_maps". Kernels take global-scope
 * parameters only; a path may continue into a global struct ("frame.exposure") or array
 * ("maps[2]"), but a ConstantBuffer global is bound as a whole. */
EN_API en_status en_kernel_find_param(const en_kernel *kernel, const char *path,
                                      en_param_info *out);
/* Offset and size of `field` within the struct-typed (or ConstantBuffer) parameter `path`. */
EN_API en_status en_kernel_find_field(const en_kernel *kernel, const char *path,
                                      const char *field, uint64_t *offset, uint64_t *size);

/* ---------------------------------------------------------------- commands */

typedef enum en_binding_kind {
    EN_BIND_DATA = 0,
    EN_BIND_BUFFER = 1,
    EN_BIND_TEXTURE = 2,
    EN_BIND_SAMPLER = 3,
} en_binding_kind;

/* Exactly the field selected by kind is read. EN_BIND_DATA writes the whole parameter:
 * size must equal its en_param_info.size. */
typedef struct en_binding {
    en_param param;
    en_binding_kind kind;
    const void *data;
    uint64_t size;
    en_buffer *buffer;
    en_texture *texture;
    en_sampler *sampler;
} en_binding;

EN_API en_status en_cmd_begin(en_device *device, en_cmd **out);
/* Consecutive dispatches in one command list are separated by a UAV barrier. */
EN_API en_status en_cmd_dispatch(en_cmd *cmd, en_kernel *kernel, const en_binding *bindings,
                                 uint32_t binding_count, uint32_t groups_x, uint32_t groups_y,
                                 uint32_t groups_z);
/* Submits and releases cmd, also on failure. */
EN_API en_status en_cmd_submit(en_cmd *cmd);

/* ---------------------------------------------------------------- surface */

typedef enum en_window_kind {
    EN_WINDOW_XLIB = 0,
    EN_WINDOW_WIN32 = 1,
    EN_WINDOW_COCOA = 2,
} en_window_kind;

/* Xlib: display = Display*, window = Window. Win32: window = HWND. Cocoa: window = NSWindow*. */
typedef struct en_window_handle {
    en_window_kind kind;
    void *display;
    uint64_t window;
} en_window_handle;

EN_API en_status en_surface_create(en_device *device, const en_window_handle *window,
                                   en_surface **out);
EN_API void en_surface_destroy(en_surface *surface);
EN_API en_status en_surface_configure(en_surface *surface, uint32_t width, uint32_t height,
                                      uint32_t vsync);
/* Draws `image` (rgba8, any size) stretched over the surface, then the ImGui frame if
 * one was begun, and presents. Returns EN_ERR_SURFACE_OUT_OF_DATE when the surface
 * must be reconfigured. */
EN_API en_status en_present(en_device *device, en_surface *surface, en_texture *image);

/* ---------------------------------------------------------------- ui */

/* One ImGui context per process. sdl_window is an SDL_Window*. Widgets use the
 * exported dcimgui API (dcimgui.h); an ImTextureID is an en_texture*. */
EN_API en_status en_ui_init(en_device *device, en_surface *surface, void *sdl_window);
EN_API void en_ui_shutdown(void);
/* sdl_event is a const SDL_Event*. */
EN_API void en_ui_process_event(const void *sdl_event);
EN_API void en_ui_new_frame(void);

#ifdef __cplusplus
}
#endif

#endif
