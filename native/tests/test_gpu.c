// Compute-side tests of libearth_native on the default backend with validation enabled.

#include "earth_native.h"

#include <dirent.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;

#define CHECK(cond)                                                                                \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond);               \
            ++failures;                                                                            \
        }                                                                                          \
    } while (0)

#define CHECK_STATUS(call, expected)                                                               \
    do {                                                                                           \
        en_status status_ = (call);                                                                \
        if (status_ != (expected)) {                                                               \
            fprintf(stderr, "%s:%d: %s returned %d, expected %d: %s\n", __FILE__, __LINE__, #call, \
                    status_, expected, en_last_error());                                           \
            ++failures;                                                                            \
        }                                                                                          \
    } while (0)

#define CHECK_OK(call) CHECK_STATUS(call, EN_OK)

// Mirror of `Frame` in kernels.slang.
struct frame {
    float tint[4];
    uint32_t counts[4];
    float scale;
    uint32_t offset;
};

static en_device *create_device(const char *cache_path) {
    const char *paths[] = {EN_TEST_DIR};
    en_device_desc desc = {EN_BACKEND_DEFAULT, 1, paths, 1, cache_path};
    en_device *device = NULL;
    CHECK_OK(en_device_create(&desc, &device));
    return device;
}

static en_param param(en_kernel *kernel, const char *path, en_param_kind kind) {
    en_param_info info = {0};
    CHECK_OK(en_kernel_find_param(kernel, path, &info));
    CHECK(info.kind == kind);
    return info.param;
}

static void test_device_info(en_device *device) {
    en_device_info info;
    en_device_get_info(device, &info);
    printf("adapter: %s, backend %d, 2D %u, 3D %u, layers %u, rgba32f filterable %u\n",
           info.adapter_name, info.backend, info.max_texture_dimension_2d,
           info.max_texture_dimension_3d, info.max_texture_array_layers, info.rgba32f_filterable);
    CHECK(info.backend != EN_BACKEND_DEFAULT);
    CHECK(info.max_texture_dimension_2d >= 4096);
}

static void test_buffer_round_trip(en_device *device) {
    uint32_t data[256], back[256];
    for (int i = 0; i < 256; ++i)
        data[i] = 0xA5000000u + i;
    const en_buffer_desc desc = {sizeof data, 4,
                                 EN_BUFFER_SHADER_RESOURCE | EN_BUFFER_COPY_SOURCE |
                                     EN_BUFFER_COPY_DESTINATION,
                                 EN_MEMORY_DEVICE, "round trip"};
    en_buffer *buffer = NULL;
    CHECK_OK(en_buffer_create(device, &desc, data, &buffer));
    CHECK_OK(en_buffer_read(device, buffer, 0, back, sizeof back));
    CHECK(memcmp(data, back, sizeof data) == 0);

    const uint32_t patch[4] = {1, 2, 3, 4};
    CHECK_OK(en_buffer_upload(device, buffer, 16, patch, sizeof patch));
    CHECK_OK(en_buffer_read(device, buffer, 16, back, sizeof patch));
    CHECK(memcmp(patch, back, sizeof patch) == 0);
    CHECK_STATUS(en_buffer_upload(device, buffer, sizeof data - 4, patch, sizeof patch),
                 EN_ERR_INVALID_ARGUMENT);
    en_buffer_destroy(buffer);

    const en_buffer_desc upload_desc = {sizeof patch, 0, EN_BUFFER_COPY_SOURCE, EN_MEMORY_UPLOAD,
                                        "upload"};
    CHECK_OK(en_buffer_create(device, &upload_desc, NULL, &buffer));
    CHECK_OK(en_buffer_upload(device, buffer, 0, patch, sizeof patch));
    CHECK_OK(en_buffer_read(device, buffer, 0, back, sizeof patch));
    CHECK(memcmp(patch, back, sizeof patch) == 0);
    en_buffer_destroy(buffer);
}

static void test_texture_round_trips(en_device *device) {
    const uint32_t usage = EN_TEXTURE_SHADER_RESOURCE | EN_TEXTURE_COPY_SOURCE |
                           EN_TEXTURE_COPY_DESTINATION;

    // 2D array: a 3x2 region at (1, 1) of layer 2, read back as the whole layer.
    const en_texture_desc array_desc = {EN_TEXTURE_2D_ARRAY, EN_FORMAT_RGBA8_UNORM, 5, 4, 3, usage,
                                        "array"};
    en_texture *array = NULL;
    CHECK_OK(en_texture_create(device, &array_desc, &array));
    uint32_t zeros[5 * 4] = {0};
    for (uint32_t layer = 0; layer < 3; ++layer) {
        const en_texture_region whole = {0, 0, layer, 5, 4, 1};
        CHECK_OK(en_texture_upload(device, array, &whole, zeros, 5 * 4));
    }
    // Source rows are padded to 4 texels to exercise row_pitch.
    uint32_t region_data[2][4];
    for (int y = 0; y < 2; ++y)
        for (int x = 0; x < 4; ++x)
            region_data[y][x] = 0xFF000000u | (uint32_t)(y << 8) | (uint32_t)x;
    const en_texture_region region = {1, 1, 2, 3, 2, 1};
    CHECK_OK(en_texture_upload(device, array, &region, region_data, 16));
    uint32_t layer2[4][5];
    CHECK_OK(en_texture_read(device, array, 2, layer2, sizeof layer2));
    CHECK(layer2[0][0] == 0 && layer2[1][0] == 0 && layer2[1][4] == 0);
    CHECK(layer2[1][1] == region_data[0][0] && layer2[2][3] == region_data[1][2]);
    uint32_t layer1[4][5];
    CHECK_OK(en_texture_read(device, array, 1, layer1, sizeof layer1));
    CHECK(layer1[1][1] == 0);
    CHECK_STATUS(en_texture_read(device, array, 3, layer1, sizeof layer1), EN_ERR_INVALID_ARGUMENT);
    en_texture_destroy(array);

    // 3D r32f: two slices uploaded at once from z = 1, slice 2 read back.
    const en_texture_desc volume_desc = {EN_TEXTURE_3D, EN_FORMAT_R32_FLOAT, 4, 3, 4, usage,
                                         "volume"};
    en_texture *volume = NULL;
    CHECK_OK(en_texture_create(device, &volume_desc, &volume));
    float slices[2][3][4];
    for (int z = 0; z < 2; ++z)
        for (int y = 0; y < 3; ++y)
            for (int x = 0; x < 4; ++x)
                slices[z][y][x] = (float)(100 * z + 10 * y + x);
    const en_texture_region volume_region = {0, 0, 1, 4, 3, 2};
    CHECK_OK(en_texture_upload(device, volume, &volume_region, slices, 4 * sizeof(float)));
    float slice2[3][4];
    CHECK_OK(en_texture_read(device, volume, 2, slice2, sizeof slice2));
    CHECK(memcmp(slice2, slices[1], sizeof slice2) == 0);
    en_texture_destroy(volume);
}

static void test_reflection(en_kernel *kernel) {
    uint32_t group[3];
    en_kernel_thread_group_size(kernel, group);
    CHECK(group[0] == 64 && group[1] == 1 && group[2] == 1);

    static const struct {
        const char *field;
        uint64_t offset, size;
    } fields[] = {{"tint", 0, 16}, {"counts", 16, 16}, {"scale", 32, 4}, {"offset", 36, 4}};
    const char *params[] = {"frame", "frame_cb"};
    for (int p = 0; p < 2; ++p) {
        for (int f = 0; f < 4; ++f) {
            uint64_t offset = ~0ull, size = ~0ull;
            CHECK_OK(en_kernel_find_field(kernel, params[p], fields[f].field, &offset, &size));
            if (offset != fields[f].offset || size != fields[f].size) {
                fprintf(stderr, "%s.%s at %llu+%llu, expected %llu+%llu\n", params[p],
                        fields[f].field, (unsigned long long)offset, (unsigned long long)size,
                        (unsigned long long)fields[f].offset, (unsigned long long)fields[f].size);
                ++failures;
            }
        }
        en_param_info info;
        CHECK_OK(en_kernel_find_param(kernel, params[p], &info));
        CHECK(info.kind == EN_PARAM_DATA && info.size == sizeof(struct frame));
    }

    en_param_info info;
    CHECK_OK(en_kernel_find_param(kernel, "frame.scale", &info));
    CHECK(info.kind == EN_PARAM_DATA && info.size == 4);
    param(kernel, "values", EN_PARAM_RW_BUFFER);
    param(kernel, "input_values", EN_PARAM_BUFFER);
    param(kernel, "target", EN_PARAM_RW_TEXTURE);
    param(kernel, "source", EN_PARAM_TEXTURE);
    param(kernel, "linear_sampler", EN_PARAM_SAMPLER);
    param(kernel, "maps[1]", EN_PARAM_TEXTURE);

    uint64_t offset, size;
    CHECK_STATUS(en_kernel_find_param(kernel, "missing", &info), EN_ERR_NOT_FOUND);
    CHECK_STATUS(en_kernel_find_param(kernel, "frame_cb.scale", &info), EN_ERR_UNSUPPORTED);
    CHECK_STATUS(en_kernel_find_param(kernel, "maps", &info), EN_ERR_UNSUPPORTED);
    CHECK_STATUS(en_kernel_find_field(kernel, "frame", "missing", &offset, &size),
                 EN_ERR_NOT_FOUND);
    CHECK_STATUS(en_kernel_find_field(kernel, "values", "x", &offset, &size),
                 EN_ERR_INVALID_ARGUMENT);
}

static void test_dispatch(en_device *device) {
    en_kernel *fill = NULL, *doubled = NULL, *copy = NULL;
    CHECK_OK(en_kernel_create(device, "kernels", "fill", NULL, 0, &fill));
    CHECK_OK(en_kernel_create(device, "kernels", "double_values", NULL, 0, &doubled));
    CHECK_OK(en_kernel_create(device, "kernels", "copy_plus_one", NULL, 0, &copy));
    if (!fill || !doubled || !copy)
        return;
    test_reflection(fill);

    enum { count = 256 };
    const en_buffer_desc desc = {count * 4, 4,
                                 EN_BUFFER_SHADER_RESOURCE | EN_BUFFER_UNORDERED_ACCESS |
                                     EN_BUFFER_COPY_SOURCE,
                                 EN_MEMORY_DEVICE, "values"};
    en_buffer *values = NULL, *copied = NULL;
    CHECK_OK(en_buffer_create(device, &desc, NULL, &values));
    CHECK_OK(en_buffer_create(device, &desc, NULL, &copied));

    const struct frame frame = {{1, 1, 1, 1}, {3, 0, 0, 0}, 1.0f, 7};
    const struct frame frame_cb = {{0}, {0}, 0.0f, 5};
    en_binding fill_bindings[2] = {0};
    fill_bindings[0].param = param(fill, "frame", EN_PARAM_DATA);
    fill_bindings[0].kind = EN_BIND_DATA;
    fill_bindings[0].data = &frame;
    fill_bindings[0].size = sizeof frame;
    fill_bindings[1].param = param(fill, "values", EN_PARAM_RW_BUFFER);
    fill_bindings[1].kind = EN_BIND_BUFFER;
    fill_bindings[1].buffer = values;

    en_binding double_bindings[2] = {0};
    double_bindings[0].param = param(doubled, "frame_cb", EN_PARAM_DATA);
    double_bindings[0].kind = EN_BIND_DATA;
    double_bindings[0].data = &frame_cb;
    double_bindings[0].size = sizeof frame_cb;
    double_bindings[1].param = param(doubled, "values", EN_PARAM_RW_BUFFER);
    double_bindings[1].kind = EN_BIND_BUFFER;
    double_bindings[1].buffer = values;

    en_binding copy_bindings[2] = {0};
    copy_bindings[0].param = param(copy, "input_values", EN_PARAM_BUFFER);
    copy_bindings[0].kind = EN_BIND_BUFFER;
    copy_bindings[0].buffer = values;
    copy_bindings[1].param = param(copy, "values", EN_PARAM_RW_BUFFER);
    copy_bindings[1].kind = EN_BIND_BUFFER;
    copy_bindings[1].buffer = copied;

    // Three dependent dispatches in one command list: UAV -> UAV -> SRV.
    en_cmd *cmd = NULL;
    CHECK_OK(en_cmd_begin(device, &cmd));
    CHECK_OK(en_cmd_dispatch(cmd, fill, fill_bindings, 2, count / 64, 1, 1));
    CHECK_OK(en_cmd_dispatch(cmd, doubled, double_bindings, 2, count / 64, 1, 1));
    CHECK_OK(en_cmd_dispatch(cmd, copy, copy_bindings, 2, count / 64, 1, 1));

    // Rejected bindings leave the command list usable.
    en_binding wrong = fill_bindings[1];
    wrong.kind = EN_BIND_TEXTURE;
    CHECK_STATUS(en_cmd_dispatch(cmd, fill, &wrong, 1, 1, 1, 1), EN_ERR_INVALID_ARGUMENT);
    en_binding too_big = fill_bindings[0];
    too_big.size = sizeof frame + 4;
    CHECK_STATUS(en_cmd_dispatch(cmd, fill, &too_big, 1, 1, 1, 1), EN_ERR_INVALID_ARGUMENT);
    en_binding too_small = fill_bindings[0];
    too_small.size = sizeof frame - 4;
    CHECK_STATUS(en_cmd_dispatch(cmd, fill, &too_small, 1, 1, 1, 1), EN_ERR_INVALID_ARGUMENT);
    CHECK_OK(en_cmd_submit(cmd));

    uint32_t result[count];
    CHECK_OK(en_buffer_read(device, copied, 0, result, sizeof result));
    int mismatches = 0;
    for (uint32_t i = 0; i < count; ++i)
        mismatches += result[i] != (i * 3 + 7) * 2 + 5 + 1;
    CHECK(mismatches == 0);

    en_buffer_destroy(values);
    en_buffer_destroy(copied);
    en_kernel_destroy(fill);
    en_kernel_destroy(doubled);
    en_kernel_destroy(copy);
}

static void test_texture_dispatch(en_device *device) {
    en_kernel *paint = NULL, *sample = NULL;
    CHECK_OK(en_kernel_create(device, "kernels", "paint", NULL, 0, &paint));
    CHECK_OK(en_kernel_create(device, "kernels", "sample_between", NULL, 0, &sample));
    if (!paint || !sample)
        return;
    uint32_t group[3];
    en_kernel_thread_group_size(paint, group);
    CHECK(group[0] == 8 && group[1] == 8 && group[2] == 1);

    enum { w = 32, h = 16 };
    const en_texture_desc desc = {EN_TEXTURE_2D, EN_FORMAT_RGBA32_FLOAT, w, h, 1,
                                  EN_TEXTURE_UNORDERED_ACCESS | EN_TEXTURE_COPY_SOURCE, "painted"};
    en_texture *target = NULL;
    CHECK_OK(en_texture_create(device, &desc, &target));
    const struct frame frame = {{1, 2, 1, 1}, {0}, 0.25f, 0};
    en_binding bindings[2] = {0};
    bindings[0].param = param(paint, "frame", EN_PARAM_DATA);
    bindings[0].kind = EN_BIND_DATA;
    bindings[0].data = &frame;
    bindings[0].size = sizeof frame;
    bindings[1].param = param(paint, "target", EN_PARAM_RW_TEXTURE);
    bindings[1].kind = EN_BIND_TEXTURE;
    bindings[1].texture = target;
    en_cmd *cmd = NULL;
    CHECK_OK(en_cmd_begin(device, &cmd));
    CHECK_OK(en_cmd_dispatch(cmd, paint, bindings, 2, w / 8, h / 8, 1));
    CHECK_OK(en_cmd_submit(cmd));

    static float texels[h][w][4];
    CHECK_OK(en_texture_read(device, target, 0, texels, sizeof texels));
    int mismatches = 0;
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x)
            mismatches += texels[y][x][0] != (float)x || texels[y][x][1] != 2.0f * y ||
                          texels[y][x][2] != 0.25f || texels[y][x][3] != 1.0f;
    CHECK(mismatches == 0);
    en_texture_destroy(target);

    // Linear filtering halfway between two r32f texels yields their mean.
    const en_texture_desc source_desc = {EN_TEXTURE_2D, EN_FORMAT_R32_FLOAT, 2, 1, 1,
                                         EN_TEXTURE_SHADER_RESOURCE | EN_TEXTURE_COPY_DESTINATION,
                                         "source"};
    en_texture *source = NULL;
    CHECK_OK(en_texture_create(device, &source_desc, &source));
    const float source_texels[2] = {1.0f, 3.0f};
    const en_texture_region whole = {0, 0, 0, 2, 1, 1};
    CHECK_OK(en_texture_upload(device, source, &whole, source_texels, sizeof source_texels));
    const en_sampler_desc sampler_desc = {EN_FILTER_LINEAR, EN_ADDRESS_CLAMP, EN_ADDRESS_CLAMP,
                                          EN_ADDRESS_CLAMP};
    en_sampler *sampler = NULL;
    CHECK_OK(en_sampler_create(device, &sampler_desc, &sampler));
    const en_buffer_desc result_desc = {4, 4, EN_BUFFER_UNORDERED_ACCESS | EN_BUFFER_COPY_SOURCE,
                                        EN_MEMORY_DEVICE, "results"};
    en_buffer *results = NULL;
    CHECK_OK(en_buffer_create(device, &result_desc, NULL, &results));

    en_binding sample_bindings[3] = {0};
    sample_bindings[0].param = param(sample, "source", EN_PARAM_TEXTURE);
    sample_bindings[0].kind = EN_BIND_TEXTURE;
    sample_bindings[0].texture = source;
    sample_bindings[1].param = param(sample, "linear_sampler", EN_PARAM_SAMPLER);
    sample_bindings[1].kind = EN_BIND_SAMPLER;
    sample_bindings[1].sampler = sampler;
    sample_bindings[2].param = param(sample, "results", EN_PARAM_RW_BUFFER);
    sample_bindings[2].kind = EN_BIND_BUFFER;
    sample_bindings[2].buffer = results;
    CHECK_OK(en_cmd_begin(device, &cmd));
    CHECK_OK(en_cmd_dispatch(cmd, sample, sample_bindings, 3, 1, 1, 1));
    CHECK_OK(en_cmd_submit(cmd));
    float mean = 0.0f;
    CHECK_OK(en_buffer_read(device, results, 0, &mean, sizeof mean));
    CHECK(fabsf(mean - 2.0f) < 1e-3f);

    en_buffer_destroy(results);
    en_sampler_destroy(sampler);
    en_texture_destroy(source);
    en_kernel_destroy(paint);
    en_kernel_destroy(sample);
}

// A generic entry compiles once per argument list: a type and a value here.
static void test_generic_entry(en_device *device) {
    enum { count = 64 };
    const en_buffer_desc desc = {count * 4, 4, EN_BUFFER_UNORDERED_ACCESS | EN_BUFFER_COPY_SOURCE,
                                 EN_MEMORY_DEVICE, "values"};
    en_buffer *values = NULL;
    CHECK_OK(en_buffer_create(device, &desc, NULL, &values));
    const char *args[] = {"AddSeven", "3"};
    en_kernel *kernel = NULL;
    CHECK_OK(en_kernel_create(device, "kernels", "fill_generic", args, 2, &kernel));
    if (!kernel || !values)
        return;
    en_binding binding = {0};
    binding.param = param(kernel, "values", EN_PARAM_RW_BUFFER);
    binding.kind = EN_BIND_BUFFER;
    binding.buffer = values;
    en_cmd *cmd = NULL;
    CHECK_OK(en_cmd_begin(device, &cmd));
    CHECK_OK(en_cmd_dispatch(cmd, kernel, &binding, 1, 1, 1, 1));
    CHECK_OK(en_cmd_submit(cmd));
    uint32_t result[count];
    CHECK_OK(en_buffer_read(device, values, 0, result, sizeof result));
    int mismatches = 0;
    for (uint32_t i = 0; i < count; ++i)
        mismatches += result[i] != i * 3 + 7;
    CHECK(mismatches == 0);
    en_kernel_destroy(kernel);
    en_buffer_destroy(values);

    const char *unknown[] = {"NoSuchType", "3"};
    CHECK_STATUS(en_kernel_create(device, "kernels", "fill_generic", unknown, 2, &kernel),
                 EN_ERR_SHADER_COMPILE);
    CHECK(kernel == NULL);
    CHECK_STATUS(en_kernel_create(device, "kernels", "fill_generic", NULL, 1, &kernel),
                 EN_ERR_INVALID_ARGUMENT);
}

static void test_diagnostics(en_device *device) {
    en_kernel *kernel = NULL;
    CHECK_STATUS(en_kernel_create(device, "broken", "main", NULL, 0, &kernel), EN_ERR_SHADER_COMPILE);
    CHECK(kernel == NULL);
    CHECK(strstr(en_last_error(), "broken.slang") != NULL);
    CHECK(strstr(en_last_error(), "undeclared_function") != NULL);
    CHECK_STATUS(en_kernel_create(device, "kernels", "no_such_entry", NULL, 0, &kernel), EN_ERR_NOT_FOUND);
    CHECK_STATUS(en_kernel_create(device, "no_such_module", "main", NULL, 0, &kernel),
                 EN_ERR_SHADER_COMPILE);
}

static int count_cache_entries(const char *dir) {
    DIR *d = opendir(dir);
    if (!d)
        return 0;
    int count = 0;
    for (struct dirent *e; (e = readdir(d));)
        count += strstr(e->d_name, ".bin") != NULL;
    closedir(d);
    return count;
}

// A second device compiling the same kernel from a populated cache must still work.
static void test_shader_cache(void) {
    char dir[] = "/tmp/earth_native_cache_XXXXXX";
    if (!mkdtemp(dir)) {
        perror("mkdtemp");
        ++failures;
        return;
    }
    int entries = 0;
    for (int run = 0; run < 2; ++run) {
        en_device *device = create_device(dir);
        en_kernel *kernel = NULL;
        CHECK_OK(en_kernel_create(device, "kernels", "fill", NULL, 0, &kernel));
        en_kernel_destroy(kernel);
        en_device_destroy(device);
        const int now = count_cache_entries(dir);
        CHECK(now > 0);
        if (run == 1)
            CHECK(now == entries);
        entries = now;
    }
    printf("shader cache: %d entries\n", entries);
    char command[96];
    snprintf(command, sizeof command, "rm -rf %s", dir);
    CHECK(system(command) == 0);
}

int main(void) {
    en_device *device = create_device(NULL);
    if (!device)
        return 1;
    test_device_info(device);
    test_buffer_round_trip(device);
    test_texture_round_trips(device);
    test_dispatch(device);
    test_texture_dispatch(device);
    test_generic_entry(device);
    test_diagnostics(device);
    CHECK_OK(en_device_wait_idle(device));
    en_device_destroy(device);
    test_shader_cache();
    printf(failures ? "FAILED: %d\n" : "all passed\n", failures);
    return failures ? 1 : 0;
}
