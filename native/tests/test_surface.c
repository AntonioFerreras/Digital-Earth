// Presentation through an SDL3 window: frames with and without ImGui, a resize, and teardown.
// Exits 77 (skipped) when no X11 display is available.

#include "earth_native.h"

#include <SDL3/SDL.h>
#include <dcimgui.h>

#include <stdio.h>

static int failures = 0;

#define CHECK_OK(call)                                                                             \
    do {                                                                                           \
        en_status status_ = (call);                                                                \
        if (status_ != EN_OK) {                                                                    \
            fprintf(stderr, "%s:%d: %s returned %d: %s\n", __FILE__, __LINE__, #call, status_,     \
                    en_last_error());                                                              \
            ++failures;                                                                            \
        }                                                                                          \
    } while (0)

static void pump_events(void) {
    SDL_Event event;
    while (SDL_PollEvent(&event))
        en_ui_process_event(&event);
}

// Presents one frame, reconfiguring once if the swapchain went out of date.
static void present(en_device *device, en_surface *surface, en_texture *image, SDL_Window *window) {
    en_status status = en_present(device, surface, image);
    if (status == EN_ERR_SURFACE_OUT_OF_DATE) {
        int w, h;
        SDL_GetWindowSizeInPixels(window, &w, &h);
        CHECK_OK(en_surface_configure(surface, (uint32_t)w, (uint32_t)h, 0));
        return;
    }
    CHECK_OK(status);
}

static void ui_frame(en_texture *image, float font_size) {
    en_ui_new_frame();
    ImGui_Begin("earth_native", NULL, 0);
    ImGui_Text("frame text");
    // A new font size makes ImGui bake glyphs, which exercises texture updates.
    ImGui_PushFontFloat(NULL, font_size);
    ImGui_Text("size %.0f", font_size);
    ImGui_PopFont();
    const ImTextureRef image_ref = {NULL, (ImTextureID)(uintptr_t)image};
    ImGui_Image(image_ref, (ImVec2){64, 64});
    ImGui_End();
}

int main(void) {
    SDL_SetHint(SDL_HINT_VIDEO_DRIVER, "x11");
    if (!SDL_Init(SDL_INIT_VIDEO)) {
        printf("skipped: %s\n", SDL_GetError());
        return 77;
    }
    SDL_Window *window = SDL_CreateWindow("earth_native test", 256, 256, SDL_WINDOW_RESIZABLE);
    if (!window) {
        printf("skipped: %s\n", SDL_GetError());
        SDL_Quit();
        return 77;
    }
    SDL_PropertiesID props = SDL_GetWindowProperties(window);
    const en_window_handle handle = {
        EN_WINDOW_XLIB, SDL_GetPointerProperty(props, SDL_PROP_WINDOW_X11_DISPLAY_POINTER, NULL),
        (uint64_t)SDL_GetNumberProperty(props, SDL_PROP_WINDOW_X11_WINDOW_NUMBER, 0)};

    const char *paths[] = {EN_SHADER_DIR};
    const en_device_desc desc = {EN_BACKEND_DEFAULT, 1, paths, 1, NULL};
    en_device *device = NULL;
    en_surface *surface = NULL;
    en_texture *image = NULL;
    CHECK_OK(en_device_create(&desc, &device));
    if (device)
        CHECK_OK(en_surface_create(device, &handle, &surface));
    if (!surface)
        goto teardown;
    CHECK_OK(en_surface_configure(surface, 256, 256, 0));

    enum { size = 64 };
    static uint8_t pixels[size][size][4];
    for (int y = 0; y < size; ++y)
        for (int x = 0; x < size; ++x) {
            pixels[y][x][0] = (uint8_t)(x * 4);
            pixels[y][x][1] = (uint8_t)(y * 4);
            pixels[y][x][2] = 128;
            pixels[y][x][3] = 255;
        }
    const en_texture_desc image_desc = {EN_TEXTURE_2D, EN_FORMAT_RGBA8_UNORM, size, size, 1,
                                        EN_TEXTURE_SHADER_RESOURCE | EN_TEXTURE_COPY_DESTINATION,
                                        "image"};
    CHECK_OK(en_texture_create(device, &image_desc, &image));
    const en_texture_region whole = {0, 0, 0, size, size, 1};
    CHECK_OK(en_texture_upload(device, image, &whole, pixels, size * 4));

    for (int i = 0; i < 4; ++i) {
        pump_events();
        present(device, surface, i == 0 ? NULL : image, window);
    }

    CHECK_OK(en_ui_init(device, surface, window));
    for (int i = 0; i < 8; ++i) {
        pump_events();
        ui_frame(image, 13.0f + (float)(4 * (i / 2)));
        present(device, surface, image, window);
    }

    SDL_SetWindowSize(window, 320, 200);
    SDL_SyncWindow(window);
    for (int i = 0; i < 6; ++i) {
        pump_events();
        ui_frame(image, 20.0f);
        present(device, surface, image, window);
    }
    // A frame begun but never presented is closed by the next one.
    ui_frame(image, 20.0f);
    ui_frame(image, 20.0f);
    present(device, surface, image, window);

    CHECK_OK(en_device_wait_idle(device));
    en_ui_shutdown();

teardown:
    en_texture_destroy(image);
    en_surface_destroy(surface);
    en_device_destroy(device);
    SDL_DestroyWindow(window);
    SDL_Quit();
    printf(failures ? "FAILED: %d\n" : "all passed\n", failures);
    return failures ? 1 : 0;
}
