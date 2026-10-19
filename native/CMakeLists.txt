cmake_minimum_required(VERSION 3.25)
project(earth_native LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_VISIBILITY_PRESET hidden)
set(CMAKE_VISIBILITY_INLINES_HIDDEN ON)

include(FetchContent)

# slang-rhi guarantees neither API nor ABI stability between revisions; bump deliberately.
set(SLANG_RHI_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(SLANG_RHI_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(SLANG_RHI_BUILD_SHARED OFF CACHE BOOL "" FORCE)
set(SLANG_RHI_ENABLE_CUDA OFF CACHE BOOL "" FORCE)
set(SLANG_RHI_ENABLE_OPTIX OFF CACHE BOOL "" FORCE)
set(SLANG_RHI_ENABLE_WGPU OFF CACHE BOOL "" FORCE)
set(SLANG_RHI_ENABLE_CPU OFF CACHE BOOL "" FORCE)
FetchContent_Declare(slang_rhi
    GIT_REPOSITORY https://github.com/shader-slang/slang-rhi.git
    GIT_TAG d825f483e48c6a4d64019347f14685d3a579d83b
    GIT_SHALLOW FALSE
    GIT_SUBMODULES_RECURSE TRUE)

# dear_bindings releases are generated against one exact ImGui tag.
FetchContent_Declare(imgui
    GIT_REPOSITORY https://github.com/ocornut/imgui.git
    GIT_TAG 4e02e84cbb85d48881388e7090c2dfa1708c02a1)
FetchContent_Declare(dear_bindings
    URL https://github.com/dearimgui/dear_bindings/releases/download/DearBindings_v0.26_ImGui_v1.93.0-docking/DearBindings_v0.26_ImGui_v1.93.0-docking.zip)

FetchContent_MakeAvailable(slang_rhi imgui dear_bindings)

find_package(SDL3 REQUIRED CONFIG)

add_library(earth_native SHARED
    src/gpu.cpp
    src/ui.cpp
    ${imgui_SOURCE_DIR}/imgui.cpp
    ${imgui_SOURCE_DIR}/imgui_demo.cpp
    ${imgui_SOURCE_DIR}/imgui_draw.cpp
    ${imgui_SOURCE_DIR}/imgui_tables.cpp
    ${imgui_SOURCE_DIR}/imgui_widgets.cpp
    ${imgui_SOURCE_DIR}/backends/imgui_impl_sdl3.cpp
    ${dear_bindings_SOURCE_DIR}/dcimgui.cpp
    ${dear_bindings_SOURCE_DIR}/backends/dcimgui_impl_sdl3.cpp)
target_include_directories(earth_native
    PUBLIC include
    PRIVATE ${imgui_SOURCE_DIR} ${imgui_SOURCE_DIR}/backends ${dear_bindings_SOURCE_DIR}
    ${dear_bindings_SOURCE_DIR}/backends)
# Exports the dcimgui C API from the shared library for the Zig host.
target_compile_definitions(earth_native PRIVATE EN_BUILDING
    "CIMGUI_API=extern \"C\" __attribute__((visibility(\"default\")))"
    "CIMGUI_IMPL_API=extern \"C\" __attribute__((visibility(\"default\")))")
target_link_libraries(earth_native PRIVATE slang-rhi SDL3::SDL3)
set_target_properties(earth_native PROPERTIES BUILD_RPATH_USE_ORIGIN ON INSTALL_RPATH "$ORIGIN")
# Hidden visibility misses libstdc++ template instantiations; the version script is exact.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_options(earth_native PRIVATE
        "-Wl,--version-script=${CMAKE_CURRENT_SOURCE_DIR}/src/exports.map")
    set_target_properties(earth_native PROPERTIES LINK_DEPENDS
        "${CMAKE_CURRENT_SOURCE_DIR}/src/exports.map")
endif()

install(TARGETS earth_native LIBRARY DESTINATION lib RUNTIME DESTINATION bin)
install(FILES include/earth_native.h DESTINATION include)
install(FILES
    ${imgui_SOURCE_DIR}/imconfig.h
    ${dear_bindings_SOURCE_DIR}/dcimgui.h
    ${dear_bindings_SOURCE_DIR}/backends/dcimgui_impl_sdl3.h
    DESTINATION include)
install(DIRECTORY shaders/ DESTINATION share/earth_native/shaders)
# slang-rhi loads the Slang compiler at runtime; ship it beside the library.
# libslang-llvm only serves CPU targets and is left out.
install(DIRECTORY ${SLANG_RHI_SLANG_BINARY_DIR}/lib/ DESTINATION lib
    FILES_MATCHING PATTERN "libslang-compiler*" PATTERN "libslang.so*" PATTERN "libslang-glsl*"
    PATTERN "libslang-rt*" PATTERN "slang-standard-module-*"
    PATTERN "cmake" EXCLUDE PATTERN "pkgconfig" EXCLUDE)
install(PROGRAMS ${SLANG_RHI_SLANG_BINARY_DIR}/bin/slangc DESTINATION bin)

option(EN_BUILD_TESTS "Build the C tests of earth_native (run with ctest)" OFF)
if(EN_BUILD_TESTS)
    enable_testing()
    foreach(name gpu surface)
        add_executable(test_${name} tests/test_${name}.c)
        target_link_libraries(test_${name} PRIVATE earth_native SDL3::SDL3 m)
        # The prebuilt Slang needs a newer glibc than the conda sysroot; the host's resolves it.
        target_link_options(test_${name} PRIVATE -Wl,--allow-shlib-undefined)
        target_include_directories(test_${name} PRIVATE ${imgui_SOURCE_DIR} ${dear_bindings_SOURCE_DIR})
        target_compile_definitions(test_${name} PRIVATE
            EN_TEST_DIR="${CMAKE_CURRENT_SOURCE_DIR}/tests"
            EN_SHADER_DIR="${CMAKE_CURRENT_SOURCE_DIR}/shaders")
        add_test(NAME ${name} COMMAND test_${name})
        # Validation messages only reach stderr, so they fail the test here.
        set_tests_properties(${name} PROPERTIES SKIP_RETURN_CODE 77
            FAIL_REGULAR_EXPRESSION "\\[earth_native\\] (layer|driver) (error|warning)")
    endforeach()
endif()
