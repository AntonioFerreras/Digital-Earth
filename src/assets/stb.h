// stb_image and stb_image_write as the Zig `stb` module sees them: memory-only
// APIs, so every file access goes through std.Io.
#define STBI_NO_STDIO
#define STBI_WRITE_NO_STDIO
#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#include "stb_image.h"
#include "stb_image_write.h"
