//! Scene presets: the flags and constants that let one path-tracer kernel render both
//! original scenes. Values are those of the Taichi branches named below.

const std = @import("std");
const uniforms = @import("uniforms.zig");

pub const Preset = enum {
    /// `main`: heightfield land with the NASA maps, clouds, and stars. Night lights are
    /// off until their look is reworked.
    textured,
    /// `data_prep`: a smooth grey sphere seen through uvwz-canonicalized rays, the
    /// scene the MLP learns and `datagen` samples.
    symmetric,

    pub fn sceneUniforms(preset: Preset) uniforms.SceneUniforms {
        return switch (preset) {
            .textured => build(.{
                .flags = .{ .terrain = true, .clouds = true, .stars = true, .earth_maps = true },
                .bounce_count_max = 25,
                .turbidity = 1.06,
                .cloud_extinction = 0.1,
                .cloud_density = 0.029,
                .rayleigh_sea_level_density = 1.225,
                .albedo_srgb = 0.3,
                .land_height_scale_m = 7800,
                .night_light_scale = 1e-4,
                .star_scale = 1e-7,
            }),
            .symmetric => build(.{
                .flags = .{ .mie_hg_after_first = true, .canonicalize = true },
                .bounce_count_max = 10,
                .turbidity = 1.03,
                .cloud_extinction = 0,
                .cloud_density = 0,
                .rayleigh_sea_level_density = 1.220554684,
                .albedo_srgb = 0.3,
                // No terrain: the scale only lifts surface points for shadow rays.
                .land_height_scale_m = 7800,
                .night_light_scale = 0,
                .star_scale = 0,
            }),
        };
    }
};

/// `scene_uniforms` with light that scatters once only: one bounce in every mode, and
/// the `raymarch` mode leaves out its multiple-scattering table.
pub fn singleScattering(scene_uniforms: uniforms.SceneUniforms) uniforms.SceneUniforms {
    var flags: uniforms.SceneFlags = @bitCast(scene_uniforms.config[0]);
    flags.single_scattering = true;
    var result = scene_uniforms;
    result.config[0] = @bitCast(flags);
    result.config[1] = 1;
    return result;
}

/// `scene_uniforms` without the cloud shell.
pub fn withoutClouds(scene_uniforms: uniforms.SceneUniforms) uniforms.SceneUniforms {
    var flags: uniforms.SceneFlags = @bitCast(scene_uniforms.config[0]);
    flags.clouds = false;
    var result = scene_uniforms;
    result.config[0] = @bitCast(flags);
    return result;
}

/// Both originals start Russian roulette once `scatter_count > 3`.
const russian_roulette_first_bounce = 4;

const Constants = struct {
    flags: uniforms.SceneFlags,
    bounce_count_max: u32,
    turbidity: f32,
    cloud_extinction: f32,
    cloud_density: f32,
    rayleigh_sea_level_density: f32,
    /// Ground albedo where `earth_maps` is off.
    albedo_srgb: f32,
    land_height_scale_m: f32,
    night_light_scale: f32,
    star_scale: f32,
};

fn build(constants: Constants) uniforms.SceneUniforms {
    return .{
        .config = .{ @bitCast(constants.flags), constants.bounce_count_max, russian_roulette_first_bounce, 0 },
        .media = .{
            constants.turbidity,
            constants.cloud_extinction,
            constants.cloud_density,
            constants.rayleigh_sea_level_density,
        },
        .surface = .{
            constants.albedo_srgb,
            constants.land_height_scale_m,
            constants.night_light_scale,
            constants.star_scale,
        },
    };
}

const testing = std.testing;

test "single scattering allows one bounce and keeps the rest of the preset" {
    for (std.enums.values(Preset)) |preset| {
        const full = preset.sceneUniforms();
        const single = singleScattering(full);
        const full_flags: uniforms.SceneFlags = @bitCast(full.config[0]);
        const single_flags: uniforms.SceneFlags = @bitCast(single.config[0]);
        try testing.expect(!full_flags.single_scattering);
        try testing.expect(single_flags.single_scattering);
        try testing.expectEqual(@as(u32, 1), single.config[1]);
        try testing.expect(full.config[1] > 1);

        var restored = single;
        restored.config[0] = full.config[0];
        restored.config[1] = full.config[1];
        try testing.expectEqualDeep(full, restored);
    }
}
