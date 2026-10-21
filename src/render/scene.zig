//! Scene presets: the flags and constants that let one path-tracer kernel render both
//! original scenes. The presets differ in scene content only; the atmosphere and the
//! light transport are the same in both. Values are those of the Taichi branches
//! named below.

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
                .cloud_extinction = 0.1,
                .cloud_density = 0.029,
                .albedo_srgb = 0.3,
                .land_height_scale_m = 7800,
                .night_light_scale = 1e-4,
                .star_scale = 1e-7,
            }),
            .symmetric => build(.{
                .flags = .{ .canonicalize = true },
                .cloud_extinction = 0,
                .cloud_density = 0,
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

/// `scene_uniforms` with the surface from the cube-face tiles.
pub fn withTiles(scene_uniforms: uniforms.SceneUniforms) uniforms.SceneUniforms {
    var flags: uniforms.SceneFlags = @bitCast(scene_uniforms.config[0]);
    flags.tiles = true;
    var result = scene_uniforms;
    result.config[0] = @bitCast(flags);
    return result;
}

/// The water model's parts, each switchable for now to compare it with the previous
/// look (shared/uniforms.slang).
pub const Water = struct {
    glint: bool = true,
    body: bool = true,
    clear_deep_water: bool = true,
    calm_lakes: bool = true,
    wind_map: bool = true,
    roughness_texture: bool = true,
    /// The sea's, in m/s at 10 m, where the wind map is off or holds none.
    wind_speed: f64 = 0.25,
    /// Factors on the roughness texture's default strength and size.
    texture_strength: f64 = 0.02,
    texture_size: f64 = 0.62,
    /// How far the wind map's speed replaces `wind_speed`, 0 to 1.
    wind_map_influence: f64 = 0.4,
    /// The weight of the sea's slope tail, 0 to 1.
    glint_tail: f64 = 1.0,
    /// The roughness texture's streaks, in wavelengths along the wind.
    streak_length: f64 = 4.4,
    /// How far the roughness texture's coordinates bend, in its coarsest wavelengths.
    texture_warp: f64 = 1.4,
    /// The factor on the slope variance the wind adds over a calm sea's.
    wind_roughness: f64 = 1.0,
};

/// `scene_uniforms` with `water`.
pub fn withWater(scene_uniforms: uniforms.SceneUniforms, water: Water) uniforms.SceneUniforms {
    var flags: uniforms.SceneFlags = @bitCast(scene_uniforms.config[0]);
    flags.water_glint = water.glint;
    flags.water_body = water.body;
    flags.clear_deep_water = water.clear_deep_water;
    flags.calm_lakes = water.calm_lakes;
    flags.wind_map = water.wind_map;
    flags.roughness_texture = water.roughness_texture;
    var result = scene_uniforms;
    result.config[0] = @bitCast(flags);
    result.water = .{ @floatCast(water.wind_speed), @floatCast(water.texture_strength), @floatCast(water.texture_size), @floatCast(water.wind_map_influence) };
    result.glint = .{ @floatCast(water.glint_tail), @floatCast(water.streak_length), @floatCast(water.texture_warp), @floatCast(water.wind_roughness) };
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

/// Scene uniforms holding only what every preset shares, for tables baked from the
/// atmosphere alone.
pub fn atmosphereUniforms() uniforms.SceneUniforms {
    return build(.{
        .flags = .{},
        .cloud_extinction = 0,
        .cloud_density = 0,
        .albedo_srgb = 0,
        .land_height_scale_m = 0,
        .night_light_scale = 0,
        .star_scale = 0,
    });
}

/// What every preset shares: the atmosphere and the light transport. The values are
/// `data_prep`'s, the newer branch; `main` had turbidity 1.06, the Rayleigh fit
/// normalized by 1.225 instead of by its own sea-level value, Klein-Nishina aerosol
/// phase on every scatter, and the same 25 bounces.
const atmosphere = struct {
    const bounce_count_max = 25;
    /// Both originals start Russian roulette once `scatter_count > 3`.
    const russian_roulette_first_bounce = 4;
    const turbidity = 1.03;
    /// The Rayleigh fit's value at sea level, so its normalized density is 1 there.
    const rayleigh_sea_level_density = 1.220554684;
};

/// Scene content: what the presets differ in.
const Constants = struct {
    flags: uniforms.SceneFlags,
    cloud_extinction: f32,
    cloud_density: f32,
    /// Ground albedo where `earth_maps` is off.
    albedo_srgb: f32,
    land_height_scale_m: f32,
    night_light_scale: f32,
    star_scale: f32,
};

fn build(constants: Constants) uniforms.SceneUniforms {
    return .{
        .config = .{ @bitCast(constants.flags), atmosphere.bounce_count_max, atmosphere.russian_roulette_first_bounce, 0 },
        .media = .{
            atmosphere.turbidity,
            constants.cloud_extinction,
            constants.cloud_density,
            atmosphere.rayleigh_sea_level_density,
        },
        .surface = .{
            constants.albedo_srgb,
            constants.land_height_scale_m,
            constants.night_light_scale,
            constants.star_scale,
        },
        .water = @splat(0),
        .glint = @splat(0),
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
