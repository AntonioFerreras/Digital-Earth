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
    return withBounceLimit(result, 1);
}

/// `scene_uniforms` with at most `limit` bounces.
pub fn withBounceLimit(scene_uniforms: uniforms.SceneUniforms, limit: u32) uniforms.SceneUniforms {
    var result = scene_uniforms;
    result.config[1] = @min(result.config[1], limit);
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

/// The clouds' multiple scattering, after Nubis Cubed (clouds/multiple_scattering.slang).
/// Depths are optical depths at full density. Nubis's published rates apply to its own
/// density sums, so the defaults are a starting point to tune from: Nubis's ratio of
/// its deep to surface rate, and a surface rate of 1 - g of the droplets' phase, the
/// rate at which scattering forgets the light's direction, with the strength that
/// brings the cloud bench's default view to the reference's mean.
pub const CloudScattering = struct {
    strength: f64 = 13.0,
    /// Attenuation rates per optical depth towards the sun: for views away from the sun,
    /// and for views towards it at a cloud's surface and `relaxed_depth` inside it.
    rate_away: f64 = 0.15,
    rate_surface: f64 = 0.15,
    rate_deep: f64 = 0.03,
    relaxed_depth: f64 = 10.0,
    /// The secondary scattering phase's Henyey-Greenstein g, which Nubis does not publish.
    phase_g: f64 = 0.6,
    /// How far the dimensional profile, 0 at a cloud's edge and 1 deep inside, dims the
    /// multiply scattered light towards its edge and the sky's light towards its core,
    /// 0 to 1; and the depth over which it rises.
    profile_strength: f64 = 1.0,
    profile_depth: f64 = 1.0,
};

/// The cloud bench (src/render/CloudBench.zig): one cloud volume in the atmosphere,
/// lit by the sun and the sky above a black planet, path traced next to the ray
/// marcher's model of it.
pub const CloudBench = struct {
    view: View = .split,
    /// Per metre at unit density; the default is a real cumulus's, whose droplets
    /// leave a mean free path of about ten metres.
    extinction: f64 = 0.1,
    /// Off leaves the sun as the only light.
    sky: bool = true,
    /// Scattering in the dense core goes on for thousands of bounces.
    bounce_limit: u32 = 4096,
    /// How the approximation takes the light its view's scatter receives from the probes.
    first_scatter: FirstScatter = .forward_peak,
    /// How the approximation turns the probes' fluence and flux into the radiance it
    /// scatters: P1, which goes negative where the light is strongly directional, or a
    /// von Mises-Fisher lobe with the same moments.
    probe_closure: Closure = .von_mises_fisher,
    /// How the probes' walks estimate the sun's transmittance at each collision.
    probe_sun_estimate: SunEstimate = .march,
    probe_bake: ProbeBake = .photons,
    /// How the photons add their paths to the probes.
    photon_deposit: PhotonDeposit = .points,
    /// The spherical harmonic order of the probes' fit of the sun's log fluence.
    probe_sun_order: SunOrder = .@"3",
    /// The probes' bake stops after this many walks per probe.
    probe_walks: u32 = 2048,
    /// Walks per probe, or rounds of photons, each frame while the probes bake.
    probe_rounds_per_frame: u32 = 8,
    /// Where the probes lie (src/render/ProbeLayout.zig).
    probe_layout: ProbeLayout = .adaptive,
    /// The adaptive layout's most probes, and the most its spacing may reach as a
    /// multiple of the depth inside the cloud.
    probe_budget: u32 = 1 << 18,
    probe_depth_ratio: f64 = 2.5,
    /// The approximation's steps along the view, as a multiple of those its march would
    /// take at MARCH_STEP_VOXELS (modes/cloud_bench.slang): fewer leave the converged
    /// image as it is, the march being jittered, and add a little noise.
    view_steps: f64 = 0.4,
    /// The majorant cells that the approximation's march walks along the view and
    /// towards the sun, as multiples of the volume's per axis: larger cells take fewer
    /// steps between them and skip less empty space.
    view_march_cells: MarchCells = .@"1",
    sun_march_cells: MarchCells = .@"4",

    pub const View = enum { reference, approximation, split, @"error", coverage };
    pub const SunEstimate = enum { march, ratio_tracking };
    /// `walks` follows the light backwards from each probe; `photons` traces it forward
    /// from outside, each path feeding every probe it passes.
    pub const ProbeBake = enum { walks, photons };
    /// `points` adds a point every finest cell along the path, `leaf_lengths` the exact
    /// length of the path in each leaf.
    pub const PhotonDeposit = enum { points, leaf_lengths };
    pub const SunOrder = enum { @"2", @"3" };
    /// `probes` reads them where the view scatters; `gather` follows one scatter there
    /// at full resolution and reads them only past it; `forward_peak` carries the
    /// phase's forward peak along the view at full resolution and reads the probes for
    /// the rest; `forward_peak_gather` gathers that rest.
    pub const FirstScatter = enum { probes, gather, forward_peak, forward_peak_gather };
    pub const Closure = enum { p1, von_mises_fisher };
    pub const ProbeLayout = enum { dense, adaptive };
    /// Tagged by its walk level.
    pub const MarchCells = enum(u32) { @"1", @"2", @"4" };

    /// Metres per volume unit, so that the data set's cloud is a cumulus a few
    /// kilometres across, and the altitude of the grid's base.
    const metres_per_unit = 10;
    const base_altitude_m = 1000;
    /// PLANET_RADIUS in common/constants.slang.
    const planet_radius_m = 6371e3;

    /// Where the frame of a grid from `min_corner` lies on the planet's +Y axis, in
    /// metres, and its metres per volume unit.
    pub fn origin(min_corner: [3]f32) [4]f32 {
        return .{ 0, planet_radius_m + base_altitude_m - min_corner[1] * metres_per_unit, 0, metres_per_unit };
    }

    /// Everything but the frame and grid, which the volume file sets, and the sample
    /// counts.
    pub fn benchUniforms(bench: CloudBench, single_scattering: bool) uniforms.CloudBenchUniforms {
        const flags: uniforms.CloudBenchFlags = .{
            .sky = bench.sky,
            .probe_gather = bench.first_scatter == .gather or bench.first_scatter == .forward_peak_gather,
            .forward_peak = bench.first_scatter == .forward_peak or bench.first_scatter == .forward_peak_gather,
            .p1_closure = bench.probe_closure == .p1,
            .ratio_tracking = bench.probe_sun_estimate == .ratio_tracking,
            .sun_order3 = bench.probe_sun_order == .@"3",
            .adaptive_probes = bench.probe_layout == .adaptive,
            .photons = bench.probe_bake == .photons,
            .leaf_lengths = bench.photon_deposit == .leaf_lengths,
        };
        return .{
            .origin = @splat(0),
            .grid_min = @splat(0),
            .grid_size = @splat(0),
            .medium = .{ @floatCast(bench.extinction), 0, 0, 0 },
            .march = .{ @floatCast(bench.view_steps), @floatFromInt(@backingInt(bench.view_march_cells)), @floatFromInt(@backingInt(bench.sun_march_cells)), 0 },
            .options = .{ @backingInt(bench.view), if (single_scattering) 1 else bench.bounce_limit, @bitCast(flags), bench.probe_walks },
            .samples = @splat(0),
            .probe_origin = .{ 0, 0, 0, @floatCast(bench.probe_depth_ratio) },
            .probe_spacing = @splat(0),
            .probe_cells = .{ 0, 0, 0, bench.probe_budget },
            .probe_counts = .{ 0, 0, bench.probe_rounds_per_frame, 0 },
        };
    }
};

/// `scene_uniforms` with `cloud_scattering`.
pub fn withCloudScattering(scene_uniforms: uniforms.SceneUniforms, cloud_scattering: CloudScattering) uniforms.SceneUniforms {
    const s = cloud_scattering;
    var result = scene_uniforms;
    result.cloud_scattering = .{ @floatCast(s.strength), @floatCast(s.rate_away), @floatCast(s.rate_surface), @floatCast(s.rate_deep) };
    result.cloud_profile = .{ @floatCast(s.profile_strength), @floatCast(s.profile_depth), @floatCast(s.relaxed_depth), @floatCast(s.phase_g) };
    return result;
}

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

/// `scene_uniforms` with the clouds alone: no air, a black surface, and no night lights.
pub fn cloudsOnly(scene_uniforms: uniforms.SceneUniforms) uniforms.SceneUniforms {
    var flags: uniforms.SceneFlags = @bitCast(scene_uniforms.config[0]);
    flags.clouds_only = true;
    flags.night_lights = false;
    var result = scene_uniforms;
    result.config[0] = @bitCast(flags);
    return result;
}

/// `scene_uniforms` with every cloud shadow marched, the ray marcher's shadow map unused.
pub fn marchedCloudShadows(scene_uniforms: uniforms.SceneUniforms) uniforms.SceneUniforms {
    var flags: uniforms.SceneFlags = @bitCast(scene_uniforms.config[0]);
    flags.march_cloud_shadows = true;
    var result = scene_uniforms;
    result.config[0] = @bitCast(flags);
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
pub const atmosphere = struct {
    pub const bounce_count_max = 25;
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
        .cloud_scattering = @splat(0),
        .cloud_profile = @splat(0),
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
