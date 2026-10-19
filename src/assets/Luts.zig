//! Spectral and camera-response lookup tables from `LUT/`, widened to f32 and
//! laid out for upload.

const Luts = @This();

const std = @import("std");
const builtin = @import("builtin");
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const Io = std.Io;

/// `CIE.dat`: f32[2][wavelength_count][3]. Row 0 holds the per-channel
/// normalized CDFs of xbar, ybar, zbar; row 1 the CIE 1931 2-degree CMFs.
cie: []f32,
/// `srgb2spec.dat`, stored as f16: f32[srgb_to_spectrum_count][3] basis
/// weights (r, g, b), one row per nanometre from 400 nm.
srgb_to_spectrum: []f32,
/// `ozone_cross_section.dat`: f32[wavelength_count], cm^2 per molecule, entry
/// i the mean over the 1 nm bin centred on 390 + i nm.
ozone_cross_section: []f32,
/// `ozone_profile.dat`: f32[ozone_profile_count], ozone number density (m^-3)
/// at altitude i * ozone_profile_step_m.
ozone_profile: []f32,
/// Camera response curves as an rgba32f texture `crf_texel_count` wide with
/// one curve per row (alpha 1), in `crf_names` order.
crf: []f32,
/// Curve names (file stems): Neutral first, then by file name.
crf_names: []const []const u8,

comptime {
    // LUT files are raw host-order floats written by numpy on x86.
    assert(builtin.cpu.arch.endian() == .little);
}

/// One sample per nanometre over 390..830 nm.
pub const wavelength_count = 441;
pub const srgb_to_spectrum_count = 300;
/// One sample per 500 m from sea level to the top of the atmosphere at 110 km.
pub const ozone_profile_count = 221;
pub const ozone_profile_step_m = 500.0;
pub const crf_texel_count = 1024;
pub const crf_dir_path = "camera_response_functions";
const crf_neutral = "Neutral.rf";
/// Byte-identical copies of other curves.
const crf_duplicates = [_][]const u8{"agfacolor-futura-100CD (1).rf"};
const crf_count_max = 256;
const crf_file_size_max = 1 << 20;

pub const LoadError = error{
    LutSizeMismatch,
    InvalidOzoneProfile,
    MissingNeutralCrf,
    TooManyCrfs,
    CrfRowCount,
    CrfColumnCount,
    InvalidCrfNumber,
} || Allocator.Error || Io.Dir.OpenError || Io.Dir.Reader.Error ||
    Io.File.OpenError || Io.File.StatError || Io.File.ReadPositionalError || Io.Dir.ReadFileAllocError;

/// Loads every table from `lut_dir`, the repository's `LUT/` directory.
pub fn load(gpa: Allocator, io: Io, lut_dir: Io.Dir) LoadError!Luts {
    const cie = try gpa.alloc(f32, 2 * wavelength_count * 3);
    errdefer gpa.free(cie);
    try readExact(io, lut_dir, "CIE.dat", std.mem.sliceAsBytes(cie));

    var srgb_to_spectrum_half: [srgb_to_spectrum_count * 3]f16 = undefined;
    try readExact(io, lut_dir, "srgb2spec.dat", std.mem.sliceAsBytes(&srgb_to_spectrum_half));
    const srgb_to_spectrum = try gpa.alloc(f32, srgb_to_spectrum_half.len);
    errdefer gpa.free(srgb_to_spectrum);
    for (srgb_to_spectrum, srgb_to_spectrum_half) |*target, source| target.* = source;

    const ozone_cross_section = try gpa.alloc(f32, wavelength_count);
    errdefer gpa.free(ozone_cross_section);
    try readExact(io, lut_dir, "ozone_cross_section.dat", std.mem.sliceAsBytes(ozone_cross_section));

    const ozone_profile = try gpa.alloc(f32, ozone_profile_count);
    errdefer gpa.free(ozone_profile);
    try readExact(io, lut_dir, "ozone_profile.dat", std.mem.sliceAsBytes(ozone_profile));
    // Delta tracking takes the largest sample as its bound, which NaN or infinity would break.
    for (ozone_profile) |density| {
        if (!(density >= 0 and density < std.math.inf(f32))) return error.InvalidOzoneProfile;
    }

    var crf_dir = try lut_dir.openDir(io, crf_dir_path, .{ .iterate = true });
    defer crf_dir.close(io);
    const crf_file_names = try listCrfFiles(gpa, io, crf_dir);
    defer freeNames(gpa, crf_file_names);

    const crf = try gpa.alloc(f32, crf_file_names.len * crf_texel_count * 4);
    errdefer gpa.free(crf);
    for (crf_file_names, 0..) |file_name, row| {
        const text = try crf_dir.readFileAlloc(io, file_name, gpa, .limited(crf_file_size_max));
        defer gpa.free(text);
        parseCrf(text, crf[row * crf_texel_count * 4 ..][0 .. crf_texel_count * 4]) catch |err| {
            std.log.scoped(.assets).err("{s}/{s}: {t}", .{ crf_dir_path, file_name, err });
            return err;
        };
    }

    const crf_names = try gpa.alloc([]const u8, crf_file_names.len);
    var crf_names_done: usize = 0;
    errdefer {
        for (crf_names[0..crf_names_done]) |name| gpa.free(name);
        gpa.free(crf_names);
    }
    for (crf_names, crf_file_names) |*name, file_name| {
        name.* = try gpa.dupe(u8, std.fs.path.stem(file_name));
        crf_names_done += 1;
    }

    return .{
        .cie = cie,
        .srgb_to_spectrum = srgb_to_spectrum,
        .ozone_cross_section = ozone_cross_section,
        .ozone_profile = ozone_profile,
        .crf = crf,
        .crf_names = crf_names,
    };
}

pub fn deinit(luts: *Luts, gpa: Allocator) void {
    gpa.free(luts.cie);
    gpa.free(luts.srgb_to_spectrum);
    gpa.free(luts.ozone_cross_section);
    gpa.free(luts.ozone_profile);
    gpa.free(luts.crf);
    freeNames(gpa, luts.crf_names);
    luts.* = undefined;
}

fn freeNames(gpa: Allocator, names: []const []const u8) void {
    for (names) |name| gpa.free(name);
    gpa.free(names);
}

/// Fills `target` with the whole file, which must be exactly that size.
fn readExact(io: Io, dir: Io.Dir, sub_path: []const u8, target: []u8) LoadError!void {
    const file = try dir.openFile(io, sub_path, .{});
    defer file.close(io);
    const stat = try file.stat(io);
    if (stat.size != target.len) {
        std.log.scoped(.assets).err("{s}: {d} bytes, expected {d}", .{ sub_path, stat.size, target.len });
        return error.LutSizeMismatch;
    }
    const read_size = try file.readPositionalAll(io, target, 0);
    if (read_size != target.len) return error.LutSizeMismatch;
}

/// `.rf` file names in texture row order: Neutral first so it is the default
/// curve, then sorted by name so the order is stable across file systems.
fn listCrfFiles(gpa: Allocator, io: Io, crf_dir: Io.Dir) LoadError![]const []const u8 {
    var names: std.ArrayList([]const u8) = .empty;
    defer names.deinit(gpa);
    errdefer for (names.items) |name| gpa.free(name);

    var iterator = crf_dir.iterate();
    while (try iterator.next(io)) |entry| {
        if (entry.kind != .file) continue;
        if (!std.mem.eql(u8, std.fs.path.extension(entry.name), ".rf")) continue;
        if (isDuplicateCrf(entry.name)) continue;
        if (names.items.len == crf_count_max) return error.TooManyCrfs;
        const name = try gpa.dupe(u8, entry.name);
        names.append(gpa, name) catch |err| {
            gpa.free(name);
            return err;
        };
    }

    std.mem.sort([]const u8, names.items, {}, crfOrderLessThan);
    if (names.items.len == 0 or !std.mem.eql(u8, names.items[0], crf_neutral)) return error.MissingNeutralCrf;
    return names.toOwnedSlice(gpa);
}

fn isDuplicateCrf(name: []const u8) bool {
    for (crf_duplicates) |duplicate| {
        if (std.mem.eql(u8, name, duplicate)) return true;
    }
    return false;
}

fn crfOrderLessThan(_: void, a: []const u8, b: []const u8) bool {
    const a_neutral = std.mem.eql(u8, a, crf_neutral);
    const b_neutral = std.mem.eql(u8, b, crf_neutral);
    if (a_neutral != b_neutral) return a_neutral;
    return std.mem.lessThan(u8, a, b);
}

/// Parses `crf_texel_count` lines of `irradiance r g b` into RGBA texels,
/// dropping the irradiance column (it is the implicit texel coordinate).
fn parseCrf(text: []const u8, texels: []f32) error{ CrfRowCount, CrfColumnCount, InvalidCrfNumber }!void {
    assert(texels.len == crf_texel_count * 4);
    var lines = std.mem.tokenizeAny(u8, text, "\r\n");
    var row: usize = 0;
    while (lines.next()) |line| : (row += 1) {
        if (row == crf_texel_count) return error.CrfRowCount;
        var columns = std.mem.tokenizeAny(u8, line, " \t");
        _ = columns.next() orelse return error.CrfColumnCount;
        const texel = texels[row * 4 ..][0..4];
        for (texel[0..3]) |*channel| {
            const column = columns.next() orelse return error.CrfColumnCount;
            channel.* = std.fmt.parseFloat(f32, column) catch return error.InvalidCrfNumber;
        }
        if (columns.next() != null) return error.CrfColumnCount;
        texel[3] = 1;
    }
    if (row != crf_texel_count) return error.CrfRowCount;
}

test "load reads the repository LUTs" {
    const gpa = std.testing.allocator;
    var lut_dir = try Io.Dir.cwd().openDir(std.testing.io, "LUT", .{});
    defer lut_dir.close(std.testing.io);
    var luts: Luts = try .load(gpa, std.testing.io, lut_dir);
    defer luts.deinit(gpa);

    // The CDF row ends at 1 for each channel; the CMF row peaks near 1 for ybar at 555 nm.
    const cdf_last = luts.cie[(wavelength_count - 1) * 3 ..][0..3];
    for (cdf_last) |value| try std.testing.expectApproxEqAbs(1.0, value, 1e-3);
    const ybar_555 = luts.cie[(wavelength_count + 555 - 390) * 3 + 1];
    try std.testing.expectApproxEqAbs(1.0, ybar_555, 1e-2);

    // srgb2spec row 0 is the first entry of the generator's table, through f16.
    try std.testing.expectApproxEqAbs(0.328251563088163, luts.srgb_to_spectrum[0], 1e-3);
    try std.testing.expectApproxEqAbs(0.544522552139376, luts.srgb_to_spectrum[luts.srgb_to_spectrum.len - 3], 1e-3);
    for (luts.ozone_cross_section) |cross_section| try std.testing.expect(cross_section >= 0 and cross_section < 1e-19);

    try std.testing.expectEqual(15, luts.crf_names.len);
    try std.testing.expectEqualStrings("Neutral", luts.crf_names[0]);
    try std.testing.expectEqualStrings("Advantix-100CD", luts.crf_names[1]);
    try std.testing.expectEqualStrings("optura981111.SLRR", luts.crf_names[14]);
    for (luts.crf_names) |name| try std.testing.expect(std.mem.indexOf(u8, name, "(1)") == null);
    try std.testing.expectEqual(luts.crf_names.len * crf_texel_count * 4, luts.crf.len);
    // Neutral is the identity curve: texel i holds i / 1023 in every channel.
    const neutral_texel_512 = luts.crf[512 * 4 ..][0..4];
    try std.testing.expectApproxEqAbs(512.0 / 1023.0, neutral_texel_512[0], 1e-6);
    try std.testing.expectEqual(1.0, neutral_texel_512[3]);
}

// The trapezoid column of the uploaded samples, which the shader's linear
// interpolation integrates exactly, against the McPeters and Labow 2011 global
// annual mean that python/lut/ozone_profile_generator.py reports (292.08 DU).
test "the ozone profile holds the global mean column" {
    const gpa = std.testing.allocator;
    var lut_dir = try Io.Dir.cwd().openDir(std.testing.io, "LUT", .{});
    defer lut_dir.close(std.testing.io);
    var luts: Luts = try .load(gpa, std.testing.io, lut_dir);
    defer luts.deinit(gpa);

    const dobson_unit = 2.6867e20;
    var column: f64 = 0;
    for (luts.ozone_profile[0 .. ozone_profile_count - 1], luts.ozone_profile[1..]) |lower, upper| {
        column += 0.5 * (@as(f64, lower) + upper) * ozone_profile_step_m;
    }
    try std.testing.expectApproxEqAbs(292.08, column / dobson_unit, 0.01);

    const peak = std.mem.indexOfMax(f32, luts.ozone_profile);
    try std.testing.expectEqual(25e3, @as(f64, @floatFromInt(peak)) * ozone_profile_step_m);
    try std.testing.expect(luts.ozone_profile[ozone_profile_count - 1] < 1e-7 * luts.ozone_profile[peak]);
}

test "parseCrf rejects malformed curves" {
    var texels: [crf_texel_count * 4]f32 = undefined;
    try std.testing.expectError(error.CrfRowCount, parseCrf("0 0 0 0\n", &texels));
    try std.testing.expectError(error.CrfColumnCount, parseCrf("0 0 0\n", &texels));
    try std.testing.expectError(error.CrfColumnCount, parseCrf("0 0 0 0 0\n", &texels));
    try std.testing.expectError(error.InvalidCrfNumber, parseCrf("0 0 x 0\n", &texels));

    const line = "0.5 0.25 0.5 0.75\n";
    var too_long: [line.len * (crf_texel_count + 1)]u8 = undefined;
    for (0..crf_texel_count + 1) |index| @memcpy(too_long[index * line.len ..][0..line.len], line);
    try std.testing.expectError(error.CrfRowCount, parseCrf(&too_long, &texels));
    try parseCrf(too_long[0 .. too_long.len - line.len], &texels);
    try std.testing.expectEqualSlices(f32, &.{ 0.25, 0.5, 0.75, 1 }, texels[4..8]);
}
