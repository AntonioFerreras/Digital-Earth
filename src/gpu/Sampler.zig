//! A texture sampler owned by the caller; destroy it before its device.

const c = @import("c");
const gpu = @import("../gpu.zig");

const Sampler = @This();

handle: *c.en_sampler,

pub const Filter = enum(c.en_filter) { nearest, linear };
pub const Address = enum(c.en_address) { repeat, clamp };

comptime {
    gpu.assertMirrorsC(Filter, "EN_FILTER_");
    gpu.assertMirrorsC(Address, "EN_ADDRESS_");
}

pub const Desc = struct {
    filter: Filter,
    address_u: Address,
    address_v: Address,
    address_w: Address,
};

pub fn init(device: gpu.Device, desc: Desc) gpu.Error!Sampler {
    const native_desc: c.en_sampler_desc = .{
        .filter = @backingInt(desc.filter),
        .address_u = @backingInt(desc.address_u),
        .address_v = @backingInt(desc.address_v),
        .address_w = @backingInt(desc.address_w),
    };
    var handle: ?*c.en_sampler = null;
    try gpu.check(c.en_sampler_create(device.handle, &native_desc, &handle), "en_sampler_create");
    return .{ .handle = handle.? };
}

pub fn deinit(sampler: Sampler) void {
    c.en_sampler_destroy(sampler.handle);
}
