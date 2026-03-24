#include <metal_stdlib>
using namespace metal;

kernel void elementwise_add_f32(
    device const float* a      [[buffer(0)]],
    device const float* b      [[buffer(1)]],
    device float*       result [[buffer(2)]],
    uint                gid    [[thread_position_in_grid]]
) {
    result[gid] = a[gid] + b[gid];
}
