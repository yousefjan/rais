#import <Metal/Metal.h>

#include <rais/metal_allocator.hpp>

#include <catch2/catch_test_macros.hpp>
#include <vector>

static id<MTLDevice> get_device() {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    REQUIRE(device != nil);
    return device;
}

TEST_CASE("MetalBufferPool: acquire and release", "[metal_pool]") {
    id<MTLDevice> device = get_device();
    rais::MetalBufferPool pool((__bridge void*)device);

    REQUIRE(pool.live_buffers() == 0);
    REQUIRE(pool.pool_size() == 0);

    // Acquire a 1KB buffer (rounds up to 4KB size class)
    void* buf = pool.acquire(1024);
    REQUIRE(buf != nullptr);
    REQUIRE(pool.live_buffers() == 1);

    id<MTLBuffer> mtl_buf = (__bridge id<MTLBuffer>)buf;
    REQUIRE(mtl_buf.length >= 1024);
    REQUIRE(mtl_buf.storageMode == MTLStorageModeShared);

    // Release back to pool
    pool.release(buf);
    REQUIRE(pool.live_buffers() == 0);
    REQUIRE(pool.pool_size() == 1);
}

TEST_CASE("MetalBufferPool: buffer reuse from pool", "[metal_pool]") {
    id<MTLDevice> device = get_device();
    rais::MetalBufferPool pool((__bridge void*)device);

    // Acquire and release a 4KB buffer
    void* buf1 = pool.acquire(4096);
    REQUIRE(buf1 != nullptr);
    pool.release(buf1);

    // Acquire again — should get the same buffer back
    void* buf2 = pool.acquire(4096);
    REQUIRE(buf2 != nullptr);
    REQUIRE(pool.pool_size() == 0); // was popped from pool

    // The underlying MTLBuffer should be the same object
    id<MTLBuffer> mtl1 = (__bridge id<MTLBuffer>)buf1;
    id<MTLBuffer> mtl2 = (__bridge id<MTLBuffer>)buf2;
    REQUIRE(mtl1 == mtl2);

    pool.release(buf2);
}

TEST_CASE("MetalBufferPool: 1000 acquire/release cycles, no leaks", "[metal_pool]") {
    id<MTLDevice> device = get_device();
    rais::MetalBufferPool pool((__bridge void*)device);

    for (int i = 0; i < 1000; ++i) {
        // Vary size to hit different buckets
        size_t sizes[] = {1024, 4096, 32768, 65536, 500000, 1048576};
        size_t sz = sizes[i % 6];

        void* buf = pool.acquire(sz);
        REQUIRE(buf != nullptr);
        pool.release(buf);
    }

    REQUIRE(pool.live_buffers() == 0);
}

TEST_CASE("MetalBufferPool: multiple outstanding buffers", "[metal_pool]") {
    id<MTLDevice> device = get_device();
    rais::MetalBufferPool pool((__bridge void*)device);

    std::vector<void*> buffers;
    constexpr int N = 50;

    for (int i = 0; i < N; ++i) {
        void* buf = pool.acquire(4096);
        REQUIRE(buf != nullptr);
        buffers.push_back(buf);
    }
    REQUIRE(pool.live_buffers() == N);

    for (auto* buf : buffers) {
        pool.release(buf);
    }
    REQUIRE(pool.live_buffers() == 0);
    REQUIRE(pool.pool_size() == N);
}

TEST_CASE("MetalBufferPool: CPU write to shared buffer", "[metal_pool]") {
    id<MTLDevice> device = get_device();
    rais::MetalBufferPool pool((__bridge void*)device);

    void* buf = pool.acquire(4096);
    REQUIRE(buf != nullptr);

    id<MTLBuffer> mtl_buf = (__bridge id<MTLBuffer>)buf;
    float* data = static_cast<float*>(mtl_buf.contents);

    // Write and read back — verifies shared memory works
    for (int i = 0; i < 1024; ++i) {
        data[i] = static_cast<float>(i);
    }
    for (int i = 0; i < 1024; ++i) {
        REQUIRE(data[i] == static_cast<float>(i));
    }

    pool.release(buf);
}
