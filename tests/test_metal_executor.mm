#import <Metal/Metal.h>

#include <catch2/catch_test_macros.hpp>

#include <rais/metal_executor.hpp>

#include <atomic>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <thread>
#include <vector>

// Path to the compiled metallib — set by CMake as a compile definition
#ifndef RAIS_METALLIB_PATH
#error "RAIS_METALLIB_PATH must be defined"
#endif

static id<MTLDevice> get_device() {
    return MTLCreateSystemDefaultDevice();
}

static std::filesystem::path metallib_path() {
    return RAIS_METALLIB_PATH;
}

TEST_CASE("Trivial kernel (elementwise add) produces correct output", "[metal]") {
    id<MTLDevice> device = get_device();
    REQUIRE(device != nil);

    rais::MetalExecutor exec((__bridge void*)device, metallib_path());

    constexpr uint32_t N = 1024;
    constexpr size_t buf_size = N * sizeof(float);

    id<MTLBuffer> buf_a = [device newBufferWithLength:buf_size
                                              options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_b = [device newBufferWithLength:buf_size
                                              options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_out = [device newBufferWithLength:buf_size
                                                options:MTLResourceStorageModeShared];

    auto* a = static_cast<float*>(buf_a.contents);
    auto* b = static_cast<float*>(buf_b.contents);
    for (uint32_t i = 0; i < N; ++i) {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(i * 2);
    }

    void* pso = exec.pipeline("elementwise_add_f32");
    REQUIRE(pso != nullptr);

    std::atomic<bool> completed{false};

    bool submitted = exec.submit(
        [&](void* /* cmd_buf */, void* enc) {
            auto encoder = (__bridge id<MTLComputeCommandEncoder>)enc;
            [encoder setComputePipelineState:(__bridge id<MTLComputePipelineState>)pso];
            [encoder setBuffer:buf_a offset:0 atIndex:0];
            [encoder setBuffer:buf_b offset:0 atIndex:1];
            [encoder setBuffer:buf_out offset:0 atIndex:2];

            MTLSize grid = MTLSizeMake(N, 1, 1);
            NSUInteger thread_width =
                ((__bridge id<MTLComputePipelineState>)pso).threadExecutionWidth;
            MTLSize tg = MTLSizeMake(thread_width, 1, 1);
            [encoder dispatchThreads:grid threadsPerThreadgroup:tg];
        },
        [&completed]() {
            completed.store(true, std::memory_order_release);
        });

    REQUIRE(submitted);
    exec.flush();
    REQUIRE(completed.load(std::memory_order_acquire));

    auto* out = static_cast<float*>(buf_out.contents);
    for (uint32_t i = 0; i < N; ++i) {
        REQUIRE(out[i] == static_cast<float>(i + i * 2));
    }
}

TEST_CASE("100 concurrent GPU tasks all complete", "[metal]") {
    id<MTLDevice> device = get_device();
    REQUIRE(device != nil);

    rais::MetalExecutor exec((__bridge void*)device, metallib_path());

    constexpr int NUM_TASKS = 100;
    constexpr uint32_t N = 256;
    constexpr size_t buf_size = N * sizeof(float);

    std::atomic<int> complete_count{0};
    void* pso = exec.pipeline("elementwise_add_f32");

    // Shared input buffers
    id<MTLBuffer> buf_a = [device newBufferWithLength:buf_size
                                              options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_b = [device newBufferWithLength:buf_size
                                              options:MTLResourceStorageModeShared];

    auto* a = static_cast<float*>(buf_a.contents);
    auto* b = static_cast<float*>(buf_b.contents);
    for (uint32_t i = 0; i < N; ++i) { a[i] = 1.0f; b[i] = 2.0f; }

    // Each task gets its own output buffer
    std::vector<id<MTLBuffer>> outputs(NUM_TASKS);
    for (int t = 0; t < NUM_TASKS; ++t) {
        outputs[t] = [device newBufferWithLength:buf_size
                                         options:MTLResourceStorageModeShared];
    }

    for (int t = 0; t < NUM_TASKS; ++t) {
        // Retry with backoff if backpressure
        while (!exec.submit(
            [&, t](void* /* cmd_buf */, void* enc) {
                auto encoder = (__bridge id<MTLComputeCommandEncoder>)enc;
                [encoder setComputePipelineState:
                    (__bridge id<MTLComputePipelineState>)pso];
                [encoder setBuffer:buf_a offset:0 atIndex:0];
                [encoder setBuffer:buf_b offset:0 atIndex:1];
                [encoder setBuffer:outputs[t] offset:0 atIndex:2];

                MTLSize grid = MTLSizeMake(N, 1, 1);
                NSUInteger tw =
                    ((__bridge id<MTLComputePipelineState>)pso).threadExecutionWidth;
                MTLSize tg = MTLSizeMake(tw, 1, 1);
                [encoder dispatchThreads:grid threadsPerThreadgroup:tg];
            },
            [&complete_count]() {
                complete_count.fetch_add(1, std::memory_order_relaxed);
            }))
        {
            std::this_thread::yield();
        }
    }

    exec.flush();
    REQUIRE(complete_count.load() == NUM_TASKS);

    // Verify all outputs
    for (int t = 0; t < NUM_TASKS; ++t) {
        auto* out = static_cast<float*>(outputs[t].contents);
        for (uint32_t i = 0; i < N; ++i) {
            REQUIRE(out[i] == 3.0f);
        }
    }
}

TEST_CASE("Backpressure — submit returns false when ring full", "[metal]") {
    id<MTLDevice> device = get_device();
    REQUIRE(device != nil);

    rais::MetalExecutor exec((__bridge void*)device, metallib_path());
    void* pso = exec.pipeline("elementwise_add_f32");

    constexpr uint32_t N = 64;
    constexpr size_t buf_size = N * sizeof(float);

    id<MTLBuffer> buf_a = [device newBufferWithLength:buf_size
                                              options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_b = [device newBufferWithLength:buf_size
                                              options:MTLResourceStorageModeShared];
    id<MTLBuffer> buf_out = [device newBufferWithLength:buf_size
                                                options:MTLResourceStorageModeShared];

    int submitted = 0;
    int rejected = 0;
    std::atomic<int> completed{0};

    // Try to submit 200 tasks rapidly — some should be rejected
    for (int i = 0; i < 200; ++i) {
        bool ok = exec.submit(
            [&](void* /* cmd_buf */, void* enc) {
                auto encoder = (__bridge id<MTLComputeCommandEncoder>)enc;
                [encoder setComputePipelineState:
                    (__bridge id<MTLComputePipelineState>)pso];
                [encoder setBuffer:buf_a offset:0 atIndex:0];
                [encoder setBuffer:buf_b offset:0 atIndex:1];
                [encoder setBuffer:buf_out offset:0 atIndex:2];

                MTLSize grid = MTLSizeMake(N, 1, 1);
                NSUInteger tw =
                    ((__bridge id<MTLComputePipelineState>)pso).threadExecutionWidth;
                MTLSize tg = MTLSizeMake(tw, 1, 1);
                [encoder dispatchThreads:grid threadsPerThreadgroup:tg];
            },
            [&completed]() {
                completed.fetch_add(1, std::memory_order_relaxed);
            });

        if (ok) ++submitted;
        else ++rejected;
    }

    // We expect some rejections (ring buffer capacity is 8)
    REQUIRE(rejected > 0);
    REQUIRE(submitted > 0);

    exec.flush();
    REQUIRE(completed.load() == submitted);
}
