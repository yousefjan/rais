#import <Metal/Metal.h>

#include <rais/metal_executor.hpp>
#include <rais/clock.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#ifndef RAIS_METALLIB_PATH
#error "RAIS_METALLIB_PATH must be defined"
#endif

static std::string today() {
    auto now = std::chrono::system_clock::now();
    auto tt = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
    localtime_r(&tt, &tm);
    char buf[16];
    std::strftime(buf, sizeof(buf), "%Y-%m-%d", &tm);
    return buf;
}

static std::string build_config() {
#ifdef NDEBUG
    return "release";
#else
    return "debug";
#endif
}

struct BenchResult {
    std::string test_name;
    std::string metric;
    double value;
    std::string units;
    std::string notes;
};

static void append_results(const std::vector<BenchResult>& results) {
    const char* path = "../experiments/results.tsv";

    bool needs_header = false;
    {
        std::ifstream check(path);
        needs_header = !check.good() || check.peek() == std::ifstream::traits_type::eof();
    }

    std::ofstream out(path, std::ios::app);
    if (!out) {
        std::cerr << "Warning: could not open " << path << " for writing\n";
        return;
    }

    if (needs_header) {
        out << "date\tbuild_config\ttest_name\tmetric\tvalue\tunits\tnotes\n";
    }

    std::string date = today();
    std::string config = build_config();
    for (const auto& r : results) {
        out << date << "\t" << config << "\t" << r.test_name << "\t"
            << r.metric << "\t" << static_cast<uint64_t>(r.value) << "\t"
            << r.units << "\t" << r.notes << "\n";
    }
}

int main() {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (!device) {
            std::fprintf(stderr, "No Metal device available\n");
            return 1;
        }

        std::printf("Metal device: %s\n", device.name.UTF8String);

        rais::MetalExecutor exec((__bridge void*)device, RAIS_METALLIB_PATH);
        void* pso = exec.pipeline("elementwise_add_f32");

        constexpr uint32_t N = 1'000'000;
        constexpr size_t buf_size = N * sizeof(float);

        id<MTLBuffer> buf_a = [device newBufferWithLength:buf_size
                                                  options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_b = [device newBufferWithLength:buf_size
                                                  options:MTLResourceStorageModeShared];
        id<MTLBuffer> buf_out = [device newBufferWithLength:buf_size
                                                    options:MTLResourceStorageModeShared];

        auto* a = static_cast<float*>(buf_a.contents);
        auto* b = static_cast<float*>(buf_b.contents);
        for (uint32_t i = 0; i < N; ++i) { a[i] = 1.0f; b[i] = 2.0f; }

        // Warm up
        for (int i = 0; i < 3; ++i) {
            exec.submit(
                [&](void*, void* enc) {
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
                });
        }
        exec.flush();

        // Measure submit-to-complete latency
        constexpr int TRIALS = 100;
        std::vector<uint64_t> latencies(TRIALS);

        for (int t = 0; t < TRIALS; ++t) {
            uint64_t t0 = rais::clock_ns();
            std::atomic<uint64_t> t1{0};

            while (!exec.submit(
                [&](void*, void* enc) {
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
                [&t1]() {
                    t1.store(rais::clock_ns(), std::memory_order_relaxed);
                }))
            {
                std::this_thread::yield();
            }
            exec.flush();
            latencies[t] = t1.load() - t0;
        }

        std::sort(latencies.begin(), latencies.end());
        auto pct = [&](double p) -> uint64_t {
            return latencies[static_cast<size_t>(p / 100.0 * (TRIALS - 1))];
        };

        uint64_t p50 = pct(50);
        uint64_t p95 = pct(95);
        uint64_t p99 = pct(99);

        std::printf("GPU task latency (1M-element add, submit→complete):\n");
        std::printf("  P50=%lluus  P95=%lluus  P99=%lluus\n",
                    static_cast<unsigned long long>(p50 / 1000),
                    static_cast<unsigned long long>(p95 / 1000),
                    static_cast<unsigned long long>(p99 / 1000));

        std::vector<BenchResult> results;
        results.push_back({"metal_add_1M", "latency_p50",
                            static_cast<double>(p50), "ns", "1M-element add"});
        results.push_back({"metal_add_1M", "latency_p95",
                            static_cast<double>(p95), "ns", "1M-element add"});
        results.push_back({"metal_add_1M", "latency_p99",
                            static_cast<double>(p99), "ns", "1M-element add"});
        append_results(results);
        std::printf("Results appended to experiments/results.tsv\n");
    }
    return 0;
}
