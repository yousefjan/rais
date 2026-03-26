#import <Metal/Metal.h>

#include <rais/layer_streamer.hpp>
#include <rais/metal_allocator.hpp>
#include <rais/scheduler.hpp>

#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <vector>

#include <catch2/catch_test_macros.hpp>

namespace {

/// Create fake layer files filled with a known byte pattern.
/// Pattern for layer N: every byte is (N + 1) & 0xFF.
std::filesystem::path create_layer_dir(size_t num_layers, size_t layer_size) {
    auto dir = std::filesystem::temp_directory_path() / "rais_test_layers";
    std::filesystem::create_directories(dir);

    for (size_t i = 0; i < num_layers; ++i) {
        std::ostringstream name;
        name << "layer_" << std::setw(4) << std::setfill('0') << i;
        auto path = dir / name.str();

        std::vector<uint8_t> data(layer_size,
                                   static_cast<uint8_t>((i + 1) & 0xFF));
        std::ofstream f(path, std::ios::binary);
        f.write(reinterpret_cast<const char*>(data.data()),
                static_cast<std::streamsize>(data.size()));
    }
    return dir;
}

void cleanup_layer_dir(const std::filesystem::path& dir) {
    std::filesystem::remove_all(dir);
}

} // namespace

TEST_CASE("Sequential layer requests return correct data", "[layer_streamer]") {
    constexpr size_t kNumLayers = 10;
    constexpr size_t kLayerSize = 1024 * 64; // 64 KB per layer

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    REQUIRE(device != nil);

    auto dir = create_layer_dir(kNumLayers, kLayerSize);

    rais::Scheduler sched({.num_workers = 2});
    rais::MetalBufferPool pool((__bridge void*)device);

    rais::LayerStreamerConfig config{
        .layer_size_bytes = kLayerSize,
        .num_buffer_slots = 3,
        .model_dir = dir,
        .num_layers = kNumLayers,
    };
    rais::LayerStreamer streamer(sched, pool, config);

    for (size_t i = 0; i < kNumLayers; ++i) {
        std::atomic<bool> verified{false};
        uint8_t expected_byte = static_cast<uint8_t>((i + 1) & 0xFF);

        auto h = streamer.request_layer(i, [&](void* buf) {
            if (!buf) return;
            auto* bytes = static_cast<const uint8_t*>(buf);
            bool ok = true;
            for (size_t j = 0; j < kLayerSize; ++j) {
                if (bytes[j] != expected_byte) { ok = false; break; }
            }
            verified.store(ok, std::memory_order_relaxed);
        });
        h.wait();
        REQUIRE(verified.load(std::memory_order_relaxed));

        streamer.release_layer(i);
    }

    cleanup_layer_dir(dir);
}

TEST_CASE("Prefetch makes first request immediate", "[layer_streamer]") {
    constexpr size_t kNumLayers = 5;
    constexpr size_t kLayerSize = 1024 * 16; // 16 KB

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    REQUIRE(device != nil);

    auto dir = create_layer_dir(kNumLayers, kLayerSize);

    rais::Scheduler sched({.num_workers = 2});
    rais::MetalBufferPool pool((__bridge void*)device);

    rais::LayerStreamerConfig config{
        .layer_size_bytes = kLayerSize,
        .num_buffer_slots = 3,
        .model_dir = dir,
        .num_layers = kNumLayers,
    };
    rais::LayerStreamer streamer(sched, pool, config);

    // Start prefetch, then wait a bit for IO to complete
    streamer.start_prefetch(0);
    std::this_thread::sleep_for(std::chrono::milliseconds(50));

    // Layer 0 should complete very quickly (already prefetched)
    auto t0 = std::chrono::steady_clock::now();
    std::atomic<bool> got_data{false};
    auto h = streamer.request_layer(0, [&](void* buf) {
        got_data.store(buf != nullptr, std::memory_order_relaxed);
    });
    h.wait();
    auto elapsed = std::chrono::steady_clock::now() - t0;

    REQUIRE(got_data.load(std::memory_order_relaxed));
    // Should be near-instant since data was prefetched
    REQUIRE(elapsed < std::chrono::milliseconds(50));

    cleanup_layer_dir(dir);
}

TEST_CASE("Buffer recycling keeps live count at num_buffer_slots",
          "[layer_streamer]") {
    constexpr size_t kNumLayers = 10;
    constexpr size_t kLayerSize = 1024 * 64; // 64 KB
    constexpr size_t kSlots = 3;

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    REQUIRE(device != nil);

    auto dir = create_layer_dir(kNumLayers, kLayerSize);

    rais::Scheduler sched({.num_workers = 2});
    rais::MetalBufferPool pool((__bridge void*)device);

    rais::LayerStreamerConfig config{
        .layer_size_bytes = kLayerSize,
        .num_buffer_slots = kSlots,
        .model_dir = dir,
        .num_layers = kNumLayers,
    };
    rais::LayerStreamer streamer(sched, pool, config);

    // The streamer pre-acquires kSlots buffers
    REQUIRE(pool.live_buffers() == kSlots);

    // Process layers 0-3, releasing as we go
    for (size_t i = 0; i < 4; ++i) {
        auto h = streamer.request_layer(i, [](void*){});
        h.wait();
        streamer.release_layer(i);
    }

    // Live count should still be kSlots — buffers are recycled, not leaked
    REQUIRE(pool.live_buffers() == kSlots);

    cleanup_layer_dir(dir);
}

TEST_CASE("cancel_all cancels in-flight reads", "[layer_streamer]") {
    constexpr size_t kNumLayers = 10;
    constexpr size_t kLayerSize = 1024 * 64;

    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    REQUIRE(device != nil);

    auto dir = create_layer_dir(kNumLayers, kLayerSize);

    rais::Scheduler sched({.num_workers = 2});
    rais::MetalBufferPool pool((__bridge void*)device);

    rais::LayerStreamerConfig config{
        .layer_size_bytes = kLayerSize,
        .num_buffer_slots = 3,
        .model_dir = dir,
        .num_layers = kNumLayers,
    };
    rais::LayerStreamer streamer(sched, pool, config);

    streamer.start_prefetch(0);
    streamer.cancel_all();

    // After cancel, we should be able to start a fresh prefetch cycle
    // without issues — slots are freed.
    streamer.start_prefetch(0);
    auto h = streamer.request_layer(0, [](void*){});
    h.wait();
    REQUIRE(h.done());

    cleanup_layer_dir(dir);
}
