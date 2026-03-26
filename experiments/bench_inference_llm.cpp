/// bench_inference_llm.cpp — benchmark real LLM weight loading: naive vs RAIS.
///
/// Reads a model directory prepared by fetch_model.py (manifest.tsv + layer_NNNN
/// files) and measures wall-clock time for two strategies:
///
///   Naive:  for each layer: open → read → close → simulate compute → next
///   RAIS:   IO-lane reads pipeline ahead of compute via triple buffering
///
/// Simulates per-layer compute proportional to layer size (matmul-bound).
/// Outputs TSV: model, approach, ttft_us, total_ms, tok_s, throughput_GBs.
///
/// Usage:
///   ./build/bench_inference_llm experiments/models/smollm2-135m \
///                                [experiments/models/tinyllama-1.1b-chat-v1.0]

#include <rais/clock.hpp>
#include <rais/scheduler.hpp>
#include <rais/streaming.hpp>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <unistd.h>
#include <vector>

namespace fs = std::filesystem;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

static constexpr int kWarmupRuns   = 1;
static constexpr int kMeasuredRuns = 5;
static constexpr size_t kNumSlots  = 3; // triple buffering

// Simulated compute: ~100 ns per KB of weights.
// This models the rough ratio of memory-bandwidth to compute
// in a typical matmul-bound transformer forward pass on Apple Silicon.
static constexpr double kComputeNsPerByte = 0.1;

static void burn_ns(uint64_t ns) {
    uint64_t start = rais::clock_ns();
    uint64_t sink = 0;
    while (rais::clock_ns() - start < ns) {
        sink += start;
    }
    if (sink == 1) std::abort();
}

// ---------------------------------------------------------------------------
// Manifest parsing
// ---------------------------------------------------------------------------

struct LayerInfo {
    size_t idx;
    size_t size_bytes;
};

struct Model {
    std::string name;
    fs::path dir;
    std::vector<LayerInfo> layers;
    size_t max_layer_bytes = 0;
    size_t total_bytes = 0;
};

static Model load_manifest(const fs::path& model_dir) {
    Model m;
    m.dir = model_dir;
    m.name = model_dir.filename().string();

    fs::path manifest = model_dir / "manifest.tsv";
    std::ifstream f(manifest);
    if (!f) {
        std::fprintf(stderr, "Cannot open %s\n", manifest.c_str());
        std::exit(1);
    }

    std::string line;
    std::getline(f, line); // skip header
    while (std::getline(f, line)) {
        std::istringstream ss(line);
        size_t idx, size, count;
        ss >> idx >> size >> count;
        m.layers.push_back({idx, size});
        m.max_layer_bytes = std::max(m.max_layer_bytes, size);
        m.total_bytes += size;
    }
    return m;
}

// ---------------------------------------------------------------------------
// Result
// ---------------------------------------------------------------------------

struct Result {
    double ttft_us;
    double total_ms;
    double tok_per_s;
    double throughput_gbs;
};

// ---------------------------------------------------------------------------
// Naive: sequential read + compute
// ---------------------------------------------------------------------------

static Result run_naive(const Model& model) {
    std::vector<uint8_t> buf(model.max_layer_bytes);

    uint64_t t0 = rais::clock_ns();
    uint64_t first_token_ns = 0;

    for (size_t i = 0; i < model.layers.size(); ++i) {
        const auto& layer = model.layers[i];
        char name[32];
        std::snprintf(name, sizeof(name), "layer_%04zu", layer.idx);
        fs::path p = model.dir / name;

        int fd = ::open(p.c_str(), O_RDONLY);
        ::fcntl(fd, F_NOCACHE, 1);

        size_t remaining = layer.size_bytes;
        uint8_t* dst = buf.data();
        while (remaining > 0) {
            ssize_t n = ::read(fd, dst, remaining);
            if (n <= 0) break;
            dst += n;
            remaining -= static_cast<size_t>(n);
        }
        ::close(fd);

        // Simulate compute proportional to layer size
        burn_ns(static_cast<uint64_t>(layer.size_bytes * kComputeNsPerByte));

        if (i == 0) {
            first_token_ns = rais::clock_ns() - t0;
        }
    }

    uint64_t total_ns = rais::clock_ns() - t0;
    size_t num_layers = model.layers.size();

    return {
        static_cast<double>(first_token_ns) / 1e3,
        static_cast<double>(total_ns) / 1e6,
        static_cast<double>(num_layers) / (static_cast<double>(total_ns) / 1e9),
        static_cast<double>(model.total_bytes) / (static_cast<double>(total_ns) / 1e9) / 1e9,
    };
}

// ---------------------------------------------------------------------------
// RAIS: pipelined IO + compute
// ---------------------------------------------------------------------------

static Result run_rais(const Model& model) {
    rais::SchedulerConfig cfg;
    cfg.num_workers     = 2;
    cfg.io_thread_count = 2;
    rais::Scheduler scheduler(cfg);

    // Pre-open all layer fds with F_NOCACHE
    std::vector<int> fds(model.layers.size());
    for (size_t i = 0; i < model.layers.size(); ++i) {
        char name[32];
        std::snprintf(name, sizeof(name), "layer_%04zu", model.layers[i].idx);
        fs::path p = model.dir / name;
        fds[i] = ::open(p.c_str(), O_RDONLY);
        ::fcntl(fds[i], F_NOCACHE, 1);
    }

    // Allocate triple-buffer slots sized to the largest layer
    std::vector<std::vector<uint8_t>> buffers(
        kNumSlots, std::vector<uint8_t>(model.max_layer_bytes));

    uint64_t t0 = rais::clock_ns();
    uint64_t first_token_ns = 0;

    size_t num_layers = model.layers.size();
    std::vector<rais::TaskHandle> read_handles(num_layers);

    // Layer 0 (embed/head) is loaded synchronously — same as naive.
    // Pipelining begins at transformer layers (layer 1+).
    {
        size_t sz = model.layers[0].size_bytes;
        size_t remaining = sz;
        uint8_t* dst = buffers[0].data();
        while (remaining > 0) {
            ssize_t n = ::pread(fds[0], dst, remaining,
                                static_cast<off_t>(sz - remaining));
            if (n <= 0) break;
            dst += n;
            remaining -= static_cast<size_t>(n);
        }
        burn_ns(static_cast<uint64_t>(sz * kComputeNsPerByte));
        first_token_ns = rais::clock_ns() - t0;
    }

    // Kick off initial reads for transformer layers
    for (size_t i = 1; i < std::min(1 + kNumSlots, num_layers); ++i) {
        read_handles[i] = rais::submit_read(
            scheduler, fds[i], 0, model.layers[i].size_bytes,
            buffers[(i - 1) % kNumSlots].data());
    }

    // Process transformer layers: wait for read, kick off next, compute
    for (size_t i = 1; i < num_layers; ++i) {
        read_handles[i].wait();

        // Kick off read-ahead
        size_t ahead = i + kNumSlots;
        if (ahead < num_layers) {
            read_handles[ahead] = rais::submit_read(
                scheduler, fds[ahead], 0, model.layers[ahead].size_bytes,
                buffers[(ahead - 1) % kNumSlots].data());
        }

        // Simulate compute
        burn_ns(static_cast<uint64_t>(model.layers[i].size_bytes * kComputeNsPerByte));
    }

    uint64_t total_ns = rais::clock_ns() - t0;

    for (int fd : fds) ::close(fd);
    scheduler.shutdown();

    return {
        static_cast<double>(first_token_ns) / 1e3,
        static_cast<double>(total_ns) / 1e6,
        static_cast<double>(num_layers) / (static_cast<double>(total_ns) / 1e9),
        static_cast<double>(model.total_bytes) / (static_cast<double>(total_ns) / 1e9) / 1e9,
    };
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "Usage: %s <model_dir> [model_dir2 ...]\n", argv[0]);
        return 1;
    }

    std::printf("model\tapproach\tttft_us\ttotal_ms\ttok_s\tthroughput_GBs\n");

    for (int a = 1; a < argc; ++a) {
        auto model = load_manifest(argv[a]);
        std::fprintf(stderr, "\n=== %s: %zu layers, %.1f MB total, max layer %.1f MB ===\n",
                     model.name.c_str(), model.layers.size(),
                     model.total_bytes / 1e6, model.max_layer_bytes / 1e6);

        // Warmup — populate kernel page cache, then F_NOCACHE bypasses it
        for (int w = 0; w < kWarmupRuns; ++w) {
            run_naive(model);
            run_rais(model);
        }
        std::fprintf(stderr, "  warmup done, measuring %d runs...\n", kMeasuredRuns);

        // Measured
        Result naive_avg{}, rais_avg{};
        for (int r = 0; r < kMeasuredRuns; ++r) {
            auto n = run_naive(model);
            auto p = run_rais(model);
            naive_avg.ttft_us        += n.ttft_us;
            naive_avg.total_ms       += n.total_ms;
            naive_avg.tok_per_s      += n.tok_per_s;
            naive_avg.throughput_gbs += n.throughput_gbs;
            rais_avg.ttft_us         += p.ttft_us;
            rais_avg.total_ms        += p.total_ms;
            rais_avg.tok_per_s       += p.tok_per_s;
            rais_avg.throughput_gbs  += p.throughput_gbs;
        }
        naive_avg.ttft_us        /= kMeasuredRuns;
        naive_avg.total_ms       /= kMeasuredRuns;
        naive_avg.tok_per_s      /= kMeasuredRuns;
        naive_avg.throughput_gbs /= kMeasuredRuns;
        rais_avg.ttft_us         /= kMeasuredRuns;
        rais_avg.total_ms        /= kMeasuredRuns;
        rais_avg.tok_per_s       /= kMeasuredRuns;
        rais_avg.throughput_gbs  /= kMeasuredRuns;

        std::printf("%s\tnaive\t%.1f\t%.1f\t%.1f\t%.2f\n",
                    model.name.c_str(),
                    naive_avg.ttft_us, naive_avg.total_ms,
                    naive_avg.tok_per_s, naive_avg.throughput_gbs);
        std::printf("%s\trais\t%.1f\t%.1f\t%.1f\t%.2f\n",
                    model.name.c_str(),
                    rais_avg.ttft_us, rais_avg.total_ms,
                    rais_avg.tok_per_s, rais_avg.throughput_gbs);
        std::fflush(stdout);

        std::fprintf(stderr, "  naive: TTFT=%.0fus  total=%.1fms  %.1f tok/s  %.2f GB/s\n",
                     naive_avg.ttft_us, naive_avg.total_ms,
                     naive_avg.tok_per_s, naive_avg.throughput_gbs);
        std::fprintf(stderr, "  rais:  TTFT=%.0fus  total=%.1fms  %.1f tok/s  %.2f GB/s\n",
                     rais_avg.ttft_us, rais_avg.total_ms,
                     rais_avg.tok_per_s, rais_avg.throughput_gbs);
        std::fprintf(stderr, "  speedup: TTFT %.2fx  throughput %.2fx\n",
                     naive_avg.ttft_us / rais_avg.ttft_us,
                     rais_avg.tok_per_s / naive_avg.tok_per_s);
    }

    return 0;
}
