#include <rais/scheduler.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <ctime>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

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
    constexpr int N = 10'000;

    std::printf("Scheduler latency benchmark: %d Interactive tasks, 4 workers\n", N);

    std::vector<uint64_t> latencies(N);
    std::vector<std::atomic<uint64_t>> start_times(N);
    for (auto& s : start_times) s.store(0, std::memory_order_relaxed);

    {
        rais::Scheduler sched({.num_workers = 4, .global_queue_capacity = 16384});

        std::vector<rais::TaskHandle> handles;
        handles.reserve(N);

        for (int i = 0; i < N; ++i) {
            uint64_t submit_ns = rais::clock_ns();
            handles.push_back(sched.submit([i, &start_times]() {
                start_times[i].store(rais::clock_ns(), std::memory_order_relaxed);
            }, rais::Lane::Interactive));
            latencies[i] = submit_ns; // temporarily store submit time
        }

        for (auto& h : handles) h.wait();

        // Compute submit-to-start latencies
        for (int i = 0; i < N; ++i) {
            uint64_t start = start_times[i].load(std::memory_order_relaxed);
            latencies[i] = start - latencies[i]; // start - submit
        }
    }

    std::sort(latencies.begin(), latencies.end());

    auto percentile = [&](double p) -> uint64_t {
        size_t idx = static_cast<size_t>(p / 100.0 * static_cast<double>(N - 1));
        return latencies[idx];
    };

    uint64_t p50 = percentile(50);
    uint64_t p95 = percentile(95);
    uint64_t p99 = percentile(99);

    std::printf("  P50=%lluns  P95=%lluns  P99=%lluns\n",
                static_cast<unsigned long long>(p50),
                static_cast<unsigned long long>(p95),
                static_cast<unsigned long long>(p99));

    std::vector<BenchResult> results;
    results.push_back({"scheduler_interactive_10k", "latency_p50",
                        static_cast<double>(p50), "ns", "4 workers"});
    results.push_back({"scheduler_interactive_10k", "latency_p95",
                        static_cast<double>(p95), "ns", "4 workers"});
    results.push_back({"scheduler_interactive_10k", "latency_p99",
                        static_cast<double>(p99), "ns", "4 workers"});
    append_results(results);
    std::printf("Results appended to experiments/results.tsv\n");

    return 0;
}
