#include <rais/deque.hpp>

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

/// Measure steal latency: owner pushes continuously, thieves steal and
/// record per-steal latency. Reports P50/P95/P99.
static void bench_steal_latency(size_t num_thieves) {
    constexpr size_t NUM_ITEMS = 100'000;

    rais::WorkStealingDeque<uintptr_t> deque;

    std::atomic<bool> start{false};
    std::atomic<bool> done{false};

    // Each thief collects its own latencies to avoid contention on a shared vector
    std::vector<std::vector<int64_t>> thief_latencies(num_thieves);

    std::vector<std::thread> thieves;
    thieves.reserve(num_thieves);
    for (size_t t = 0; t < num_thieves; ++t) {
        thieves.emplace_back([&, t]() {
            while (!start.load(std::memory_order_acquire)) {}
            auto& latencies = thief_latencies[t];
            latencies.reserve(NUM_ITEMS / num_thieves);

            while (!done.load(std::memory_order_relaxed)) {
                auto t0 = std::chrono::high_resolution_clock::now();
                uintptr_t val = deque.steal();
                auto t1 = std::chrono::high_resolution_clock::now();
                if (val != 0) {
                    latencies.push_back(
                        std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
                }
            }
            // Drain
            while (true) {
                auto t0 = std::chrono::high_resolution_clock::now();
                uintptr_t val = deque.steal();
                auto t1 = std::chrono::high_resolution_clock::now();
                if (val == 0) break;
                latencies.push_back(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
            }
        });
    }

    start.store(true, std::memory_order_release);

    // Owner pushes items
    for (size_t i = 0; i < NUM_ITEMS; ++i) {
        deque.push(i + 1); // +1 so 0 means empty
    }

    // Give thieves time to drain, then signal done
    while (deque.size_approx() > 0) {
        std::this_thread::yield();
    }
    done.store(true, std::memory_order_release);

    for (auto& t : thieves) t.join();

    // Merge latencies
    std::vector<int64_t> all_latencies;
    for (auto& v : thief_latencies) {
        all_latencies.insert(all_latencies.end(), v.begin(), v.end());
    }

    if (all_latencies.empty()) {
        std::printf("  %zu thieves: no successful steals\n", num_thieves);
        return;
    }

    std::sort(all_latencies.begin(), all_latencies.end());

    auto percentile = [&](double p) -> int64_t {
        size_t idx = static_cast<size_t>(p / 100.0 * static_cast<double>(all_latencies.size() - 1));
        return all_latencies[idx];
    };

    int64_t p50 = percentile(50);
    int64_t p95 = percentile(95);
    int64_t p99 = percentile(99);

    std::printf("  %zu thieves: P50=%lldns  P95=%lldns  P99=%lldns  (n=%zu)\n",
                num_thieves, static_cast<long long>(p50),
                static_cast<long long>(p95), static_cast<long long>(p99),
                all_latencies.size());

    std::string label = "deque_steal_" + std::to_string(num_thieves) + "t";
    std::vector<BenchResult> results;
    results.push_back({label, "latency_p50", static_cast<double>(p50), "ns",
                       std::to_string(num_thieves) + " thieves"});
    results.push_back({label, "latency_p95", static_cast<double>(p95), "ns",
                       std::to_string(num_thieves) + " thieves"});
    results.push_back({label, "latency_p99", static_cast<double>(p99), "ns",
                       std::to_string(num_thieves) + " thieves"});
    append_results(results);
}

int main() {
    std::printf("Steal latency benchmark:\n");
    bench_steal_latency(4);
    bench_steal_latency(8);
    bench_steal_latency(16);
    std::printf("Results appended to experiments/results.tsv\n");
    return 0;
}
