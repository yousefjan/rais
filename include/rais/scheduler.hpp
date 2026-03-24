#pragma once

#include <rais/clock.hpp>
#include <rais/deque.hpp>
#include <rais/queue.hpp>
#include <rais/task.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <thread>
#include <vector>

namespace rais {

enum class ShutdownPolicy { Drain, Cancel };

struct SchedulerConfig {
    size_t num_workers          = 0; // 0 = hardware_concurrency() - 1
    size_t global_queue_capacity = 65536; // must be power of two
};

class Scheduler {
public:
    explicit Scheduler(SchedulerConfig config = {});
    ~Scheduler();

    Scheduler(const Scheduler&) = delete;
    Scheduler& operator=(const Scheduler&) = delete;

    /// Submit a task to the scheduler. Sets enqueue_time_ns and returns
    /// a handle the caller can use to wait/cancel.
    TaskHandle submit(std::function<void()> fn, Lane lane = Lane::Background);

    /// Shut down the scheduler. Drain finishes all pending tasks;
    /// Cancel marks pending tasks as cancelled and stops immediately.
    void shutdown(ShutdownPolicy policy = ShutdownPolicy::Drain);

    /// Number of tasks currently in a given lane (approximate).
    int32_t lane_count(Lane lane) const;

private:
    struct Worker {
        std::thread thread;
        WorkStealingDeque<Task*> deque;
        size_t id = 0;

        Worker() = default;
        Worker(const Worker&) = delete;
        Worker& operator=(const Worker&) = delete;
    };

    void worker_loop(size_t worker_id);
    Task* try_steal(size_t worker_id, std::mt19937& rng);
    void check_starvation_promotions(Task* task);

    MPMCQueue<Task*> global_queue_;
    std::vector<std::unique_ptr<Worker>> workers_;

    // Per-lane admission counters. Indexed by static_cast<int>(Lane).
    alignas(64) std::atomic<int32_t> lane_counts_[4] = {};

    std::atomic<bool> stop_flag_{false};
    std::atomic<bool> shutdown_called_{false};
};

} // namespace rais
