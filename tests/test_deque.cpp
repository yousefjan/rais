#include <catch2/catch_test_macros.hpp>

#include <rais/deque.hpp>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <numeric>
#include <thread>
#include <vector>

TEST_CASE("Owner push/pop preserves LIFO order for 10k elements", "[deque]") {
    rais::WorkStealingDeque<int*> deque;

    constexpr int N = 10'000;
    std::vector<int> storage(N);
    std::iota(storage.begin(), storage.end(), 0);

    for (int i = 0; i < N; ++i) {
        deque.push(&storage[i]);
    }

    // Pop should return in reverse (LIFO) order
    for (int i = N - 1; i >= 0; --i) {
        int* val = deque.pop();
        REQUIRE(val != nullptr);
        REQUIRE(*val == i);
    }

    // Deque should now be empty
    REQUIRE(deque.pop() == nullptr);
}

TEST_CASE("Pop from empty deque returns nullptr", "[deque]") {
    rais::WorkStealingDeque<int*> deque;
    REQUIRE(deque.pop() == nullptr);
}

TEST_CASE("Steal from empty deque returns nullptr", "[deque]") {
    rais::WorkStealingDeque<int*> deque;
    REQUIRE(deque.steal() == nullptr);
}

TEST_CASE("Single owner + 8 thieves — 100k tasks each consumed exactly once", "[deque]") {
    constexpr int NUM_TASKS = 100'000;
    constexpr int NUM_THIEVES = 8;

    // One atomic counter per task slot to verify exactly-once consumption
    std::vector<std::atomic<int>> consumed(NUM_TASKS);
    for (auto& c : consumed) c.store(0, std::memory_order_relaxed);

    // Task IDs stored as tagged pointers: encode the index into the pointer value.
    // We use uintptr_t cast to int* (never dereferenced — just used as an ID).
    rais::WorkStealingDeque<int*> deque;

    std::atomic<bool> done{false};

    // Thief threads
    std::vector<std::thread> thieves;
    thieves.reserve(NUM_THIEVES);
    for (int t = 0; t < NUM_THIEVES; ++t) {
        thieves.emplace_back([&]() {
            while (!done.load(std::memory_order_acquire)) {
                int* val = deque.steal();
                if (val != nullptr) {
                    auto idx = reinterpret_cast<uintptr_t>(val) - 1;
                    consumed[idx].fetch_add(1, std::memory_order_relaxed);
                }
            }
            // Drain remaining
            while (true) {
                int* val = deque.steal();
                if (val == nullptr) break;
                auto idx = reinterpret_cast<uintptr_t>(val) - 1;
                consumed[idx].fetch_add(1, std::memory_order_relaxed);
            }
        });
    }

    // Owner pushes all tasks, then pops any remaining
    for (int i = 0; i < NUM_TASKS; ++i) {
        // Encode task index as pointer value (offset by 1 so 0 means nullptr)
        deque.push(reinterpret_cast<int*>(static_cast<uintptr_t>(i + 1)));
    }

    // Owner pops what it can
    while (true) {
        int* val = deque.pop();
        if (val == nullptr) break;
        auto idx = reinterpret_cast<uintptr_t>(val) - 1;
        consumed[idx].fetch_add(1, std::memory_order_relaxed);
    }

    done.store(true, std::memory_order_release);

    for (auto& t : thieves) t.join();

    // Verify every task was consumed exactly once
    for (int i = 0; i < NUM_TASKS; ++i) {
        REQUIRE(consumed[i].load(std::memory_order_relaxed) == 1);
    }
}

TEST_CASE("Resize under concurrent steal pressure — capacity 4, push 10k with 4 thieves", "[deque]") {
    constexpr int NUM_TASKS = 10'000;
    constexpr int NUM_THIEVES = 4;

    // Start with capacity 4 to force multiple resizes
    rais::WorkStealingDeque<int*, 4> deque;

    std::vector<std::atomic<int>> consumed(NUM_TASKS);
    for (auto& c : consumed) c.store(0, std::memory_order_relaxed);

    std::atomic<bool> done{false};

    std::vector<std::thread> thieves;
    thieves.reserve(NUM_THIEVES);
    for (int t = 0; t < NUM_THIEVES; ++t) {
        thieves.emplace_back([&]() {
            while (!done.load(std::memory_order_acquire)) {
                int* val = deque.steal();
                if (val != nullptr) {
                    auto idx = reinterpret_cast<uintptr_t>(val) - 1;
                    consumed[idx].fetch_add(1, std::memory_order_relaxed);
                }
            }
            // Drain
            while (true) {
                int* val = deque.steal();
                if (val == nullptr) break;
                auto idx = reinterpret_cast<uintptr_t>(val) - 1;
                consumed[idx].fetch_add(1, std::memory_order_relaxed);
            }
        });
    }

    // Owner pushes with thieves running — forces resize under contention
    for (int i = 0; i < NUM_TASKS; ++i) {
        deque.push(reinterpret_cast<int*>(static_cast<uintptr_t>(i + 1)));
    }

    // Owner drains
    while (true) {
        int* val = deque.pop();
        if (val == nullptr) break;
        auto idx = reinterpret_cast<uintptr_t>(val) - 1;
        consumed[idx].fetch_add(1, std::memory_order_relaxed);
    }

    done.store(true, std::memory_order_release);
    for (auto& t : thieves) t.join();

    // Every task consumed exactly once
    for (int i = 0; i < NUM_TASKS; ++i) {
        REQUIRE(consumed[i].load(std::memory_order_relaxed) == 1);
    }
}

TEST_CASE("Wraparound correctness — repeated push/pop cycles", "[deque]") {
    rais::WorkStealingDeque<int*, 4> deque;

    int storage[4] = {10, 20, 30, 40};

    for (int round = 0; round < 200; ++round) {
        for (int i = 0; i < 4; ++i) deque.push(&storage[i]);
        for (int i = 3; i >= 0; --i) {
            int* val = deque.pop();
            REQUIRE(val == &storage[i]);
        }
        REQUIRE(deque.pop() == nullptr);
    }
}
