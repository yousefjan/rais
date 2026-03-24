#include <catch2/catch_test_macros.hpp>

#include <rais/queue.hpp>

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <thread>
#include <vector>

TEST_CASE("SPSC correctness — produce N then consume N preserves order", "[queue]") {
    constexpr size_t N = 1024;
    rais::MPMCQueue<uint64_t> q(N);

    for (uint64_t i = 0; i < N; ++i) {
        REQUIRE(q.push(i));
    }

    for (uint64_t i = 0; i < N; ++i) {
        uint64_t val = 0;
        REQUIRE(q.pop(val));
        REQUIRE(val == i);
    }
}

TEST_CASE("Push to full queue returns false", "[queue]") {
    rais::MPMCQueue<int> q(4);

    REQUIRE(q.push(1));
    REQUIRE(q.push(2));
    REQUIRE(q.push(3));
    REQUIRE(q.push(4));
    REQUIRE_FALSE(q.push(5));
}

TEST_CASE("Pop from empty queue returns false", "[queue]") {
    rais::MPMCQueue<int> q(4);
    int val = 0;
    REQUIRE_FALSE(q.pop(val));
}

TEST_CASE("MPMC stress — 4 producers 4 consumers 1M items all received exactly once", "[queue]") {
    constexpr size_t NUM_PRODUCERS = 4;
    constexpr size_t NUM_CONSUMERS = 4;
    constexpr size_t ITEMS_PER_PRODUCER = 250'000;
    constexpr size_t TOTAL_ITEMS = NUM_PRODUCERS * ITEMS_PER_PRODUCER;

    rais::MPMCQueue<uint64_t> q(65536);

    std::vector<std::thread> producers;
    producers.reserve(NUM_PRODUCERS);
    for (size_t i = 0; i < NUM_PRODUCERS; ++i) {
        producers.emplace_back([&q, i]() {
            uint64_t base = i * ITEMS_PER_PRODUCER;
            for (uint64_t j = 0; j < ITEMS_PER_PRODUCER; ++j) {
                while (!q.push(base + j)) {
                    // spin until space available
                }
            }
        });
    }

    std::vector<std::vector<uint64_t>> consumed(NUM_CONSUMERS);
    std::vector<std::thread> consumers;
    consumers.reserve(NUM_CONSUMERS);
    std::atomic<bool> done{false};

    for (size_t i = 0; i < NUM_CONSUMERS; ++i) {
        consumed[i].reserve(TOTAL_ITEMS / NUM_CONSUMERS + 1024);
        consumers.emplace_back([&q, &consumed, &done, i]() {
            uint64_t val = 0;
            while (!done.load(std::memory_order_relaxed)) {
                if (q.pop(val)) {
                    consumed[i].push_back(val);
                }
            }
            while (q.pop(val)) {
                consumed[i].push_back(val);
            }
        });
    }

    for (auto& t : producers) t.join();
    done.store(true, std::memory_order_relaxed);
    for (auto& t : consumers) t.join();

    std::vector<uint64_t> all;
    all.reserve(TOTAL_ITEMS);
    for (auto& v : consumed) {
        all.insert(all.end(), v.begin(), v.end());
    }

    REQUIRE(all.size() == TOTAL_ITEMS);

    std::sort(all.begin(), all.end());
    for (size_t i = 0; i < TOTAL_ITEMS; ++i) {
        REQUIRE(all[i] == i);
    }
}

TEST_CASE("Wraparound correctness — push/pop beyond capacity", "[queue]") {
    rais::MPMCQueue<uint64_t> q(4);

    for (int round = 0; round < 100; ++round) {
        for (uint64_t i = 0; i < 4; ++i) {
            REQUIRE(q.push(round * 4 + i));
        }
        REQUIRE_FALSE(q.push(999));

        for (uint64_t i = 0; i < 4; ++i) {
            uint64_t val = 0;
            REQUIRE(q.pop(val));
            REQUIRE(val == static_cast<uint64_t>(round * 4 + i));
        }
        uint64_t tmp = 0;
        REQUIRE_FALSE(q.pop(tmp));
    }
}
