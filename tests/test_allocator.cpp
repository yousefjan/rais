#include <rais/allocator.hpp>

#include <algorithm>
#include <atomic>
#include <catch2/catch_test_macros.hpp>
#include <numeric>
#include <set>
#include <thread>
#include <vector>

// ---------------------------------------------------------------------------
// SlabAllocator tests
// ---------------------------------------------------------------------------

TEST_CASE("Slab: alloc N, free all, realloc N — pointer reuse", "[slab]") {
    constexpr size_t N = 128;
    rais::SlabAllocator<uint64_t, N> slab;

    // Allocate all N slots
    std::vector<uint64_t*> ptrs;
    ptrs.reserve(N);
    for (size_t i = 0; i < N; ++i) {
        uint64_t* p = slab.allocate();
        REQUIRE(p != nullptr);
        *p = i;
        ptrs.push_back(p);
    }

    // Slab should be exhausted
    REQUIRE(slab.allocate() == nullptr);

    // All pointers should be unique
    std::set<uint64_t*> unique_ptrs(ptrs.begin(), ptrs.end());
    REQUIRE(unique_ptrs.size() == N);

    // Free all
    for (auto* p : ptrs) {
        slab.free(p);
    }

    // Reallocate N — all should succeed, and pointers should be reused
    std::set<uint64_t*> realloc_ptrs;
    for (size_t i = 0; i < N; ++i) {
        uint64_t* p = slab.allocate();
        REQUIRE(p != nullptr);
        realloc_ptrs.insert(p);
    }
    REQUIRE(realloc_ptrs.size() == N);

    // Verify pointer reuse: the set of reallocated pointers should be
    // the same as the original set
    REQUIRE(realloc_ptrs == unique_ptrs);

    // Cleanup
    for (auto* p : realloc_ptrs) {
        slab.free(p);
    }
}

TEST_CASE("Slab: owns() correctness", "[slab]") {
    rais::SlabAllocator<int, 16> slab;

    int* p = slab.allocate();
    REQUIRE(p != nullptr);
    REQUIRE(slab.owns(p));

    int stack_val = 42;
    REQUIRE_FALSE(slab.owns(&stack_val));

    slab.free(p);
}

TEST_CASE("Slab: 8-thread concurrent alloc/free", "[slab]") {
    constexpr size_t N = 8192;
    constexpr size_t NUM_THREADS = 8;
    constexpr size_t OPS_PER_THREAD = N / NUM_THREADS;

    rais::SlabAllocator<uint64_t, N> slab;

    // Each thread will allocate OPS_PER_THREAD slots, write a unique value,
    // verify the value, then free.
    std::atomic<uint64_t> total_allocated{0};
    std::atomic<uint64_t> total_freed{0};
    std::atomic<bool> start{false};

    std::vector<std::thread> threads;
    threads.reserve(NUM_THREADS);

    for (size_t t = 0; t < NUM_THREADS; ++t) {
        threads.emplace_back([&, t]() {
            while (!start.load(std::memory_order_acquire)) {}

            std::vector<uint64_t*> my_ptrs;
            my_ptrs.reserve(OPS_PER_THREAD);

            for (size_t i = 0; i < OPS_PER_THREAD; ++i) {
                uint64_t* p = nullptr;
                while (!(p = slab.allocate())) {
                    std::this_thread::yield(); // contention — retry
                }
                *p = t * OPS_PER_THREAD + i;
                my_ptrs.push_back(p);
                total_allocated.fetch_add(1, std::memory_order_relaxed);
            }

            // Verify all values
            for (size_t i = 0; i < OPS_PER_THREAD; ++i) {
                REQUIRE(*my_ptrs[i] == t * OPS_PER_THREAD + i);
            }

            // Free all
            for (auto* p : my_ptrs) {
                slab.free(p);
                total_freed.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }

    start.store(true, std::memory_order_release);
    for (auto& th : threads) th.join();

    REQUIRE(total_allocated.load() == N);
    REQUIRE(total_freed.load() == N);
}

TEST_CASE("Slab: repeated concurrent alloc/free cycles", "[slab]") {
    constexpr size_t N = 256;
    constexpr size_t NUM_THREADS = 8;
    constexpr size_t CYCLES = 100;
    constexpr size_t SLOTS_PER_THREAD = N / NUM_THREADS;

    rais::SlabAllocator<uint64_t, N> slab;
    std::atomic<bool> start{false};

    std::vector<std::thread> threads;
    for (size_t t = 0; t < NUM_THREADS; ++t) {
        threads.emplace_back([&]() {
            while (!start.load(std::memory_order_acquire)) {}

            for (size_t c = 0; c < CYCLES; ++c) {
                std::vector<uint64_t*> my_ptrs;
                for (size_t i = 0; i < SLOTS_PER_THREAD; ++i) {
                    uint64_t* p = nullptr;
                    while (!(p = slab.allocate())) {
                        std::this_thread::yield();
                    }
                    *p = 0xDEADBEEF;
                    my_ptrs.push_back(p);
                }
                for (auto* p : my_ptrs) {
                    slab.free(p);
                }
            }
        });
    }

    start.store(true, std::memory_order_release);
    for (auto& th : threads) th.join();
}

// ---------------------------------------------------------------------------
// ArenaAllocator tests
// ---------------------------------------------------------------------------

TEST_CASE("Arena: basic alloc and reset", "[arena]") {
    rais::ArenaAllocator arena(4096);

    void* a = arena.allocate(100);
    REQUIRE(a != nullptr);
    REQUIRE(arena.used() >= 100);

    void* b = arena.allocate(200);
    REQUIRE(b != nullptr);
    REQUIRE(b != a);

    arena.reset();
    REQUIRE(arena.used() == 0);

    // After reset, allocations reuse the same memory
    void* c = arena.allocate(100);
    REQUIRE(c == a);
}

TEST_CASE("Arena: exhaustion returns nullptr", "[arena]") {
    rais::ArenaAllocator arena(128);

    void* a = arena.allocate(64);
    REQUIRE(a != nullptr);

    void* b = arena.allocate(64);
    REQUIRE(b != nullptr);

    // Should be exhausted (or very close)
    void* c = arena.allocate(64);
    REQUIRE(c == nullptr);
}

TEST_CASE("Arena: alignment correctness", "[arena]") {
    rais::ArenaAllocator arena(8192);

    for (size_t align : {1, 4, 8, 16, 64}) {
        // Allocate a small object to potentially misalign
        arena.allocate(1, 1);

        void* p = arena.allocate(32, align);
        REQUIRE(p != nullptr);
        auto addr = reinterpret_cast<uintptr_t>(p);
        REQUIRE((addr % align) == 0);
    }
}

TEST_CASE("Arena: remaining() tracks space", "[arena]") {
    rais::ArenaAllocator arena(1024);

    REQUIRE(arena.remaining() == 1024);
    arena.allocate(256);
    REQUIRE(arena.remaining() <= 768);
    REQUIRE(arena.capacity() == 1024);
}
