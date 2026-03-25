#include <rais/scheduler.hpp>

#include <atomic>
#include <mutex>
#include <thread>
#include <vector>

#include <catch2/catch_test_macros.hpp>

TEST_CASE("Linear chain A -> B -> C", "[continuations]") {
    rais::Scheduler sched({.num_workers = 2});

    std::mutex mu;
    std::vector<int> order;

    auto a = sched.submit([&] {
        std::lock_guard<std::mutex> lock(mu);
        order.push_back(1);
    }, rais::Lane::Background);

    auto b = sched.then(a, [&] {
        std::lock_guard<std::mutex> lock(mu);
        order.push_back(2);
    });

    auto c = sched.then(b, [&] {
        std::lock_guard<std::mutex> lock(mu);
        order.push_back(3);
    });

    c.wait();

    std::lock_guard<std::mutex> lock(mu);
    REQUIRE(order == std::vector<int>{1, 2, 3});
}

TEST_CASE("Fan-in: C depends on A and B", "[continuations]") {
    rais::Scheduler sched({.num_workers = 4});

    std::atomic<int> count{0};
    std::atomic<bool> c_ran{false};

    auto a = sched.submit([&] {
        count.fetch_add(1, std::memory_order_relaxed);
    }, rais::Lane::Background);

    auto b = sched.submit([&] {
        count.fetch_add(1, std::memory_order_relaxed);
    }, rais::Lane::Background);

    auto c = sched.submit_after([&] {
        // Both A and B must have completed
        REQUIRE(count.load(std::memory_order_relaxed) == 2);
        c_ran.store(true, std::memory_order_relaxed);
    }, rais::Lane::Background, {a, b});

    c.wait();
    REQUIRE(c_ran.load(std::memory_order_relaxed));
}

TEST_CASE("Fan-out: A has two dependents B and C", "[continuations]") {
    rais::Scheduler sched({.num_workers = 4});

    std::atomic<bool> a_done{false};
    std::atomic<bool> b_done{false};
    std::atomic<bool> c_done{false};

    auto a = sched.submit([&] {
        a_done.store(true, std::memory_order_release);
    }, rais::Lane::Background);

    auto b = sched.then(a, [&] {
        REQUIRE(a_done.load(std::memory_order_acquire));
        b_done.store(true, std::memory_order_release);
    });

    auto c = sched.then(a, [&] {
        REQUIRE(a_done.load(std::memory_order_acquire));
        c_done.store(true, std::memory_order_release);
    });

    b.wait();
    c.wait();
    REQUIRE(b_done.load(std::memory_order_relaxed));
    REQUIRE(c_done.load(std::memory_order_relaxed));
}

TEST_CASE("Cascading cancel: A -> B -> C", "[continuations]") {
    rais::Scheduler sched({.num_workers = 2});

    std::atomic<bool> b_fn_ran{false};
    std::atomic<bool> c_fn_ran{false};

    // A sleeps briefly so we can cancel it before it completes
    auto a = sched.submit([&] {
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }, rais::Lane::Background);

    auto b = sched.then(a, [&] {
        b_fn_ran.store(true, std::memory_order_relaxed);
    });

    auto c = sched.then(b, [&] {
        c_fn_ran.store(true, std::memory_order_relaxed);
    });

    // Cancel A before it finishes
    a.cancel();
    c.wait();

    // B and C should be completed (for waiter unblocking) but their fns
    // should not have executed.
    REQUIRE(b.done());
    REQUIRE(c.done());
    REQUIRE_FALSE(b_fn_ran.load(std::memory_order_relaxed));
    REQUIRE_FALSE(c_fn_ran.load(std::memory_order_relaxed));
}

TEST_CASE("Cross-lane continuation", "[continuations]") {
    rais::Scheduler sched({.num_workers = 2});

    std::atomic<bool> interactive_done{false};
    std::atomic<bool> bg_done{false};

    auto a = sched.submit([&] {
        interactive_done.store(true, std::memory_order_release);
    }, rais::Lane::Interactive);

    auto b = sched.then(a, [&] {
        REQUIRE(interactive_done.load(std::memory_order_acquire));
        bg_done.store(true, std::memory_order_release);
    }, rais::Lane::Background);

    b.wait();
    REQUIRE(bg_done.load(std::memory_order_relaxed));
}

TEST_CASE("Zero deps behaves like submit", "[continuations]") {
    rais::Scheduler sched({.num_workers = 2});

    std::atomic<bool> ran{false};

    auto h = sched.submit_after([&] {
        ran.store(true, std::memory_order_relaxed);
    }, rais::Lane::Background, {});

    h.wait();
    REQUIRE(ran.load(std::memory_order_relaxed));
}

TEST_CASE("Stress: fan-in of 100 predecessors", "[continuations]") {
    rais::Scheduler sched({.num_workers = 4});

    std::atomic<int> pred_count{0};
    std::atomic<int> final_runs{0};

    std::vector<rais::TaskHandle> preds;
    preds.reserve(100);

    for (int i = 0; i < 100; ++i) {
        preds.push_back(sched.submit([&] {
            pred_count.fetch_add(1, std::memory_order_relaxed);
        }, rais::Lane::Background));
    }

    auto final_task = sched.submit_after([&] {
        // All 100 predecessors should have completed
        REQUIRE(pred_count.load(std::memory_order_relaxed) == 100);
        final_runs.fetch_add(1, std::memory_order_relaxed);
    }, rais::Lane::Background, preds);

    final_task.wait();

    // The final task must run exactly once
    REQUIRE(final_runs.load(std::memory_order_relaxed) == 1);
}
