#pragma once

#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <new>
#include <type_traits>
#include <utility>

namespace rais {

/// Bounded lock-free multi-producer multi-consumer ring buffer.
///
/// Each slot carries a sequence number that serves two purposes:
///   1. It tells producers/consumers whether the slot is available for writing/reading.
///   2. It solves the ABA problem — a slow thread that stalled mid-operation will see
///      a sequence mismatch when it resumes, rather than corrupting the queue.
///
/// Capacity must be a power of two so the index mask (capacity - 1) replaces
/// expensive modulo with a bitwise AND.
template <typename T>
class MPMCQueue {
    static_assert(std::is_trivially_copyable_v<T> || std::is_move_constructible_v<T>,
                  "T must be trivially copyable or move-constructible");

public:
    explicit MPMCQueue(size_t capacity)
        : capacity_(capacity),
          mask_(capacity - 1),
          slots_(static_cast<Slot*>(::operator new(sizeof(Slot) * capacity, std::align_val_t{alignof(Slot)}))) {
        assert(capacity >= 2 && "Capacity must be at least 2");
        assert((capacity & (capacity - 1)) == 0 && "Capacity must be a power of two");

        for (size_t i = 0; i < capacity_; ++i) {
            slots_[i].sequence.store(i, std::memory_order_relaxed); // initial sequence = slot index
        }
    }

    ~MPMCQueue() {
        // Destroy any remaining elements in the queue
        T tmp;
        while (pop(tmp)) {}
        ::operator delete(slots_, std::align_val_t{alignof(Slot)});
    }

    MPMCQueue(const MPMCQueue&) = delete;
    MPMCQueue& operator=(const MPMCQueue&) = delete;

    /// Non-blocking push. Returns false if the queue is full.
    bool push(const T& value) {
        return emplace_impl(value);
    }

    /// Non-blocking push (move). Returns false if the queue is full.
    bool push(T&& value) {
        return emplace_impl(std::move(value));
    }

    /// Non-blocking pop. Returns false if the queue is empty.
    bool pop(T& value) {
        size_t pos = head_.load(std::memory_order_relaxed); // relaxed: will be validated by sequence check
        for (;;) {
            Slot& slot = slots_[pos & mask_];
            size_t seq = slot.sequence.load(std::memory_order_acquire); // acquire: need to see the stored data
            intptr_t diff = static_cast<intptr_t>(seq) - static_cast<intptr_t>(pos + 1);

            if (diff == 0) {
                // Slot is ready for reading (sequence == pos + 1, set by a completed push)
                if (head_.compare_exchange_weak(pos, pos + 1,
                        std::memory_order_relaxed, // relaxed: the acquire on sequence already synchronizes
                        std::memory_order_relaxed)) {
                    value = std::move(slot.data);
                    // Advance sequence by capacity to mark slot as writable for the next cycle
                    slot.sequence.store(pos + mask_ + 1, std::memory_order_release); // release: make our read visible before slot reuse
                    return true;
                }
            } else if (diff < 0) {
                // Slot not yet written — queue is empty (or we're behind)
                return false;
            } else {
                // Another consumer took this slot; reload and retry
                pos = head_.load(std::memory_order_relaxed);
            }
        }
    }

    size_t capacity() const noexcept { return capacity_; }

private:
    template <typename U>
    bool emplace_impl(U&& value) {
        size_t pos = tail_.load(std::memory_order_relaxed); // relaxed: will be validated by sequence check
        for (;;) {
            Slot& slot = slots_[pos & mask_];
            size_t seq = slot.sequence.load(std::memory_order_acquire); // acquire: need to see prior consumer's release
            intptr_t diff = static_cast<intptr_t>(seq) - static_cast<intptr_t>(pos);

            if (diff == 0) {
                // Slot is writable (sequence matches our expected position)
                if (tail_.compare_exchange_weak(pos, pos + 1,
                        std::memory_order_relaxed, // relaxed: the release on sequence below provides ordering
                        std::memory_order_relaxed)) {
                    slot.data = std::forward<U>(value);
                    // Set sequence to pos + 1 to signal consumers this slot is readable
                    slot.sequence.store(pos + 1, std::memory_order_release); // release: data must be visible before sequence update
                    return true;
                }
            } else if (diff < 0) {
                // Slot still occupied by unconsumed data — queue is full
                return false;
            } else {
                // Another producer claimed this slot; reload and retry
                pos = tail_.load(std::memory_order_relaxed);
            }
        }
    }

    struct alignas(64) Slot { // 64-byte alignment: one slot per cache line to reduce false sharing between adjacent slots
        std::atomic<size_t> sequence;
        T data;
    };

    const size_t capacity_;
    const size_t mask_;
    Slot* const slots_;

    // Each of these lives on its own cache line to prevent false sharing between producers and consumers.
    // Without this padding, producer and consumer threads would bounce the same cache line between cores
    // on every push/pop, destroying throughput.
    alignas(64) std::atomic<size_t> tail_{0};
    alignas(64) std::atomic<size_t> head_{0};
};

/// Variant without cache-line padding, used to measure the impact of false sharing.
/// This exists solely for benchmarking — do not use in production.
template <typename T>
class MPMCQueueUnpadded {
    static_assert(std::is_trivially_copyable_v<T> || std::is_move_constructible_v<T>,
                  "T must be trivially copyable or move-constructible");

public:
    explicit MPMCQueueUnpadded(size_t capacity)
        : capacity_(capacity),
          mask_(capacity - 1),
          slots_(new SlotUnpadded[capacity]) {
        assert(capacity >= 2 && "Capacity must be at least 2");
        assert((capacity & (capacity - 1)) == 0 && "Capacity must be a power of two");

        for (size_t i = 0; i < capacity_; ++i) {
            slots_[i].sequence.store(i, std::memory_order_relaxed);
        }
    }

    ~MPMCQueueUnpadded() {
        T tmp;
        while (pop(tmp)) {}
        delete[] slots_;
    }

    MPMCQueueUnpadded(const MPMCQueueUnpadded&) = delete;
    MPMCQueueUnpadded& operator=(const MPMCQueueUnpadded&) = delete;

    bool push(const T& value) {
        return emplace_impl(value);
    }

    bool push(T&& value) {
        return emplace_impl(std::move(value));
    }

    bool pop(T& value) {
        size_t pos = head_.load(std::memory_order_relaxed);
        for (;;) {
            SlotUnpadded& slot = slots_[pos & mask_];
            size_t seq = slot.sequence.load(std::memory_order_acquire);
            intptr_t diff = static_cast<intptr_t>(seq) - static_cast<intptr_t>(pos + 1);

            if (diff == 0) {
                if (head_.compare_exchange_weak(pos, pos + 1,
                        std::memory_order_relaxed, std::memory_order_relaxed)) {
                    value = std::move(slot.data);
                    slot.sequence.store(pos + mask_ + 1, std::memory_order_release);
                    return true;
                }
            } else if (diff < 0) {
                return false;
            } else {
                pos = head_.load(std::memory_order_relaxed);
            }
        }
    }

    size_t capacity() const noexcept { return capacity_; }

private:
    template <typename U>
    bool emplace_impl(U&& value) {
        size_t pos = tail_.load(std::memory_order_relaxed);
        for (;;) {
            SlotUnpadded& slot = slots_[pos & mask_];
            size_t seq = slot.sequence.load(std::memory_order_acquire);
            intptr_t diff = static_cast<intptr_t>(seq) - static_cast<intptr_t>(pos);

            if (diff == 0) {
                if (tail_.compare_exchange_weak(pos, pos + 1,
                        std::memory_order_relaxed, std::memory_order_relaxed)) {
                    slot.data = std::forward<U>(value);
                    slot.sequence.store(pos + 1, std::memory_order_release);
                    return true;
                }
            } else if (diff < 0) {
                return false;
            } else {
                pos = tail_.load(std::memory_order_relaxed);
            }
        }
    }

    // No padding — slots, tail, and head may share cache lines
    struct SlotUnpadded {
        std::atomic<size_t> sequence;
        T data;
    };

    const size_t capacity_;
    const size_t mask_;
    SlotUnpadded* const slots_;
    std::atomic<size_t> tail_{0};
    std::atomic<size_t> head_{0};
};

} // namespace rais
