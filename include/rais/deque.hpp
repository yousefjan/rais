#pragma once

#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace rais {

/// Chase-Lev dynamic circular work-stealing deque.
///
/// Reference: Chase & Lev, "Dynamic Circular Work-Stealing Deque", SPAA 2005.
///
/// The owner thread pushes and pops from the bottom (LIFO).
/// Thief threads steal from the top (FIFO).
/// The internal circular buffer doubles when full and never shrinks.
///
/// Retired buffers are kept in a deferred-free list and only freed at
/// destruction. A production system would use hazard pointers or epoch-based
/// reclamation to safely reclaim retired buffers while thieves may still
/// hold stale pointers into them.
template <typename T, size_t InitialCapacity = 64>
class WorkStealingDeque {
    static_assert(InitialCapacity >= 2, "Capacity must be at least 2");
    static_assert((InitialCapacity & (InitialCapacity - 1)) == 0,
                  "Capacity must be a power of two");

    struct CircularBuffer {
        const size_t capacity;
        const size_t mask; // capacity - 1

        explicit CircularBuffer(size_t cap)
            : capacity(cap),
              mask(cap - 1),
              slots_(new std::atomic<T>[cap]) {}

        ~CircularBuffer() { delete[] slots_; }

        CircularBuffer(const CircularBuffer&) = delete;
        CircularBuffer& operator=(const CircularBuffer&) = delete;

        T load(int64_t i) const {
            // relaxed: the caller is responsible for establishing ordering
            // through the top/bottom indices
            return slots_[static_cast<size_t>(i) & mask].load(std::memory_order_relaxed);
        }

        void store(int64_t i, T val) {
            // relaxed: the owner publishes via a release store on bottom_
            slots_[static_cast<size_t>(i) & mask].store(val, std::memory_order_relaxed);
        }

        /// Allocate a new buffer with double capacity and copy the live
        /// range [top, bottom) from this buffer into it.
        CircularBuffer* grow(int64_t top, int64_t bottom) const {
            auto* buf = new CircularBuffer(capacity * 2);
            for (int64_t i = top; i < bottom; ++i) {
                buf->store(i, load(i));
            }
            return buf;
        }

    private:
        std::atomic<T>* slots_;
    };

public:
    WorkStealingDeque()
        : bottom_(0),
          top_(0),
          buffer_(new CircularBuffer(InitialCapacity)) {}

    ~WorkStealingDeque() {
        delete buffer_.load(std::memory_order_relaxed);
        for (auto* buf : retired_) {
            delete buf;
        }
    }

    WorkStealingDeque(const WorkStealingDeque&) = delete;
    WorkStealingDeque& operator=(const WorkStealingDeque&) = delete;

    /// Push an item onto the bottom of the deque. Owner thread only.
    void push(T val) {
        int64_t b = bottom_.load(std::memory_order_relaxed);
        // acquire: need to see the most recent top written by a thief's
        // successful CAS, so we correctly detect a full buffer
        int64_t t = top_.load(std::memory_order_acquire);
        CircularBuffer* buf = buffer_.load(std::memory_order_relaxed);

        if (b - t >= static_cast<int64_t>(buf->capacity)) {
            // Buffer full — grow. Copy live range into a new doubled buffer.
            CircularBuffer* new_buf = buf->grow(t, b);
            retired_.push_back(buf);
            // release: thieves loading the buffer pointer with acquire will
            // see all the data we copied into the new buffer
            buffer_.store(new_buf, std::memory_order_release);
            buf = new_buf;
        }

        buf->store(b, val);
        // release: makes the stored item visible to thieves who will
        // observe this new bottom value via their acquire load
        std::atomic_thread_fence(std::memory_order_release);
        bottom_.store(b + 1, std::memory_order_relaxed);
    }

    /// Pop an item from the bottom of the deque. Owner thread only.
    /// Returns nullptr (or the null value of T) if the deque is empty.
    T pop() {
        int64_t b = bottom_.load(std::memory_order_relaxed) - 1;
        CircularBuffer* buf = buffer_.load(std::memory_order_relaxed);
        // release: publish the decremented bottom so that a concurrent
        // thief's acquire load of bottom sees it, preventing both the
        // owner and thief from taking the same last element
        bottom_.store(b, std::memory_order_relaxed);
        // seq_cst fence: establishes a total order between the owner's
        // write to bottom and the thief's write to top. Without this,
        // both could read stale values and both claim the last item.
        std::atomic_thread_fence(std::memory_order_seq_cst);
        int64_t t = top_.load(std::memory_order_relaxed);

        if (t <= b) {
            // Non-empty
            T val = buf->load(b);
            if (t == b) {
                // Last element — race with thieves. CAS top from t to t+1.
                // seq_cst: must be totally ordered with steals to prevent
                // both owner and thief from claiming this element.
                if (!top_.compare_exchange_strong(t, t + 1,
                        std::memory_order_seq_cst,
                        std::memory_order_relaxed)) {
                    // A thief took it. Use b+1 (== original t+1) rather than
                    // the CAS-modified t+1: compare_exchange_strong overwrites
                    // t with the actual value on failure, so t+1 would overshoot
                    // bottom and create a phantom entry in the deque.
                    bottom_.store(b + 1, std::memory_order_relaxed);
                    return T{};
                }
                bottom_.store(b + 1, std::memory_order_relaxed);
            }
            return val;
        }

        // Empty
        bottom_.store(t, std::memory_order_relaxed);
        return T{};
    }

    /// Steal an item from the top of the deque. Any thread may call this.
    /// Returns nullptr (or the null value of T) if the deque is empty
    /// or the steal lost a race.
    T steal() {
        // acquire: need to see the top value before reading the buffer,
        // pairs with the CAS release below
        int64_t t = top_.load(std::memory_order_acquire);
        // seq_cst fence: establishes ordering between our read of top
        // and our read of bottom. Without seq_cst here, we could read
        // a stale (larger) bottom and attempt to steal an element that
        // the owner has already popped, leading to a double-take.
        std::atomic_thread_fence(std::memory_order_seq_cst);
        int64_t b = bottom_.load(std::memory_order_acquire);

        if (t >= b) {
            return T{}; // empty
        }

        // acquire: pairs with the release store in push() after a resize,
        // ensuring we see the data in the new buffer
        CircularBuffer* buf = buffer_.load(std::memory_order_acquire);
        T val = buf->load(t);

        // seq_cst CAS on top: must be totally ordered with the owner's
        // pop CAS and other thieves' steal CAS. A relaxed CAS here would
        // be unsafe because a thief could succeed the CAS but read stale
        // data from the buffer — the seq_cst provides the necessary
        // ordering to guarantee the load of val happened-before any
        // subsequent modification of the slot.
        if (!top_.compare_exchange_strong(t, t + 1,
                std::memory_order_seq_cst,
                std::memory_order_relaxed)) {
            // Lost race to another thief or the owner's pop
            return T{};
        }

        return val;
    }

    /// Returns the approximate number of items in the deque.
    /// Not linearizable — intended for diagnostics only.
    size_t size_approx() const {
        int64_t b = bottom_.load(std::memory_order_relaxed);
        int64_t t = top_.load(std::memory_order_relaxed);
        return static_cast<size_t>(b - t > 0 ? b - t : 0);
    }

private:
    // bottom_ is written only by the owner. Aligned to its own cache line
    // to avoid false sharing with top_ (which is contended by thieves).
    alignas(64) std::atomic<int64_t> bottom_;
    // top_ is contended: CAS'd by both owner (pop of last element) and
    // all thief threads (steal). Separate cache line from bottom_.
    alignas(64) std::atomic<int64_t> top_;

    std::atomic<CircularBuffer*> buffer_;

    // Retired buffers awaiting deferred free. Only the owner thread
    // modifies this (during push → grow), so no synchronization needed.
    std::vector<CircularBuffer*> retired_;
};

} // namespace rais
