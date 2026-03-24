#pragma once

#include <atomic>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <new>

namespace rais {

// ---------------------------------------------------------------------------
// SlabAllocator<T, N>
//
// Lock-free fixed-size object pool. Uses a tagged free list with upper 16-bit
// generation counter to prevent ABA. O(1) allocate and free, thread-safe.
//
// A production system might use hazard pointers or epoch-based reclamation
// for the free list nodes; the tagged pointer approach here is simple and
// sufficient for 48-bit virtual address spaces (x86-64 and AArch64).
// ---------------------------------------------------------------------------

template <typename T, size_t N = 1024>
class SlabAllocator {
public:
    SlabAllocator() {
        static_assert(N > 0, "Slab capacity must be > 0");
        // Allocate raw storage for N objects, cache-line aligned
        storage_ = static_cast<Slot*>(
            ::operator new(N * sizeof(Slot), std::align_val_t{alignof(Slot)}));

        // Build the free list: each slot points to the next
        for (size_t i = 0; i + 1 < N; ++i) {
            storage_[i].next.store(&storage_[i + 1], std::memory_order_relaxed);
        }
        storage_[N - 1].next.store(nullptr, std::memory_order_relaxed);

        // Tag starts at 0
        head_.store(pack(storage_, 0), std::memory_order_relaxed);
    }

    ~SlabAllocator() {
        // Caller must have freed all objects before destroying the allocator
        ::operator delete(storage_, std::align_val_t{alignof(Slot)});
    }

    SlabAllocator(const SlabAllocator&) = delete;
    SlabAllocator& operator=(const SlabAllocator&) = delete;

    /// Allocate one T-sized slot. Returns nullptr if the slab is exhausted.
    T* allocate() {
        TaggedPtr old_head = head_.load(std::memory_order_acquire);
        for (;;) {
            Slot* slot = ptr(old_head);
            if (!slot) return nullptr; // exhausted

            // Read next before CAS — another thread might free slot between
            // our load and CAS, but the tag prevents ABA.
            Slot* next = slot->next.load(std::memory_order_acquire);
            uint16_t new_tag = static_cast<uint16_t>(tag(old_head) + 1);
            TaggedPtr new_head = pack(next, new_tag);

            if (head_.compare_exchange_weak(old_head, new_head,
                    std::memory_order_acq_rel, std::memory_order_acquire)) {
                return reinterpret_cast<T*>(slot->data);
            }
        }
    }

    /// Return a previously allocated slot to the free list.
    void free(T* obj) {
        assert(owns(obj) && "SlabAllocator::free: pointer not from this slab");
        auto addr = reinterpret_cast<uintptr_t>(obj);
        auto base = reinterpret_cast<uintptr_t>(storage_);
        Slot* slot = &storage_[(addr - base) / sizeof(Slot)];

        TaggedPtr old_head = head_.load(std::memory_order_acquire);
        for (;;) {
            slot->next.store(ptr(old_head), std::memory_order_release);
            uint16_t new_tag = static_cast<uint16_t>(tag(old_head) + 1);
            TaggedPtr new_head = pack(slot, new_tag);

            if (head_.compare_exchange_weak(old_head, new_head,
                    std::memory_order_acq_rel, std::memory_order_acquire)) {
                return;
            }
        }
    }

    /// Check whether a pointer belongs to this slab.
    bool owns(const T* obj) const {
        auto addr = reinterpret_cast<uintptr_t>(obj);
        auto base = reinterpret_cast<uintptr_t>(&storage_[0].data);
        if (addr < base) return false;
        size_t offset = addr - base;
        return offset < N * sizeof(Slot) && offset % sizeof(Slot) == 0;
    }

    static constexpr size_t capacity() { return N; }

private:
    // Each slot is either occupied (contains a T) or free (contains a next ptr).
    // next is atomic to avoid TSan reports on the speculative read in allocate()
    // that races with writes in free(). The CAS on head_ ensures we never act on
    // a stale next value, but the read itself must be data-race-free.
    struct Slot {
        union {
            alignas(T) unsigned char data[sizeof(T)];
            char pad[sizeof(T) < sizeof(std::atomic<Slot*>)
                     ? sizeof(std::atomic<Slot*>) : sizeof(T)];
        };
        std::atomic<Slot*> next{nullptr};
    };

    // Tagged pointer: upper 16 bits = generation tag, lower 48 bits = pointer.
    // This prevents ABA on the lock-free free list.
    using TaggedPtr = uint64_t;

    static TaggedPtr pack(Slot* p, uint16_t t) {
        return (static_cast<uint64_t>(t) << 48) |
               (reinterpret_cast<uintptr_t>(p) & 0x0000FFFFFFFFFFFF);
    }

    static Slot* ptr(TaggedPtr tp) {
        // Sign-extend bit 47 for canonical addresses on x86-64
        uintptr_t raw = tp & 0x0000FFFFFFFFFFFF;
        if (raw & (1ULL << 47)) {
            raw |= 0xFFFF000000000000ULL;
        }
        return reinterpret_cast<Slot*>(raw);
    }

    static uint16_t tag(TaggedPtr tp) {
        return static_cast<uint16_t>(tp >> 48);
    }

    Slot* storage_ = nullptr;
    alignas(64) std::atomic<TaggedPtr> head_{0};
};

// ---------------------------------------------------------------------------
// ArenaAllocator
//
// Bump-pointer allocator with fixed capacity. Not thread-safe — designed for
// per-worker usage. O(1) allocate, O(1) bulk reset.
// ---------------------------------------------------------------------------

class ArenaAllocator {
public:
    explicit ArenaAllocator(size_t capacity)
        : capacity_(capacity), offset_(0) {
        // Over-align to support up to 64-byte alignment requests
        buf_ = static_cast<unsigned char*>(
            ::operator new(capacity, std::align_val_t{64}));
    }

    ~ArenaAllocator() {
        ::operator delete(buf_, std::align_val_t{64});
    }

    ArenaAllocator(const ArenaAllocator&) = delete;
    ArenaAllocator& operator=(const ArenaAllocator&) = delete;

    /// Allocate `bytes` with given alignment. Returns nullptr if out of space.
    void* allocate(size_t bytes, size_t align = alignof(std::max_align_t)) {
        assert((align & (align - 1)) == 0 && "alignment must be power of two");
        // Round up offset to alignment
        size_t aligned = (offset_ + align - 1) & ~(align - 1);
        if (aligned + bytes > capacity_) return nullptr;
        void* result = buf_ + aligned;
        offset_ = aligned + bytes;
        return result;
    }

    /// Reset the arena, reclaiming all memory at once. Does not call
    /// destructors — caller must manage object lifetimes.
    void reset() { offset_ = 0; }

    size_t capacity() const { return capacity_; }
    size_t used() const { return offset_; }
    size_t remaining() const { return capacity_ - offset_; }

private:
    unsigned char* buf_ = nullptr;
    size_t capacity_;
    size_t offset_;
};

} // namespace rais
