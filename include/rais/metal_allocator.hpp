#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

namespace rais {

/// GPU buffer pool with size-class bucketing.
///
/// All Objective-C types are hidden behind PIMPL so this header can be
/// included from pure C++ translation units. The implementation lives
/// in metal_allocator.mm (Objective-C++).
///
/// Size classes (powers of two): 4KB, 64KB, 1MB, 16MB, 256MB.
/// All buffers use MTLStorageModeShared — runtime assert against
/// Managed mode on Apple Silicon (unified memory makes it pointless).
class MetalBufferPool {
public:
    /// Construct with an existing MTLDevice (passed as void* to avoid
    /// exposing Objective-C types).
    explicit MetalBufferPool(void* device);
    ~MetalBufferPool();

    MetalBufferPool(const MetalBufferPool&) = delete;
    MetalBufferPool& operator=(const MetalBufferPool&) = delete;

    /// Acquire a buffer of at least `bytes` size. Rounds up to the next
    /// size class and returns a pooled buffer if available, or allocates
    /// a new one. Returns id<MTLBuffer> as void*. Returns nullptr on
    /// allocation failure.
    void* acquire(size_t bytes);

    /// Return a buffer to the pool. Does not deallocate or zero memory.
    /// The buffer must have been acquired from this pool.
    void release(void* buffer);

    /// Total number of buffers currently held in pool free lists
    /// (not including buffers that are in use).
    size_t pool_size() const;

    /// Total number of buffers currently in use (acquired but not released).
    size_t live_buffers() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace rais
