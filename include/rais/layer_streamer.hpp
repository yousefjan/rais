#pragma once

#include <rais/metal_allocator.hpp>
#include <rais/scheduler.hpp>
#include <rais/streaming.hpp>

#include <cstddef>
#include <filesystem>
#include <functional>
#include <mutex>
#include <vector>

namespace rais {

struct LayerStreamerConfig {
    size_t layer_size_bytes;         // size of one model layer's weights
    size_t num_buffer_slots = 3;     // triple buffering: read / compute / recycle.
                                     // 3 is the minimum for full pipelining.
    std::filesystem::path model_dir; // directory containing per-layer weight files
    size_t num_layers;               // total layers in the model
};

/// Manages a ring of GPU buffers for layer-by-layer model streaming.
///
/// The pipeline per layer:
///   1. Acquire a buffer from MetalBufferPool
///   2. Submit an IO-lane read to fill it from SSD
///   3. When the read completes, the buffer is ready for GPU compute
///   4. After GPU compute, release the buffer back to the pool
///
/// LayerStreamer maintains `num_buffer_slots` buffers and pipelines reads
/// ahead of compute so the GPU never stalls waiting for SSD.
///
/// Layer weight files are expected at `model_dir / "layer_NNNN"` where NNNN
/// is the zero-padded (4 digits) layer index. This convention is documented
/// here and can be adapted to match the inference engine's naming.
class LayerStreamer {
public:
    LayerStreamer(Scheduler& scheduler,
                 MetalBufferPool& pool,
                 LayerStreamerConfig config);
    ~LayerStreamer();

    LayerStreamer(const LayerStreamer&) = delete;
    LayerStreamer& operator=(const LayerStreamer&) = delete;

    /// Request layer `layer_idx` for GPU use. Returns a TaskHandle that
    /// completes when the layer's weights are in a GPU buffer ready for
    /// compute. The caller receives the buffer pointer via `on_ready`.
    ///
    /// LayerStreamer may have already prefetched this layer — if so,
    /// the TaskHandle completes immediately.
    TaskHandle request_layer(size_t layer_idx,
                             std::function<void(void* buffer)> on_ready);

    /// Begin prefetching layers starting from `start_layer`. Call this
    /// when inference begins to fill the pipeline ahead of compute.
    void start_prefetch(size_t start_layer);

    /// Signal that the GPU is done with a previously requested layer's
    /// buffer. The buffer returns to the ring for reuse. Automatically
    /// advances the prefetch window to the next not-yet-prefetched layer.
    void release_layer(size_t layer_idx);

    /// Cancel all in-flight prefetch reads (e.g., user started typing
    /// and the current inference is being abandoned).
    void cancel_all();

private:
    struct Slot {
        void* buffer   = nullptr;  // MTLBuffer (as void*) from MetalBufferPool
        void* contents = nullptr;  // buffer's CPU-visible pointer (MTLBuffer.contents)
        size_t layer_idx = SIZE_MAX; // which layer is loaded (SIZE_MAX = none)
        TaskHandle read_handle;     // in-flight read, if any
        bool occupied  = false;     // true if buffer holds valid/in-progress data
    };

    std::filesystem::path layer_path(size_t idx) const;
    void prefetch_layer(size_t layer_idx);

    Scheduler& scheduler_;
    MetalBufferPool& pool_;
    LayerStreamerConfig config_;

    // Protects slots_, next_prefetch_, and slot assignment logic.
    // Contention is low: held briefly for bookkeeping, never during IO.
    std::mutex mu_;
    std::vector<Slot> slots_;
    // Next layer to prefetch. Initialized past-the-end so release_layer
    // doesn't auto-prefetch unless start_prefetch has been called.
    size_t next_prefetch_ = SIZE_MAX;
};

} // namespace rais
