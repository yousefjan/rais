#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <string_view>

namespace rais {

/// GPU execution lane wrapping Metal's command queue.
///
/// All Objective-C types are hidden behind PIMPL so this header can be
/// included from pure C++ translation units. The implementation lives
/// in metal_executor.mm (Objective-C++).
///
/// MetalExecutor does not create its own MTLDevice — it accepts one from
/// outside so the device can be shared across the system.
class MetalExecutor {
public:
    /// Construct with an existing MTLDevice (passed as void* to avoid
    /// exposing Objective-C types) and path to a compiled .metallib.
    MetalExecutor(void* device, std::filesystem::path metallib_path);
    ~MetalExecutor();

    MetalExecutor(const MetalExecutor&) = delete;
    MetalExecutor& operator=(const MetalExecutor&) = delete;

    /// Submit GPU work. encode_fn receives an id<MTLComputeCommandEncoder>
    /// (as void*) to encode compute commands into. on_complete is called
    /// on Metal's completion thread when the GPU finishes.
    /// Returns false if the in-flight ring buffer is full (backpressure).
    bool submit(std::function<void(void* command_buffer, void* encoder)> encode_fn,
                std::function<void()> on_complete = {});

    /// Block until all submitted command buffers have completed.
    void flush();

    /// Expose the MTLDevice (as void*) for callers who need to allocate
    /// MTLBuffers directly.
    void* device() const;

    /// Load a named compute pipeline from the metallib. Returns
    /// id<MTLComputePipelineState> as void*. Cached after first call.
    void* pipeline(std::string_view kernel_name);

    /// Query GPU family support for runtime feature gating.
    /// family values: 1008 = MTLGPUFamilyApple8 (M3), etc.
    bool supports_family(int family) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace rais
