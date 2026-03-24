#import <Metal/Metal.h>
#import <Foundation/Foundation.h>

#include <rais/metal_executor.hpp>

#include <mutex>
#include <semaphore>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace rais {

struct MetalExecutor::Impl {
    id<MTLDevice> device;
    id<MTLCommandQueue> command_queue;
    id<MTLLibrary> library;

    std::unordered_map<std::string, id<MTLComputePipelineState>> pipeline_cache;
    std::mutex pipeline_mutex; // guards pipeline_cache

    // Bounded in-flight command buffer ring. Apple recommends not exceeding
    // ~16 in-flight command buffers on a single queue; we use 8.
    static constexpr int kMaxInFlight = 8;
    std::counting_semaphore<kMaxInFlight> in_flight_sem{kMaxInFlight};

    // On M3/M4 (Apple9+), use command buffer descriptors with
    // retainedReferences = NO for reduced overhead. The caller must
    // keep encoder/buffer references alive until commit.
    bool use_unretained = false;
};

MetalExecutor::MetalExecutor(void* device, std::filesystem::path metallib_path)
    : impl_(std::make_unique<Impl>()) {

    impl_->device = (__bridge id<MTLDevice>)device;
    if (!impl_->device) {
        throw std::runtime_error("MetalExecutor: nil MTLDevice");
    }

    impl_->command_queue = [impl_->device newCommandQueue];
    if (!impl_->command_queue) {
        throw std::runtime_error("MetalExecutor: failed to create command queue");
    }

    // Load the compiled Metal library from disk
    NSError* error = nil;
    NSString* path = [NSString stringWithUTF8String:metallib_path.c_str()];
    impl_->library = [impl_->device newLibraryWithURL:[NSURL fileURLWithPath:path]
                                                error:&error];
    if (!impl_->library) {
        std::string msg = "MetalExecutor: failed to load metallib: ";
        msg += error.localizedDescription.UTF8String;
        throw std::runtime_error(msg);
    }

    // Detect M3/M4 for unretained references optimization.
    // MTLGPUFamilyApple9 corresponds to M3 and later.
    if ([impl_->device supportsFamily:MTLGPUFamilyApple9]) {
        impl_->use_unretained = true;
    }
}

MetalExecutor::~MetalExecutor() {
    flush();
}

bool MetalExecutor::submit(
        std::function<void(void* command_buffer, void* encoder)> encode_fn,
        std::function<void()> on_complete) {

    // Non-blocking acquire — returns false immediately if all slots are taken
    if (!impl_->in_flight_sem.try_acquire()) {
        return false; // backpressure
    }

    id<MTLCommandBuffer> cmd_buf;
    if (impl_->use_unretained) {
        // retainedReferences = NO: lower overhead on M3/M4, but the caller
        // must ensure all resources stay alive until the command buffer commits.
        MTLCommandBufferDescriptor* desc = [[MTLCommandBufferDescriptor alloc] init];
        desc.retainedReferences = NO;
        cmd_buf = [impl_->command_queue commandBufferWithDescriptor:desc];
    } else {
        cmd_buf = [impl_->command_queue commandBuffer];
    }

    if (!cmd_buf) {
        impl_->in_flight_sem.release();
        return false;
    }

    id<MTLComputeCommandEncoder> encoder = [cmd_buf computeCommandEncoder];
    if (!encoder) {
        impl_->in_flight_sem.release();
        return false;
    }

    // Let the caller encode compute commands
    encode_fn((__bridge void*)cmd_buf, (__bridge void*)encoder);

    [encoder endEncoding];

    // Capture on_complete and semaphore for the completion handler.
    // The block must own the function to keep it alive.
    auto complete_fn = std::move(on_complete);
    auto* sem = &impl_->in_flight_sem;

    [cmd_buf addCompletedHandler:^(id<MTLCommandBuffer> /* buf */) {
        if (complete_fn) {
            complete_fn();
        }
        sem->release();
    }];

    [cmd_buf commit];
    return true;
}

void MetalExecutor::flush() {
    // Acquire all semaphore slots to wait for all in-flight buffers,
    // then release them all.
    for (int i = 0; i < Impl::kMaxInFlight; ++i) {
        impl_->in_flight_sem.acquire();
    }
    for (int i = 0; i < Impl::kMaxInFlight; ++i) {
        impl_->in_flight_sem.release();
    }
}

void* MetalExecutor::device() const {
    return (__bridge void*)impl_->device;
}

void* MetalExecutor::pipeline(std::string_view kernel_name) {
    std::string name(kernel_name);

    {
        std::lock_guard lock(impl_->pipeline_mutex);
        auto it = impl_->pipeline_cache.find(name);
        if (it != impl_->pipeline_cache.end()) {
            return (__bridge void*)it->second;
        }
    }

    NSString* fn_name = [NSString stringWithUTF8String:name.c_str()];
    id<MTLFunction> function = [impl_->library newFunctionWithName:fn_name];
    if (!function) {
        throw std::runtime_error("MetalExecutor: kernel not found: " + name);
    }

    NSError* error = nil;
    id<MTLComputePipelineState> pso =
        [impl_->device newComputePipelineStateWithFunction:function error:&error];
    if (!pso) {
        std::string msg = "MetalExecutor: pipeline creation failed: ";
        msg += error.localizedDescription.UTF8String;
        throw std::runtime_error(msg);
    }

    {
        std::lock_guard lock(impl_->pipeline_mutex);
        impl_->pipeline_cache[name] = pso;
    }

    return (__bridge void*)pso;
}

bool MetalExecutor::supports_family(int family) const {
    return [impl_->device supportsFamily:static_cast<MTLGPUFamily>(family)];
}

} // namespace rais
