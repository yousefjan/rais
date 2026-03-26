#pragma once

#include <rais/scheduler.hpp>

#include <cstddef>
#include <filesystem>
#include <functional>
#include <sys/types.h>

namespace rais {

/// Submits an IO-lane task that reads `length` bytes from `fd` at `offset`
/// into `dest`. Uses pread() for thread safety (no shared file offset).
/// On macOS, the caller should set F_NOCACHE on the fd to bypass the kernel
/// page cache — layer weights are used once per forward pass and shouldn't
/// pollute cache.
///
/// Returns a TaskHandle so callers can chain GPU compute as a continuation.
TaskHandle submit_read(Scheduler& scheduler,
                       int fd,
                       off_t offset,
                       size_t length,
                       void* dest);

/// Convenience: opens the file, reads, and closes. Suitable for one-shot
/// layer loads. Sets F_NOCACHE on the fd. Caller owns `dest` memory
/// (typically a MetalBufferPool buffer).
TaskHandle submit_file_read(Scheduler& scheduler,
                            const std::filesystem::path& path,
                            size_t offset,
                            size_t length,
                            void* dest);

} // namespace rais
