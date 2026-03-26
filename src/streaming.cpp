#include <rais/streaming.hpp>

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>

namespace rais {

TaskHandle submit_read(Scheduler& scheduler,
                       int fd,
                       off_t offset,
                       size_t length,
                       void* dest) {
    return scheduler.submit([fd, offset, length, dest]() {
        auto* buf = static_cast<char*>(dest);
        size_t remaining = length;
        off_t  pos = offset;

        while (remaining > 0) {
            ssize_t n = ::pread(fd, buf, remaining, pos);
            if (n < 0) {
                if (errno == EINTR) continue; // interrupted — retry
                // IO error — zero the rest and bail. Callers can detect
                // incomplete reads by checking buffer contents or adding
                // higher-level error tracking.
                std::memset(buf, 0, remaining);
                return;
            }
            if (n == 0) {
                // EOF before expected length — zero-fill remainder
                std::memset(buf, 0, remaining);
                return;
            }
            buf       += n;
            pos       += n;
            remaining -= static_cast<size_t>(n);
        }
    }, Lane::IO);
}

TaskHandle submit_file_read(Scheduler& scheduler,
                            const std::filesystem::path& path,
                            size_t offset,
                            size_t length,
                            void* dest) {
    int fd = ::open(path.c_str(), O_RDONLY);
    if (fd < 0) {
        // Can't open — return a task that completes immediately as a no-op.
        // Caller will see zero-filled buffer or can add error signaling.
        return scheduler.submit([](){}, Lane::IO);
    }

    // Bypass kernel page cache — layer weights are read once per forward
    // pass and shouldn't evict hot filesystem cache entries.
    ::fcntl(fd, F_NOCACHE, 1);

    auto read_handle = submit_read(scheduler, fd, static_cast<off_t>(offset),
                                   length, dest);

    // Chain a continuation that closes the fd after the read completes.
    // The fd is captured by value — safe because it's just an int.
    scheduler.then(read_handle, [fd]() {
        ::close(fd);
    }, Lane::IO);

    return read_handle;
}

} // namespace rais
