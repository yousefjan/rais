# RAIS

RAIS (Runtime for AI Scheduling; رئيس) is a userspace C++20 task scheduler built for
concurrent local AI workloads on a single developer machine.

## Architecture

```
                    ┌──────────────────┐
                    │ External Submit  │
                    └────────┬─────────┘
                             │
                             ▼
                 ┌───────────────────────┐
                 │  Global MPMC Queue    │
                 │  (lock-free ring buf) │
                 └─────┬─────┬─────┬────┘
                       │     │     │
              ┌────────┘     │     └────────┐
              ▼              ▼              ▼
       ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
       │  Worker 0   │ │  Worker 1   │ │  Worker N   │
       │ ┌─────────┐ │ │ ┌─────────┐ │ │ ┌─────────┐ │
       │ │Chase-Lev│ │ │ │Chase-Lev│ │ │ │Chase-Lev│ │
       │ │  Deque  │ │ │ │  Deque  │ │ │ │  Deque  │ │
       │ └─────────┘ │ │ └─────────┘ │ │ └─────────┘ │
       └──────┼──────┘ └──────┼──────┘ └──────┼──────┘
              │               │               │
              └───── steal ───┴───── steal ────┘
```

## Results

| Date       | Build   | Test               | Metric     | Value     | Units   | Notes       |
|------------|---------|--------------------|------------|-----------|---------|-------------|
| 2026-03-23 | release | queue_mpmc_4p4c    | throughput | 5,874,574 | ops/sec | with padding|
| 2026-03-23 | release | queue_mpmc_4p4c    | throughput | 3,141,639 | ops/sec | no padding  |

Cache-line padding on the MPMC queue indices delivers ~87% higher throughput by eliminating false sharing between producer and consumer cores.

## Design Decisions

### Lock-free MPMC ring buffer with per-slot sequence numbers
**Why not a mutex-based queue?** The global submission queue is the single hottest contention point in the scheduler — every external submitter and every worker thread touches it. A mutex would serialize all access. The Dmitry Vyukov–style sequence-number design gives us wait-free reads of slot state and lock-free push/pop via CAS on shared indices, with no ABA risk because each slot's sequence counter is monotonically increasing.

### Power-of-two capacity
Replacing modulo (`pos % capacity`) with bitwise AND (`pos & mask`) eliminates an expensive integer division on every push/pop. The capacity constraint is enforced at construction with an assert.

### Cache-line padding between indices
The producer index (`tail_`) and consumer index (`head_`) are accessed by disjoint sets of threads. Without padding, they share a cache line and cause constant cross-core invalidation traffic. Benchmarks confirm a ~87% throughput improvement from `alignas(64)` padding.

### Separate padded and unpadded queue variants
The unpadded variant (`MPMCQueueUnpadded`) exists solely to measure the false-sharing penalty. It is not intended for production use.

