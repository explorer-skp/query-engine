// ported from raft-rsm/bench/bench_util.h — rigor infra, Raft-specifics stripped
#pragma once

// Small shared utilities for the benchmark harness. Everything here is
// allocation-free and self-contained: seeded PRNG streams (workload
// determinism), monotonic-nanosecond helpers, and best-effort thread pinning.

#include <chrono>
#include <cstdint>

#if defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

namespace qe::bench {

// splitmix64 — the standard seed-derivation mix: cheap, well-distributed,
// deterministic. Used to fan a single master seed into independent streams.
inline std::uint64_t splitmix64(std::uint64_t& state) {
    state += 0x9E3779B97F4A7C15ULL;
    std::uint64_t z = state;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    return z ^ (z >> 31);
}

// Derives an independent stream seed from (master, streamId).
inline std::uint64_t deriveSeed(std::uint64_t master, std::uint64_t streamId) {
    std::uint64_t s = master ^ (streamId * 0xD6E8FEB86659FD93ULL);
    return splitmix64(s);
}

// xorshift64* — the per-thread workload RNG (one u64 of state, no allocation;
// quality is ample for index picks and branch decisions).
struct XorShift64 {
    std::uint64_t state;
    explicit XorShift64(std::uint64_t seed) : state(seed ? seed : 0x9E3779B9ULL) {}
    std::uint64_t next() {
        std::uint64_t x = state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        state = x;
        return x * 0x2545F4914F6CDD1DULL;
    }
};

// All bench timestamps are steady_clock nanoseconds-since-epoch as u64 — one
// clock everywhere, so cross-thread time arithmetic is always valid.
inline std::uint64_t nowNs() {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now().time_since_epoch())
            .count());
}

inline std::uint64_t toNs(std::chrono::steady_clock::time_point tp) {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            tp.time_since_epoch())
            .count());
}

// Pins the calling thread to one CPU. Thread affinity is a Linux-only concept
// (sched.h / cpu_set_t / pthread_setaffinity_np); macOS exposes no portable
// equivalent, so there we change nothing and report the no-op. Callers treat
// pinning as best-effort and record the returned outcome either way.
inline bool pinSelfToCpu(int cpu) {
    if (cpu < 0) return false;
#if defined(__linux__)
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(static_cast<unsigned>(cpu), &set);
    return pthread_setaffinity_np(pthread_self(), sizeof(set), &set) == 0;
#else
    (void)cpu;
    return false;  // best-effort no-op on non-Linux (e.g. macOS); recorded
#endif
}

}  // namespace qe::bench
