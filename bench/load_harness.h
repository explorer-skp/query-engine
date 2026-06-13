// ported from raft-rsm/bench/load_gen.{h,cpp} — rigor infra, Raft-specifics stripped
#pragma once

// Domain-free load harness with coordinated-omission correction.
//
// The original generator welded this mechanism to a specific RPC client and
// workload. Stripped to the rigor core, the "work" is an injectable callable
// and nothing here knows what it does — it could drive a query operator, a
// kernel microbench, or a sleep. What survives is the part that makes a tail
// latency honest:
//
// Closed-loop: N threads, each issue -> wait -> issue. Finds saturation
// throughput and *service* latency, but SELF-THROTTLES: when the system slows
// down a closed-loop client offers less load, so queueing delay that real
// arrivals would experience never appears. Closed-loop latency is therefore
// service latency only — never the tail story.
//
// Open-loop: work units are scheduled at a fixed target rate, independent of
// completions (deterministic fixed-rate arrivals striped round-robin across T
// threads, thread j offset by j/rate). Each unit's latency is measured FROM
// ITS INTENDED (scheduled) SEND TIME — the coordinated-omission correction:
// when the system falls behind, the units queueing behind the slowdown record
// the queueing they actually suffered instead of being silently rescheduled.
// The harness never skips a scheduled unit (it keeps issuing late ones until
// the schedule is exhausted or the drain cap trips), so no sample is omitted
// and no back-fill is needed. The actual-send-time histogram is kept alongside
// to SHOW the correction working: the stall self-test asserts the intended
// tail moves while the actual tail hides the stall.
//
// Bounded in-flight caveat (documented honestly): each thread issues
// synchronously, so at most T units are in flight — a true open system queues
// without bound. Measuring from intended time still charges every unit its
// full schedule-to-response delay, so stalls and saturation appear in the
// tail; size T well above the in-flight demand at every rate below saturation.

#include <atomic>
#include <cstdint>
#include <functional>

#include "bench_util.h"
#include "histogram.h"

namespace qe::bench {

struct LoadConfig {
    int threads = 4;
    std::uint64_t seed = 1;
    // Open-loop only: total offered rate across all threads (units/sec).
    double ratePerSec = 1000.0;
    // Open-loop only: give up replaying a late schedule this long past the
    // window end (the abandoned count marks the run as oversaturated).
    std::uint64_t drainCapNs = 10ull * 1000 * 1000 * 1000;
};

struct LoadStats {
    qe::metrics::LatencyHistogram intended;  // latency from scheduled send (open)
    qe::metrics::LatencyHistogram actual;    // latency from actual send
    std::uint64_t scheduled = 0;  // units whose time fell in the window
    std::uint64_t oks = 0;
    std::uint64_t failures = 0;
    std::uint64_t abandoned = 0;          // open: schedule given up at drain cap
    std::uint64_t sumSendLatenessNs = 0;  // open: actual - intended send time
    std::uint64_t maxSendLatenessNs = 0;
    std::uint64_t sends = 0;              // in-window sends (lateness samples)

    void merge(const LoadStats& o) {
        intended.merge(o.intended);
        actual.merge(o.actual);
        scheduled += o.scheduled;
        oks += o.oks;
        failures += o.failures;
        abandoned += o.abandoned;
        sumSendLatenessNs += o.sumSendLatenessNs;
        maxSendLatenessNs = std::max(maxSendLatenessNs, o.maxSendLatenessNs);
        sends += o.sends;
    }
};

// One unit of work. Receives the calling thread's seeded RNG (so the workload
// is reproducible: same seed => same sequence) and the thread index. Returns
// true on success, false on a counted failure. The callable is responsible
// for whatever it models; the harness only times it. It MUST be safe to call
// concurrently from `cfg.threads` threads.
using WorkFn = std::function<bool(XorShift64& rng, int threadId)>;

// Deterministic per-thread index draw — the workload's reproducibility helper
// (same seed => same sequence; unit-tested). Kept here so injected work can
// share the harness's notion of a seeded draw.
inline std::uint32_t nextIndex(XorShift64& rng, int n) {
    return static_cast<std::uint32_t>(rng.next() %
                                      static_cast<std::uint64_t>(n <= 0 ? 1 : n));
}

// Runs closed-loop load from `cfg.threads` threads until measureEndNs (or
// `stop`, if non-null, for run-until-told modes). Records actual-send-time
// latency for units fully inside [measureStartNs, measureEndNs).
LoadStats runClosedLoop(const WorkFn& work, const LoadConfig& cfg,
                        std::uint64_t measureStartNs,
                        std::uint64_t measureEndNs,
                        const std::atomic<bool>* stop = nullptr);

// Runs open-loop load at cfg.ratePerSec from startNs until the schedule
// reaches measureEndNs. Units with intended time in [measureStartNs,
// measureEndNs) are recorded (intended- and actual-send-time histograms).
LoadStats runOpenLoop(const WorkFn& work, const LoadConfig& cfg,
                      std::uint64_t startNs, std::uint64_t measureStartNs,
                      std::uint64_t measureEndNs);

}  // namespace qe::bench
