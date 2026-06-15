//  WP-10 GATE: the validity gate must be shown to BITE. A gate that cannot reject
//  proves nothing (RIGOR.md rule 4, applied to the measurement instrument). Two
//  proofs:
//
//   1. STATIC decision (deterministic, host-independent): a MachineState carrying
//      background load / a hot thermal zone / a non-performance governor is
//      REJECTED, with a reason naming the cause. No timing — pure decision logic.
//
//   2. DYNAMIC quiescence probe (the macOS-side teeth): with the machine
//      deliberately OVERSUBSCRIBED by busy threads, the fixed compute probe's
//      sample spread blows past the ceiling and the gate rejects the run. This is
//      the "a deliberately loaded run is rejected" demonstration — the same signal
//      that catches thermal throttle on a host with no governor knob.
//
//  Determinism: the static cases are exact; the dynamic case asserts a direction
//  (loaded => more jitter) that holds regardless of absolute host speed.

#include <atomic>
#include <thread>
#include <vector>

#include "doctest/doctest.h"

#include "bench_report.h"
#include "validity_gate.h"

using namespace qe::bench;

namespace {

bool hasReasonContaining(const GateVerdict& v, const char* needle) {
    for (const auto& r : v.reasons)
        if (r.find(needle) != std::string::npos) return true;
    return false;
}

MachineState quietHost() {
    MachineState m;
    m.cpuModel = "synthetic";
    m.logicalCpus = 8;
    m.loadavg1 = 0.0;
    m.governors = "n/a-macOS";  // macOS: governor knob absent
    m.maxTempC = -1;            // thermal n/a
    return m;
}

}  // namespace

TEST_CASE("validity gate: background load is rejected (static, deterministic)") {
    MachineState m = quietHost();
    m.loadavg1 = 100.0;  // 12.5 per cpu, way over the 0.6 ceiling
    const GateVerdict v = evaluateGate(m);
    CHECK_FALSE(v.accepted);
    CHECK(hasReasonContaining(v, "background load"));
}

TEST_CASE("validity gate: hot thermal zone is rejected (x86 knob)") {
    MachineState m = quietHost();
    m.maxTempC = 120;  // above the 90C ceiling
    const GateVerdict v = evaluateGate(m);
    CHECK_FALSE(v.accepted);
    CHECK(v.thermal_checked);
    CHECK(hasReasonContaining(v, "thermal"));
}

TEST_CASE("validity gate: non-performance governor is rejected (x86 knob)") {
    MachineState m = quietHost();
    m.governors = "powersave";  // not 'performance'
    const GateVerdict v = evaluateGate(m);
    CHECK_FALSE(v.accepted);
    CHECK(v.governor_checked);
    CHECK(hasReasonContaining(v, "governor"));
}

TEST_CASE("validity gate: a macOS-shaped quiet host checks only what it can") {
    // No governor, no thermal: the gate must NOT invent a reason from absent
    // knobs. (It may still reject on the live probe if THIS host is busy, so we
    // assert the absence of phantom static reasons, not unconditional acceptance.)
    MachineState m = quietHost();
    const GateVerdict v = evaluateGate(m);
    CHECK_FALSE(v.thermal_checked);   // n/a not mistaken for a reading
    CHECK_FALSE(v.governor_checked);  // n/a not mistaken for a reading
    CHECK_FALSE(hasReasonContaining(v, "background load"));
    CHECK_FALSE(hasReasonContaining(v, "thermal"));
    CHECK_FALSE(hasReasonContaining(v, "governor"));
}

TEST_CASE("validity gate: deliberate oversubscription trips the quiescence probe") {
    // Oversubscribe every logical core with spin loops, then run the probe under
    // contention: the per-rep spread must exceed the ceiling. This is the
    // portable signal that bites on macOS (and on any throttling host).
    const unsigned hw = std::thread::hardware_concurrency();
    const int hogs = static_cast<int>(hw ? hw * 2 : 8);
    std::atomic<bool> stop{false};
    std::vector<std::thread> busy;
    busy.reserve(static_cast<std::size_t>(hogs));
    for (int i = 0; i < hogs; ++i)
        busy.emplace_back([&stop] {
            volatile std::uint64_t x = 1;
            while (!stop.load(std::memory_order_relaxed))
                x = x * 6364136223846793005ull + 1;
        });

    GateConfig cfg;
    cfg.probe_reps = 15;
    std::uint64_t sink = 0;
    std::vector<std::uint64_t> s = runQuiescenceProbe(cfg, sink);
    std::uint64_t mn = s[0], mx = s[0];
    for (auto x : s) {
        mn = x < mn ? x : mn;
        mx = x > mx ? x : mx;
    }
    const double spread =
        mn > 0 ? static_cast<double>(mx) / static_cast<double>(mn) : 0.0;

    stop.store(true, std::memory_order_relaxed);
    for (auto& th : busy) th.join();

    CAPTURE(spread);
    CAPTURE(cfg.max_probe_spread);
    // Under heavy oversubscription the scheduler steals cycles unevenly => jitter.
    CHECK(spread > cfg.max_probe_spread);
    // Touch sink so the probe work cannot be elided.
    CHECK(sink != 0xFFFFFFFFFFFFFFFFull);
}
