//  WP-10: the VALIDITY GATE. A measurement taken on a contaminated host is not a
//  measurement — it is noise wearing a number's clothes. Before any latency is
//  reported, the harness asks this gate "is this machine fit to benchmark on right
//  now?", and a NO means the run is rejected, never reported (RIGOR.md rule 7 /
//  §10). The verdict travels WITH the run in its JSON so a reader can see the gate
//  fired.
//
//  TWO SIGNALS, one portable and one host-rich:
//   1. STATIC host state (from bench_report.h MachineState, detected at runtime —
//      never hardcoded): background load (loadavg1 per logical CPU) on every OS;
//      and, where the OS exposes them, the x86-box knobs — scaling governor must
//      be "performance", intel_pstate turbo state known, hottest thermal zone
//      under a ceiling. macOS exposes none of the frequency/thermal knobs (no
//      cpufreq governor, SMC thermal needs entitlements), so there the gate is
//      HONEST: it checks only what it can (load) and marks thermal "n/a".
//   2. A QUIESCENCE PROBE that works the SAME on Mac and x86: time a fixed,
//      deterministic compute kernel a handful of times and look at the SPREAD
//      (max/min). On a quiet, un-throttled core the spread is tiny; background
//      contention or thermal throttling steals cycles unevenly and the spread
//      blows past the ceiling. This is the signal that bites on macOS where the
//      governor/thermal knobs are blind — and it is what the gate self-test
//      deliberately trips by loading the machine.
//
//  Nothing here hardcodes core count, width, cache, or ISA: topology comes from
//  MachineState (runtime sysctl/procfs) and the probe is width-agnostic scalar
//  arithmetic. The same gate reruns on the x86 box unchanged (M5).
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "bench_report.h"  // MachineState, JsonWriter

namespace qe::bench {

// Thresholds the gate judges against — named, not magic, and all overridable so a
// noisier CI host can loosen them explicitly rather than silently. Defaults are
// tuned for an idle developer machine.
struct GateConfig {
    // Reject if loadavg1 / logical_cpus exceeds this (background load).
    double max_loadavg_per_cpu = 0.60;
    // Reject if the quiescence probe's max/min sample ratio exceeds this
    // (un-quiet core: contention or thermal throttle stealing cycles unevenly).
    // Platform-defaulted HONESTLY (§2): macOS cannot pin a thread and migrates it
    // across heterogeneous P/E cores, so a quiet Mac core still shows ~1.5x jitter
    // — the ceiling there is looser AND every Mac run is already labeled
    // preliminary. The x86 box pins to one fixed-frequency core, so it is strict;
    // that is where the credible absolute timing lives.
#if defined(__APPLE__)
    double max_probe_spread = 2.00;
#else
    double max_probe_spread = 1.35;
#endif
    // x86 thermal ceiling, °C (hottest zone). Ignored where temp is n/a (macOS).
    long max_temp_c = 90;
    // x86: require a fixed-frequency governor for credible absolute timing. macOS
    // has no governor concept, so this is enforced only where governors are read.
    bool require_perf_governor = true;

    // Probe shape: odd rep count (median is well-defined); work units per rep big
    // enough to be many microseconds yet sub-millisecond on a modern core.
    int probe_reps = 11;
    std::uint64_t probe_work = 1u << 20;
};

// The gate's decision plus the evidence behind it (all serialized into the run).
struct GateVerdict {
    bool accepted = false;
    std::vector<std::string> reasons;  // why rejected; empty iff accepted

    // Evidence.
    double loadavg_per_cpu = 0.0;
    double probe_spread = 0.0;  // max/min of the probe samples
    std::uint64_t probe_min_ns = 0;
    std::uint64_t probe_max_ns = 0;
    std::uint64_t probe_median_ns = 0;
    bool thermal_checked = false;  // false on macOS (knob is n/a)
    bool governor_checked = false;
};

// Run the quiescence probe alone (exposed for the self-test): returns the per-rep
// nanosecond samples of a fixed deterministic compute. `sink` is XORed with the
// kernel output so the optimizer cannot elide the work.
std::vector<std::uint64_t> runQuiescenceProbe(const GateConfig& cfg,
                                              std::uint64_t& sink);

// Evaluate the gate against an already-captured host state. Pure decision given
// `m` + a fresh probe; does not capture the machine itself (the caller does, so
// the SAME MachineState is what gets written into the run record).
GateVerdict evaluateGate(const MachineState& m, const GateConfig& cfg = {});

// Serialize a verdict as a JSON object (caller has already written the key).
void writeGateVerdict(JsonWriter& w, const GateVerdict& v);

}  // namespace qe::bench
