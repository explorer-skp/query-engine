//  WP-10: the validity gate implementation. See validity_gate.h for the rationale.
#include "validity_gate.h"

#include <algorithm>
#include <cstdio>

#include "bench_util.h"  // nowNs

namespace qe::bench {

namespace {

// A fixed, deterministic, width-agnostic compute kernel: a dependent multiply-xor
// chain so the optimizer cannot vectorize away the data dependency, and the whole
// loop cannot be folded to a constant (state feeds the return value -> the caller
// XORs it into a sink). Pure scalar arithmetic: no ISA/width assumption, so the
// probe behaves the same on NEON and AVX-512.
std::uint64_t probeKernel(std::uint64_t work, std::uint64_t seed) {
    std::uint64_t x = seed | 1ull;
    for (std::uint64_t i = 0; i < work; ++i) {
        x ^= x >> 12;
        x *= 0x2545F4914F6CDD1Dull;
        x += i;
    }
    return x;
}

}  // namespace

std::vector<std::uint64_t> runQuiescenceProbe(const GateConfig& cfg,
                                              std::uint64_t& sink) {
    const int reps = cfg.probe_reps > 0 ? cfg.probe_reps : 1;
    std::vector<std::uint64_t> samples;
    samples.reserve(static_cast<std::size_t>(reps));
    // One warmup rep (page-in, frequency ramp) excluded from the samples.
    sink ^= probeKernel(cfg.probe_work, 0xC0FFEEull);
    for (int r = 0; r < reps; ++r) {
        const std::uint64_t t0 = nowNs();
        sink ^= probeKernel(cfg.probe_work, static_cast<std::uint64_t>(r) + 1);
        const std::uint64_t t1 = nowNs();
        samples.push_back(t1 - t0);
    }
    return samples;
}

GateVerdict evaluateGate(const MachineState& m, const GateConfig& cfg) {
    GateVerdict v;

    // --- Signal 1: static host state ---------------------------------------
    const int cpus = m.logicalCpus > 0 ? m.logicalCpus : 1;
    v.loadavg_per_cpu = m.loadavg1 / static_cast<double>(cpus);
    if (v.loadavg_per_cpu > cfg.max_loadavg_per_cpu) {
        char buf[160];
        std::snprintf(buf, sizeof(buf),
                      "background load: loadavg1/cpu=%.2f > %.2f",
                      v.loadavg_per_cpu, cfg.max_loadavg_per_cpu);
        v.reasons.emplace_back(buf);
    }

    // Governor / turbo: only meaningful where the OS exposes cpufreq (x86 Linux).
    // macOS reports "n/a-macOS" and we honestly skip it rather than pretend.
    if (m.governors != "n/a" && m.governors != "n/a-macOS" && !m.governors.empty()) {
        v.governor_checked = true;
        if (cfg.require_perf_governor && m.governors.find("performance") ==
                                             std::string::npos) {
            v.reasons.emplace_back("governor not 'performance': " + m.governors);
        }
    }

    // Thermal: -1 means n/a (macOS). >= 0 is a real reading we can ceiling.
    if (m.maxTempC >= 0) {
        v.thermal_checked = true;
        if (m.maxTempC > cfg.max_temp_c) {
            char buf[120];
            std::snprintf(buf, sizeof(buf), "thermal: %ld°C > %ld°C ceiling",
                          m.maxTempC, cfg.max_temp_c);
            v.reasons.emplace_back(buf);
        }
    }

    // --- Signal 2: quiescence probe (portable; the macOS-side teeth) --------
    std::uint64_t sink = 0;
    std::vector<std::uint64_t> s = runQuiescenceProbe(cfg, sink);
    // Touch the sink so the probe cannot be optimized to nothing.
    if (sink == 0xDEADBEEFull) std::fprintf(stderr, " ");  // never taken in practice
    std::sort(s.begin(), s.end());
    v.probe_min_ns = s.front();
    v.probe_max_ns = s.back();
    v.probe_median_ns = s[s.size() / 2];
    v.probe_spread = v.probe_min_ns > 0
                         ? static_cast<double>(v.probe_max_ns) /
                               static_cast<double>(v.probe_min_ns)
                         : 0.0;
    if (v.probe_spread > cfg.max_probe_spread) {
        char buf[160];
        std::snprintf(buf, sizeof(buf),
                      "probe jitter: max/min=%.2f > %.2f (core not quiescent: "
                      "contention or thermal throttle)",
                      v.probe_spread, cfg.max_probe_spread);
        v.reasons.emplace_back(buf);
    }

    v.accepted = v.reasons.empty();
    return v;
}

void writeGateVerdict(JsonWriter& w, const GateVerdict& v) {
    w.beginObject();
    w.kv("accepted", v.accepted);
    w.kv("loadavg_per_cpu", v.loadavg_per_cpu);
    w.kv("probe_spread", v.probe_spread);
    w.kv("probe_min_ns", v.probe_min_ns);
    w.kv("probe_median_ns", v.probe_median_ns);
    w.kv("probe_max_ns", v.probe_max_ns);
    w.kv("thermal_checked", v.thermal_checked);
    w.kv("governor_checked", v.governor_checked);
    w.key("reasons");
    w.beginArray();
    for (const auto& r : v.reasons) w.value(r);
    w.endArray();
    w.endObject();
}

}  // namespace qe::bench
