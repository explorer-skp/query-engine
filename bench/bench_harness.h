//  WP-10: shared harness helpers — the reusable spine every WP-10 bench driver is
//  built from, so the timing discipline (raw samples in, percentiles out) and the
//  honesty discipline (host+isa tags, Mac preliminary, relative ratios) live in
//  ONE place and cannot drift between operators.
//
//  WHAT THIS GIVES A DRIVER:
//   * timeSamples()  — run a callable N times, recording RAW per-call latencies
//     into a LatencyHistogram (warmup excluded). Percentiles are computed from the
//     samples LATER (bench_report.h), never averaged at capture time (rule 7).
//   * throughputFrom() — rows/s AND GB/s scanned, derived from a representative
//     latency and the known work size.
//   * classifyRoofline() — arithmetic-intensity framing (flops/byte) + a compute-
//     vs-bandwidth label. HONESTY (§2): on Mac the machine-balance ridge point is
//     ASSUMED, not measured, so the classification is flagged preliminary; the
//     credible roofline is the x86 box.
//   * writeVecScalarDoc() — emit ONE complete vec-vs-scalar JSON document in the
//     bench_sort schema family (host/isa/seed/preliminary/machine_start/validity/
//     percentile blocks + the scalar/vec ratio), so the plot layer reads every
//     operator uniformly.
//
//  Nothing here references an engine type, a vector width, a cache size, or a core
//  count: it is pure timing+reporting, so it reruns on x86 unchanged.
#pragma once

#include <cstdint>
#include <string_view>

#include "bench_report.h"
#include "bench_util.h"  // nowNs (timeSamples records raw per-call latencies)
#include "histogram.h"
#include "validity_gate.h"

namespace qe::bench {

// Run `f` `iters` times after `warmup` untimed reps, recording each call's
// nanosecond latency as a RAW sample. `f` returns a checksum that is XORed into
// `checksum` so dead-store elimination cannot delete the work being timed.
template <typename Fn>
qe::metrics::LatencyHistogram timeSamples(int iters, int warmup,
                                          std::uint64_t& checksum, Fn&& f) {
    for (int w = 0; w < warmup; ++w) checksum ^= f();
    qe::metrics::LatencyHistogram h;
    for (int it = 0; it < iters; ++it) {
        const std::uint64_t t0 = nowNs();
        const std::uint64_t c = f();
        const std::uint64_t t1 = nowNs();
        h.record(t1 - t0);
        checksum ^= c;
    }
    return h;
}

// rows/s and GB/s from a representative per-call latency (ns) and the work moved.
struct Throughput {
    double rows_per_s = 0.0;
    double gb_per_s = 0.0;  // GB = 1e9 bytes (decimal, stated)
    std::uint64_t bytes_scanned = 0;
};

Throughput throughputFrom(std::uint64_t rows, std::uint64_t bytesScanned,
                          double latencyNs);

// Arithmetic-intensity roofline framing. `preliminary` is set on hosts where the
// ridge point is assumed rather than measured (macOS — §2).
struct Roofline {
    std::uint64_t flops = 0;
    std::uint64_t bytes = 0;
    double arithmetic_intensity = 0.0;  // flops / bytes
    double ridge_flops_per_byte = 0.0;  // machine balance (assumed on Mac)
    bool compute_bound = false;         // AI > ridge
    bool preliminary = true;
};

Roofline classifyRoofline(std::uint64_t flops, std::uint64_t bytes,
                          double ridge_flops_per_byte, bool preliminary);

// scalar/vec (or slow/fast) speedup at quantile q, from the raw histograms.
double speedupAt(const qe::metrics::LatencyHistogram& slow,
                 const qe::metrics::LatencyHistogram& fast, double q);

// A fully-derived vec-vs-scalar operator measurement, ready to serialize.
struct VecScalarResult {
    qe::metrics::LatencyHistogram vec;
    qe::metrics::LatencyHistogram scalar;
    std::uint64_t rows = 0;
    std::uint64_t bytes_scanned = 0;  // bytes moved per call (for GB/s)
    Roofline roofline;
};

// Emit a complete vec-vs-scalar JSON doc (kind="bench_vec_scalar"). `op` is the
// operator family ("expr"/"hashtable"/"aggregate"/"sort"); `label` the run label.
// `valid` is gate.accepted AND-ed with any caller condition: a rejected run is
// written with valid=false and NEVER reported as a credible ratio.
void writeVecScalarDoc(JsonWriter& w, std::string_view label, std::string_view op,
                       std::string_view host, std::string_view isa,
                       std::uint64_t seed, bool preliminary,
                       std::string_view note, int iters, std::uint64_t checksum,
                       const MachineState& machine, const GateVerdict& gate,
                       const VecScalarResult& r);

// Writes the shared top-of-document tags (kind/label/host/isa/seed/preliminary/
// note). Caller has already called beginObject(). Used by drivers that emit a
// non-vec/scalar shape (engine-vs-duckdb, sustained CO) but want identical tags.
void writeRunTags(JsonWriter& w, std::string_view kind, std::string_view label,
                  std::string_view host, std::string_view isa,
                  std::uint64_t seed, bool preliminary, std::string_view note);

}  // namespace qe::bench
